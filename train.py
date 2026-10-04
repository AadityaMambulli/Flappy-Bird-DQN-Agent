"""
Deep Q-Network (DQN) Training Script for Flappy Bird.

This script trains a DQN agent to play Flappy Bird using experience replay,
a target network for Q-value stability, epsilon-greedy exploration, and Huber loss.
"""

import argparse
import os
import random
import shutil
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure current workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.dqn import DQN, normalize_observation
from agent.replay_buffer import ReplayBuffer
from env.flappy_bird_env import FlappyBirdEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Flappy Bird")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer (default: 1e-3)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer capacity (default: 50000)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Random exploration steps before training (default: 500)")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration (default: 1.0)")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon for exploration (default: 0.01)")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, help="Epsilon multiplicative decay per episode (default: 0.998)")
    parser.add_argument("--target_update_freq", type=int, default=10, help="Frequency (in episodes) to update target network (default: 10)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints (default: models)")
    parser.add_argument("--save_freq", type=int, default=100, help="Checkpoint save interval in episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--render", action="store_true", help="Render environment during training (slows down training)")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action(state: np.ndarray, policy_net: nn.Module, epsilon: float, action_space, device: torch.device) -> int:
    """Select action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return action_space.sample()

    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
    gamma: float,
    device: torch.device
) -> float:
    """Sample a batch from replay buffer and perform one gradient descent step."""
    if len(replay_buffer) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device=device)

    # Current Q-values predicted by policy net for the taken actions
    current_q = policy_net(states).gather(1, actions)

    # Compute target Q-values using target net (Bellman optimality equation)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (1.0 - dones) * gamma * max_next_q

    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())


def plot_training_results(
    scores: list,
    rewards: list,
    losses: list,
    epsilons: list,
    save_path: str
):
    """Plot and save training performance curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scores (Pipes passed)
    axes[0, 0].plot(scores, label="Episode Score", alpha=0.4, color="tab:blue")
    if len(scores) >= 20:
        rolling_score = np.convolve(scores, np.ones(20) / 20, mode="valid")
        axes[0, 0].plot(range(19, len(scores)), rolling_score, label="20-Ep Moving Avg", color="tab:blue", linewidth=2)
    axes[0, 0].set_title("Pipes Passed per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode Rewards
    axes[0, 1].plot(rewards, label="Episode Reward", alpha=0.4, color="tab:green")
    if len(rewards) >= 20:
        rolling_reward = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        axes[0, 1].plot(range(19, len(rewards)), rolling_reward, label="20-Ep Moving Avg", color="tab:green", linewidth=2)
    axes[0, 1].set_title("Total Reward per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Training Loss
    axes[1, 0].plot(losses, label="Mean Loss", alpha=0.7, color="tab:red")
    axes[1, 0].set_title("Loss per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Huber Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Epsilon Decay
    axes[1, 1].plot(epsilons, label="Epsilon", color="tab:orange", linewidth=2)
    axes[1, 1].set_title("Epsilon Exploration Schedule")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon (ε)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Flappy Bird Reinforcement Learning - DQN Training")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Gamma: {args.gamma} | Buffer Capacity: {args.buffer_size}")
    print(f"Warmup Steps: {args.warmup_steps} | Target Update Freq: {args.target_update_freq} eps")
    print(f"Checkpoint Directory: {args.save_dir}")
    print("=" * 60)

    render_mode = "human" if args.render else None
    env = FlappyBirdEnv(render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    epsilon = args.epsilon_start
    total_steps = 0
    best_score = 0
    best_episode = 1
    best_avg_score = 0.0

    recent_scores = deque(maxlen=20)
    recent_rewards = deque(maxlen=20)

    all_scores = []
    all_rewards = []
    all_losses = []
    all_epsilons = []

    # Warmup Phase: Seed replay with transitions from a gap-aware controller
    print(f"\n[Phase 1] Collecting {args.warmup_steps} warmup transitions into Replay Buffer...")
    warmup_obs, _ = env.reset(seed=args.seed)
    warmup_pipe_passes = 0
    warmup_deaths = 0
    warmup_alive_rewards = 0
    for _ in range(args.warmup_steps):
        bird_y, bird_velocity, _, vertical_distance = warmup_obs
        gap_y = bird_y - vertical_distance
        projected_y = bird_y + 3 * bird_velocity + 0.4 * 3 * 4
        action = int(projected_y > gap_y + 20)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        norm_warmup_obs = normalize_observation(warmup_obs)
        norm_next_obs = normalize_observation(next_obs)
        replay_buffer.push(norm_warmup_obs, action, reward, norm_next_obs, done)
        if reward == 1.0:
            warmup_pipe_passes += 1
        elif reward == -10.0:
            warmup_deaths += 1
        elif reward == 0.01:
            warmup_alive_rewards += 1
        total_steps += 1
        if done:
            warmup_obs, _ = env.reset()
        else:
            warmup_obs = next_obs
    print(
        f"Warmup completed. Buffer size: {len(replay_buffer)} | "
        f"Transitions: {args.warmup_steps} | "
        f"Pipe passes: {warmup_pipe_passes} | "
        f"Deaths: {warmup_deaths} | "
        f"Alive rewards: {warmup_alive_rewards}"
    )

    # Main Training Loop
    print("\n[Phase 2] Starting Main Training Loop...")
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        optimization_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            norm_obs = normalize_observation(obs)
            action = select_action(norm_obs, policy_net, epsilon, env.action_space, device)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            norm_next_obs = normalize_observation(next_obs)
            replay_buffer.push(norm_obs, action, reward, norm_next_obs, done)
            obs = next_obs
            total_steps += 1
            episode_reward += reward

            # Train network
            loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device=device
            )
            if loss > 0.0:
                episode_loss += loss
                optimization_steps += 1

        # Episode finished
        ep_score = info["score"]
        recent_scores.append(ep_score)
        recent_rewards.append(episode_reward)

        avg_loss = (episode_loss / optimization_steps) if optimization_steps > 0 else 0.0
        all_scores.append(ep_score)
        all_rewards.append(episode_reward)
        all_losses.append(avg_loss)
        all_epsilons.append(epsilon)

        avg_score = float(np.mean(recent_scores))
        avg_reward = float(np.mean(recent_rewards))

        # Update target network periodically
        if episode % args.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Logging
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:4d}/{args.episodes:4d} | "
                f"Score: {ep_score:2d} (Best: {best_score:2d}, Avg20: {avg_score:.2f}) | "
                f"Reward: {episode_reward:6.2f} (Avg20: {avg_reward:6.2f}) | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Steps: {total_steps} | "
                f"Time: {elapsed:.1f}s"
            )

        best_path = os.path.join(args.save_dir, "flappy_bird_dqn_best.pth")
        # Check and save best model
        if not os.path.exists(best_path) or ep_score > best_score or (ep_score == best_score and avg_score > best_avg_score):
            best_score = max(best_score, ep_score)
            best_episode = episode
            best_avg_score = max(best_avg_score, avg_score)
            torch.save({
                "episode": episode,
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": best_score,
                "avg_score": best_avg_score,
                "epsilon": epsilon
            }, best_path)

        # Periodic checkpoint
        if episode % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f"flappy_bird_dqn_ep{episode}.pth")
            torch.save({
                "episode": episode,
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": ep_score,
                "avg_score": avg_score,
                "epsilon": epsilon
            }, ckpt_path)

    total_duration = time.time() - start_time

    # Save final model
    final_path = os.path.join(args.save_dir, "flappy_bird_dqn_final.pth")
    torch.save({
        "episode": args.episodes,
        "model_state_dict": policy_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": ep_score,
        "avg_score": avg_score,
        "epsilon": epsilon
    }, final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"Best model saved to {best_path} (Best Score: {best_score}, Best Episode: {best_episode})")

    # Save standard deployment model to models/flappy_dqn.pth
    standard_path = os.path.join(args.save_dir, "flappy_dqn.pth")
    source_path = best_path if os.path.exists(best_path) else final_path
    shutil.copyfile(source_path, standard_path)
    print(f"Deployment model saved to {standard_path}")

    # Plot training metrics
    plot_path = os.path.join(args.save_dir, "training_curves.png")
    plot_training_results(all_scores, all_rewards, all_losses, all_epsilons, plot_path)

    env.close()

    print("\n" + "=" * 60)
    print("TRAINING RUN SUMMARY:")
    print(f"  Total Episodes Trained    : {args.episodes}")
    print(f"  Best Score Achieved       : {best_score}")
    print(f"  Best Episode              : {best_episode}")
    print(f"  Final Episode Score       : {ep_score}")
    print(f"  Initial Reward (Ep 1)     : {all_rewards[0]:.2f}")
    print(f"  Final Reward (Ep {args.episodes})   : {all_rewards[-1]:.2f}")
    print(f"  Average Reward (First 50) : {np.mean(all_rewards[:50]):.2f}")
    print(f"  Average Reward (Last 50)  : {np.mean(all_rewards[-50:]):.2f}")
    print(f"  Standard Model Saved      : {os.path.exists(standard_path)} ({standard_path})")
    print(f"  Training Duration         : {total_duration:.2f}s ({total_duration/60:.2f}m)")
    print("=" * 60)
    print("Training finished successfully.")


if __name__ == "__main__":
    train()
