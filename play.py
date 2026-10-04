"""
Evaluation and Play Script for Flappy Bird DQN Agent.

Runs the trained DQN agent or a baseline random agent in the Pygame visual environment.
"""

import argparse
import os
import sys
import time
import numpy as np
import pygame
import torch

# Ensure current workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.dqn import DQN, normalize_observation
from env.flappy_bird_env import FlappyBirdEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Flappy Bird")
    parser.add_argument(
        "--model",
        type=str,
        default="models/flappy_dqn.pth",
        help="Path to the model checkpoint (.pth). Defaults to models/flappy_dqn.pth"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play (default: 5)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering (default: 30)")
    parser.add_argument("--random", action="store_true", help="Run a random agent instead of loading a model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (default: cpu)")
    return parser.parse_args()


def load_model(model_path: str, state_dim: int, action_dim: int, device: torch.device) -> DQN:
    """Load DQN model weights from file."""
    model = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at: '{model_path}'.\n"
            f"Please run `train.py` first to train a model, or pass `--random` to watch random actions."
        )

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        score = checkpoint.get("score", "N/A")
        ep = checkpoint.get("episode", "N/A")
        print(f"Loaded checkpoint from {model_path} (Episode: {ep}, Best Score: {score})")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {model_path}")

    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    env = FlappyBirdEnv(render_mode="human")
    env.metadata["render_fps"] = args.fps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = None
    if not args.random:
        # Determine model path (defaults to models/flappy_dqn.pth, with fallback if needed)
        model_path = args.model
        if not os.path.exists(model_path):
            if os.path.exists("models/flappy_bird_dqn_best.pth"):
                model_path = "models/flappy_bird_dqn_best.pth"
            elif os.path.exists("models/flappy_bird_dqn_final.pth"):
                model_path = "models/flappy_bird_dqn_final.pth"

        try:
            model = load_model(model_path, state_dim, action_dim, device)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            env.close()
            sys.exit(1)
    else:
        print("Running in RANDOM AGENT mode (baseline)...")

    print("\n" + "=" * 55)
    print(f"Playing {args.episodes} episode(s) at {args.fps} FPS")
    print("Press window close button [X] at any time to exit.")
    print("=" * 55 + "\n")

    episode_scores = []
    episode_rewards = []
    episode_steps = []

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Pygame window exit handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nWindow closed by user. Exiting...")
                        env.close()
                        return

                # Choose action
                if model is not None:
                    with torch.no_grad():
                        norm_obs = normalize_observation(obs)
                        state_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values = model(state_tensor)
                        action = int(q_values.argmax(dim=1).item())
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            score = info.get("score", 0)
            episode_scores.append(score)
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            print(f"Episode {ep:2d}/{args.episodes} | Score: {score:2d} | Reward: {total_reward:6.2f} | Steps: {steps:3d}")
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nPlay session interrupted by user.")
    finally:
        env.close()

    if episode_scores:
        print("\n" + "=" * 55)
        print("SUMMARY PERFORMANCE:")
        print(f"  Episodes Played : {len(episode_scores)}")
        print(f"  Scores          : {episode_scores}")
        print(f"  Max Score       : {max(episode_scores)}")
        print(f"  Mean Score      : {np.mean(episode_scores):.2f}")
        print(f"  Mean Reward     : {np.mean(episode_rewards):.2f}")
        print(f"  Mean Steps      : {np.mean(episode_steps):.1f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
