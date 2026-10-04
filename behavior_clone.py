"""Train the existing DQN network to imitate a gap-aware teacher policy."""

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.dqn import DQN, normalize_observation
from env.flappy_bird_env import FlappyBirdEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural policy with supervised teacher demonstrations."
    )
    parser.add_argument("--rollouts", type=int, default=256, help="Teacher episodes to generate")
    parser.add_argument("--max_steps", type=int, default=180, help="Maximum steps per teacher episode")
    parser.add_argument("--epochs", type=int, default=20, help="Supervised training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--validation_fraction", type=float, default=0.2, help="Episode-level validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="models/flappy_bc.pth", help="New checkpoint path")
    return parser.parse_args()


def teacher_action(observation: np.ndarray) -> int:
    """Apply the verified three-step gap-aware controller to a raw observation."""
    bird_y, bird_velocity, _, vertical_distance = observation
    gap_y = bird_y - vertical_distance
    projected_y = bird_y + 3 * bird_velocity + 0.4 * 3 * 4
    return int(projected_y > gap_y + 20)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_demonstrations(env, rollout_count: int, max_steps: int, seed: int):
    successful_rollouts = []
    total_pipe_passes = 0

    for rollout_index in range(rollout_count):
        observation, _ = env.reset(seed=seed + rollout_index)
        states = []
        actions = []
        episode_passes = 0

        for _ in range(max_steps):
            action = teacher_action(observation)
            states.append(normalize_observation(observation))
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            if reward == 1.0:
                episode_passes += 1
            if terminated or truncated:
                break

        if episode_passes:
            successful_rollouts.append(
                (np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64))
            )
            total_pipe_passes += episode_passes

    return successful_rollouts, total_pipe_passes


def flatten_rollouts(rollouts):
    states = np.concatenate([rollout[0] for rollout in rollouts], axis=0)
    actions = np.concatenate([rollout[1] for rollout in rollouts], axis=0)
    return states, actions


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            logits = model(states)
            total_loss += loss_fn(logits, actions).item() * len(actions)
            correct += (logits.argmax(dim=1) == actions).sum().item()
            sample_count += len(actions)

    return total_loss / sample_count, correct / sample_count


def main():
    args = parse_args()
    if args.rollouts < 2 or args.max_steps <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("rollouts must be at least 2; steps, epochs, and batch size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlappyBirdEnv()
    try:
        rollouts, pipe_passes = collect_demonstrations(
            env, args.rollouts, args.max_steps, args.seed
        )
    finally:
        env.close()

    if len(rollouts) < 2:
        raise RuntimeError(
            f"Only {len(rollouts)} successful teacher rollouts; need at least 2 for a train/validation split."
        )

    permutation = np.random.permutation(len(rollouts))
    validation_count = max(1, int(round(len(rollouts) * args.validation_fraction)))
    validation_count = min(validation_count, len(rollouts) - 1)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    training_rollouts = [rollouts[index] for index in training_indices]
    validation_rollouts = [rollouts[index] for index in validation_indices]
    training_states, training_actions = flatten_rollouts(training_rollouts)
    validation_states, validation_actions = flatten_rollouts(validation_rollouts)

    action_counts = np.bincount(
        np.concatenate([rollout[1] for rollout in rollouts]), minlength=2
    )
    print(
        f"Teacher data: {len(rollouts)} successful rollouts, {len(training_actions) + len(validation_actions)} "
        f"samples, {pipe_passes} pipe passes"
    )
    print(f"Action counts: 0={action_counts[0]}, 1={action_counts[1]}")
    print(f"Train samples: {len(training_actions)} | Validation samples: {len(validation_actions)}")

    train_dataset = TensorDataset(
        torch.from_numpy(training_states), torch.from_numpy(training_actions)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_states), torch.from_numpy(validation_actions)
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=loader_generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = DQN(state_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(states), actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(actions)
            sample_count += len(actions)

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn, device
        )
        training_loss = total_loss / sample_count
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.3f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Supervised training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    final_validation_loss, final_validation_accuracy = evaluate(
        model, validation_loader, loss_fn, device
    )
    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "validation_loss": final_validation_loss,
            "validation_accuracy": final_validation_accuracy,
            "sample_count": len(training_actions) + len(validation_actions),
            "action_counts": action_counts.tolist(),
            "seed": args.seed,
        },
        args.output,
    )
    print(
        f"Saved behavior-cloned policy to {args.output} | "
        f"Validation loss: {final_validation_loss:.4f} | "
        f"Validation accuracy: {final_validation_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
