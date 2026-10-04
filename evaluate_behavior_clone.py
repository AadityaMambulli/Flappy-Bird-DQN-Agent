"""Evaluate a behavior-cloned policy without invoking its teacher."""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.dqn import DQN, normalize_observation
from env.flappy_bird_env import FlappyBirdEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run autonomous argmax inference with a behavior-cloned policy."
    )
    parser.add_argument("--model", type=str, default="models/flappy_bc.pth", help="Behavior-cloning checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Behavior-cloning checkpoint not found: {args.model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid behavior-cloning checkpoint: {args.model}")

    model = DQN(state_dim=4, action_dim=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    env = FlappyBirdEnv(render_mode="human" if args.render else None)
    total_passes = 0
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                normalized_state = normalize_observation(observation)
                state_tensor = torch.as_tensor(
                    normalized_state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action = int(model(state_tensor).argmax(dim=1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            score = int(info.get("score", 0))
            total_passes += score
            print(
                f"Episode {episode + 1:02d}/{args.episodes} | "
                f"Score: {score} | Reward: {total_reward:.2f} | Steps: {steps}"
            )
    finally:
        env.close()

    print(f"Total pipe passes: {total_passes}")


if __name__ == "__main__":
    main()
