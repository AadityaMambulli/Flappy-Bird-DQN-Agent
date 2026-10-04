"""Run the behavior-cloned neural policy in the rendered Flappy Bird game."""

import argparse
import os
import random
import secrets
import sys

import numpy as np
import pygame
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.dqn import DQN, normalize_observation
from env.flappy_bird_env import FlappyBirdEnv


MODEL_PATH = os.path.join("models", "flappy_bc.pth")


class BehaviorCloneVisualEnv(FlappyBirdEnv):
    """Render the existing game surface centered inside a resizable window."""

    def __init__(self):
        super().__init__(render_mode="rgb_array")
        pygame.init()
        pygame.display.set_caption("Flappy Bird - Behavior Cloned AI")
        self.display_surface = pygame.display.set_mode((432, 768), pygame.RESIZABLE)
        self.window = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22, bold=True)
        self.status_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.episode_number = 1
        self.best_score = 0
        self.completed_scores = []
        self.total_pipe_passes = 0

    def render(self):
        rendered_frame = super().render()
        del rendered_frame

        average_score = (
            sum(self.completed_scores) / len(self.completed_scores)
            if self.completed_scores
            else 0.0
        )
        lines = (
            "AI: Behavior Cloned Neural Network",
            f"Episode: {self.episode_number}",
            f"Score: {self.score}",
            f"Best: {self.best_score}",
            f"Average: {average_score:.2f}",
            f"Total pipe passes: {self.total_pipe_passes}",
        )
        for index, text in enumerate(lines):
            label = self.status_font.render(text, True, (255, 255, 255))
            self.window.blit(label, (15, 45 + index * 20))

        display_width, display_height = self.display_surface.get_size()
        scale = min(
            display_width / self.screen_width,
            display_height / self.screen_height,
        )
        scaled_size = (
            max(1, round(self.screen_width * scale)),
            max(1, round(self.screen_height * scale)),
        )
        scaled_game = pygame.transform.scale(self.window, scaled_size)
        self.display_surface.fill((24, 29, 33))
        destination = (
            (display_width - scaled_size[0]) // 2,
            (display_height - scaled_size[1]) // 2,
        )
        self.display_surface.blit(scaled_game, destination)
        pygame.display.flip()

    def resize_display(self, size):
        self.display_surface = pygame.display.set_mode(size, pygame.RESIZABLE)

    def close(self):
        super().close()
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Flappy Bird autonomously with the behavior-cloned network."
    )
    parser.add_argument("--fps", type=int, default=30, help="Render frame rate")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base environment seed (omit for a fresh random seed each launch)",
    )
    return parser.parse_args()


def load_policy(device: torch.device) -> DQN:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Behavior-cloning checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid behavior-cloning checkpoint: {MODEL_PATH}")

    model = DQN(state_dim=4, action_dim=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded autonomous neural policy from {MODEL_PATH}")
    return model


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    actual_seed = args.seed if args.seed is not None else secrets.randbits(32)
    if not 0 <= actual_seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(device)

    env = BehaviorCloneVisualEnv()
    env.metadata["render_fps"] = args.fps
    action_counts = [0, 0]
    episode = 1
    previous_score = 0
    episode_seed = actual_seed
    env.episode_number = episode
    print(f"Using base seed: {actual_seed}")
    print(f"Episode {episode} seed: {episode_seed}")
    observation, _ = env.reset(seed=episode_seed)
    env.render()
    print("AI: Behavior Cloned Neural Network | Close window to exit | R to restart")
    print(f"Human display opened: {pygame.display.get_surface() is not None}")

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    env.resize_display(event.size)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            normalized_state = normalize_observation(observation)
            state_tensor = torch.as_tensor(
                normalized_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state_tensor).argmax(dim=1).item())
            action_counts[action] += 1

            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            score = int(info.get("score", 0))
            if score > previous_score:
                passed = score - previous_score
                env.total_pipe_passes += passed
                print(
                    f"Episode {episode}: neural policy passed pipe "
                    f"(score {score}, action {action}, reward {reward})"
                )
                previous_score = score

            if terminated or truncated:
                env.completed_scores.append(score)
                env.best_score = max(env.best_score, score)
                print(
                    f"Episode {episode} ended | Score: {score} | "
                    f"Actions: do nothing={action_counts[0]}, flap={action_counts[1]}"
                )
                print("Press R to restart or close the window to exit.")
                env.render()
                while running:
                    restart = False
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.VIDEORESIZE:
                            env.resize_display(event.size)
                            env.render()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_r:
                                restart = True
                                break
                    env.clock.tick(args.fps)
                    if restart:
                        episode += 1
                        episode_seed = (actual_seed + episode - 1) % 2**32
                        random.seed(episode_seed)
                        np.random.seed(episode_seed)
                        env.episode_number = episode
                        print(f"Episode {episode} seed: {episode_seed}")
                        observation, _ = env.reset(seed=episode_seed)
                        env.render()
                        previous_score = 0
                        break
    finally:
        env.close()

    print(f"Total pipe passes: {env.total_pipe_passes}")
    print(f"Neural action counts: do nothing={action_counts[0]}, flap={action_counts[1]}")


if __name__ == "__main__":
    main()
