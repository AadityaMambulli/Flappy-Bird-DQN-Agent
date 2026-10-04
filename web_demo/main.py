"""Pygbag entry point for the autonomous behavior-cloned Flappy Bird demo."""

import asyncio
import os
import random
import secrets

import pygame

from agent.dqn import DQN, argmax, normalize_observation
from env.flappy_bird_env import FlappyBirdEnv


SCREEN_SIZE = (288, 512)
FPS = 30


def load_policy():
    app_dir = os.path.dirname(__file__)
    model_path = os.path.join(app_dir, "models", "flappy_bc.json")
    model = DQN(state_dim=4, action_dim=2)
    model.load_state_dict(model_path)
    model.eval()
    print(f"Loaded behavior-cloned neural policy from {model_path}")
    return model


def draw_status(surface, font, episode, score, best_score, completed_scores, total_passes):
    average_score = (
        sum(completed_scores) / len(completed_scores) if completed_scores else 0.0
    )
    lines = (
        "AI: Behavior Cloned Neural Network",
        f"Episode: {episode}",
        f"Score: {score}",
        f"Best: {best_score}",
        f"Average: {average_score:.2f}",
        f"Total pipe passes: {total_passes}",
    )
    panel = pygame.Surface((SCREEN_SIZE[0], 92), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    surface.blit(panel, (0, 34))
    for row, text in enumerate(lines):
        label = font.render(text, True, (255, 255, 255))
        surface.blit(label, (7, 37 + row * 14))


async def main():
    pygame.init()
    pygame.display.set_caption("Flappy Bird - Behavior Cloned AI")
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 11, bold=True)
    env = FlappyBirdEnv()
    model = load_policy()

    seed_override = os.environ.get("FLAPPY_BC_SEED")
    base_seed = (
        int(seed_override) % 2**32
        if seed_override is not None
        else secrets.randbits(32)
    )
    print(f"Using base environment seed: {base_seed}")

    episode = 1
    episode_seed = base_seed
    random.seed(episode_seed)
    observation = env.reset(seed=episode_seed)
    best_score = 0
    completed_scores = []
    total_pipe_passes = 0
    previous_score = 0
    game_over = False
    running = True

    while running:
        restart = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif game_over and event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                    restart = True
            elif game_over and event.type == pygame.MOUSEBUTTONDOWN:
                restart = True

        if not running:
            break

        if restart:
            episode += 1
            episode_seed = (base_seed + episode - 1) % 2**32
            random.seed(episode_seed)
            observation = env.reset(seed=episode_seed)
            previous_score = 0
            game_over = False
            print(f"Episode {episode} seed: {episode_seed}")

        if not game_over:
            normalized_state = normalize_observation(observation)
            action = argmax(model(normalized_state))
            observation, reward, terminated, truncated, info = env.step(action)
            score = info["score"]
            if score > previous_score:
                total_pipe_passes += score - previous_score
                print(f"Episode {episode}: policy passed a pipe (reward={reward})")
                previous_score = score
            if terminated or truncated:
                completed_scores.append(score)
                best_score = max(best_score, score)
                game_over = True
                print(f"Episode {episode} ended with score {score}")

        env.render(screen)
        draw_status(
            screen,
            font,
            episode,
            env.score,
            best_score,
            completed_scores,
            total_pipe_passes,
        )
        if game_over:
            prompt = font.render("Game over - press R or click to restart", True, (255, 255, 255))
            screen.blit(prompt, (8, SCREEN_SIZE[1] - 22))

        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(0)


asyncio.run(main())