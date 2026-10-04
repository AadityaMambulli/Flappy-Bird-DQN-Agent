"""Browser-compatible copy of the project's Flappy Bird environment mechanics."""

import random
import struct

import pygame


class FlappyBirdEnv:
    def __init__(self):
        self.screen_width = 288
        self.screen_height = 512
        self.gravity = 0.8
        self.flap_strength = -7.5
        self.max_velocity = 10.0
        self.bird_x = 60
        self.bird_radius = 12
        self.pipe_width = 52
        self.pipe_gap = 105
        self.pipe_speed = 3.0
        self.min_pipe_y = 100
        self.max_pipe_y = 350
        self.pipe_spacing = 180
        self.font = None
        self.reset()

    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe["x"] + self.pipe_width >= self.bird_x - self.bird_radius:
                return pipe
        return self.pipes[0] if self.pipes else None

    def _get_obs(self):
        def float32(value):
            return struct.unpack("<f", struct.pack("<f", value))[0]

        next_pipe = self._get_next_pipe()
        if next_pipe is None:
            horizontal_distance = float(self.screen_width)
            vertical_distance = 0.0
        else:
            horizontal_distance = float(next_pipe["x"] - self.bird_x)
            vertical_distance = float(self.bird_y - next_pipe["gap_y"])
        return (
            float32(self.bird_y),
            float32(self.bird_velocity),
            float32(horizontal_distance),
            float32(vertical_distance),
        )

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.bird_y = self.screen_height / 2.0
        self.bird_velocity = 0.0
        self.score = 0
        self.step_count = 0
        first_pipe_x = 140
        gap_y1 = random.randint(self.min_pipe_y, self.max_pipe_y)
        self.pipes = [{"x": first_pipe_x, "gap_y": gap_y1, "scored": False}]
        gap_y2 = random.randint(self.min_pipe_y, self.max_pipe_y)
        self.pipes.append(
            {
                "x": first_pipe_x + self.pipe_spacing,
                "gap_y": gap_y2,
                "scored": False,
            }
        )
        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        reward = 0.01
        terminated = False
        truncated = False

        if action == 1:
            self.bird_velocity = self.flap_strength
        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity:
            self.bird_velocity = self.max_velocity
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["scored"] = True
                self.score += 1
                reward = 1.0

        if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
            self.pipes.pop(0)
            new_x = self.pipes[-1]["x"] + self.pipe_spacing
            new_gap_y = random.randint(self.min_pipe_y, self.max_pipe_y)
            self.pipes.append({"x": new_x, "gap_y": new_gap_y, "scored": False})

        if (
            self.bird_y - self.bird_radius <= 0
            or self.bird_y + self.bird_radius >= self.screen_height
        ):
            terminated = True
            reward = -10.0

        if not terminated:
            bird_rect = pygame.Rect(
                int(self.bird_x - self.bird_radius),
                int(self.bird_y - self.bird_radius),
                int(self.bird_radius * 2),
                int(self.bird_radius * 2),
            )
            for pipe in self.pipes:
                pipe_x = int(pipe["x"])
                gap_y = int(pipe["gap_y"])
                gap_half = self.pipe_gap / 2.0
                top_pipe_rect = pygame.Rect(
                    pipe_x, 0, self.pipe_width, int(gap_y - gap_half)
                )
                bottom_pipe_rect = pygame.Rect(
                    pipe_x,
                    int(gap_y + gap_half),
                    self.pipe_width,
                    int(self.screen_height - (gap_y + gap_half)),
                )
                if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(
                    bottom_pipe_rect
                ):
                    terminated = True
                    reward = -10.0
                    break

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self, surface):
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 22, bold=True)
        surface.fill((112, 197, 206))
        pipe_color = (115, 191, 46)
        pipe_border = (83, 140, 34)
        for pipe in self.pipes:
            pipe_x = int(pipe["x"])
            gap_y = int(pipe["gap_y"])
            gap_half = self.pipe_gap / 2.0
            top_height = int(gap_y - gap_half)
            if top_height > 0:
                pygame.draw.rect(
                    surface, pipe_color, (pipe_x, 0, self.pipe_width, top_height)
                )
                pygame.draw.rect(
                    surface,
                    pipe_border,
                    (pipe_x, 0, self.pipe_width, top_height),
                    2,
                )
                pygame.draw.rect(
                    surface,
                    pipe_color,
                    (pipe_x - 3, top_height - 16, self.pipe_width + 6, 16),
                )
                pygame.draw.rect(
                    surface,
                    pipe_border,
                    (pipe_x - 3, top_height - 16, self.pipe_width + 6, 16),
                    2,
                )
            bottom_y = int(gap_y + gap_half)
            bottom_height = int(self.screen_height - bottom_y)
            if bottom_height > 0:
                pygame.draw.rect(
                    surface,
                    pipe_color,
                    (pipe_x, bottom_y, self.pipe_width, bottom_height),
                )
                pygame.draw.rect(
                    surface,
                    pipe_border,
                    (pipe_x, bottom_y, self.pipe_width, bottom_height),
                    2,
                )
                pygame.draw.rect(
                    surface,
                    pipe_color,
                    (pipe_x, bottom_y, self.pipe_width + 6, 16),
                )
                pygame.draw.rect(
                    surface,
                    pipe_border,
                    (pipe_x, bottom_y, self.pipe_width + 6, 16),
                    2,
                )

        bird_center = (int(self.bird_x), int(self.bird_y))
        pygame.draw.circle(surface, (250, 218, 94), bird_center, self.bird_radius)
        pygame.draw.circle(surface, (220, 160, 20), bird_center, self.bird_radius, 2)
        pygame.draw.circle(
            surface, (255, 255, 255), (int(self.bird_x + 4), int(self.bird_y - 4)), 4
        )
        pygame.draw.circle(
            surface, (0, 0, 0), (int(self.bird_x + 4), int(self.bird_y - 4)), 2
        )
        pygame.draw.polygon(
            surface,
            (244, 114, 52),
            (
                (int(self.bird_x + 8), int(self.bird_y - 2)),
                (int(self.bird_x + 16), int(self.bird_y + 2)),
                (int(self.bird_x + 8), int(self.bird_y + 5)),
            ),
        )
        pygame.draw.rect(
            surface,
            (222, 216, 149),
            (0, self.screen_height - 12, self.screen_width, 12),
        )
        pygame.draw.line(
            surface,
            (115, 191, 46),
            (0, self.screen_height - 12),
            (self.screen_width, self.screen_height - 12),
            3,
        )
        score = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        surface.blit(score, (15, 15))
