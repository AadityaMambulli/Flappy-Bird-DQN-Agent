import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class FlappyBirdEnv(gym.Env):
    """
    Custom Flappy Bird Gymnasium Environment for Reinforcement Learning.
    
    Observation Space (Box(4,)):
        [0] bird_y: vertical position of the bird center
        [1] bird_velocity: current vertical velocity of the bird
        [2] horizontal_distance_to_pipe: horizontal distance from bird to the next upcoming pipe
        [3] vertical_distance_to_gap: vertical distance from bird to the center of the next pipe gap
        
    Action Space (Discrete(2)):
        0: Do nothing (let gravity pull the bird down)
        1: Flap (push the bird upward)
        
    Rewards:
        +0.01: staying alive each step
        +1.0 : successfully passing a pipe
        -10.0: dying (hitting the ground, ceiling, or a pipe)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Screen dimensions
        self.screen_width = 288
        self.screen_height = 512
        self.render_mode = render_mode
        
        # Physics constants
        self.gravity = 0.8
        self.flap_strength = -7.5
        self.max_velocity = 10.0
        self.bird_x = 60
        self.bird_radius = 12
        
        # Pipe constants
        self.pipe_width = 52
        self.pipe_gap = 105
        self.pipe_speed = 3.0
        self.min_pipe_y = 100
        self.max_pipe_y = 350
        self.pipe_spacing = 180  # horizontal distance between consecutive pipes

        # Spaces
        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation space: 4 continuous values
        # [bird_y, bird_velocity, horizontal_distance_to_pipe, vertical_distance_to_gap]
        # horizontal_distance can become negative as the pipe passes through the bird
        min_h_dist = -(self.pipe_width + self.bird_radius)
        low = np.array([0.0, -self.max_velocity, min_h_dist, -self.screen_height], dtype=np.float32)
        high = np.array([self.screen_height, self.max_velocity, self.screen_width + self.pipe_width, self.screen_height], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Pygame rendering attributes
        self.window = None
        self.clock = None
        self.font = None

        # Game state variables
        self.bird_y = self.screen_height / 2.0
        self.bird_velocity = 0.0
        self.pipes = []
        self.score = 0
        self.step_count = 0

    def _get_next_pipe(self):
        """Find the closest pipe ahead of the bird."""
        for pipe in self.pipes:
            # Check if bird is still to the left of the pipe's right edge
            if pipe["x"] + self.pipe_width >= (self.bird_x - self.bird_radius):
                return pipe
        # Fallback if no pipe is ahead
        return self.pipes[0] if self.pipes else None

    def _get_obs(self):
        """Construct the 4-element observation vector."""
        next_pipe = self._get_next_pipe()
        if next_pipe is not None:
            horizontal_distance = float(next_pipe["x"] - self.bird_x)
            vertical_distance = float(self.bird_y - next_pipe["gap_y"])
        else:
            horizontal_distance = float(self.screen_width)
            vertical_distance = 0.0

        obs = np.array([
            float(self.bird_y),
            float(self.bird_velocity),
            horizontal_distance,
            vertical_distance
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        
        self.bird_y = self.screen_height / 2.0
        self.bird_velocity = 0.0
        self.score = 0
        self.step_count = 0

        # Initialize first two pipes
        self.pipes = []
        first_pipe_x = 140
        gap_y1 = random.randint(self.min_pipe_y, self.max_pipe_y)
        self.pipes.append({"x": first_pipe_x, "gap_y": gap_y1, "scored": False})

        gap_y2 = random.randint(self.min_pipe_y, self.max_pipe_y)
        self.pipes.append({"x": first_pipe_x + self.pipe_spacing, "gap_y": gap_y2, "scored": False})

        observation = self._get_obs()
        info = {"score": self.score}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """
        Execute one action in the environment.
        
        Args:
            action: 0 (do nothing) or 1 (flap)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = 0.01  # Base reward for staying alive
        terminated = False
        truncated = False

        # Apply action
        if action == 1:
            self.bird_velocity = self.flap_strength

        # Apply gravity
        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity:
            self.bird_velocity = self.max_velocity

        # Update bird position
        self.bird_y += self.bird_velocity

        # Move pipes
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        # Check for scoring (bird passed pipe center/right edge)
        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["scored"] = True
                self.score += 1
                reward = 1.0  # Reward for passing pipe

        # Recycle pipes that went off-screen and spawn new ones
        if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
            self.pipes.pop(0)
            new_x = self.pipes[-1]["x"] + self.pipe_spacing
            new_gap_y = random.randint(self.min_pipe_y, self.max_pipe_y)
            self.pipes.append({"x": new_x, "gap_y": new_gap_y, "scored": False})

        # Check collision with floor or ceiling
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.screen_height:
            terminated = True
            reward = -10.0

        # Check collision with pipes
        if not terminated:
            bird_rect = pygame.Rect(
                int(self.bird_x - self.bird_radius),
                int(self.bird_y - self.bird_radius),
                int(self.bird_radius * 2),
                int(self.bird_radius * 2)
            )

            for pipe in self.pipes:
                pipe_x = int(pipe["x"])
                gap_y = int(pipe["gap_y"])
                gap_half = self.pipe_gap / 2.0

                top_pipe_rect = pygame.Rect(pipe_x, 0, self.pipe_width, int(gap_y - gap_half))
                bottom_pipe_rect = pygame.Rect(
                    pipe_x,
                    int(gap_y + gap_half),
                    self.pipe_width,
                    int(self.screen_height - (gap_y + gap_half))
                )

                if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                    terminated = True
                    reward = -10.0
                    break

        observation = self._get_obs()
        info = {"score": self.score}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment using Pygame."""
        if self.render_mode is None:
            return None

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Flappy Bird RL - DQN")
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            elif self.render_mode == "rgb_array":
                self.window = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 22, bold=True)

        # Handle Pygame window events so window stays responsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Background (sky blue)
        self.window.fill((112, 197, 206))

        # Draw pipes (green tubes)
        pipe_color = (115, 191, 46)
        pipe_border = (83, 140, 34)
        for pipe in self.pipes:
            pipe_x = int(pipe["x"])
            gap_y = int(pipe["gap_y"])
            gap_half = self.pipe_gap / 2.0

            # Upper pipe
            top_h = int(gap_y - gap_half)
            if top_h > 0:
                pygame.draw.rect(self.window, pipe_color, (pipe_x, 0, self.pipe_width, top_h))
                pygame.draw.rect(self.window, pipe_border, (pipe_x, 0, self.pipe_width, top_h), 2)
                # Upper pipe lip
                pygame.draw.rect(self.window, pipe_color, (pipe_x - 3, top_h - 16, self.pipe_width + 6, 16))
                pygame.draw.rect(self.window, pipe_border, (pipe_x - 3, top_h - 16, self.pipe_width + 6, 16), 2)

            # Lower pipe
            bot_y = int(gap_y + gap_half)
            bot_h = int(self.screen_height - bot_y)
            if bot_h > 0:
                pygame.draw.rect(self.window, pipe_color, (pipe_x, bot_y, self.pipe_width, bot_h))
                pygame.draw.rect(self.window, pipe_border, (pipe_x, bot_y, self.pipe_width, bot_h), 2)
                # Lower pipe lip
                pygame.draw.rect(self.window, pipe_color, (pipe_x - 3, bot_y, self.pipe_width + 6, 16))
                pygame.draw.rect(self.window, pipe_border, (pipe_x - 3, bot_y, self.pipe_width + 6, 16), 2)

        # Draw bird (yellow circle with eye and beak)
        bird_center = (int(self.bird_x), int(self.bird_y))
        pygame.draw.circle(self.window, (250, 218, 94), bird_center, self.bird_radius)
        pygame.draw.circle(self.window, (220, 160, 20), bird_center, self.bird_radius, 2)
        # Eye
        eye_center = (int(self.bird_x + 4), int(self.bird_y - 4))
        pygame.draw.circle(self.window, (255, 255, 255), eye_center, 4)
        pygame.draw.circle(self.window, (0, 0, 0), eye_center, 2)
        # Beak
        beak_pts = [
            (int(self.bird_x + 8), int(self.bird_y - 2)),
            (int(self.bird_x + 16), int(self.bird_y + 2)),
            (int(self.bird_x + 8), int(self.bird_y + 5))
        ]
        pygame.draw.polygon(self.window, (244, 114, 52), beak_pts)

        # Draw ground line
        pygame.draw.rect(self.window, (222, 216, 149), (0, self.screen_height - 12, self.screen_width, 12))
        pygame.draw.line(self.window, (115, 191, 46), (0, self.screen_height - 12), (self.screen_width, self.screen_height - 12), 3)

        # Score text
        score_surf = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_surf, (15, 15))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up Pygame window and resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None
