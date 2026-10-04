import numpy as np
import torch
import torch.nn as nn


def normalize_observation(obs):
    """
    Normalize the 4 observation values:
        bird_y / 512.0
        bird_velocity / 10.0
        horizontal_distance / 288.0
        vertical_distance / 512.0
    Clip normalized values appropriately to [-1.0, 1.0] so the network receives numerically balanced inputs.
    """
    if isinstance(obs, torch.Tensor):
        scale = torch.tensor([512.0, 10.0, 288.0, 512.0], device=obs.device, dtype=obs.dtype)
        return torch.clamp(obs / scale, -1.0, 1.0)
    norm = np.array([
        obs[0] / 512.0,
        obs[1] / 10.0,
        obs[2] / 288.0,
        obs[3] / 512.0
    ], dtype=np.float32)
    return np.clip(norm, -1.0, 1.0)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) for Flappy Bird.
    
    Architecture:
        Input Layer  : 4 inputs [bird_y, bird_velocity, horizontal_dist, vertical_dist]
        Hidden Layer 1: 128 units + ReLU
        Hidden Layer 2: 128 units + ReLU
        Output Layer : 2 outputs [Q(s, do_nothing), Q(s, flap)]
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        """
        Initialize the DQN model layers.
        
        Args:
            state_dim: Dimension of input state vector (default: 4)
            action_dim: Number of discrete actions (default: 2)
        """
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Q-values tensor of shape (batch_size, 2)
        """
        return self.network(x)
