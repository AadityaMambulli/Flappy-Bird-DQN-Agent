import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for Deep Q-Learning (DQN).
    
    Stores transitions (state, action, reward, next_state, done)
    and allows uniform random mini-batch sampling to break temporal correlations.
    """

    def __init__(self, capacity: int = 50000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience tuple to the buffer.
        
        Args:
            state: Current state observation (array-like of shape (4,))
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state observation (array-like of shape (4,))
            done: Whether the episode terminated (bool)
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int, device: str = "cpu"):
        """
        Sample a random mini-batch of experiences.
        
        Args:
            batch_size: Number of transitions to sample.
            device: Target torch device ('cpu' or 'cuda').
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        """Return the current number of experiences stored in the buffer."""
        return len(self.buffer)
