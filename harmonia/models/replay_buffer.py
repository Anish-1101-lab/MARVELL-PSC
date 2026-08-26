"""
Experience Replay Buffer for Harmonia RL Agents.

Maintains fixed-size DRAM buffer (1,000 experiences per agent)
as specified in Harmonia Section 4.5 and Table 2.
"""

from collections import deque
import numpy as np
import torch
from typing import Tuple, List, Optional


class ReplayBuffer:
    def __init__(self, capacity: int = 1000, seed: int = 42):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ) -> None:
        """Adds a new experience tuple to the replay buffer."""
        self.buffer.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(done)
        ))

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Samples a random batch of experiences.
        Returns:
            (states, actions, rewards, next_states, dones) tensors
        """
        if len(self.buffer) < batch_size:
            return None

        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
