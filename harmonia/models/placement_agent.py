"""
Data-Placement Agent for Harmonia.

Optimizes immediate write request latency on the I/O critical path.
Hyperparameters from Harmonia Section 4.5 & Table 2:
  - Discount Factor (gamma): 0.9
  - Learning Rate (alpha): 1e-3
  - Exploration Rate (epsilon): 0.001
  - Batch Size: 128
  - Experience Buffer Size: 1000
  - Sync Interval (K): 1000 requests
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any

from .q_network import SwishFeedForwardQNetwork
from .replay_buffer import ReplayBuffer


class PlacementAgent:
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 10,
        num_tiers: int = 4,
        gamma: float = 0.9,
        alpha: float = 1e-3,
        epsilon: float = 0.001,
        batch_size: int = 128,
        buffer_size: int = 1000,
        sync_interval: int = 1000,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.num_tiers = num_tiers
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.sync_interval = sync_interval
        
        self.rng = np.random.default_rng(seed)
        
        # Dual network architecture (inference thread on critical path, training thread in background)
        self.inference_net = SwishFeedForwardQNetwork(input_dim, hidden_dim, num_tiers)
        self.training_net = SwishFeedForwardQNetwork(input_dim, hidden_dim, num_tiers)
        self.training_net.load_state_dict(self.inference_net.state_dict())
        
        self.optimizer = optim.Adam(self.training_net.parameters(), lr=alpha)
        self.criterion = nn.SmoothL1Loss() # Robust Huber loss
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        self.step_count = 0
        self.total_train_steps = 0
        self.last_loss = 0.0

    def select_placement_tier(self, state: np.ndarray) -> int:
        """
        Inference on the I/O critical path (fast forward pass ~240 ns).
        Selects target tier (0=HBM, 1=CXL_DRAM, 2=NVMe_SSD, 3=Cold_Storage).
        """
        action = self.inference_net.get_action(state, epsilon=self.epsilon, rng=self.rng)
        return action

    def compute_reward(self, latency_us: float) -> float:
        """
        Placement reward: R_placement = 1 / L_t (Harmonia Eq. 2)
        Avoids division by zero using small epsilon.
        """
        lat = max(latency_us, 0.01) # in microseconds
        return float(1.0 / lat)

    def record_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ) -> None:
        """Stores experience into replay buffer and triggers background training."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        
        # Background training step
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
            
        # Periodic weight synchronization
        if self.step_count % self.sync_interval == 0:
            self.inference_net.sync_weights_from(self.training_net)

    def _train_step(self) -> Optional[float]:
        """Performs one mini-batch Bellman Q-learning update."""
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        
        # Current Q estimates
        current_q = self.training_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q calculation with target network / inference network
        with torch.no_grad():
            next_q = self.training_net(next_states).max(dim=1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.training_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.last_loss = float(loss.item())
        self.total_train_steps += 1
        return self.last_loss
