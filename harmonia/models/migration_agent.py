"""
Data-Migration Agent for Harmonia.

Asynchronously monitors previously placed pages, identifies migration candidates,
and selects target tiers with a delayed reward structure to optimize long-term HSS performance.

Hyperparameters from Harmonia Section 4.5 & Table 2:
  - Discount Factor (gamma): 0.1
  - Learning Rate (alpha): 1e-2
  - Exploration Rate (epsilon): 0.001
  - Batch Size: 256
  - Experience Buffer Size: 1000
  - Sync Interval (K): 1000 requests
  - Delayed Reward Window (n): 50 future requests
  - Migration Batch Size (x): 10 pages
"""

from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List, Tuple

from .q_network import SwishFeedForwardQNetwork
from .replay_buffer import ReplayBuffer


class MigrationAgent:
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 10,
        num_tiers: int = 4,
        gamma: float = 0.1,
        alpha: float = 1e-2,
        epsilon: float = 0.001,
        batch_size: int = 256,
        buffer_size: int = 1000,
        sync_interval: int = 1000,
        delayed_window_n: int = 50,
        migration_batch_x: int = 10,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.num_tiers = num_tiers
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.sync_interval = sync_interval
        self.delayed_window_n = delayed_window_n
        self.migration_batch_x = migration_batch_x
        
        self.rng = np.random.default_rng(seed)
        
        # Dual network architecture (inference thread in background, training thread in background)
        self.inference_net = SwishFeedForwardQNetwork(input_dim, hidden_dim, num_tiers)
        self.training_net = SwishFeedForwardQNetwork(input_dim, hidden_dim, num_tiers)
        self.training_net.load_state_dict(self.inference_net.state_dict())
        
        self.optimizer = optim.Adam(self.training_net.parameters(), lr=alpha)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Delayed reward tracking pipeline
        # Stores pending migrations awaiting subsequent n I/O request latencies
        self.pending_migrations: List[Dict[str, Any]] = []
        self.recent_latencies: deque = deque(maxlen=delayed_window_n * 2)
        
        self.step_count = 0
        self.total_train_steps = 0
        self.last_loss = 0.0

    def select_migration_tier(self, state: np.ndarray) -> int:
        """
        Background inference to determine target device for a candidate page.
        """
        action = self.inference_net.get_action(state, epsilon=self.epsilon, rng=self.rng)
        return action

    def register_migration(
        self,
        block_id: int,
        state: np.ndarray,
        target_tier: int,
        avg_access_interval: float,
        avg_migr_interval: float
    ) -> None:
        """
        Registers a committed migration to receive a delayed reward after n subsequent I/O requests.
        """
        self.pending_migrations.append({
            "block_id": block_id,
            "state": np.array(state, dtype=np.float32, copy=True),
            "action": target_tier,
            "acc_intr": avg_access_interval,
            "migr_intr": avg_migr_interval,
            "latency_collector": []
        })

    def record_io_latency(self, latency_us: float, current_state: np.ndarray) -> None:
        """
        Feeds incoming I/O latency to pending migration trackers to fulfill delayed rewards.
        """
        self.recent_latencies.append(latency_us)
        
        # Advance latency collectors for pending migrations
        completed_indices = []
        for idx, migr in enumerate(self.pending_migrations):
            migr["latency_collector"].append(latency_us)
            if len(migr["latency_collector"]) >= self.delayed_window_n:
                # Compute Delayed Reward: R_migration = n / sum(L_i) - P_migr (Harmonia Eq. 3)
                sum_lat = sum(migr["latency_collector"])
                if sum_lat > 0:
                    base_reward = (self.delayed_window_n / sum_lat) * 100.0  # scaled to match Q-scale
                else:
                    base_reward = 1.0
                
                # Ping-pong penalty (inversely proportional to migration & access intervals)
                stability = max(1.0, migr["acc_intr"] + migr["migr_intr"])
                p_migr = float(10.0 / stability)
                delayed_reward = float(base_reward - p_migr)
                
                # Push experience to migration replay buffer
                self.replay_buffer.push(
                    migr["state"],
                    migr["action"],
                    delayed_reward,
                    current_state,
                    False
                )
                self.step_count += 1
                completed_indices.append(idx)
                
                # Train if batch is ready
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step()
                    
                if self.step_count % self.sync_interval == 0:
                    self.inference_net.sync_weights_from(self.training_net)

        # Remove completed migrations in reverse order
        for idx in reversed(completed_indices):
            del self.pending_migrations[idx]

    def _train_step(self) -> Optional[float]:
        """Performs mini-batch Q-learning update for migration policy."""
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        
        current_q = self.training_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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
