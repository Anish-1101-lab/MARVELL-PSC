"""
Harmonia Neural Network Architecture.

Implements the 7-10-N Feed-Forward Q-Network with Swish activation
as specified in Harmonia Section 4.5.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Swish(nn.Module):
    """Swish activation function: f(x) = x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwishFeedForwardQNetwork(nn.Module):
    """
    Lightweight 7-10-N feed-forward neural network for Harmonia RL agents.
    
    Architecture:
      - Input: 7 features (state observation vector)
      - Hidden Layer: 10 neurons with Swish activation
      - Output Layer: N neurons (Q-values for N storage tiers/devices)
    """
    def __init__(self, input_dim: int = 7, hidden_dim: int = 10, num_actions: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = Swish()
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
        # Initialize weights with standard normal for stable Q-learning
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim) or (input_dim,)
        Returns:
            q_values: Tensor of shape (batch_size, num_actions)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.act(self.fc1(x))
        q_values = self.fc2(h)
        return q_values

    def get_action(self, state: np.ndarray, epsilon: float = 0.0, rng: Optional[np.random.Generator] = None) -> int:
        """
        Selects an action using epsilon-greedy policy.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        if rng.random() < epsilon:
            return int(rng.integers(0, self.num_actions))
        
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32)
            q_vals = self.forward(state_t)
            action = int(torch.argmax(q_vals, dim=1).item())
        return action

    def sync_weights_from(self, source_net: nn.Module) -> None:
        """
        Copies weights from training network to inference network.
        Harmonia performs this synchronization every K=1000 requests.
        """
        self.load_state_dict(source_net.state_dict())
