import numpy as np
import os
from collections import deque
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

WINDOW_SIZE = 32
NUM_PHASES = 3
NUM_TIERS = 4
BLOCK_ID_NORM = 1e6
PHASE_NAMES = {0: "zipfian", 1: "sequential", 2: "random_crop"}

if _HAS_TORCH:
    class LSTMPhaseClassifier(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, num_classes: int = NUM_PHASES):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    class MLPConditionedPolicy(nn.Module):
        def __init__(self, input_size: int = WINDOW_SIZE + NUM_PHASES, hidden1: int = 128, hidden2: int = 64, output_size: int = NUM_TIERS + 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden1), nn.ReLU(),
                nn.Linear(hidden1, hidden2), nn.ReLU(),
                nn.Linear(hidden2, output_size), nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

class TieredCachePredictor:
    def __init__(self, lstm_path: Optional[str] = None, mlp_path: Optional[str] = None):
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._rng = np.random.default_rng(seed=0)

        if lstm_path is None or mlp_path is None:
            self._mode = "random"
            self._lstm = None
            self._mlp = None
        else:
            if not _HAS_TORCH:
                raise ImportError("PyTorch is required for model mode.")
            self._mode = "model"
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lstm = LSTMPhaseClassifier().to(self._device)
            self._mlp = MLPConditionedPolicy().to(self._device)
            self._lstm.load_state_dict(torch.load(lstm_path, map_location=self._device, weights_only=True))
            self._mlp.load_state_dict(torch.load(mlp_path, map_location=self._device, weights_only=True))
            self._lstm.eval()
            self._mlp.eval()

    def predict(self, block_id: int) -> Tuple[int, float, int]:
        self._window.append(block_id)
        if self._mode == "random":
            return (int(self._rng.integers(0, NUM_TIERS)), 0.5, int(self._rng.integers(0, NUM_PHASES)))
        return self._predict_model()

    @torch.no_grad()
    def _predict_model(self) -> Tuple[int, float, int]:
        raw = list(self._window)
        pad_len = WINDOW_SIZE - len(raw)
        if pad_len > 0:
            raw = [0] * pad_len + raw
        window = np.array(raw, dtype=np.float32) / BLOCK_ID_NORM
        
        win_tensor = torch.from_numpy(window).to(self._device)
        lstm_input = win_tensor.unsqueeze(0).unsqueeze(-1)
        phase_logits = self._lstm(lstm_input)
        phase_label = int(torch.argmax(phase_logits, dim=-1).item())

        phase_onehot = torch.zeros(NUM_PHASES, dtype=torch.float32, device=self._device)
        phase_onehot[phase_label] = 1.0

        mlp_input = torch.cat([win_tensor, phase_onehot]).unsqueeze(0)
        policy_out = self._mlp(mlp_input)
        tier_id = int(torch.argmax(policy_out[0, :NUM_TIERS]).item())
        prefetch_prob = float(policy_out[0, NUM_TIERS].item())

        return (tier_id, prefetch_prob, phase_label)

    def reset(self):
        self._window.clear()
