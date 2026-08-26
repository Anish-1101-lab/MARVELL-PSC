"""
Migration Queue for Harmonia.

Implements the fixed-size migration queue (x=10 entries) that stores
candidate pages and target devices identified by the data-migration agent
as described in Harmonia Section 4.4 and Section 6.1.
"""

from collections import deque
from typing import Optional, Dict, Any, List
import numpy as np


class MigrationQueue:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)

    def push(
        self,
        block_id: int,
        target_tier: int,
        current_tier: int,
        size_bytes: int,
        state: np.ndarray,
        acc_intr: float,
        migr_intr: float
    ) -> bool:
        """
        Pushes a migration candidate to the queue if target != current.
        Returns True if candidate was pushed, False otherwise.
        """
        if target_tier == current_tier:
            return False
            
        # Avoid duplicate entries already in queue
        for item in self.queue:
            if item["block_id"] == block_id:
                return False
                
        self.queue.append({
            "block_id": block_id,
            "target_tier": target_tier,
            "current_tier": current_tier,
            "size_bytes": size_bytes,
            "state": state,
            "acc_intr": acc_intr,
            "migr_intr": migr_intr
        })
        return True

    def pop_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        """Pops up to count migration candidates to be executed."""
        batch = []
        while self.queue and len(batch) < count:
            batch.append(self.queue.popleft())
        return batch

    def is_full(self) -> bool:
        return len(self.queue) >= self.max_size

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)

    def clear(self) -> None:
        self.queue.clear()
