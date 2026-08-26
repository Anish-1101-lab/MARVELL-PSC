"""
State Encoder and Feature Extractor for Harmonia.

Extracts and quantizes the 7-dimensional observation vector Ot
as specified in Harmonia Table 1 and Section 4.2:
  1. req_type  : Read (0) or Write (1)                     [1 bit / 2 bins]
  2. req_size  : Request size in 4 KiB pages               [3 bits / 8 bins]
  3. acc_intr  : Access interval between page accesses     [8 bits / 64 bins]
  4. acc_freq  : Total access count of page                [8 bits / 64 bins]
  5. fast_cap  : Remaining free capacity in fast tier      [3 bits / 8 bins]
  6. curr_dev  : Storage device/tier where page resides    [1-2 bits]
  7. migr_intr : Inter-migration interval of page          [8 bits / 64 bins]
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional


class HarmoniaStateEncoder:
    def __init__(self, page_size_bytes: int = 4096, num_tiers: int = 4):
        self.page_size_bytes = page_size_bytes
        self.num_tiers = num_tiers
        
        # Metadata tracking (simulating DRAM metadata table)
        # block_id -> {last_access_step, access_count, last_migration_step, migration_count}
        self.page_metadata: Dict[int, Dict[str, int]] = {}
        self.global_access_step = 0
        self.global_migration_step = 0

    def quantize_req_size(self, size_bytes: int) -> int:
        """Quantizes request size in pages (4 KiB) into 8 bins (0-7)."""
        pages = max(1, size_bytes // self.page_size_bytes)
        if pages <= 1:
            return 0
        elif pages <= 2:
            return 1
        elif pages <= 4:
            return 2
        elif pages <= 8:
            return 3
        elif pages <= 16:
            return 4
        elif pages <= 32:
            return 5
        elif pages <= 64:
            return 6
        else:
            return 7

    def quantize_interval(self, interval: int) -> int:
        """Quantizes temporal access or migration interval into 64 bins (0-63)."""
        if interval <= 1:
            return 0
        # Logarithmic/exponential binning up to 65,536 intervals
        bin_idx = int(np.clip(np.log2(max(1, interval)) * 4, 0, 63))
        return bin_idx

    def quantize_frequency(self, freq: int) -> int:
        """Quantizes access frequency count into 64 bins (0-63)."""
        if freq <= 1:
            return 0
        bin_idx = int(np.clip(np.log2(max(1, freq)) * 4, 0, 63))
        return bin_idx

    def quantize_fast_capacity(self, current_occupancy: int, max_capacity: int) -> int:
        """
        Quantizes remaining free capacity percentage in fast tier into 8 bins (0-7).
        0: 0-12.5% free, 7: 87.5-100% free.
        """
        if max_capacity <= 0:
            return 0
        free_ratio = max(0.0, 1.0 - (current_occupancy / max_capacity))
        return int(np.clip(free_ratio * 8.0, 0, 7))

    def update_access(self, block_id: int) -> Tuple[int, int]:
        """
        Updates access history for block_id.
        Returns: (access_interval, access_frequency)
        """
        self.global_access_step += 1
        if block_id not in self.page_metadata:
            self.page_metadata[block_id] = {
                "last_access": self.global_access_step,
                "access_count": 1,
                "last_migration": 0,
                "migration_count": 0,
            }
            return 64, 1  # Initial cold interval
        
        meta = self.page_metadata[block_id]
        interval = self.global_access_step - meta["last_access"]
        meta["last_access"] = self.global_access_step
        meta["access_count"] += 1
        return interval, meta["access_count"]

    def update_migration(self, block_id: int) -> int:
        """
        Updates migration history for block_id.
        Returns: migration_interval
        """
        self.global_migration_step += 1
        if block_id not in self.page_metadata:
            self.page_metadata[block_id] = {
                "last_access": self.global_access_step,
                "access_count": 1,
                "last_migration": self.global_migration_step,
                "migration_count": 1,
            }
            return 64
        
        meta = self.page_metadata[block_id]
        interval = self.global_migration_step - meta["last_migration"] if meta["last_migration"] > 0 else 64
        meta["last_migration"] = self.global_migration_step
        meta["migration_count"] += 1
        return interval

    def extract_state(
        self,
        block_id: int,
        size_bytes: int,
        is_write: bool,
        current_tier: int,
        fast_occupancy: int,
        fast_capacity: int
    ) -> Tuple[np.ndarray, int]:
        """
        Extracts the 7-dimensional observation vector Ot.
        Returns:
            normalized_state: np.ndarray (float32, shape=(7,), range [0, 1])
            packed_32bit: int (32-bit compact integer metadata)
        """
        # 1. req_type (0=read, 1=write)
        q_type = 1 if is_write else 0
        
        # 2. req_size (0-7)
        q_size = self.quantize_req_size(size_bytes)
        
        # 3. acc_intr & 4. acc_freq
        acc_intr, acc_freq = self.update_access(block_id)
        q_acc_intr = self.quantize_interval(acc_intr)
        q_acc_freq = self.quantize_frequency(acc_freq)
        
        # 5. fast_cap (0-7)
        q_fast_cap = self.quantize_fast_capacity(fast_occupancy, fast_capacity)
        
        # 6. curr_dev (0..num_tiers-1)
        q_curr_dev = max(0, min(current_tier, self.num_tiers - 1))
        
        # 7. migr_intr (0-63)
        meta = self.page_metadata.get(block_id, {})
        last_migr = meta.get("last_migration", 0)
        migr_intr = self.global_migration_step - last_migr if last_migr > 0 else 64
        q_migr_intr = self.quantize_interval(migr_intr)
        
        # Build normalized float vector for neural network
        norm_state = np.array([
            float(q_type),
            float(q_size) / 7.0,
            float(q_acc_intr) / 63.0,
            float(q_acc_freq) / 63.0,
            float(q_fast_cap) / 7.0,
            float(q_curr_dev) / float(max(1, self.num_tiers - 1)),
            float(q_migr_intr) / 63.0
        ], dtype=np.float32)
        
        # 32-bit bit-packed integer representation
        # [q_type: 1b][q_size: 3b][q_acc_intr: 6b][q_acc_freq: 6b][q_fast_cap: 3b][q_curr_dev: 2b][q_migr_intr: 6b] = 27 bits
        packed_32bit = (
            (q_type & 0x1) << 26 |
            (q_size & 0x7) << 23 |
            (q_acc_intr & 0x3F) << 17 |
            (q_acc_freq & 0x3F) << 11 |
            (q_fast_cap & 0x7) << 8 |
            (q_curr_dev & 0x3) << 6 |
            (q_migr_intr & 0x3F)
        )
        
        return norm_state, packed_32bit

    def get_raw_intervals(self, block_id: int) -> Tuple[float, float]:
        """Returns raw (access_interval, migr_interval) for ping-pong penalty calculation."""
        meta = self.page_metadata.get(block_id, {})
        last_acc = meta.get("last_access", self.global_access_step)
        last_migr = meta.get("last_migration", self.global_migration_step)
        
        acc_intr = max(1.0, float(self.global_access_step - last_acc))
        migr_intr = max(1.0, float(self.global_migration_step - last_migr))
        return acc_intr, migr_intr
