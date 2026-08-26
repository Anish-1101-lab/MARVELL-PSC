import enum
from typing import Dict, List, Optional

class Tier(enum.Enum):
    HBM = 0
    DRAM = 1
    NVME = 2
    SSD = 3

class TierConfig:
    def __init__(self, name: str, base_latency: float, storage_cost: float, capacity: int, access_penalty: float = 0.0):
        self.name = name
        self.base_latency = base_latency
        self.storage_cost = storage_cost
        self.capacity = capacity
        self.access_penalty = access_penalty

class DataObject:
    def __init__(self, obj_id: int, size: int = 1):
        self.obj_id = obj_id
        self.size = size
        self.tier = Tier.SSD # Default to lowest tier
        
        # Access History signals
        self.last_accessed = -1
        self.access_count = 0
        
        # Oracle / Prediction signals (Optional, populated by Workload Generator / Predictor)
        self.time_to_next_access: Optional[int] = None
        self.reuse_probability: Optional[float] = None
        
    def update_access(self, current_time: int):
        self.last_accessed = current_time
        self.access_count += 1
        
    def __repr__(self):
        return f"DataObject(id={self.obj_id}, tier={self.tier.name})"

class StorageHierarchy:
    """
    Manages the tiers and object placements.
    Latency_i = L_{tier(i)} + AccessPenalty_i
    """
    def __init__(self, configs: Optional[Dict[Tier, TierConfig]] = None):
        if configs is None:
            # Default mock configuration for AI workloads
            # Capacities are in terms of 'units' or blocks for simplicity
            self.configs = {
                Tier.HBM: TierConfig("HBM", base_latency=1.0, storage_cost=100.0, capacity=5000, access_penalty=0.1),
                Tier.DRAM: TierConfig("DRAM", base_latency=10.0, storage_cost=10.0, capacity=10000, access_penalty=1.0),
                Tier.NVME: TierConfig("NVME", base_latency=100.0, storage_cost=1.0, capacity=100000, access_penalty=10.0),
                Tier.SSD: TierConfig("SSD", base_latency=1000.0, storage_cost=0.1, capacity=float('inf'), access_penalty=100.0)
            }
        else:
            self.configs = configs
            
        # Mapping tier -> Dict[obj_id, DataObject]
        self.tiers: Dict[Tier, Dict[int, DataObject]] = {tier: {} for tier in Tier}
        self.tier_usage: Dict[Tier, int] = {tier: 0 for tier in Tier}
        
    def add_object(self, obj: DataObject, initial_tier: Tier = Tier.SSD) -> bool:
        """Adds a new object directly into the hierarchy."""
        if self.tier_usage[initial_tier] + obj.size > self.configs[initial_tier].capacity:
            return False # Capacity Exceeded
            
        obj.tier = initial_tier
        self.tiers[initial_tier][obj.obj_id] = obj
        self.tier_usage[initial_tier] += obj.size
        return True

    def get_latency(self, obj: DataObject) -> float:
        """Latency_i = L_{tier(i)} + AccessPenalty_i"""
        config = self.configs[obj.tier]
        return config.base_latency + config.access_penalty
        
    def get_storage_cost(self) -> float:
        """Total storage cost: sum of (usage * cost_per_unit) for each tier"""
        return sum(self.tier_usage[tier] * self.configs[tier].storage_cost for tier in Tier)
        
    def print_status(self):
        print("Storage Hierarchy Status:")
        for tier in Tier:
            usage = self.tier_usage[tier]
            cap = self.configs[tier].capacity
            percentage = (usage / cap * 100) if cap > 0 and cap != float('inf') else 0
            print(f"[{tier.name}] Usage: {usage} / {cap} ({percentage:.2f}%)")

    def move_object(self, obj: DataObject, new_tier: Tier) -> bool:
        """Moves object to a new tier. Returns True if successful, False if capacity exceeded."""
        if obj.tier == new_tier:
            return True
            
        # Check capacity
        if self.tier_usage[new_tier] + obj.size > self.configs[new_tier].capacity:
            return False
            
        # Remove from old tier
        if obj.obj_id in self.tiers[obj.tier]:
            del self.tiers[obj.tier][obj.obj_id]
            self.tier_usage[obj.tier] -= obj.size
            if self.tier_usage[obj.tier] < 0:
                self.tier_usage[obj.tier] = 0
            
        # Add to new tier
        obj.tier = new_tier
        self.tiers[new_tier][obj.obj_id] = obj
        self.tier_usage[new_tier] += obj.size
        
        return True

    def get_objects_in_tier(self, tier: Tier) -> List[DataObject]:
        return list(self.tiers[tier].values())
