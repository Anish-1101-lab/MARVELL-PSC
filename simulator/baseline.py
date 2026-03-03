import numpy as np
from typing import Dict
from collections import OrderedDict

from hierarchy import Tier, StorageHierarchy, DataObject

class HarmoniaBaselineEngine:
    """
    Implements a strict frequency-threshold and LRU-based hierarchical cache
    representing standard caching approaches (Baseline).
    """
    def __init__(self, hierarchy: StorageHierarchy):
        self.hierarchy = hierarchy
        
        # Track metrics
        self.total_latency = 0.0
        self.total_storage_cost = 0.0
        self.total_migration_cost = 0.0
        self.migration_cost_base = 50.0  # Must match RL env
        
        # LRU queues per tier
        self.lru_queues: Dict[Tier, OrderedDict] = {tier: OrderedDict() for tier in Tier}
        self.objects: Dict[int, DataObject] = {}
        self.time_step = 0
        
        # Frequency thresholds for promotion
        self.thresholds = {
            Tier.HBM: 15,
            Tier.DRAM: 5,
            Tier.NVME: 2,
            Tier.SSD: 0
        }

    def _evict_lru(self, from_tier: Tier, required_space: int):
        if from_tier == Tier.SSD:
            return
            
        lower_tier = Tier(from_tier.value + 1)
        freed = 0
        
        objects_to_evict = []
        for obj_id, obj in list(self.lru_queues[from_tier].items()):
            objects_to_evict.append(obj)
            freed += obj.size
            if freed >= required_space:
                break
                
        for obj in objects_to_evict:
            self.hierarchy.move_object(obj, lower_tier)
            self.total_migration_cost += self.migration_cost_base * obj.size
            
            del self.lru_queues[from_tier][obj.obj_id]
            self.lru_queues[lower_tier][obj.obj_id] = obj
            
    def _touch_lru(self, obj: DataObject):
        if obj.obj_id in self.lru_queues[obj.tier]:
            del self.lru_queues[obj.tier][obj.obj_id]
        self.lru_queues[obj.tier][obj.obj_id] = obj

    def process_access(self, obj_id: int, size: int):
        if obj_id not in self.objects:
            obj = DataObject(obj_id, size=size)
            self.hierarchy.add_object(obj, initial_tier=Tier.SSD)
            self.objects[obj_id] = obj
            self.lru_queues[Tier.SSD][obj_id] = obj
        else:
            obj = self.objects[obj_id]
            
        obj.update_access(self.time_step)
        self.time_step += 1
        
        self.total_latency += self.hierarchy.get_latency(obj)
        self._touch_lru(obj)
        self.total_storage_cost += self.hierarchy.get_storage_cost() / 1000.0
        
        # Determine target tier based on frequency thresholds
        target_tier = obj.tier
        for t in [Tier.HBM, Tier.DRAM, Tier.NVME]:
            if obj.access_count >= self.thresholds[t]:
                if t.value < target_tier.value:  # E.g., HBM (0) < DRAM (1)
                    target_tier = t
        
        # Promote if eligible
        if target_tier != obj.tier:
            current_usage = self.hierarchy.tier_usage[target_tier]
            capacity = self.hierarchy.configs[target_tier].capacity
            
            if current_usage + obj.size > capacity:
                self._evict_lru(target_tier, obj.size)
                
            success = self.hierarchy.move_object(obj, target_tier)
            if success:
                self.total_migration_cost += self.migration_cost_base * obj.size
                
                for t in Tier:
                    if t != target_tier and obj.obj_id in self.lru_queues[t]:
                        del self.lru_queues[t][obj.obj_id]
                self._touch_lru(obj)

    def run_simulation(self, trace: np.ndarray, sizes: np.ndarray):
        history = {'steps': [], 'latency': [], 'storage_cost': [], 'migration_cost': [], 'total_cost': [], 'hbm_usage': []}
        print(f"Starting Baseline simulation with {len(trace)} accesses...")
        for i, obj_id in enumerate(trace):
            size = sizes[obj_id]
            self.process_access(obj_id, size)
            if i % 500 == 0:
                history['steps'].append(i)
                history['latency'].append(self.total_latency)
                history['storage_cost'].append(self.total_storage_cost)
                history['migration_cost'].append(self.total_migration_cost)
                history['total_cost'].append(self.total_latency + self.total_storage_cost + self.total_migration_cost)
                hbm_cap = self.hierarchy.configs[Tier.HBM].capacity
                hbm_usage = self.hierarchy.tier_usage[Tier.HBM]
                history['hbm_usage'].append((hbm_usage / hbm_cap * 100) if hbm_cap > 0 else 0)
                
            if i > 0 and i % 10000 == 0:
                print(f"Baseline: Processed {i} accesses...")
                
        print("Baseline Simulation complete.")
        self.hierarchy.print_status()
        print(f"Total Latency: {self.total_latency:.2f}")
        print(f"Total Storage Cost (Amortized): {self.total_storage_cost:.2f}")
        print(f"Total Migration Cost: {self.total_migration_cost:.2f}")
        
        return {
            'latency': self.total_latency,
            'storage_cost': self.total_storage_cost,
            'migration_cost': self.total_migration_cost,
            'total_cost': self.total_latency + self.total_storage_cost + self.total_migration_cost,
            'history': history
        }

class PureLRUBaselineEngine:
    """
    Implements a pure LRU caching approach without capacity tiers/frequency thresholds.
    Everything is promoted to HBM on access.
    """
    def __init__(self, hierarchy: StorageHierarchy):
        self.hierarchy = hierarchy
        
        # Track metrics
        self.total_latency = 0.0
        self.total_storage_cost = 0.0
        self.total_migration_cost = 0.0
        self.migration_cost_base = 50.0
        
        self.lru_queues: Dict[Tier, OrderedDict] = {tier: OrderedDict() for tier in Tier}
        self.objects: Dict[int, DataObject] = {}
        self.time_step = 0

    def _evict_lru(self, from_tier: Tier, required_space: int):
        if from_tier == Tier.SSD:
            return
            
        lower_tier = Tier(from_tier.value + 1)
        
        # Ensure lower tier has space
        current_lower_usage = self.hierarchy.tier_usage[lower_tier]
        capacity_lower = self.hierarchy.configs[lower_tier].capacity
        if current_lower_usage + required_space > capacity_lower:
            # Need to free space in lower tier recursively
            self._evict_lru(lower_tier, (current_lower_usage + required_space) - capacity_lower)

        freed = 0
        objects_to_evict = []
        for obj_id, obj in list(self.lru_queues[from_tier].items()):
            objects_to_evict.append(obj)
            freed += obj.size
            if freed >= required_space:
                break
                
        for obj in objects_to_evict:
            success = self.hierarchy.move_object(obj, lower_tier)
            if success:
                self.total_migration_cost += self.migration_cost_base * obj.size
                del self.lru_queues[from_tier][obj.obj_id]
                self.lru_queues[lower_tier][obj.obj_id] = obj
            
    def _touch_lru(self, obj: DataObject):
        if obj.obj_id in self.lru_queues[obj.tier]:
            del self.lru_queues[obj.tier][obj.obj_id]
        self.lru_queues[obj.tier][obj.obj_id] = obj

    def process_access(self, obj_id: int, size: int):
        if obj_id not in self.objects:
            obj = DataObject(obj_id, size=size)
            self.hierarchy.add_object(obj, initial_tier=Tier.SSD)
            self.objects[obj_id] = obj
            self.lru_queues[Tier.SSD][obj_id] = obj
        else:
            obj = self.objects[obj_id]
            
        obj.update_access(self.time_step)
        self.time_step += 1
        
        self.total_latency += self.hierarchy.get_latency(obj)
        self._touch_lru(obj)
        self.total_storage_cost += self.hierarchy.get_storage_cost() / 1000.0
        
        # Pure LRU ALWAYS promotes to HBM on access
        target_tier = Tier.HBM
        
        if target_tier != obj.tier:
            current_usage = self.hierarchy.tier_usage[target_tier]
            capacity = self.hierarchy.configs[target_tier].capacity
            
            if current_usage + obj.size > capacity:
                self._evict_lru(target_tier, obj.size)
                
            success = self.hierarchy.move_object(obj, target_tier)
            if success:
                self.total_migration_cost += self.migration_cost_base * obj.size
                
                for t in Tier:
                    if t != target_tier and obj.obj_id in self.lru_queues[t]:
                        del self.lru_queues[t][obj.obj_id]
                self._touch_lru(obj)

    def run_simulation(self, trace: np.ndarray, sizes: np.ndarray):
        history = {'steps': [], 'latency': [], 'storage_cost': [], 'migration_cost': [], 'total_cost': [], 'hbm_usage': []}
        print(f"Starting Pure LRU Baseline simulation with {len(trace)} accesses...")
        for i, obj_id in enumerate(trace):
            size = sizes[obj_id]
            self.process_access(obj_id, size)
            if i % 500 == 0:
                history['steps'].append(i)
                history['latency'].append(self.total_latency)
                history['storage_cost'].append(self.total_storage_cost)
                history['migration_cost'].append(self.total_migration_cost)
                history['total_cost'].append(self.total_latency + self.total_storage_cost + self.total_migration_cost)
                hbm_cap = self.hierarchy.configs[Tier.HBM].capacity
                hbm_usage = self.hierarchy.tier_usage[Tier.HBM]
                history['hbm_usage'].append((hbm_usage / hbm_cap * 100) if hbm_cap > 0 else 0)
                
            if i > 0 and i % 10000 == 0:
                print(f"Pure LRU Baseline: Processed {i} accesses...")
                
        print("Pure LRU Baseline Simulation complete.")
        self.hierarchy.print_status()
        print(f"Total Latency: {self.total_latency:.2f}")
        print(f"Total Storage Cost (Amortized): {self.total_storage_cost:.2f}")
        print(f"Total Migration Cost: {self.total_migration_cost:.2f}")
        
        return {
            'latency': self.total_latency,
            'storage_cost': self.total_storage_cost,
            'migration_cost': self.total_migration_cost,
            'total_cost': self.total_latency + self.total_storage_cost + self.total_migration_cost,
            'history': history
        }

