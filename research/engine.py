import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict

from hierarchy import Tier, StorageHierarchy, DataObject
from predictor import LightGBMPredictor
from rl_controller import MigrationEnv

class SimulationEngine:
    def __init__(self, hierarchy: StorageHierarchy, predictor: LightGBMPredictor, rl_model, env: MigrationEnv):
        self.hierarchy = hierarchy
        self.predictor = predictor
        self.rl_model = rl_model
        self.env = env
        
        # Track metrics
        self.total_latency = 0.0
        self.total_storage_cost = 0.0
        self.total_migration_cost = 0.0
        
        # LRU queues per tier (Mapping object_id -> DataObject for fast deletion)
        self.lru_queues: Dict[Tier, OrderedDict] = {tier: OrderedDict() for tier in Tier}
        
        
        self.objects: Dict[int, DataObject] = {}
        self.time_step = 0
        self._predict_cache = {}
        
    def _get_tier_usages(self) -> List[float]:
        usages = []
        for tier in [Tier.HBM, Tier.DRAM, Tier.NVME, Tier.SSD]:
            cap = self.hierarchy.configs[tier].capacity
            if cap == float('inf') or cap == 0:
                usages.append(0.0)
            else:
                usages.append(self.hierarchy.tier_usage[tier] / cap)
        return usages

    def _evict_lru(self, from_tier: Tier, required_space: int):
        """Evicts objects from 'from_tier' to the next lower tier until required_space is freed."""
        if from_tier == Tier.SSD:
            return # SSD is unbounded bottom tier
            
        lower_tier = Tier(from_tier.value + 1)
        freed = 0
        
        # Iterate LRU queue (oldest first)
        objects_to_evict = []
        for obj_id, obj in [ (k,v) for k,v in self.lru_queues[from_tier].items() ]:
            objects_to_evict.append(obj)
            freed += obj.size
            if freed >= required_space:
                break
                
        for obj in objects_to_evict:
            # Move in hierarchy
            self.hierarchy.move_object(obj, lower_tier)
            self.total_migration_cost += self.env.migration_cost_base * obj.size
            
            # Update LRU tracking
            del self.lru_queues[from_tier][obj.obj_id]
            self.lru_queues[lower_tier][obj.obj_id] = obj
            
    def _touch_lru(self, obj: DataObject):
        """Updates obj position to strictly MRU in its current tier's queue."""
        if obj.obj_id in self.lru_queues[obj.tier]:
            del self.lru_queues[obj.tier][obj.obj_id]
        self.lru_queues[obj.tier][obj.obj_id] = obj

    def process_access(self, obj_id: int, size: int):
        """Handles a single access event."""
        # 1. Fetch or Create Object
        if obj_id not in self.objects:
            obj = DataObject(obj_id, size=size)
            self.hierarchy.add_object(obj, initial_tier=Tier.SSD)
            self.objects[obj_id] = obj
            self.lru_queues[Tier.SSD][obj_id] = obj
        else:
            obj = self.objects[obj_id]
            
        # Recency calc before update
        recency = self.time_step - obj.last_accessed if obj.last_accessed != -1 else -1
        
        # Update state
        obj.update_access(self.time_step)
        self.time_step += 1
        
        # 2. Add Latency for access (based on current tier)
        self.total_latency += self.hierarchy.get_latency(obj)
        self._touch_lru(obj)
        
        # 3. Add ongoing storage cost (amortized per access for simplicity)
        self.total_storage_cost += self.hierarchy.get_storage_cost() / 1000.0
        
        # 4. Predictive Filtering
        state_key = (obj.size, recency, obj.access_count)
        if state_key in self._predict_cache:
            p_i, t_i = self._predict_cache[state_key]
        else:
            p_i, t_i = self.predictor.predict(size=obj.size, recency=recency, frequency=obj.access_count)
            self._predict_cache[state_key] = (p_i, t_i)
        
        obj.reuse_probability = p_i
        obj.time_to_next_access = t_i
        
        # Fallback to standard LRU (no promotion) if predict says low reuse
        if p_i < 0.5:
            # For low reuse objects, if they are stuck in a high tier, LRU will evict them later
            return
            
        # 5. State Observation & RL Action
        usages = self._get_tier_usages()
        self.env.set_state(usages, obj.tier.value, obj.size, p_i, t_i)
        
        # Get action from model
        action, _states = self.rl_model.predict(self.env.current_state, deterministic=True)
        target_tier = Tier(int(action))
        
        # 6. Action Execution & Cost Validation
        if target_tier != obj.tier:
            # Check capacity
            current_usage = self.hierarchy.tier_usage[target_tier]
            capacity = self.hierarchy.configs[target_tier].capacity
            
            if current_usage + obj.size > capacity:
                # Need to evict
                self._evict_lru(target_tier, obj.size)
                
            # Now try to move
            success = self.hierarchy.move_object(obj, target_tier)
            if success:
                # Valid move - pay migration cost
                self.total_migration_cost += self.env.migration_cost_base * obj.size
                
                # Update old queue and new queue
                # Old queue was already handled safely except if it missed removing, 
                # but move_object changes obj.tier, so we must clean up
                for t in Tier:
                    if t != target_tier and obj.obj_id in self.lru_queues[t]:
                        del self.lru_queues[t][obj.obj_id]
                self._touch_lru(obj)

    def _prefetch(self, obj_id: int, size: int):
        """Predictively pre-fetch an object to HBM if reuse probability is high."""
        if obj_id not in self.objects:
            obj = DataObject(obj_id, size=size)
            self.hierarchy.add_object(obj, initial_tier=Tier.SSD)
            self.objects[obj_id] = obj
            self.lru_queues[Tier.SSD][obj_id] = obj
        else:
            obj = self.objects[obj_id]
            
        if obj.tier == Tier.HBM:
            return  # Already in HBM
            
        recency = self.time_step - obj.last_accessed if obj.last_accessed != -1 else -1
        
        # Predict reuse probability (use cache to prevent massive overhead during lookahead)
        state_key = (obj.size, recency, obj.access_count)
        if state_key in self._predict_cache:
            p_i, t_i = self._predict_cache[state_key]
        else:
            p_i, t_i = self.predictor.predict(size=obj.size, recency=recency, frequency=obj.access_count)
            self._predict_cache[state_key] = (p_i, t_i)
        
        if p_i > 0.8:
            # Trigger migration to HBM now
            target_tier = Tier.HBM
            current_usage = self.hierarchy.tier_usage[target_tier]
            capacity = self.hierarchy.configs[target_tier].capacity
            
            if current_usage + obj.size > capacity:
                self._evict_lru(target_tier, obj.size)
                
            success = self.hierarchy.move_object(obj, target_tier)
            if success:
                self.total_migration_cost += self.env.migration_cost_base * obj.size
                for t in Tier:
                    if t != target_tier and obj.obj_id in self.lru_queues[t]:
                        del self.lru_queues[t][obj.obj_id]
                self._touch_lru(obj)

    def run_simulation(self, trace: np.ndarray, sizes: np.ndarray, lookahead: int = 10):
        """Runs the complete phase loop, including predictive pre-warming."""
        history = {'steps': [], 'latency': [], 'storage_cost': [], 'migration_cost': [], 'total_cost': [], 'hbm_usage': []}
        print(f"Starting simulation with {len(trace)} accesses...")
        for i, obj_id in enumerate(trace):
            # Predictive Pre-warming
            for w in range(1, lookahead + 1):
                future_idx = i + w
                if future_idx < len(trace):
                    fut_obj_id = trace[future_idx]
                    fut_size = sizes[fut_obj_id]
                    self._prefetch(fut_obj_id, fut_size)
            
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
                
                dram_cap = self.hierarchy.configs[Tier.DRAM].capacity
                dram_usage = self.hierarchy.tier_usage[Tier.DRAM]
                history.setdefault('dram_usage', []).append((dram_usage / dram_cap * 100) if dram_cap > 0 else 0)
                
                # SSD has 'inf' capacity, track absolute usage
                ssd_usage = self.hierarchy.tier_usage[Tier.SSD]
                history.setdefault('ssd_usage', []).append(ssd_usage)
                
            if i > 0 and i % 10000 == 0:
                print(f"Processed {i} accesses...")
        
        # Final evaluation metrics
        print("Simulation complete.")
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
