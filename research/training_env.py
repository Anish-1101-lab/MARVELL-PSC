import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hierarchy import Tier, StorageHierarchy
from engine import SimulationEngine
from rl_controller import MigrationEnv

class TraceTrainingEnv(gym.Env):
    def __init__(self, trace, sizes, predictor, alpha=10.0, beta=1.0, gamma=5.0, lookahead=10):
        super().__init__()
        self.trace = trace
        self.sizes = sizes
        self.predictor = predictor
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lookahead = lookahead
        
        self.action_space = spaces.Discrete(4)
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 3.0, float('inf'), 1.0, float('inf')], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.idx = 0
        self.engine = None
        self.migration_env = None
        self._generator = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.hierarchy = StorageHierarchy()
        self.hierarchy.configs[Tier.HBM].capacity = 5000
        
        self.migration_env = MigrationEnv(self.hierarchy, self.alpha, self.beta, self.gamma)
        
        class DummyModel:
            def predict(self, state, deterministic=True): return 0, None
            
        self.engine = SimulationEngine(self.hierarchy, self.predictor, DummyModel(), self.migration_env)
        
        self._generator = self._trace_stepper()
        try:
            state = next(self._generator)
            return state, {}
        except StopIteration:
            return np.zeros(8, dtype=np.float32), {}

    def _trace_stepper(self):
        while self.idx < len(self.trace):
            i = self.idx
            obj_id = self.trace[i]
            
            # Predict prewarming
            for w in range(1, self.lookahead + 1):
                future_idx = i + w
                if future_idx < len(self.trace):
                    fut_obj_id = self.trace[future_idx]
                    fut_size = self.sizes[fut_obj_id]
                    self.engine._prefetch(fut_obj_id, fut_size)
                    
            size = self.sizes[obj_id]
            
            # Engine logic replicated to yield state and inject action
            if obj_id not in self.engine.objects:
                obj = self.engine.objects.get(obj_id) # Won't exist
                obj = self.engine.hierarchy.__class__.__module__ # Wait, need DataObject
                # Let's import it
                from hierarchy import DataObject
                obj = DataObject(obj_id, size=size)
                self.engine.hierarchy.add_object(obj, initial_tier=Tier.SSD)
                self.engine.objects[obj_id] = obj
                self.engine.lru_queues[Tier.SSD][obj_id] = obj
            else:
                obj = self.engine.objects[obj_id]
                
            recency = self.engine.time_step - obj.last_accessed if obj.last_accessed != -1 else -1

            # Engine updates access at the start
            
            # Wait, self.engine._predict_cache needs to be populated BEFORE we do obj.update_access
            # Let's match SimulationEngine exactly
            
            # Recency calc before update
            recency = self.engine.time_step - obj.last_accessed if obj.last_accessed != -1 else -1
            
            # Update state
            obj.update_access(self.engine.time_step)
            self.engine.time_step += 1
            
            # Latency and Storage
            self.engine.total_latency += self.engine.hierarchy.get_latency(obj)
            self.engine._touch_lru(obj)
            self.engine.total_storage_cost += self.engine.hierarchy.get_storage_cost() / 1000.0
            
            state_key = (obj.size, recency, obj.access_count)
            if state_key in self.engine._predict_cache:
                p_i, t_i = self.engine._predict_cache[state_key]
            else:
                p_i, t_i = self.engine.predictor.predict(size=obj.size, recency=recency, frequency=obj.access_count)
                self.engine._predict_cache[state_key] = (p_i, t_i)
                
            obj.reuse_probability = p_i
            obj.time_to_next_access = t_i
            
            if p_i < 0.5:
                # Fallback to LRU, no RL needed here
                self.idx += 1
                continue
                
            usages = self.engine._get_tier_usages()
            self.migration_env.set_state(usages, obj.tier.value, obj.size, p_i, t_i)
            
            # Wait for PPO to give us an action based on this state
            action = yield self.migration_env.current_state
            
            target_tier = Tier(int(action))
            if target_tier != obj.tier:
                current_usage = self.engine.hierarchy.tier_usage[target_tier]
                capacity = self.engine.hierarchy.configs[target_tier].capacity
                if current_usage + obj.size > capacity:
                    self.engine._evict_lru(target_tier, obj.size)
                    
                success = self.engine.hierarchy.move_object(obj, target_tier)
                if success:
                    self.engine.total_migration_cost += self.migration_env.migration_cost_base * obj.size
                    for t in Tier:
                        if t != target_tier and obj.obj_id in self.engine.lru_queues[t]:
                            del self.engine.lru_queues[t][obj.obj_id]
                    self.engine._touch_lru(obj)
                    
            _, reward, _, _, _ = self.migration_env.step(int(action))
            self.last_reward = float(reward)
            
            self.idx += 1
            
        yield None

    def step(self, action):
        try:
            state = self._generator.send(action)
            if state is None:
                return np.zeros(8, dtype=np.float32), self.last_reward, True, False, {}
            return state, self.last_reward, False, False, {}
        except StopIteration:
            return np.zeros(8, dtype=np.float32), 0.0, True, False, {}
