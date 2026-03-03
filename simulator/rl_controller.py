import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any

from hierarchy import Tier, StorageHierarchy

class MigrationEnv(gym.Env):
    """
    A custom Gymnasium environment for the RL Migration Controller.
    The agent receives state information about an object and the current storage hierarchy's utilization,
    and returns an action corresponding to the target tier for the object.
    
    Action Space: 
    0: HBM
    1: DRAM
    2: NVME
    3: SSD
    
    Observation Space:
    [HBM_usage_pct, DRAM_usage_pct, NVME_usage_pct, SSD_usage_pct,
     obj_current_tier, obj_size, p_i (reuse prob), t_i (time to next access)]
    """
    def __init__(self, hierarchy, alpha=1.0, beta=1.0, gamma=1.0, max_t_i=10000.0):
        super(MigrationEnv, self).__init__()
        
        self.action_space = spaces.Discrete(4)
        
        # Observation space bounds (8 features)
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 3.0, float('inf'), 1.0, float('inf')], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_t_i = max_t_i
        self.hierarchy = hierarchy
        
        # Tier configs matching hierarchy for reward calculation
        # Latency, Cost, Migration Base Cost
        self.tier_params = {
            t.value: {'latency': config.base_latency + config.access_penalty, 
                      'cost': config.storage_cost}
            for t, config in hierarchy.configs.items()
        }
        self.migration_cost_base = 50.0  # Penalty for moving data
        self.capacity_penalty = 10000.0  # Penalty for trying to place in a full tier
        
        self.current_state = None
        self.current_tier = None
        self.obj_size = None
        
    def set_state(self, usages: list, current_tier: int, size: int, p_i: float, t_i: float):
        """Called by the simulator before each step to inject the current physical state."""
        # Normalize t_i for neural network stability
        t_i_norm = min(t_i, self.max_t_i) / self.max_t_i
        
        # usages here is passed as raw values, we must convert it to percentages based on dynamic capacity
        usage_pcts = []
        for i, t in enumerate([Tier.HBM, Tier.DRAM, Tier.NVME, Tier.SSD]):
            cap = self.hierarchy.configs[t].capacity
            usage_pcts.append(usages[i] / cap if cap > 0 and cap != float('inf') else 0.0)
            
        self.current_state = np.array([
            usage_pcts[0], usage_pcts[1], usage_pcts[2], usage_pcts[3],
            float(current_tier), float(size), float(p_i), float(t_i_norm)
        ], dtype=np.float32)
        
        self.usages = usage_pcts
        self.current_tier = current_tier
        self.obj_size = size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.current_state is None:
            self.current_state = np.zeros(8, dtype=np.float32)
        return self.current_state, {}

    def step(self, action: int):
        target_tier = action
        
        # Extrapolate metrics
        latency_penalty = self.tier_params[target_tier]['latency']
        storage_cost = self.tier_params[target_tier]['cost'] * self.obj_size
        
        migration_penalty = 0.0
        if target_tier != self.current_tier:
            migration_penalty = self.migration_cost_base * self.obj_size
            
        # Capacity check - if target_tier is full (usage > 0.99)
        # We penalize heavily to train agent to avoid full tiers
        capacity_violation = 0.0
        if self.usages[target_tier] >= 0.99 and target_tier != self.current_tier:
             capacity_violation = self.capacity_penalty
             
        # Demotion penalty: If the predictor says high reuse (p_i > 0.8) but agent picks > Tier.HBM
        # Penalize it heavily so it stops evicting pre-warmed items
        demotion_penalty = 0.0
        p_i = self.current_state[6]
        if p_i > 0.8 and target_tier > 0:
            demotion_penalty = 50000.0
             
        # J = alpha * L_t + beta * C_t + gamma * M_t + Capacity_Penalty + Demotion_Penalty
        # Reward is -J
        J = (self.alpha * latency_penalty) + \
            (self.beta * storage_cost) + \
            (self.gamma * migration_penalty) + \
            capacity_violation + \
            demotion_penalty
            
        reward = -J
        
        # This is essentially a contextual bandit setup (episode length 1) 
        # But we can keep it running and just step with new states injected by the engine
        # Let's say it doesn't terminate from the Env perspective unless max steps, 
        # but the logic happens externally.
        terminated = False
        truncated = False
        info = {'target_tier': target_tier, 'cost': J}
        
        return self.current_state, float(reward), terminated, truncated, info
