import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

# Define the custom reward function requested by user
def compute_reward(action, ground_truth_tier, hit):
    if hit:
        base = +1.0                          # cache hit is always good
    else:
        base = -0.5                          # miss penalty
    
    # Migration cost penalty (proportional to tier distance)
    # Tier 0 = HBM, 1 = SSD, 2 = NVMe
    tier_distances = {(0,1): 0.1, (0,2): 0.3, (1,2): 0.2, 
                      (1,0): 0.15, (2,0): 0.4, (2,1): 0.25}
    cost = tier_distances.get((ground_truth_tier, action), 0)
    
    return base - cost

class CacheEnv(gym.Env):
    """
    A Gym environment simulating trace replay. 
    State: Sequence of latest N access blocks (padded if needed)
    Action: 0 (HBM), 1 (SSD), or 2 (NVMe)
    """
    def __init__(self, trace, labels, window=50):
        super(CacheEnv, self).__init__()
        self.trace = trace
        self.labels = labels
        self.window = window
        self.current_step = window
        
        # Action space: 0, 1, 2
        self.action_space = spaces.Discrete(3)
        # Observation space: array of block IDs (integer tokens up to 100k)
        self.observation_space = spaces.Box(low=0, high=100000, 
                                            shape=(window,), dtype=np.int32)
                                            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_obs(), {}
        
    def _get_obs(self):
        seq = self.trace[self.current_step - self.window : self.current_step]
        return np.array(seq, dtype=np.int32) % 100000
        
    def step(self, action):
        blk = self.trace[self.current_step]
        gt_tier = self.labels.get((blk, self.current_step), 2)
        
        # A hit means the model matched the GT tier or placed it in a faster tier than needed
        hit = action <= gt_tier
        
        reward = compute_reward(action, gt_tier, hit)
        
        self.current_step += 1
        done = self.current_step >= len(self.trace) - 1
        truncated = False
        
        obs = self._get_obs() if not done else np.zeros((self.window,), dtype=np.int32)
        return obs, reward, done, truncated, {}

if __name__ == "__main__":
    from week2_oracle import belady_3tier_labels
    
    # Generate random trace of length 2000 for RL training
    np.random.seed(42)
    dummy_trace = np.random.randint(0, 500, size=2000).tolist()
    labels = belady_3tier_labels(dummy_trace, hbm_size=10, ssd_size=20)
    
    print("Initializing RL Environment...")
    env = CacheEnv(dummy_trace, labels, window=50)
    
    # Use PPO from stable_baselines3
    # MlpPolicy is fine because observations are flattened windows of block IDs,
    # though eventually we'd extract embeddings from our pretrained CacheTransformer.
    print("Setting up PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    
    # Fine-tune for ~1000 steps locally (user requested 50K for actual, using 1K here for speed dummy testing)
    print("Fine-tuning model via RL...")
    model.learn(total_timesteps=1000)
    
    model.save("ppo_cache_agent")
    print("Saved RL agent to ppo_cache_agent.zip")
