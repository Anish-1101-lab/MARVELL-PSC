import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

class WorkloadGenerator:
    def __init__(self, num_objects: int = 10000, zipf_alpha: float = 1.2, seed: int = 42):
        self.num_objects = num_objects
        self.zipf_alpha = zipf_alpha
        self.rng = np.random.default_rng(seed)
        
        # Zipf distribution for object IDs (1 to num_objects)
        ranks = np.arange(1, self.num_objects + 1)
        weights = 1.0 / (ranks ** self.zipf_alpha)
        self.probs = weights / np.sum(weights)
        
        # Object properties
        # Random sizes between 1 and 10 units for variability
        self.object_sizes = self.rng.integers(1, 11, size=self.num_objects + 1)
        
    def generate_trace(self, num_accesses: int) -> np.ndarray:
        """Generates a sequence of object IDs based on Zipfian distribution."""
        trace = self.rng.choice(np.arange(1, self.num_objects + 1), size=num_accesses, p=self.probs)
        return trace

    def generate_dataset(self, num_accesses: int, reuse_window: int = 1000) -> pd.DataFrame:
        """
        Generates a sequence of accesses and computes retrospective features 
        and prospective targets (T_i, P_i) for training the predictor.
        
        Returns a DataFrame with columns:
        [time_step, obj_id, size, recency, frequency, time_to_next_access, reuse_probability]
        """
        trace = self.generate_trace(num_accesses)
        
        # Maintain state
        last_seen: Dict[int, int] = {}
        frequency: Dict[int, int] = {}
        
        records = []
        
        # First pass to compute historical features
        for t, obj_id in enumerate(trace):
            size = self.object_sizes[obj_id]
            recency = t - last_seen.get(obj_id, -1) if obj_id in last_seen else -1
            freq = frequency.get(obj_id, 0)
            
            records.append({
                'time_step': t,
                'obj_id': obj_id,
                'size': size,
                'recency': recency,
                'frequency': freq
            })
            
            last_seen[obj_id] = t
            frequency[obj_id] = freq + 1
            
        df = pd.DataFrame(records)
        
        # Second pass (reverse) to compute future targets (T_i, P_i)
        next_seen: Dict[int, int] = {}
        t_i = np.full(num_accesses, -1)
        p_i = np.zeros(num_accesses)
        
        for t in range(num_accesses - 1, -1, -1):
            obj_id = trace[t]
            if obj_id in next_seen:
                t_next = next_seen[obj_id]
                time_to_next = t_next - t
                t_i[t] = time_to_next
                if time_to_next <= reuse_window:
                    p_i[t] = 1.0
            next_seen[obj_id] = t
            
        df['time_to_next_access'] = t_i
        df['reuse_probability'] = p_i
        
        return df
