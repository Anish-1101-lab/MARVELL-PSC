import numpy as np
import json
from typing import List, Dict

class OracleGenerator:
    """
    Generates 'Ground Truth' storage tiering actions by looking into the future.
    Used to create datasets for SLM Supervised Fine-Tuning (SFT).
    """
    def __init__(self, trace: np.ndarray, sizes: np.ndarray, hbm_capacity: int):
        self.trace = trace
        self.sizes = sizes
        self.hbm_capacity = hbm_capacity
        self.next_access_map = self._build_next_access_map()
        
    def _build_next_access_map(self) -> Dict[int, List[int]]:
        """Pre-calculates all access indices for every object."""
        m = {}
        for idx, obj_id in enumerate(self.trace):
            if obj_id not in m:
                m[obj_id] = []
            m[obj_id].append(idx)
        return m

    def get_optimal_action(self, current_step: int, obj_id: int) -> int:
        """
        Oracle Logic (Offline Optimal):
        1. Calculate DTNA (Distance To Next Access).
        2. If DTNA < HBM_Threshold, the optimal tier is HBM (Action 0).
        3. This simulates Belady's Algorithm for multi-tier storage:
           Always keep the objects with the smallest DTNA in the fastest tiers.
        """
        # Find the next index after 'current_step' where this obj_id appears
        future_accesses = [i for i in self.next_access_map.get(obj_id, []) if i > current_step]
        
        if not future_accesses:
            # Never accessed again in this trace -> Move to slowest tier
            return 3 # SSD (Cold)
            
        dtna = future_accesses[0] - current_step
        
        # Mapping DTNA to Tiers (Thresholds can be tuned based on hardware speeds)
        if dtna < 100:
            return 0 # HBM (Hot)
        elif dtna < 500:
            return 1 # DRAM (Warm)
        else:
            return 2 # NVME (Cool)

    def generate_dataset(self, output_path: str):
        """Exports pairs of (Context, OptimalAction) to JSONL."""
        dataset = []
        # We sample points to avoid massive redundant data
        for i in range(0, len(self.trace), 10):
            obj_id = self.trace[i]
            action = self.get_optimal_action(i, obj_id)
            
            # Simple context: Last 5 accesses
            history = self.trace[max(0, i-5):i].tolist()
            
            entry = {
                "prompt": f"History: {history}, CurrentObj: {obj_id}",
                "optimal_tier": action
            }
            dataset.append(entry)
            
        with open(output_path, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Generated {len(dataset)} ground truth samples to {output_path}")

if __name__ == "__main__":
    # Example usage
    trace = np.random.randint(0, 100, 1000)
    sizes = np.ones(100) * 4096
    generator = OracleGenerator(trace, sizes, hbm_capacity=1024*1024)
    generator.generate_dataset("ground_truth_trace.jsonl")
