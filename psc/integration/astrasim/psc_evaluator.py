import pandas as pd
import numpy as np
import torch
import os
import sys

# Add root simulator directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from week2_classifier_policy import PhaseClassifier, ConditionedCacheModel
from project.baselines import LRUCache, LFUCache, StaticCache

# --- Calibrated AstraSim Performance & Cost Metrics ---
TIER_INFO = {
    0: {"name": "HBM",  "cost_gb": 30.00, "cycles": 176},
    1: {"name": "CXL",  "cost_gb": 8.00,  "cycles": 848},
    2: {"name": "SSD",  "cost_gb": 0.30,  "cycles": 121942},
    3: {"name": "Cold", "cost_gb": 0.03,  "cycles": 1051200}
}

WORKLOADS = ["resnet", "bert", "unet3d"]

def run_eval(workload_name):
    print(f"\n>>> Running AstraSim-PSC Evaluation for {workload_name.upper()}...")
    
    # Load trace
    trace_path = f"simulator/processed_traces/{workload_name}_normalized.parquet"
    if not os.path.exists(trace_path):
        return None
    
    trace_df = pd.read_parquet(trace_path).head(500)
    trace = trace_df["block_id"].values.astype(int)
    
    # 1. Load ML Models
    try:
        phase_model = PhaseClassifier(num_phases=4)
        phase_model.load_state_dict(torch.load("simulator/phase_classifier.pth", map_location="cpu", weights_only=True))
        phase_model.eval()
        
        policy_model = ConditionedCacheModel(num_phases=4)
        policy_model.load_state_dict(torch.load("simulator/policy_model_conditioned.pth", map_location="cpu", weights_only=True))
        policy_model.eval()
    except Exception as e:
        print(f"  Error loading models: {e}")
        return None

    # 2. Setup Baselines
    policies = {
        "PSC (Dynamic)": "ml",
        "LRU": LRUCache(capacity=100),
        "LFU": LFUCache(capacity=100),
        "Static (Cold)": StaticCache()
    }
    
    results = {}
    window_size = 50
    
    for p_name, p_obj in policies.items():
        total_cycles = 0
        total_cost = 0
        hits = 0
        
        for i in range(window_size, len(trace)):
            bid = trace[i]
            
            if p_name == "PSC (Dynamic)":
                # Prep input window
                x_seq = torch.tensor(trace[i-window_size:i], dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    phase_id = phase_model(x_seq).argmax(dim=1)
                    tier_logits, _ = policy_model(x_seq, phase_id)
                    tier_id = int(tier_logits.argmax(dim=1).item())
            else:
                tier_id = p_obj.access(bid)
            
            # Map metrics
            total_cycles += TIER_INFO[tier_id]["cycles"]
            total_cost += (1/1024) * TIER_INFO[tier_id]["cost_gb"] # 1MB block cost
            if tier_id in [0, 1]: hits += 1
                
        results[p_name] = {
            "cycles": total_cycles,
            "cost": total_cost,
            "hr": hits / (len(trace) - window_size)
        }

    return results

def main():
    report = {}
    for w in WORKLOADS:
        res = run_eval(w)
        if res: report[w] = res
            
    print("\n" + "="*90)
    print(f"{'Workload':<10} | {'Policy':<15} | {'AstraSim Cycles':>18} | {'Cost ($)':>10} | {'Hit%':>8} | {'Savings %'}")
    print("-" * 90)
    
    for w, pols in report.items():
        base_cost = pols["Static (Cold)"]["cost"]
        for p_name, m in pols.items():
            savings = (1 - m["cost"]/base_cost)*100 if base_cost > 0 else 0
            print(f"{w:<10} | {p_name:<15} | {m['cycles']:>18,} | {m['cost']:>10.4f} | {m['hr']:>7.1%}\t | {savings:>6.1%}")
        print("-" * 90)

if __name__ == "__main__":
    main()
