import pandas as pd
import numpy as np
import time
import torch
import os
from week2_classifier_policy import PhaseClassifier, ConditionedCacheModel
from week6_eval import simulate_ml_cache_4_tier, evaluate_baseline

def get_predictions_full_trace(model_phase, model_policy, trace, window_size=50, batch_size=2048):
    """
    Generates predictions for EVERY block in the trace (after window_size).
    """
    model_phase.eval()
    model_policy.eval()
    
    n_accesses = len(trace)
    preds_tier = []
    preds_prefetch = []
    
    # 1. Prepare sequences for all indices [window_size, n_accesses]
    indices = np.arange(window_size, n_accesses)
    all_preds_tier = np.zeros(len(indices), dtype=int)
    all_preds_prefetch = np.zeros(len(indices))
    
    start_time = time.perf_counter_ns()
    
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_x = []
            for idx in batch_indices:
                batch_x.append(trace[idx - window_size : idx])
            
            x_tensor = torch.tensor(np.array(batch_x), dtype=torch.long)
            
            # Phase prediction
            logits_phase = model_phase(x_tensor)
            p_phase = logits_phase.argmax(dim=1)
            
            # Policy prediction
            t_logits, p_vals = model_policy(x_tensor, p_phase)
            
            all_preds_tier[i : i + batch_size] = t_logits.argmax(dim=1).numpy()
            all_preds_prefetch[i : i + batch_size] = p_vals.squeeze().numpy()
            
    end_time = time.perf_counter_ns()
    latency_us = (end_time - start_time) / (len(indices) * 1000) if len(indices) > 0 else 0
    
    return all_preds_tier, all_preds_prefetch, latency_us

def simulate_profile(trace_name, cache_size=10000):
    path = f"processed_traces/{trace_name}_labeled.parquet" if trace_name != "mixed" else "dlio_mixed_labeled.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None

    print(f"\n--- Evaluating {trace_name.upper()} Profile (4-Tier) ---")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    trace = df["block_id"].values.astype(int)

    # 1. Load ML Models
    try:
        phase_model = PhaseClassifier(num_phases=4)
        phase_model.load_state_dict(torch.load("phase_classifier.pth"))
        
        policy_model = ConditionedCacheModel(num_phases=4)
        policy_model.load_state_dict(torch.load("policy_model_conditioned.pth"))
        
        # Get predictions for ALL steps (batching to avoid OOM and speed up)
        # Limiting to first 100k for faster simulation in this demo
        trace_subset = trace[:100000]
        preds_tier, preds_prefetch, latency_us = get_predictions_full_trace(phase_model, policy_model, trace_subset)
        
        # Run Simulation
        model_hr = simulate_ml_cache_4_tier(trace_subset, preds_tier, preds_prefetch, cache_size, window=50)
        print(f"ML Model Hit Rate: {model_hr:.2%}")
    except Exception as e:
        print(f"Model Error: {e}")
        import traceback
        traceback.print_exc()
        model_hr, latency_us = 0.0, 0.0
        
    # Baselines
    lru_hr = evaluate_baseline("LRU", trace_subset, cache_size)
    lfu_hr = evaluate_baseline("LFU", trace_subset, cache_size)
    naive_hr = evaluate_baseline("Naive Prefetch", trace_subset, cache_size)
    
    print(f"LRU Hit Rate: {lru_hr:.2%}")
    print(f"LFU Hit Rate: {lfu_hr:.2%}")
    print(f"Naive Prefetch: {naive_hr:.2%}")
    
    return {
        "Ours": model_hr,
        "LRU": lru_hr,
        "LFU": lfu_hr,
        "Naive": naive_hr,
        "Latency_us": latency_us
    }

if __name__ == "__main__":
    results = {}
    for workload in ["resnet", "bert", "unet3d"]:
        res = simulate_profile(workload, cache_size=10000)
        if res:
            results[workload] = res
            
    import pickle
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(results, f)
