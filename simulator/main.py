import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import argparse

from hierarchy import StorageHierarchy
from workload import WorkloadGenerator
from rocksdb_workload import RocksDBWorkloadGenerator
from predictor import LightGBMPredictor
from rl_controller import MigrationEnv
from engine import SimulationEngine
from baseline import HarmoniaBaselineEngine

def run_evaluation(trace_mode='synthetic', rocksdb_trace_path=None):
    # 1. Generate Workload
    if trace_mode == 'synthetic':
        print("Generating synthetic DLRM/LLM workload trace...")
        wg = WorkloadGenerator(num_objects=5000, zipf_alpha=1.2, seed=42)
        df_train = wg.generate_dataset(num_accesses=10000, reuse_window=500)
        eval_trace = wg.generate_trace(num_accesses=20000)
        eval_sizes = wg.object_sizes
    else:
        print(f"Loading RocksDB trace from {rocksdb_trace_path}...")
        wg = RocksDBWorkloadGenerator(rocksdb_trace_path)
        # Load up to 50,000 accesses for speed
        df_full = wg.load_trace(max_rows=50000)
        # We need to split into train and eval
        train_len = min(10000, len(df_full) // 3)
        
        df_train_raw = df_full.iloc[:train_len].copy()
        df_eval_raw = df_full.iloc[train_len:].copy()
        
        df_train = wg.generate_dataset(df_train_raw, reuse_window=500)
        
        # We only need the trace array and sizes for evaluation
        eval_trace, eval_sizes = wg.get_eval_arrays(df_eval_raw)
    
    # 2. Train Predictor
    print("Training LightGBM Predictor...")
    predictor = LightGBMPredictor()
    predictor.train(df_train)
    
    # 3. Setup Integrated Framework (RL + Predictor)
    print("\n=======================================================")
    print("   Running Integrated Framework (Predictor + RL)")
    print("=======================================================")
    hierarchy_rl = StorageHierarchy()
    env = MigrationEnv(hierarchy_rl, alpha=10.0, beta=1.0, gamma=5.0)
    
    import os
    # Load the trained RL model if it exists
    model_path = "ppo_migration_model.zip"
    if os.path.exists(model_path):
        print(f"Loading trained RL model from {model_path}...")
        rl_agent = PPO.load(model_path, env=env)
    else:
        print("Initializing untrained dummy RL model...")
        rl_agent = PPO("MlpPolicy", env, verbose=0, seed=42)
    
    engine_rl = SimulationEngine(hierarchy_rl, predictor, rl_agent, env)
    res_rl = engine_rl.run_simulation(eval_trace, eval_sizes)
    
    # 4. Setup Baseline Framework (Harmonia Threshold/LRU)
    print("\n=======================================================")
    print("   Running Baseline Framework (Harmonia Threshold/LRU)")
    print("=======================================================")
    hierarchy_base = StorageHierarchy()
    engine_base = HarmoniaBaselineEngine(hierarchy_base)
    res_base = engine_base.run_simulation(eval_trace, eval_sizes)
    
    # 5. Setup Pure LRU Baseline Framework
    print("\n=======================================================")
    print("   Running Baseline Framework (Pure LRU)")
    print("=======================================================")
    from baseline import PureLRUBaselineEngine
    hierarchy_pure_lru = StorageHierarchy()
    engine_pure_lru = PureLRUBaselineEngine(hierarchy_pure_lru)
    res_pure_lru = engine_pure_lru.run_simulation(eval_trace, eval_sizes)
    
    # 6. Display Comparisons
    print("\n--- Execution Multi-Metric Comparison ---")
    print(f"{'Metric':<25} | {'Integrated (Ours)':<20} | {'Baseline (Harmonia)':<20} | {'Baseline (Pure LRU)':<20}")
    print("-" * 95)
    for key in res_rl.keys():
        if key == 'history': continue
        val_rl = res_rl[key]
        val_base = res_base[key]
        val_pure_lru = res_pure_lru[key]
        print(f"{key.replace('_', ' ').capitalize():<25} | {val_rl:<20.2f} | {val_base:<20.2f} | {val_pure_lru:<20.2f}")
        
    # Validation against Goals
    print("\n--- Validation Checklist ---")
    print("1. Reuse Prediction (`P_i`, `T_i`): Integrated via LightGBM predictor.")
    print("2. Thrashing Control ('Foresight'): LRU fallback applied when P_i < threshold; Migration Costs penalized in RL state.")
    print("3. Workload Awareness: Trace generated explicitly using DLRM/LLM Zipfian distribution skew.")

    # 7. Generate Plots
    print("\n--- Generating Plots ---")
    import os
    artifact_dir = "/Users/anish/.gemini/antigravity/brain/1f0d34bf-42e1-4db7-b5ef-55ba80194dce"
    os.makedirs(artifact_dir, exist_ok=True)
    
    hist_rl = res_rl['history']
    hist_base = res_base['history']
    hist_pure = res_pure_lru['history']
    steps = hist_rl['steps']
    
    # Plot 1: Just Integrated (Ours) - Latency & Migration Cost over time
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Accesses')
    ax1.set_ylabel('Total Latency', color='tab:blue')
    ax1.plot(steps, hist_rl['latency'], color='tab:blue', label='Latency')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Migration Cost', color='tab:red')
    ax2.plot(steps, hist_rl['migration_cost'], color='tab:red', label='Migration Cost')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Integrated RL Model: Latency & Migration Cost over Time')
    fig.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'rl_model_metrics.png'))
    plt.close()
    
    # Plot 2: Compare Latency
    plt.figure(figsize=(10, 6))
    plt.plot(steps, hist_rl['latency'], label='Integrated (Ours)')
    plt.plot(steps, hist_base['latency'], label='Harmonia')
    plt.plot(steps, hist_pure['latency'], label='Pure LRU')
    plt.xlabel('Accesses')
    plt.ylabel('Total Latency')
    plt.title('Comparison: Latency over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifact_dir, 'compare_latency.png'))
    plt.close()

    # Plot 3: Compare Total Cost
    plt.figure(figsize=(10, 6))
    plt.plot(steps, hist_rl['total_cost'], label='Integrated (Ours)')
    plt.plot(steps, hist_base['total_cost'], label='Harmonia')
    plt.plot(steps, hist_pure['total_cost'], label='Pure LRU')
    plt.xlabel('Accesses')
    plt.ylabel('Total Cost ($J$)')
    plt.title('Comparison: Total Cost over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifact_dir, 'compare_total_cost.png'))
    plt.close()
    
    # Plot 4: Compare HBM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(steps, hist_rl['hbm_usage'], label='Integrated (Ours)')
    plt.plot(steps, hist_base['hbm_usage'], label='Harmonia')
    plt.plot(steps, hist_pure['hbm_usage'], label='Pure LRU')
    plt.xlabel('Accesses')
    plt.ylabel('HBM Usage (%)')
    plt.title('Comparison: HBM Tier Usage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifact_dir, 'compare_hbm_usage.png'))
    plt.close()
    
    # Plot 5: RL Framework Tier Usages
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Accesses')
    ax1.set_ylabel('Usage (%)', color='black')
    ax1.plot(steps, hist_rl['hbm_usage'], color='tab:blue', label='HBM Usage (%)')
    ax1.plot(steps, hist_rl['dram_usage'], color='tab:green', label='DRAM Usage (%)')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('SSD Usage (Absolute count)', color='tab:red')
    ax2.plot(steps, hist_rl['ssd_usage'], color='tab:red', linestyle='--', label='SSD Usage')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Ask matplotlib for the plotted lines and labels to combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.title('Integrated RL Model: Tier Placement Over Time')
    fig.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'rl_tier_usage.png'))
    plt.close()
    
    print(f"Plots saved to {artifact_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Storage Simulator")
    parser.add_argument('--mode', choices=['synthetic', 'rocksdb'], default='synthetic', 
                        help="Trace mode to use (synthetic or rocksdb)")
    parser.add_argument('--trace_path', type=str, default=None,
                        help="Path to RocksDB human-readable trace CSV")
    args = parser.parse_args()
    
    if args.mode == 'rocksdb' and not args.trace_path:
        print("Error: --trace_path is required when --mode is rocksdb")
        exit(1)
        
    run_evaluation(trace_mode=args.mode, rocksdb_trace_path=args.trace_path)
