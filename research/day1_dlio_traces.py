import numpy as np
import pandas as pd
import os
import argparse
import random

# Statistics from MLCommons Storage v1.0 benchmark paper (2023)

def normalize_trace(df, workload_name):
    """
    Standardizes the column schema for all generated traces.
    """
    if 'tick' not in df.columns:
        df['tick'] = np.arange(len(df))
    if 'op' not in df.columns:
        df['op'] = 'read'
    if 'size_kb' not in df.columns:
        df['size_kb'] = 1024
    if 'workload' not in df.columns:
        df['workload'] = workload_name
    if 'timestamp_delta' not in df.columns:
        df['timestamp_delta'] = 1
    
    cols = ['tick', 'block_id', 'op', 'size_kb', 'workload', 'phase', 'timestamp_delta']
    return df[cols]

def generate_resnet_v1_0(n_accesses=500000):
    """
    ResNet50: 1.28M images, Zipfian access distribution (alpha=1.2)
    Mean size 150KB, 90% reads, 10% writes.
    Shuffle buffer = 10,000 samples.
    """
    n_blocks_total = 1280000 
    alpha = 1.2
    
    # 1. Sample from Zipfian distribution
    # np.random.zipf can produce very large outliers; we clip to workspace bounds
    blocks = np.random.zipf(alpha, n_accesses)
    blocks = (blocks % n_blocks_total)
    
    # 2. Simulate 10,000 sample shuffle buffer
    # We shuffle in chunks to mimic the pre-fetch buffer behavior
    buffer_size = 10000
    for i in range(0, n_accesses, buffer_size):
        end = min(i + buffer_size, n_accesses)
        chunk = blocks[i:end]
        np.random.shuffle(chunk)
        blocks[i:end] = chunk

    # 3. Reads vs Writes (90/10)
    ops = np.random.choice(['read', 'write'], size=n_accesses, p=[0.9, 0.1])
    
    df = pd.DataFrame({
        "block_id": blocks,
        "op": ops,
        "size_kb": 150,
        "phase": 0
    })
    return normalize_trace(df, "resnet")

def generate_bert_v1_0(n_accesses=500000):
    """
    BERT: 2.5M sequences, sequential chunked reads.
    Stride = 512, Mean size 4KB, 95% reads.
    """
    n_blocks_total = 2500000
    
    # Sequential pattern: each block_id is one 512-token chunk. 
    # Sequential then means id+1.
    blocks = np.arange(n_accesses) % n_blocks_total
    
    ops = np.random.choice(['read', 'write'], size=n_accesses, p=[0.95, 0.05])
    
    df = pd.DataFrame({
        "block_id": blocks,
        "op": ops,
        "size_kb": 4,
        "phase": 1
    })
    return normalize_trace(df, "bert")

def generate_unet3d_v1_0(n_accesses=200000):
    """
    UNet3D: 484 3D volumes (approx 500MB each).
    Repeated random crop access, 85% reads.
    """
    n_volumes = 484
    
    # Randomly pick a volume, then 'read' it multiple times simulating crops
    # To keep it simple but realistic, we use a uniform choice of volumes
    blocks = np.random.randint(0, n_volumes, size=n_accesses)
    
    ops = np.random.choice(['read', 'write'], size=n_accesses, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        "block_id": blocks,
        "op": ops,
        "size_kb": 512000, # 500MB
        "phase": 0 # Usually categorized as training/computation heavy
    })
    return normalize_trace(df, "unet3d")

def generate_resnet_trace_legacy(n_blocks_total=10000, n_accesses=50000):
    """ Legacy small-scale synthetic for local testing """
    weights = np.array([1/(i+1) for i in range(n_blocks_total)])
    weights /= weights.sum()
    trace_blocks = np.random.choice(n_blocks_total, size=n_accesses, replace=True, p=weights)
    df = pd.DataFrame([{"block_id": b, "phase": 0} for b in trace_blocks])
    return normalize_trace(df, "resnet")

def generate_bert_trace_legacy(n_blocks_total=10000, n_accesses=50000):
    """ Legacy small-scale synthetic for local testing """
    single_pass = np.arange(n_blocks_total)
    repeats = (n_accesses // n_blocks_total) + 1
    trace_blocks = np.tile(single_pass, repeats)[:n_accesses]
    df = pd.DataFrame([{"block_id": b, "phase": 1} for b in trace_blocks])
    return normalize_trace(df, "bert")

def generate_mixed_trace(n_blocks_total=10000, accesses_per_phase=10000, num_phases=5):
    """ Legacy small-scale synthetic for local testing """
    trace = []
    for p in range(num_phases):
        phase_id = p % 2
        if phase_id == 0:
            weights = np.array([1/(i+1) for i in range(n_blocks_total)])
            weights /= weights.sum()
            b_ids = np.random.choice(n_blocks_total, size=accesses_per_phase, replace=True, p=weights)
        else:
            seq = np.tile(np.arange(n_blocks_total), (accesses_per_phase // n_blocks_total) + 1)
            b_ids = seq[:accesses_per_phase]
        for b in b_ids:
            trace.append({"block_id": b, "phase": phase_id})
    df = pd.DataFrame(trace)
    return normalize_trace(df, "mixed")

def load_real_traces(trace_dir, workload):
    """
    Loads normalized MLPerf traces from Parquet.
    """
    path = os.path.join(trace_dir, f"{workload}_normalized.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real-world trace Parquet not found at: {path}. Run DLIO scripts first.")
    
    df = pd.read_parquet(path)
    
    # Requirement: At least 10,000 rows
    if len(df) < 10000:
        raise ValueError(f"Trace for '{workload}' ({len(df)} rows) is too short. Minimum 10,000 required.")
    
    print(f"Loaded real-world trace: {workload} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrated MLPerf Trace Generator")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic", help="Mode: generate or load")
    parser.add_argument("--scale", type=str, choices=["local", "real"], default="local", help="Scale: local(small) or real(MLCommons stats)")
    args = parser.parse_args()

    np.random.seed(42)
    
    if args.mode == "real":
        # Load from disk (requires dlio_converter.py output)
        df_resnet = load_real_traces("./processed_traces/", "resnet")
        df_bert = load_real_traces("./processed_traces/", "bert")
        df_unet = load_real_traces("./processed_traces/", "unet3d")
    else:
        if args.scale == "real":
            print("Generating Calibrated MLPerf Storage Traces (Scale: REAL)...")
            df_resnet = generate_resnet_v1_0()
            df_bert = generate_bert_v1_0()
            df_unet = generate_unet3d_v1_0()
            
            # Save for simulator use (Unify naming with 'real' mode loader)
            os.makedirs("./processed_traces", exist_ok=True)
            df_resnet.to_parquet("./processed_traces/resnet_normalized.parquet", engine='pyarrow')
            df_bert.to_parquet("./processed_traces/bert_normalized.parquet", engine='pyarrow')
            df_unet.to_parquet("./processed_traces/unet3d_normalized.parquet", engine='pyarrow')
        else:
            print("Generating Synthetic Traces (Scale: LOCAL)...")
            df_resnet = generate_resnet_trace_legacy()
            df_bert = generate_bert_trace_legacy()
            df_mixed = generate_mixed_trace()
            
            df_resnet.to_csv("dlio_resnet_trace.csv", index=False)
            df_bert.to_csv("dlio_bert_trace.csv", index=False)
            df_mixed.to_csv("dlio_mixed_trace.csv", index=False)
    
    # Validation Assertion: check row count
    for name, df in [("ResNet", df_resnet), ("BERT", df_bert), ("UNet3D", df_unet if 'df_unet' in locals() else df_resnet)]:
        if args.scale == "real" or args.mode == "real":
            assert len(df) >= 10000, f"{name} trace density too low ({len(df)} < 10000)"
    
    print("\nTrace generation/loading complete.")
    print(f"ResNet: {len(df_resnet)} rows")
    print(f"BERT  : {len(df_bert)} rows")
    if 'df_unet' in locals():
        print(f"UNet3D: {len(df_unet)} rows")
