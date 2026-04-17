import pandas as pd
import numpy as np
import os

def compute_frequency_tiers(trace):
    """
    Assigns tiers based on access frequency quartiles:
    Top 25% frequent -> Tier 0 (HBM)
    Next 25% -> Tier 1 (CXL)
    Next 25% -> Tier 2 (SSD)
    Bottom 25% -> Tier 3 (Cold)
    """
    counts = pd.Series(trace).value_counts()
    n_blocks = len(counts)
    
    # Calculate quartile boundaries
    q1 = int(0.25 * n_blocks)
    q2 = int(0.50 * n_blocks)
    q3 = int(0.75 * n_blocks)
    
    tier_map = {}
    for i, (blk, count) in enumerate(counts.items()):
        if i < q1:
            tier_map[blk] = 0 # HBM
        elif i < q2:
            tier_map[blk] = 1 # CXL
        elif i < q3:
            tier_map[blk] = 2 # SSD
        else:
            tier_map[blk] = 3 # Cold
            
    # Map back to the full sequence
    return [tier_map.get(blk, 3) for blk in trace]

def label_traces_with_oracle(csv_path="dlio_synthetic_trace.csv", output_path="dlio_labeled_trace.csv"):
    print(f"\nLoading traces from {csv_path}...")
    if csv_path.endswith(".parquet"):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)
    
    trace_sequence = df["block_id"].tolist()
    
    print("Running Frequency Quartile Mapping for 4-Tier Ground Truth...")
    tier_labels = compute_frequency_tiers(trace_sequence)
    
    print("Mapping 4-tier labels to dataframe...")
    df["optimal_tier"] = tier_labels
    
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, engine='pyarrow')
    else:
        df.to_csv(output_path, index=False)
    print(f"Labeled traces saved to {output_path}")

if __name__ == "__main__":
    # Process local synthetic traces
    if os.path.exists("dlio_resnet_trace.csv"):
        label_traces_with_oracle("dlio_resnet_trace.csv", "dlio_resnet_labeled.csv")
    if os.path.exists("dlio_bert_trace.csv"):
        label_traces_with_oracle("dlio_bert_trace.csv", "dlio_bert_labeled.csv")
    if os.path.exists("dlio_mixed_trace.csv"):
        label_traces_with_oracle("dlio_mixed_trace.csv", "dlio_mixed_labeled.csv")
        
    # Process high-fidelity calibrated traces
    if os.path.exists("processed_traces/resnet_normalized.parquet"):
        label_traces_with_oracle("processed_traces/resnet_normalized.parquet", "processed_traces/resnet_labeled.parquet")
    if os.path.exists("processed_traces/bert_normalized.parquet"):
        label_traces_with_oracle("processed_traces/bert_normalized.parquet", "processed_traces/bert_labeled.parquet")
    if os.path.exists("processed_traces/unet3d_normalized.parquet"):
        label_traces_with_oracle("processed_traces/unet3d_normalized.parquet", "processed_traces/unet3d_labeled.parquet")
