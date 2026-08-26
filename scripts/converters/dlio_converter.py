import pandas as pd
import numpy as np
import os
import re
import sys
import glob

def extract_block_id(name):
    """
    Extracts the digit part from filenames like 'img_176_of_512.npz' or 'part_200_of_512.npz'.
    """
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    return int(hash(name) % 1000000)

def convert_dlio_trace(folder_path, workload_name):
    print(f"Processing workload: {workload_name} from {folder_path}...")
    
    # 1. Read all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {folder_path}")
        return 0

    df_list = []
    for f in csv_files:
        try:
            df_part = pd.read_csv(f)
            if df_part.empty:
                continue
            df_list.append(df_part)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list:
        print(f"No valid data found for {workload_name}")
        return 0
        
    df = pd.concat(df_list, ignore_index=True)

    # 2. Verify columns
    required_cols = ['ts', 'name', 'cat']
    for col in required_cols:
        if col not in df.columns:
            # Handle possible header corruption if it's there
            print(f"Error: Missing column '{col}' in {workload_name}. Found: {df.columns.tolist()}")
            return 0

    # 3. Normalize schema
    df = df.sort_values(by='ts').reset_index(drop=True)
    
    # tick: integer, monotonically increasing
    df['tick'] = df.index
    
    # block_id: extracted from filename
    df['block_id'] = df['name'].astype(str).apply(extract_block_id)
    
    # op: "read" or "write"
    df['op'] = df['cat'].str.lower()
    
    # size_kb: assume 1MB (1024KB)
    df['size_kb'] = 1024
    
    # workload: the label
    df['workload'] = workload_name
    
    # phase: -1 for unknown
    df['phase'] = -1
    
    # timestamp_delta: difference in ticks (literal request)
    df['timestamp_delta'] = df['tick'].diff().fillna(0).astype(int)

    # 4. Filter to standard schema
    final_cols = ['tick', 'block_id', 'op', 'size_kb', 'workload', 'phase', 'timestamp_delta']
    df = df[final_cols]

    # 5. Save as Parquet
    os.makedirs("./processed_traces", exist_ok=True)
    output_path = f"./processed_traces/{workload_name}_normalized.parquet"
    df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print(f"Successfully saved {output_path} ({len(df)} rows)")
    return len(df)

if __name__ == "__main__":
    workloads = {
        "resnet": "./real_traces/resnet/",
        "bert": "./real_traces/bert/",
        "unet3d": "./real_traces/unet3d/"
    }
    
    total_rows = {}
    for name, path in workloads.items():
        try:
            count = convert_dlio_trace(path, name)
            total_rows[name] = count
        except Exception as e:
            print(f"Failed to convert {name}: {e}")

    print("\n=========================================")
    print("Conversion Summary (Row Counts):")
    for name, count in total_rows.items():
        print(f"{name.upper():<10}: {count:>8}")
    print("=========================================")
