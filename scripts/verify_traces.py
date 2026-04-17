import os
import pandas as pd

def verify_trace(workload_dir, workload_name):
    # Expecting dlio_trace.csv in the directory based on the custom python wrapper
    csv_file = os.path.join(workload_dir, "dlio_trace.csv")
    
    if not os.path.exists(csv_file):
        print(f"[{workload_name.upper()}] No trace CSV found at {csv_file}!")
        return

    try:
        df = pd.read_csv(csv_file)
        print(f"\n=========================================")
        print(f"{workload_name.upper()} Trace Validation")
        print(f"=========================================")
        print(f"File: {csv_file}")
        print(f"Total Rows (Accesses): {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 Rows:")
        print(df.head(5).to_string(index=False))
        
    except Exception as e:
        print(f"[{workload_name.upper()}] Failed to read {csv_file}: {e}")

if __name__ == "__main__":
    base_dir = "./real_traces"
    workloads = ["resnet", "bert", "unet3d"]
    
    for w in workloads:
        workload_dir = os.path.join(base_dir, w)
        if os.path.exists(workload_dir):
            verify_trace(workload_dir, w)
        else:
            print(f"[{w.upper()}] Missing directory {workload_dir}")
