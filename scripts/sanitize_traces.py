import os
import pandas as pd

def sanitize_csv(file_path):
    if not os.path.exists(file_path):
        return
    
    print(f"Sanitizing {file_path}...")
    lines = []
    header = "ts,pid,tid,name,cat,dur"
    
    with open(file_path, "r") as f:
        # Read everything
        all_lines = f.readlines()
        
    clean_lines = [header + "\n"]
    for line in all_lines:
        line = line.strip()
        if not line or line == header:
            continue
        # Check if it has exactly 5 commas (6 fields)
        if line.count(",") == 5:
            # Check if first field is a timestamp (float-able)
            try:
                parts = line.split(",")
                float(parts[0]) # ts
                int(parts[1])   # pid
                clean_lines.append(line + "\n")
            except:
                continue
                
    with open(file_path, "w") as f:
        f.writelines(clean_lines)
    print(f"Done. Kept {len(clean_lines)-1} valid rows.")

if __name__ == "__main__":
    workloads = ["resnet", "bert", "unet3d"]
    for w in workloads:
        path = f"./real_traces/{w}/dlio_trace.csv"
        sanitize_csv(path)
