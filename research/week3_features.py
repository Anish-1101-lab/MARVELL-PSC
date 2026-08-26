import numpy as np
from collections import deque
import pandas as pd

WINDOW = 50  # last 50 accesses as context

def extract_features(trace, idx, window=WINDOW):
    block = trace[idx]
    # Handle the beginning of the trace gracefully by padding if needed
    start_idx = max(0, idx - window)
    history = trace[start_idx:idx]
    
    if not history:
        # Edge case: first item
        return np.array([0, window, 0, 0.0, block % 1000])

    freq = history.count(block)                          # access frequency
    
    # Recency: steps since last hit (default to window size if not found)
    recency = window
    for i, b in enumerate(reversed(history)):
        if b == block:
            recency = i + 1  # 1-indexed steps backward
            break
            
    unique_blocks = len(set(history))                    # working set size
    
    seq_score = sum(1 for i in range(len(history)-1)    # sequentiality
                    if abs(history[i+1]-history[i]) == 1) / max(len(history), 1)
    
    return np.array([freq, recency, unique_blocks, seq_score, block % 1000])


def build_dataset(trace, labels, window=WINDOW):
    X, y = [], []
    # Start at 'window' to have full context, though we can pad earlier accesses
    for idx in range(window, len(trace)):
        X.append(extract_features(trace, idx, window))
        # labels mapping is (block_id, time_step) -> 0,1,2
        y.append(labels.get((trace[idx], idx), 2))  # default to cold tier
    return np.array(X), np.array(y)

if __name__ == "__main__":
    from week2_oracle import belady_3tier_labels
    dummy_trace = [1, 2, 3, 1, 4, 2, 5, 1, 6, 2, 3, 4] * 5  # larger trace
    
    labels = belady_3tier_labels(dummy_trace, hbm_size=2, ssd_size=2)
    X, y = build_dataset(dummy_trace, labels, window=5)
    print("Feature Shape:", X.shape)
    print("Labels Shape:", y.shape)
    print("Sample Features:", X[0])
    print("Sample Label:", y[0])
