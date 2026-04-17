import numpy as np
import time
from collections import OrderedDict, defaultdict

# Storage Tier Stack definitions (MLCommons Storage v1.0 / Project Specs)
TIER_STATS = {
    0: {"name": "HBM", "lat_ns": 100.0, "bw_gb": 2000.0, "cost": 30.0},
    1: {"name": "CXL", "lat_ns": 80.0, "bw_gb": 200.0, "cost": 8.0},
    2: {"name": "SSD", "lat_ns": 100000.0, "bw_gb": 7.0, "cost": 0.30},
    3: {"name": "Cold", "lat_ns": 1000000.0, "bw_gb": 3.0, "cost": 0.03}
}

def lru_cache_sim(trace, cache_size):
    cache = OrderedDict()
    hits = 0
    for block in trace:
        if block in cache:
            cache.move_to_end(block)
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[block] = True
    return hits / len(trace) if len(trace) > 0 else 0

def lfu_cache_sim(trace, cache_size):
    cache = set()
    freqs = defaultdict(int)
    hits = 0
    for block in trace:
        freqs[block] += 1
        if block in cache:
            hits += 1
        else:
            if len(cache) >= cache_size:
                least_frequent = min(cache, key=lambda b: (freqs[b], b))
                cache.remove(least_frequent)
            cache.add(block)
    return hits / len(trace) if len(trace) > 0 else 0

def naive_sequential_prefetch(trace, cache_size, prefetch_n=4):
    cache = OrderedDict()
    hits = 0
    for i, block in enumerate(trace):
        if block in cache:
            cache.move_to_end(block)
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[block] = True
            
        # Unconditional Pre-fetch of the next N logical block IDs
        for j in range(1, prefetch_n + 1):
            next_b = block + j
            if next_b not in cache:
                if len(cache) >= cache_size:
                    cache.popitem(last=False)
                cache[next_b] = True
    return hits / len(trace) if len(trace) > 0 else 0

def evaluate_baseline(baseline_name, trace_sequence, cache_size):
    if baseline_name == "LRU":
        return lru_cache_sim(trace_sequence, cache_size)
    elif baseline_name == "LFU":
        return lfu_cache_sim(trace_sequence, cache_size)
    elif baseline_name == "Naive Prefetch":
        return naive_sequential_prefetch(trace_sequence, cache_size, prefetch_n=8)
    return 0.0

def simulate_ml_cache_4_tier(trace, predicted_tiers, predicted_prefetch, cache_size, window=50):
    """
    Simulates a 4-tier storage hierarchy where Tier 0/1 are the 'cacheable' tiers.
    predicted_tiers: categorical (0, 1, 2, 3)
    predicted_prefetch: scalar (0 to 8)
    """
    # For simplification of hit-rate, we treat Tier 0 (HBM) and Tier 1 (CXL) as the 'cached' state
    # Tier 2 and 3 are the 'slow' backing stores.
    cache = OrderedDict()
    hits = 0
    
    # We start after the window used by the classifier
    for i in range(window, len(trace)):
        block = trace[i]
        tier_action = predicted_tiers[i - window]
        prefetch_n = int(np.clip(predicted_prefetch[i - window], 0, 8))
        
        # 1. Check for Hit (in HBM/CXL)
        if block in cache:
            cache.move_to_end(block)
            hits += 1
        else:
            # 2. Admission Decision: Only admit to cache if model predicts Tier 0 or 1
            if tier_action in [0, 1]:
                if len(cache) >= cache_size:
                    cache.popitem(last=False)
                cache[block] = True
        
        # 3. Prefetch Decision: Proactively cache next N blocks
        if prefetch_n > 0:
            for j in range(1, prefetch_n + 1):
                next_b = block + j
                if next_b not in cache:
                    if len(cache) >= cache_size:
                        cache.popitem(last=False)
                    cache[next_b] = True
                    
    return hits / (len(trace) - window) if (len(trace) - window) > 0 else 0
