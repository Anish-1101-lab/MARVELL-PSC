#!/usr/bin/env python3
"""
MARVELL-PSC vs. Harmonia: Comprehensive Multi-Dimension Benchmark Runner

Completely rewritten from scratch to correctly implement Oracle Upper Bound,
incorporate inference overhead into latency, and accurately track migrations.

Compares:
  1. Marvell PSC (Phase-Conditioned Prefetching + 42.5us overhead)
  2. Harmonia MARL (RL Placement & Migration + 0.24us overhead)
  3. LRU Baseline
  4. LFU Baseline
  5. Oracle Upper Bound (Perfect offline clairvoyance + prefetching)

Evaluates across BERT, ResNet50, UNet3D, Strided, and GNN workloads.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any
from collections import OrderedDict, Counter, deque
import numpy as np
import pandas as pd
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psc.core.loader import generate_synthetic_trace
from psc.core.config import TIERS
from harmonia.core.harmonia_controller import HarmoniaHSSController
from scripts.benchmark_inference_overhead import BenchmarkPSCController

def print_banner(title: str):
    print("\n" + "=" * 105)
    print(f"  {title.upper()}")
    print("=" * 105)

def load_all_workloads(n_samples: int = 100000) -> Dict[str, Tuple[List[Dict[str, Any]], int]]:
    """Loads standardized workload traces along with calibrated HBM capacities."""
    workloads = {}
    
    # ResNet50
    res_p = "processed_traces/resnet_normalized.parquet"
    if os.path.exists(res_p):
        df = pd.read_parquet(res_p).head(n_samples)
        workloads["ResNet50 (Zipfian)"] = ([{"block_id": int(r["block_id"]), "size_bytes": int(r["size_kb"])*1024} for _, r in df.iterrows()], 10000)
    else:
        workloads["ResNet50 (Zipfian)"] = (generate_synthetic_trace("zipfian", n_samples), 10000)

    # BERT
    bert_p = "processed_traces/bert_normalized.parquet"
    if os.path.exists(bert_p):
        df = pd.read_parquet(bert_p).head(n_samples)
        workloads["BERT (Sequential)"] = ([{"block_id": int(r["block_id"]), "size_bytes": int(r["size_kb"])*1024} for _, r in df.iterrows()], 1000)
    else:
        workloads["BERT (Sequential)"] = (generate_synthetic_trace("sequential", n_samples), 1000)

    # UNet3D
    unet_p = "processed_traces/unet3d_normalized.parquet"
    if os.path.exists(unet_p):
        df = pd.read_parquet(unet_p).head(n_samples)
        workloads["UNet3D (Random Crop)"] = ([{"block_id": int(r["block_id"]), "size_bytes": int(r["size_kb"])*1024} for _, r in df.iterrows()], 10000)
    else:
        workloads["UNet3D (Random Crop)"] = (generate_synthetic_trace("random_crop", n_samples), 10000)

    # Strided
    str_p = "processed_traces/strided_normalized.parquet"
    if os.path.exists(str_p):
        df = pd.read_parquet(str_p).head(n_samples)
        workloads["Strided (Multi-Stride)"] = ([{"block_id": int(r["block_id"]), "size_bytes": int(r["size_kb"])*1024} for _, r in df.iterrows()], 1000)
    else:
        workloads["Strided (Multi-Stride)"] = (generate_synthetic_trace("strided", n_samples, stride=4), 1000)

    # GNN
    gnn_p = "processed_traces/graph_walk_normalized.parquet"
    if os.path.exists(gnn_p):
        df = pd.read_parquet(gnn_p).head(n_samples)
        workloads["GNN (Graph Walk)"] = ([{"block_id": int(r["block_id"]), "size_bytes": int(r["size_kb"])*1024} for _, r in df.iterrows()], 1000)
    else:
        workloads["GNN (Graph Walk)"] = (generate_synthetic_trace("graph_walk", n_samples), 1000)

    return workloads

# --- SIMULATORS ---

def run_oracle(trace, capacity_hbm):
    """
    True Oracle Upper Bound: Knows all future accesses and can perfectly prefetch
    into the cache in the background. Hit rate is 100%. Migrations are only the
    unavoidable cold misses (unique blocks) plus capacity evictions based on Belady's.
    """
    unique_blocks = set()
    total_bytes = 0
    for ev in trace:
        unique_blocks.add(ev["block_id"])
        total_bytes += ev.get("size_bytes", 4096)
    
    # Calculate Belady optimal misses just to get the accurate minimum migrations
    future_indices = {}
    for idx, ev in enumerate(trace):
        bid = ev["block_id"]
        if bid not in future_indices:
            future_indices[bid] = deque()
        future_indices[bid].append(idx)

    cache = set()
    misses = 0
    for idx, ev in enumerate(trace):
        bid = ev["block_id"]
        future_indices[bid].popleft()
        if bid not in cache:
            misses += 1
            if len(cache) >= capacity_hbm:
                victim = max(cache, key=lambda b: future_indices[b][0] if future_indices[b] else float('inf'))
                cache.remove(victim)
            cache.add(bid)

    total = len(trace)
    return {
        "hit_rate_pct": 100.0,  # Prefetched perfectly
        "avg_latency_us": 0.1,  # Tier 0 (HBM) latency is 100ns = 0.1us
        "p99_latency_us": 0.1,
        "migrations_count": misses,
        "migrated_mb": (misses * 4096) / (1024*1024),
        "write_amplification": 1.0
    }

def run_lru(trace, capacity_hbm):
    cache = OrderedDict()
    hits, misses = 0, 0
    migrations = 0
    migrated_bytes = 0
    latencies = []
    
    for ev in trace:
        bid = ev["block_id"]
        sz = ev.get("size_bytes", 4096)
        
        if bid in cache:
            hits += 1
            cache.move_to_end(bid)
            latencies.append(0.1) # 100 ns HBM
        else:
            misses += 1
            if len(cache) >= capacity_hbm:
                cache.popitem(last=False)
                migrations += 1
                migrated_bytes += sz
            
            cache[bid] = sz
            migrations += 1
            migrated_bytes += sz
            lat_us = (TIERS[2]["latency_ns"] + (sz / TIERS[2]["bandwidth_gbps"])) / 1000.0
            latencies.append(lat_us)
            
    return {
        "hit_rate_pct": hits / len(trace) * 100,
        "avg_latency_us": np.mean(latencies),
        "p99_latency_us": np.percentile(latencies, 99),
        "migrations_count": migrations,
        "migrated_mb": migrated_bytes / (1024*1024),
        "write_amplification": 1.0 + (migrated_bytes / sum(e.get("size_bytes",4096) for e in trace))
    }

def run_lfu(trace, capacity_hbm):
    cache = {}
    freqs = Counter()
    hits, misses = 0, 0
    migrations = 0
    migrated_bytes = 0
    latencies = []
    
    for ev in trace:
        bid = ev["block_id"]
        sz = ev.get("size_bytes", 4096)
        
        if bid in cache:
            hits += 1
            freqs[bid] += 1
            latencies.append(0.1)
        else:
            misses += 1
            if len(cache) >= capacity_hbm:
                victim = min(cache.keys(), key=lambda b: freqs[b])
                cache.pop(victim)
                del freqs[victim]
                migrations += 1
                migrated_bytes += sz
                
            cache[bid] = sz
            freqs[bid] = 1
            migrations += 1
            migrated_bytes += sz
            lat_us = (TIERS[2]["latency_ns"] + (sz / TIERS[2]["bandwidth_gbps"])) / 1000.0
            latencies.append(lat_us)
            
    return {
        "hit_rate_pct": hits / len(trace) * 100,
        "avg_latency_us": np.mean(latencies),
        "p99_latency_us": np.percentile(latencies, 99),
        "migrations_count": migrations,
        "migrated_mb": migrated_bytes / (1024*1024),
        "write_amplification": 1.0 + (migrated_bytes / sum(e.get("size_bytes",4096) for e in trace))
    }

def run_psc(trace, capacity_hbm):
    controller = BenchmarkPSCController(vocab_size=100000, window_size=50)
    cache = OrderedDict()
    tier_map = {}
    hits, misses = 0, 0
    migrations = 0
    migrated_bytes = 0
    latencies = []
    window = deque(maxlen=50)
    overhead_us = 42.5  # PSC inference overhead
    
    for ev in trace:
        bid = ev["block_id"]
        sz = ev.get("size_bytes", 4096)
        
        current_tier = tier_map.get(bid, 2)
        
        # Access
        if bid in cache:
            hits += 1
            cache.move_to_end(bid)
            latencies.append(0.1 + overhead_us)
            current_tier = 0
        else:
            misses += 1
            
        # Predict & Prefetch
        window.append(bid)
        if len(window) >= 50:
            pred = controller.predict(list(window))
            tier_id = pred[0]
            prefetch_n = pred[1]
            stride = pred[3] if len(pred) == 4 else 1
        else:
            tier_id = 0
            prefetch_n = 0
            stride = 1
            
        # Placement of missed block based on tier_id
        if bid not in cache:
            if tier_id == 0:
                if len(cache) >= capacity_hbm:
                    ev_bid, ev_sz = cache.popitem(last=False)
                    tier_map[ev_bid] = 2
                    migrations += 1
                    migrated_bytes += ev_sz
                cache[bid] = sz
                migrations += 1
                migrated_bytes += sz
                lat_us = (TIERS[current_tier]["latency_ns"] + (sz / TIERS[current_tier]["bandwidth_gbps"])) / 1000.0
                latencies.append(lat_us + overhead_us)
            else:
                tier_map[bid] = tier_id
                lat_us = (TIERS[current_tier]["latency_ns"] + (sz / TIERS[current_tier]["bandwidth_gbps"])) / 1000.0
                latencies.append(lat_us + overhead_us)
                if current_tier != tier_id:
                    migrations += 1
                    migrated_bytes += sz
            
        # Execute Prefetching
        if prefetch_n > 0:
            for offset in range(1, prefetch_n + 1):
                pf_bid = bid + offset * stride
                if pf_bid not in cache:
                    if len(cache) >= capacity_hbm:
                        ev_bid, ev_sz = cache.popitem(last=False)
                        tier_map[ev_bid] = 2
                        migrations += 1
                        migrated_bytes += ev_sz
                    cache[pf_bid] = sz
                    tier_map[pf_bid] = 0
                    migrations += 1
                    migrated_bytes += sz
                        
    return {
        "hit_rate_pct": hits / len(trace) * 100,
        "avg_latency_us": np.mean(latencies),
        "p99_latency_us": np.percentile(latencies, 99),
        "migrations_count": migrations,
        "migrated_mb": migrated_bytes / (1024*1024),
        "write_amplification": 1.0 + (migrated_bytes / sum(e.get("size_bytes",4096) for e in trace))
    }

def run_harmonia(trace, capacity_hbm):
    # Use Harmonia Controller directly to manage overhead correctly
    controller = HarmoniaHSSController(cache_capacity_hbm=capacity_hbm, num_tiers=4)
    controller.reset()
    latencies = []
    hits, misses = 0, 0
    migrations = 0
    migrated_bytes = 0
    overhead_us = 0.24  # Harmonia critical path overhead
    
    for ev in trace:
        res = controller.handle_request(ev, inter_arrival_mean_us=100.0)
        
        if res["is_hit"]:
            hits += 1
        else:
            misses += 1
            
        latencies.append(res["latency_us"] + overhead_us)
        migrations += res["migrations_count"]
        migrated_bytes += res["migrated_bytes"]
        
    return {
        "hit_rate_pct": hits / len(trace) * 100,
        "avg_latency_us": np.mean(latencies),
        "p99_latency_us": np.percentile(latencies, 99),
        "migrations_count": migrations,
        "migrated_mb": migrated_bytes / (1024*1024),
        "write_amplification": 1.0 + (migrated_bytes / sum(e.get("size_bytes",4096) for e in trace))
    }

def main():
    print_banner("MARVELL-PSC vs. HARMONIA: Comprehensive Benchmark Suite")
    print("Evaluating with strictly tracked Oracle bounds and inference overheads.\n")
    
    workloads = load_all_workloads(n_samples=100000)
    
    for wname, (trace, cap) in workloads.items():
        print_banner(f"Workload: {wname.upper()} (Accesses: {len(trace):,}, HBM Capacity: {cap:,})")
        
        oracle = run_oracle(trace, cap)
        lru = run_lru(trace, cap)
        lfu = run_lfu(trace, cap)
        psc = run_psc(trace, cap)
        harmonia = run_harmonia(trace, cap)

        print(f"\n{'Policy / Model':<22} | {'Hit Rate (%)':>12} | {'Avg Lat (µs)':>14} | {'P99 Lat (µs)':>14} | {'Migrations':>12} | {'Migrated MB':>14} | {'WA':>6}")
        print("-" * 105)
        print(f"{'Oracle Upper Bound':<22} | {oracle['hit_rate_pct']:>11.2f}% | {oracle['avg_latency_us']:>14.2f} | {oracle['p99_latency_us']:>14.2f} | {oracle['migrations_count']:>12,} | {oracle['migrated_mb']:>14.2f} | {oracle['write_amplification']:>6.2f}")
        print(f"{'LRU Baseline':<22} | {lru['hit_rate_pct']:>11.2f}% | {lru['avg_latency_us']:>14.2f} | {lru['p99_latency_us']:>14.2f} | {lru['migrations_count']:>12,} | {lru['migrated_mb']:>14.2f} | {lru['write_amplification']:>6.2f}")
        print(f"{'LFU Baseline':<22} | {lfu['hit_rate_pct']:>11.2f}% | {lfu['avg_latency_us']:>14.2f} | {lfu['p99_latency_us']:>14.2f} | {lfu['migrations_count']:>12,} | {lfu['migrated_mb']:>14.2f} | {lfu['write_amplification']:>6.2f}")
        print(f"{'Harmonia (MARL)':<22} | {harmonia['hit_rate_pct']:>11.2f}% | {harmonia['avg_latency_us']:>14.2f} | {harmonia['p99_latency_us']:>14.2f} | {harmonia['migrations_count']:>12,} | {harmonia['migrated_mb']:>14.2f} | {harmonia['write_amplification']:>6.2f}")
        print(f"{'Marvell PSC (2-Stage)':<22} | {psc['hit_rate_pct']:>11.2f}% | {psc['avg_latency_us']:>14.2f} | {psc['p99_latency_us']:>14.2f} | {psc['migrations_count']:>12,} | {psc['migrated_mb']:>14.2f} | {psc['write_amplification']:>6.2f}")
        print("-" * 105)

if __name__ == "__main__":
    main()
