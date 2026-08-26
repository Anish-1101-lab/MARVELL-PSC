"""
Harmonia Simulation Engine.

Simulates end-to-end execution of I/O workload traces using the Harmonia
Multi-Agent RL controller and computes all key performance, latency, bandwidth,
tail latency, and write amplification (WA) metrics.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time

from ..core.harmonia_controller import HarmoniaHSSController
from psc.core.config import NUM_TIERS, compute_cycles, compute_cost


class HarmoniaSimulator:
    def __init__(
        self,
        cache_capacity_hbm: int = 1000,
        clock_ghz: float = 1.0,
        seed: int = 42
    ):
        self.capacity_hbm = cache_capacity_hbm
        self.clock_ghz = clock_ghz
        self.controller = HarmoniaHSSController(
            cache_capacity_hbm=cache_capacity_hbm,
            num_tiers=NUM_TIERS,
            clock_ghz=clock_ghz,
            seed=seed
        )

    def run(self, trace: List[Dict[str, Any]], inter_arrival_mean_us: float = 100.0, verbose: bool = False) -> Dict[str, Any]:
        """
        Executes trace simulation.
        Args:
            trace: list of event dicts with 'block_id', 'size_bytes', and optional 'is_write'
            inter_arrival_mean_us: mean inter-arrival interval in microseconds
            verbose: whether to print progress
        Returns:
            Dict containing complete performance and overhead statistics.
        """
        self.controller.reset()
        
        latencies_us: List[float] = []
        hits = 0
        misses = 0
        total_cycles = 0.0
        total_cost_usd = 0.0
        total_migrations = 0
        total_migrated_bytes = 0
        total_bus_transit_ns = 0.0
        workload_write_bytes = 0
        
        t_start = time.perf_counter()
        
        for step, event in enumerate(trace):
            is_write = event.get("is_write", False)
            size_bytes = int(event.get("size_bytes", 4096))
            if is_write:
                workload_write_bytes += size_bytes
                
            res = self.controller.handle_request(event, inter_arrival_mean_us=inter_arrival_mean_us)
            
            if res["is_hit"]:
                hits += 1
            else:
                misses += 1
                
            latencies_us.append(res["latency_us"])
            total_cycles += res["cycles"]
            total_cost_usd += res["cost_usd"]
            total_migrations += res["migrations_count"]
            total_migrated_bytes += res["migrated_bytes"]
            total_bus_transit_ns += res["bus_transit_ns"]
            
            if verbose and (step % max(1, len(trace) // 10) == 0):
                print(f"[{step:>6d}/{len(trace)}] Latency: {res['latency_us']:.2f} µs | Hits: {hits}/{step+1}")
                
        elapsed_sec = max(time.perf_counter() - t_start, 1e-6)
        total_accesses = hits + misses
        hit_rate_pct = (hits / total_accesses * 100.0) if total_accesses > 0 else 0.0
        
        lat_arr = np.array(latencies_us)
        avg_latency_us = float(np.mean(lat_arr)) if len(lat_arr) > 0 else 0.0
        
        # Write Amplification = (Workload Writes + Migration Writes) / Workload Writes
        total_workload_bytes = sum(e.get("size_bytes", 4096) for e in trace)
        if workload_write_bytes > 0:
            wa = (workload_write_bytes + total_migrated_bytes) / workload_write_bytes
        else:
            wa = 1.0 + (total_migrated_bytes / max(1, total_workload_bytes))
        
        return {
            "policy": "harmonia_marl",
            "total_accesses": total_accesses,
            "hit_rate_pct": hit_rate_pct,
            "avg_latency_us": avg_latency_us,
            "p50_latency_us": float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0.0,
            "p90_latency_us": float(np.percentile(lat_arr, 90)) if len(lat_arr) > 0 else 0.0,
            "p99_latency_us": float(np.percentile(lat_arr, 99)) if len(lat_arr) > 0 else 0.0,
            "p9999_latency_us": float(np.percentile(lat_arr, 99.99)) if len(lat_arr) > 0 else 0.0,
            "throughput_iops": total_accesses / (np.sum(lat_arr) * 1e-6) if np.sum(lat_arr) > 0 else 0.0,
            "total_cycles": total_cycles,
            "total_cost_usd": total_cost_usd,
            "migrations_count": total_migrations,
            "migrated_mb": total_migrated_bytes / (1024 * 1024),
            "bus_transit_ms": total_bus_transit_ns / 1e6,
            "write_amplification": wa,
            "elapsed_simulation_sec": elapsed_sec
        }
