#!/usr/bin/env python3
import argparse
import csv
import os
import sys

from psc.core.loader import load_trace, generate_synthetic_trace
from psc.models.tiered_predictor import TieredCachePredictor
from psc.core.simulator import run_simulation

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

def main():
    parser = argparse.ArgumentParser(description="PSC Simulator CLI")
    parser.add_argument("--trace", type=str, default="synthetic", help="Path to trace file or 'synthetic'")
    parser.add_argument("--pattern", type=str, default="zipfian", choices=["zipfian", "sequential", "random_crop"])
    parser.add_argument("--n_accesses", type=int, default=10000)
    parser.add_argument("--lstm", type=str, default=None, help="Path to LSTM weights")
    parser.add_argument("--mlp", type=str, default=None, help="Path to MLP weights")
    parser.add_argument("--clock", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.trace.lower() == "synthetic":
        trace = generate_synthetic_trace(pattern=args.pattern, n_accesses=args.n_accesses)
        workload_name = args.pattern
    else:
        trace = load_trace(args.trace)
        workload_name = os.path.basename(args.trace)

    predictor = TieredCachePredictor(lstm_path=args.lstm, mlp_path=args.mlp)
    
    results = []
    for pol in ["ml", "lru", "lfu", "static"]:
        predictor.reset()
        res = run_simulation(
            trace=trace,
            policy=pol,
            predictor=predictor if pol == "ml" else None,
            clock_ghz=args.clock,
            prefetch_threshold=args.threshold
        )
        results.append(res)

    print_results(results, workload_name, len(trace), args.clock)

    if args.output:
        save_csv(results, args.output)

def print_results(results, workload, n_accesses, clock):
    ml_saved = results[0]["cycles_saved_vs_static"]
    static_cycles = results[3]["total_cycles"]
    ml_saved_pct = (ml_saved / static_cycles) if static_cycles > 0 else 0

    if _HAS_RICH:
        console = Console()
        console.print(Panel(f"Workload: {workload} | Accesses: {n_accesses:,} | Clock: {clock} GHz", style="bold cyan"))
        table = Table(header_style="bold magenta")
        table.add_column("Policy")
        table.add_column("Hit Rate", justify="right")
        table.add_column("Total Cycles", justify="right")
        table.add_column("Cost $USD", justify="right")
        table.add_column("Migrations", justify="right")
        table.add_column("Prefetches", justify="right")

        for r in results:
            style = "bold green" if r["policy"] == "ml" else None
            table.add_row(
                r["policy"], f"{r['hit_rate']:.2%}", f"{r['total_cycles']:,.0f}",
                f"${r['total_cost_usd']:.4f}", f"{r['migrations']:,}", f"{r['prefetch_count']:,}",
                style=style
            )
        console.print(table)
        color = "green" if ml_saved >= 0 else "red"
        console.print(f"[{color}]ML cycles saved vs static: {ml_saved:,.0f} ({ml_saved_pct:.2%} reduction)[/{color}]")
    else:
        print(f"\nWorkload: {workload} | Accesses: {n_accesses:,} | Clock: {clock} GHz")
        print("-" * 80)
        for r in results:
            print(f"{r['policy']:<10} | Hit: {r['hit_rate']:.2%} | Cycles: {r['total_cycles']:>15,.0f} | Cost: ${r['total_cost_usd']:.4f}")
        print("-" * 80)
        print(f"ML cycles saved vs static: {ml_saved:,.0f} ({ml_saved_pct:.2%} reduction)")

def save_csv(results, path):
    keys = ["policy", "hit_rate", "total_cycles", "total_cost_usd", "migrations", "prefetch_count", "cycles_saved_vs_static"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"Saved results to {path}")

if __name__ == "__main__":
    main()
