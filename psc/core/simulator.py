from typing import Dict, List, Optional
from .config import compute_cycles, compute_cost, NUM_TIERS
from .baselines import LRUCache, LFUCache, StaticCache

DEFAULT_TIER = 3  # Cold Storage

def run_simulation(
    trace: List[Dict],
    policy: str,
    predictor=None,
    clock_ghz: float = 1.0,
    prefetch_threshold: float = 0.7,
    cache_capacity: int = 500,
    verbose: bool = False,
) -> Dict:
    valid_policies = ("ml", "lru", "lfu", "static")
    if policy not in valid_policies:
        raise ValueError(f"Unknown policy '{policy}'. Expected one of {valid_policies}")

    if policy == "ml" and predictor is None:
        raise ValueError("predictor must be provided when policy='ml'")

    lru_cache: Optional[LRUCache] = None
    lfu_cache: Optional[LFUCache] = None
    static_cache: Optional[StaticCache] = None

    if policy == "lru":
        lru_cache = LRUCache(capacity=cache_capacity)
    elif policy == "lfu":
        lfu_cache = LFUCache(capacity=cache_capacity)
    elif policy == "static":
        static_cache = StaticCache()

    cache_state: Dict[int, int] = {}
    hits = 0
    misses = 0
    total_cycles = 0.0
    total_cost_usd = 0.0
    migrations = 0
    prefetch_count = 0
    static_total_cycles = 0.0

    n_events = len(trace)

    for step, event in enumerate(trace):
        block_id: int = event["block_id"]
        size_bytes: int = event["size_bytes"]
        current_tier = cache_state.get(block_id, DEFAULT_TIER)

        static_total_cycles += compute_cycles(size_bytes, DEFAULT_TIER, clock_ghz)

        prefetch_prob = 0.0
        phase_label = -1

        if policy == "ml":
            tier_id, prefetch_prob, phase_label = predictor.predict(block_id)
            target_tier = tier_id
        elif policy == "lru":
            target_tier = lru_cache.access(block_id)
        elif policy == "lfu":
            target_tier = lfu_cache.access(block_id)
        else:
            target_tier = static_cache.access(block_id)

        target_tier = max(0, min(target_tier, NUM_TIERS - 1))

        if current_tier == target_tier:
            hits += 1
        else:
            misses += 1
            evict_cycles = compute_cycles(size_bytes, current_tier, clock_ghz)
            load_cycles = compute_cycles(size_bytes, target_tier, clock_ghz)
            total_cycles += evict_cycles + load_cycles
            migrations += 1
            cache_state[block_id] = target_tier

        actual_tier = cache_state.get(block_id, DEFAULT_TIER)
        total_cycles += compute_cycles(size_bytes, actual_tier, clock_ghz)
        total_cost_usd += compute_cost(size_bytes, actual_tier)

        if policy == "ml" and prefetch_prob > prefetch_threshold:
            for offset in range(1, 4):
                pf_bid = block_id + offset
                pf_current = cache_state.get(pf_bid, DEFAULT_TIER)

                if pf_current != 0:
                    pf_evict = compute_cycles(size_bytes, pf_current, clock_ghz)
                    pf_load = compute_cycles(size_bytes, 0, clock_ghz)
                    total_cycles += pf_evict + pf_load
                    cache_state[pf_bid] = 0
                    prefetch_count += 1
                    migrations += 1

        if verbose and (step < 10 or step % (n_events // 10 + 1) == 0):
            status = "HIT " if current_tier == target_tier else "MISS"
            print(f"[{step:>7d}] bid={block_id:<8d} {status} tier {current_tier}->{target_tier} cycles={total_cycles:>14.0f} cost=${total_cost_usd:.6f}")

    total_accesses = hits + misses
    hit_rate = hits / total_accesses if total_accesses > 0 else 0.0
    cycles_saved = static_total_cycles - total_cycles

    return {
        "policy": policy,
        "hit_rate": hit_rate,
        "total_cycles": total_cycles,
        "total_cost_usd": total_cost_usd,
        "migrations": migrations,
        "prefetch_count": prefetch_count,
        "cycles_saved_vs_static": cycles_saved,
        "hits": hits,
        "misses": misses,
        "total_accesses": total_accesses,
    }
