import heapq
from collections import defaultdict

def belady_oracle(access_sequence, cache_size):
    """
    Given a flat list of block IDs in access order,
    returns a dict: {(block_id, time_step) -> optimal_tier}
    Tier 0 = hot (cache), Tier 1 = warm, Tier 2 = cold
    """
    n = len(access_sequence)
    # Build next-use index: for each position, when is this block next used?
    next_use = {}
    last_seen = {}
    for i in range(n - 1, -1, -1):
        blk = access_sequence[i]
        next_use[(blk, i)] = last_seen.get(blk, float('inf'))
        last_seen[blk] = i

    cache = set()
    labels = {}
    for t, blk in enumerate(access_sequence):
        if blk in cache:
            labels[(blk, t)] = 0  # already in hot tier
        else:
            labels[(blk, t)] = 1  # cache miss — should have been prefetched
            if len(cache) >= cache_size:
                # Evict block with furthest next use (Bélády optimal)
                victim = max(cache, key=lambda b: next_use.get((b, t), float('inf')))
                cache.discard(victim)
            cache.add(blk)
    return labels

def belady_3tier_labels(access_sequence, hbm_size, ssd_size):
    """
    Runs the oracle twice to generate 3-tier labels.
    - First pass: HBM-tier (hit = 0, miss = passed downstream)
    - Second pass over misses: SSD-tier (hit = 1, miss = 2)
    Returns: dict mapping (block_id, time_step) -> 0/1/2
    """
    # First pass: HBM level
    tier_0_labels = belady_oracle(access_sequence, hbm_size)
    
    # Collect all accesses that missed in HBM
    miss_sequence = []
    miss_times = []
    for t, blk in enumerate(access_sequence):
        if tier_0_labels[(blk, t)] == 1:
            miss_sequence.append(blk)
            miss_times.append(t)
            
    # Second pass: SSD level (over the misses from HBM)
    tier_1_labels_raw = belady_oracle(miss_sequence, ssd_size)
    
    # Merge the labels into a single sequence mapping
    final_labels = {}
    
    for t, blk in enumerate(access_sequence):
        lbl_hbm = tier_0_labels[(blk, t)]
        if lbl_hbm == 0:
            final_labels[(blk, t)] = 0 # HBM tier
        else:
            # It was a miss in HBM, so find its index in the miss_sequence to consult tier 1
            # Actually, to make this linear, we can just track the miss index counter
            pass
            
    # Let's fix the logic for merged:
    miss_idx = 0
    for t, blk in enumerate(access_sequence):
        lbl_hbm = tier_0_labels[(blk, t)]
        if lbl_hbm == 0:
            final_labels[(blk, t)] = 0
        else:
            lbl_ssd = tier_1_labels_raw[(blk, miss_idx)]
            if lbl_ssd == 0:
                final_labels[(blk, t)] = 1 # hits in SSD
            else:
                final_labels[(blk, t)] = 2 # misses in SSD, goes to NVMe (cold)
            miss_idx += 1
            
    return final_labels

if __name__ == "__main__":
    # Test script with dummy trace
    dummy_trace = [1, 2, 3, 1, 4, 2, 5, 1, 6, 2, 3, 4]
    labels = belady_3tier_labels(dummy_trace, hbm_size=2, ssd_size=2)
    print("Dummy Trace Labels:")
    for i, blk in enumerate(dummy_trace):
        print(f"Step {i}, Block {blk}: Tier {labels[(blk, i)]}")
