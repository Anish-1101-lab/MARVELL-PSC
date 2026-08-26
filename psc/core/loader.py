import csv
import json
import os
import numpy as np
from typing import List, Dict

def _load_chakra(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    accesses: List[Dict] = []
    for node in nodes:
        if node.get("type", "") != "memory_load":
            continue
        block_id = node.get("tensor_id", node.get("node_id", 0))
        size_bytes = node.get("tensor_size", 0)
        op = node.get("op", "load")
        accesses.append({
            "block_id": int(block_id),
            "size_bytes": int(size_bytes),
            "op": str(op),
        })
    return accesses

def _load_csv(path: str) -> List[Dict]:
    accesses: List[Dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accesses.append({
                "block_id": int(row["block_id"]),
                "size_bytes": int(row["size_bytes"]),
                "op": row.get("op", "load").strip(),
            })
    return accesses

def load_trace(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trace file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".et", ".json"):
        return _load_chakra(path)
    elif ext == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported trace file extension '{ext}'. Expected .et, .json (Chakra) or .csv.")

def generate_synthetic_trace(
    pattern: str,
    n_accesses: int = 10_000,
    seed: int = 42,
    stride: int = 4,
    base: int = 0,
    total_blocks: int = 10_000,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    accesses: List[Dict] = []

    if pattern == "zipfian":
        n_blocks = 5000
        alpha = 1.2
        raw = rng.zipf(alpha, size=n_accesses)
        block_ids = (raw - 1) % n_blocks
        size_bytes = 150 * 1024
        for bid in block_ids:
            accesses.append({"block_id": int(bid), "size_bytes": size_bytes, "op": "load"})
    elif pattern == "sequential":
        size_bytes = 4 * 1024
        for i in range(n_accesses):
            accesses.append({"block_id": i, "size_bytes": size_bytes, "op": "load"})
    elif pattern == "random_crop":
        size_bytes = 500 * 1024 * 1024
        block_ids = rng.integers(0, 1001, size=n_accesses)
        for bid in block_ids:
            accesses.append({"block_id": int(bid), "size_bytes": size_bytes, "op": "load"})
    elif pattern in ("strided", "multi_stride", "strided_access"):
        size_bytes = 4 * 1024
        for k in range(n_accesses):
            bid = (base + (k * stride)) % total_blocks
            accesses.append({"block_id": int(bid), "size_bytes": size_bytes, "op": "load"})
    elif pattern in ("graph_walk", "pointer_chasing", "gnn", "graph"):
        # Pointer-Chasing / Irregular Graph Walks (GNNs, SpMV/CSR, B-Trees)
        # Address(t+1) = Data[Address(t)] via adjacency index & feature lookups
        size_bytes = 4 * 1024
        n_nodes = min(total_blocks, 10000)
        # Generate power-law graph adjacency table (scale-free network with hubs)
        degrees = np.clip(rng.zipf(1.5, size=n_nodes), 2, 64)
        adj_table = []
        for d in degrees:
            neighbors = rng.integers(0, n_nodes, size=int(d))
            adj_table.append(neighbors)
        
        curr_node = 0
        for _ in range(n_accesses):
            # Pointer jump: pick next node from current node's pointer list
            neighbors = adj_table[curr_node]
            next_node = int(rng.choice(neighbors))
            # 90% follow pointer jump, 10% teleport/restart walk
            if rng.random() < 0.10:
                next_node = int(rng.integers(0, n_nodes))
            accesses.append({"block_id": int(curr_node), "size_bytes": size_bytes, "op": "load"})
            curr_node = next_node
    else:
        raise ValueError(f"Unknown pattern '{pattern}'. Expected: 'zipfian', 'sequential', 'random_crop', 'strided', 'graph_walk'.")
    return accesses


