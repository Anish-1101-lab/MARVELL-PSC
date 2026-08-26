import json
import os
from typing import Dict, Any

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
_SYSTEM_CFG_PATH = os.path.join(_CONFIG_DIR, "system.json")

def _load_system_config() -> list[dict[str, Any]]:
    if not os.path.exists(_SYSTEM_CFG_PATH):
        raise FileNotFoundError(f"System config not found at {_SYSTEM_CFG_PATH}")
    with open(_SYSTEM_CFG_PATH, "r") as f:
        cfg = json.load(f)
    return cfg["tiers"]

TIERS: list[dict[str, Any]] = _load_system_config()
_TIER_MAP: Dict[int, dict[str, Any]] = {t["id"]: t for t in TIERS}
NUM_TIERS = len(TIERS)

def compute_cycles(size_bytes: int, tier_id: int, clock_ghz: float = 1.0) -> float:
    if tier_id not in _TIER_MAP:
        raise ValueError(f"Unknown tier_id={tier_id}. Valid: {list(_TIER_MAP.keys())}")

    tier = _TIER_MAP[tier_id]
    latency_ns: float = tier["latency_ns"]
    bw_bytes_per_ns: float = tier["bandwidth_gbps"]

    transfer_ns = size_bytes / bw_bytes_per_ns
    total_ns = latency_ns + transfer_ns
    return total_ns * clock_ghz

def compute_cost(size_bytes: int, tier_id: int) -> float:
    if tier_id not in _TIER_MAP:
        raise ValueError(f"Unknown tier_id={tier_id}. Valid: {list(_TIER_MAP.keys())}")

    tier = _TIER_MAP[tier_id]
    cost_per_gb: float = tier["cost_per_gb"]
    size_gb = size_bytes / (1024 ** 3)
    return size_gb * cost_per_gb

def get_tier_name(tier_id: int) -> str:
    if tier_id not in _TIER_MAP:
        raise ValueError(f"Unknown tier_id={tier_id}. Valid: {list(_TIER_MAP.keys())}")
    return _TIER_MAP[tier_id]["name"]
