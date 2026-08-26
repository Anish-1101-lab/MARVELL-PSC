"""
Harmonia Multi-Agent HSS Controller.

Orchestrates the Data-Placement Agent and Data-Migration Agent,
manages DRAM metadata, extracts 7-D state observations, coordinates delayed rewards,
and executes idle-time proactive migrations across the 4-tier storage hierarchy.
"""

from collections import OrderedDict, deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..models.placement_agent import PlacementAgent
from ..models.migration_agent import MigrationAgent
from .state_encoder import HarmoniaStateEncoder
from .migration_queue import MigrationQueue
from psc.core.config import NUM_TIERS, compute_cycles, compute_cost, TIERS, get_tier_name


class HarmoniaHSSController:
    """
    Co-optimizes data placement and data migration via Multi-Agent RL.
    
    Tiers (from system.json):
      0: HBM
      1: CXL_DRAM
      2: NVMe_SSD
      3: Cold_Storage
    """
    def __init__(
        self,
        cache_capacity_hbm: int = 1000,
        num_tiers: int = NUM_TIERS,
        clock_ghz: float = 1.0,
        migration_interval_requests: int = 10,
        seed: int = 42
    ):
        self.num_tiers = num_tiers
        self.capacity_hbm = cache_capacity_hbm
        self.clock_ghz = clock_ghz
        self.migration_interval_requests = migration_interval_requests
        
        # State Encoder
        self.state_encoder = HarmoniaStateEncoder(page_size_bytes=4096, num_tiers=num_tiers)
        
        # RL Agents
        self.placement_agent = PlacementAgent(
            input_dim=7,
            hidden_dim=10,
            num_tiers=num_tiers,
            gamma=0.9,
            alpha=1e-3,
            epsilon=0.001,
            batch_size=128,
            buffer_size=1000,
            sync_interval=1000,
            seed=seed
        )
        
        self.migration_agent = MigrationAgent(
            input_dim=7,
            hidden_dim=10,
            num_tiers=num_tiers,
            gamma=0.1,
            alpha=1e-2,
            epsilon=0.001,
            batch_size=256,
            buffer_size=1000,
            sync_interval=1000,
            delayed_window_n=50,
            migration_batch_x=10,
            seed=seed + 1
        )
        
        # Asynchronous Migration Queue
        self.migration_queue = MigrationQueue(max_size=10)
        
        # Hierarchy Storage Tracking (block_id -> tier_id)
        self.block_tier_map: Dict[int, int] = {}
        # HBM Cache set (OrderedDict for LRU-assisted capacity management when full)
        self.hbm_cache: OrderedDict[int, int] = OrderedDict() # bid -> size_bytes
        
        # Candidate pool tracking for migration agent
        self.active_blocks: deque = deque(maxlen=2000)
        self.req_counter = 0
        
        # Discrete-event timing & bus state
        self.current_time_us = 0.0
        self.bus_available_time_us = 0.0
        self.rng = np.random.default_rng(seed)

    def get_hbm_occupancy(self) -> int:
        return len(self.hbm_cache)

    def handle_request(self, event: Dict[str, Any], inter_arrival_mean_us: float = 100.0) -> Dict[str, Any]:
        """
        Processes an incoming I/O request through Harmonia's dual-agent pipeline.
        
        Args:
            event: dict containing 'block_id', 'size_bytes', and optional 'is_write'
            inter_arrival_mean_us: mean inter-arrival time in microseconds
        Returns:
            result dict with request latency, tier placements, hits/misses, and migrations.
        """
        self.req_counter += 1
        block_id = int(event["block_id"])
        size_bytes = int(event.get("size_bytes", 4096))
        is_write = bool(event.get("is_write", False))
        
        # Advance simulation arrival time
        inter_arrival = float(self.rng.exponential(scale=max(1.0, inter_arrival_mean_us)))
        self.current_time_us += inter_arrival
        
        # Track active blocks for migration agent inspection
        if block_id not in self.active_blocks:
            self.active_blocks.append(block_id)
            
        current_tier = self.block_tier_map.get(block_id, 2) # Default NVMe SSD
        hbm_occupancy = self.get_hbm_occupancy()
        
        # 1. State Extraction (7 features)
        state_t, _ = self.state_encoder.extract_state(
            block_id=block_id,
            size_bytes=size_bytes,
            is_write=is_write,
            current_tier=current_tier,
            fast_occupancy=hbm_occupancy,
            fast_capacity=self.capacity_hbm
        )
        
        is_hit = False
        migrations_count = 0
        migrated_bytes = 0
        bus_transit_ns = 0.0
        evict_service_ns = 0.0
        
        queueing_delay_us = max(0.0, self.bus_available_time_us - self.current_time_us)
        
        # 2. Decision Logic
        if is_write:
            # Placement Agent chooses destination tier on write (critical path)
            target_tier = self.placement_agent.select_placement_tier(state_t)
            if target_tier == 0:
                if block_id in self.hbm_cache:
                    self.hbm_cache.move_to_end(block_id)
                    is_hit = True
                else:
                    if len(self.hbm_cache) >= self.capacity_hbm:
                        evicted_bid, evicted_size = self.hbm_cache.popitem(last=False)
                        self.block_tier_map[evicted_bid] = 2 # demote to NVMe SSD
                        migrations_count += 1
                        migrated_bytes += evicted_size
                        evict_bus_ns = (evicted_size / TIERS[2]["bandwidth_gbps"])
                        bus_transit_ns += evict_bus_ns
                        evict_service_ns = evict_bus_ns + TIERS[2]["latency_ns"] * 0.3
                        self.state_encoder.update_migration(evicted_bid)
                    self.hbm_cache[block_id] = size_bytes
                    self.block_tier_map[block_id] = 0
                    migrations_count += 1
                    migrated_bytes += size_bytes
                    load_bus_ns = (size_bytes / TIERS[2]["bandwidth_gbps"])
                    bus_transit_ns += load_bus_ns
            else:
                if block_id in self.hbm_cache:
                    del self.hbm_cache[block_id]
                self.block_tier_map[block_id] = target_tier
                if current_tier != target_tier:
                    migrations_count += 1
                    migrated_bytes += size_bytes
                    bus_transit_ns += (size_bytes / TIERS[target_tier]["bandwidth_gbps"])
            
            accessed_tier = target_tier
        else:
            # Read request: Check if resident in HBM
            if block_id in self.hbm_cache:
                is_hit = True
                accessed_tier = 0
                self.hbm_cache.move_to_end(block_id)
            else:
                is_hit = False
                accessed_tier = current_tier
                # Demand caching into HBM (promoting accessed page)
                if len(self.hbm_cache) >= self.capacity_hbm:
                    evicted_bid, evicted_size = self.hbm_cache.popitem(last=False)
                    self.block_tier_map[evicted_bid] = 2 # demote to NVMe SSD
                    migrations_count += 1
                    migrated_bytes += evicted_size
                    evict_bus_ns = (evicted_size / TIERS[2]["bandwidth_gbps"])
                    bus_transit_ns += evict_bus_ns
                    evict_service_ns = evict_bus_ns + TIERS[2]["latency_ns"] * 0.3
                    self.state_encoder.update_migration(evicted_bid)
                self.hbm_cache[block_id] = size_bytes
                self.block_tier_map[block_id] = 0
                migrations_count += 1
                migrated_bytes += size_bytes
                bus_bw = TIERS[accessed_tier]["bandwidth_gbps"] if accessed_tier in TIERS else TIERS[2]["bandwidth_gbps"]
                bus_transit_ns += (size_bytes / bus_bw)

        # Calculate I/O request latency
        if is_hit and accessed_tier == 0:
            access_cycles = compute_cycles(size_bytes, 0, self.clock_ghz)
            access_latency_us = (access_cycles / self.clock_ghz) / 1000.0
        else:
            tier_spec = TIERS[accessed_tier] if accessed_tier in TIERS else TIERS[2]
            demand_service_ns = tier_spec["latency_ns"] + (size_bytes / tier_spec["bandwidth_gbps"])
            total_service_ns = demand_service_ns + evict_service_ns
            access_latency_us = queueing_delay_us + (total_service_ns / 1000.0)
            access_cycles = total_service_ns * self.clock_ghz
            
            # Update bus occupancy
            bus_busy_ns = (size_bytes / tier_spec["bandwidth_gbps"]) + (evict_service_ns if evict_service_ns > 0 else 0)
            self.bus_available_time_us = max(self.current_time_us, self.bus_available_time_us) + (bus_busy_ns / 1000.0)
        
        # 3. Next State Extraction & Placement Agent Experience Recording
        next_hbm_occupancy = self.get_hbm_occupancy()
        next_state, _ = self.state_encoder.extract_state(
            block_id=block_id,
            size_bytes=size_bytes,
            is_write=is_write,
            current_tier=self.block_tier_map.get(block_id, 2),
            fast_occupancy=next_hbm_occupancy,
            fast_capacity=self.capacity_hbm
        )
        
        if is_write:
            placement_reward = self.placement_agent.compute_reward(access_latency_us)
            self.placement_agent.record_experience(
                state=state_t,
                action=accessed_tier,
                reward=placement_reward,
                next_state=next_state,
                done=False
            )

        # 4. Feed I/O latency to Migration Agent delayed reward tracker
        self.migration_agent.record_io_latency(access_latency_us, next_state)

        # 5. Background Data-Migration Candidate Selection & Execution
        if self.req_counter % self.migration_interval_requests == 0 and len(self.active_blocks) > 0:
            self._run_migration_step()

        # Check if migration queue has candidates to execute during idle slice
        migr_results = self._drain_migration_queue()
        migrations_count += migr_results["migrations_count"]
        migrated_bytes += migr_results["migrated_bytes"]
        bus_transit_ns += migr_results["bus_transit_ns"]

        return {
            "block_id": block_id,
            "is_hit": is_hit,
            "accessed_tier": accessed_tier,
            "latency_us": access_latency_us,
            "cycles": access_cycles,
            "cost_usd": compute_cost(size_bytes, accessed_tier),
            "migrations_count": migrations_count,
            "migrated_bytes": migrated_bytes,
            "bus_transit_ns": bus_transit_ns
        }

    def _run_migration_step(self) -> None:
        """
        Background Migration Agent scans a small subset of candidate pages,
        predicts optimal target tiers, and pushes promising candidates to MigrationQueue.
        """
        sample_size = min(10, len(self.active_blocks))
        if sample_size == 0:
            return
            
        candidate_bids = np.random.choice(list(self.active_blocks), size=sample_size, replace=False)
        for c_bid in candidate_bids:
            if self.migration_queue.is_full():
                break
                
            c_tier = self.block_tier_map.get(int(c_bid), 3)
            c_state, _ = self.state_encoder.extract_state(
                block_id=int(c_bid),
                size_bytes=4096,
                is_write=False,
                current_tier=c_tier,
                fast_occupancy=self.get_hbm_occupancy(),
                fast_capacity=self.capacity_hbm
            )
            
            target_tier = self.migration_agent.select_migration_tier(c_state)
            if target_tier != c_tier:
                acc_intr, migr_intr = self.state_encoder.get_raw_intervals(int(c_bid))
                self.migration_queue.push(
                    block_id=int(c_bid),
                    target_tier=target_tier,
                    current_tier=c_tier,
                    size_bytes=4096,
                    state=c_state,
                    acc_intr=acc_intr,
                    migr_intr=migr_intr
                )

    def _drain_migration_queue(self) -> Dict[str, Any]:
        """
        Drains up to 10 candidates from the migration queue in background.
        """
        batch = self.migration_queue.pop_batch(count=10)
        migr_count = 0
        migr_bytes = 0
        bus_ns = 0.0
        
        for item in batch:
            bid = item["block_id"]
            tgt_tier = item["target_tier"]
            cur_tier = self.block_tier_map.get(bid, 3)
            size = item["size_bytes"]
            
            if tgt_tier == cur_tier:
                continue
                
            # Perform physical migration
            if tgt_tier == 0:
                # Migrate to HBM
                if len(self.hbm_cache) >= self.capacity_hbm:
                    evicted_bid, evicted_size = self.hbm_cache.popitem(last=False)
                    self.block_tier_map[evicted_bid] = 2 # Evict to NVMe
                    migr_count += 1
                    migr_bytes += evicted_size
                    bus_ns += (evicted_size / TIERS[2]["bandwidth_gbps"])
                    self.state_encoder.update_migration(evicted_bid)
                
                self.hbm_cache[bid] = size
                self.block_tier_map[bid] = 0
            else:
                # Migrate down to CXL/NVMe/Cold
                if bid in self.hbm_cache:
                    del self.hbm_cache[bid]
                self.block_tier_map[bid] = tgt_tier

            migr_count += 1
            migr_bytes += size
            bus_bw = TIERS[tgt_tier]["bandwidth_gbps"]
            bus_ns += (size / bus_bw)
            
            self.state_encoder.update_migration(bid)
            
            # Register with Migration Agent for delayed reward
            self.migration_agent.register_migration(
                block_id=bid,
                state=item["state"],
                target_tier=tgt_tier,
                avg_access_interval=item["acc_intr"],
                avg_migr_interval=item["migr_intr"]
            )
            
        return {
            "migrations_count": migr_count,
            "migrated_bytes": migr_bytes,
            "bus_transit_ns": bus_ns
        }

    def reset(self) -> None:
        """Resets simulation state for new trace."""
        self.block_tier_map.clear()
        self.hbm_cache.clear()
        self.active_blocks.clear()
        self.migration_queue.clear()
        self.req_counter = 0
