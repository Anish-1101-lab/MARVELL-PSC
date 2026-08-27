import argparse

def measure_harmonia_overheads(capacity_gb: float, page_size_bytes: int = 4096, cpu_freq_ghz: float = 3.5):
    """
    Measures the 4 dimensions of overhead introduced by Harmonia based on Section 4.6 of the paper:
    1. Storage Overhead (Memory footprint of NN, buffers, queue, and metadata)
    2. Inference Latency Overhead
    3. Training Latency Overhead
    4. Metadata Collection Latency Overhead
    """
    print(f"=== Harmonia Overhead Analysis for {capacity_gb} GB Storage Capacity ===")
    
    # 1. Storage Overhead
    # Neural Networks: 2 agents * 2 networks (target+policy) * 90 weights * 16-bit (2 bytes) = 720 Bytes (~0.7 KiB)
    nn_storage_bytes = 2 * 2 * 90 * 2
    
    # Experience Buffers: 2 buffers * 1000 experiences * ~50 Bytes = 100,000 Bytes (~100 KiB)
    exp_buffer_bytes = 2 * 1000 * 50
    
    # Migration Queue: 10 candidates * (32-bit LBA + 4-bit Target ID ~ 5 Bytes) = 50 Bytes
    mig_queue_bytes = 10 * 5
    
    # Migration Reward Buffer: 50 latencies * 16-bit (2 Bytes) = 100 Bytes
    mig_reward_bytes = 50 * 2
    
    # Metadata Table: 1 entry per page, each entry is 32 bits (4 Bytes)
    num_pages = int((capacity_gb * 1024**3) / page_size_bytes)
    metadata_bytes = num_pages * 4
    
    total_storage_bytes = nn_storage_bytes + exp_buffer_bytes + mig_queue_bytes + mig_reward_bytes + metadata_bytes
    total_storage_mb = total_storage_bytes / (1024**2)
    
    print("\n1. Storage Overhead:")
    print(f"  - Neural Networks:      {nn_storage_bytes} Bytes")
    print(f"  - Experience Buffers:   {exp_buffer_bytes} Bytes")
    print(f"  - Migration Queue:      {mig_queue_bytes} Bytes")
    print(f"  - Migration Reward:     {mig_reward_bytes} Bytes")
    print(f"  - Metadata Table:       {metadata_bytes / 1024**2:.2f} MiB ({num_pages} pages)")
    print(f"  - Total Storage:        {total_storage_mb:.3f} MiB ({(total_storage_bytes / (capacity_gb * 1024**3)) * 100:.4f}% of total capacity)")
    
    # 2. Inference Latency Overhead
    # 90 MAC operations -> ~90 CPU cycles on evaluated system
    inference_cycles = 90
    inference_latency_ns = (inference_cycles / cpu_freq_ghz)
    
    print("\n2. Inference Latency Overhead (Critical Path for Placement):")
    print(f"  - Compute Complexity:   90 MAC operations")
    print(f"  - CPU Cycles:           ~{inference_cycles} cycles")
    print(f"  - Estimated Latency:    {inference_latency_ns:.2f} ns (at {cpu_freq_ghz} GHz)")
    
    # 3. Training Latency Overhead
    # 16 batches * 128 samples * 90 MACs = 184,320 MAC ops -> ~200,000 CPU cycles
    training_mac_ops = 16 * 128 * 90
    training_cycles = 200000
    training_latency_us = (training_cycles / cpu_freq_ghz) / 1000.0
    
    print("\n3. Training Latency Overhead (Background/Off Critical Path):")
    print(f"  - Compute Complexity:   {training_mac_ops:,} MAC operations")
    print(f"  - CPU Cycles:           ~{training_cycles:,} cycles")
    print(f"  - Estimated Latency:    {training_latency_us:.2f} us (at {cpu_freq_ghz} GHz)")
    
    # 4. Metadata Collection Latency Overhead
    # Extraction: 100 ns, DRAM Update: 1 us (1000 ns) -> 1.1 us total
    extraction_ns = 100
    dram_update_ns = 1000
    total_metadata_ns = extraction_ns + dram_update_ns
    
    print("\n4. Metadata Collection Latency Overhead (Critical Path):")
    print(f"  - Feature Extraction:   {extraction_ns} ns")
    print(f"  - DRAM Table Update:    {dram_update_ns} ns")
    print(f"  - Total Meta Latency:   {total_metadata_ns / 1000.0:.2f} us per I/O request")
    print("====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure all 4 dimensions of Harmonia's overhead.")
    parser.add_argument("--capacity_gb", type=float, default=1024.0, help="Total HSS capacity in GB (default: 1024 GB = 1 TB)")
    parser.add_argument("--page_size", type=int, default=4096, help="Page size in bytes (default: 4096)")
    parser.add_argument("--cpu_freq_ghz", type=float, default=3.5, help="CPU frequency in GHz (default: 3.5)")
    
    args = parser.parse_args()
    measure_harmonia_overheads(args.capacity_gb, args.page_size, args.cpu_freq_ghz)
