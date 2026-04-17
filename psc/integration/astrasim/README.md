# AstraSim Integration for PSC


This directory contains the necessary components to bridge the Phase-Conditioned ML Storage Controller (PSC) with AstraSim for high-fidelity performance simulation.

## Workflow

1. **Setup AstraSim**:
   Clone and build AstraSim as per the instructions:
   ```bash
   git clone https://github.com/astra-sim/astra-sim.git
   cd astra-sim
   git submodule init && git submodule update
   ./build/astra_analytical/build.sh -c
   ```

2. **Generate Chakra Execution Traces**:
   Convert the PSC DLIO traces into the protobuf-based Chakra format:
   ```bash
   ./venv/bin/python astrasim_integration/chakra_converter.py \
     --input ./processed_traces/resnet_normalized.parquet \
     --output ./astrasim_integration/chakra_traces/resnet_et
   ```

3. **Configure Tiers**:
   The `astrasim_integration/inputs/remote_memory/analytical/` directory contains JSON configurations for:
   - `hbm.json` (High Bandwidth Memory)
   - `cxl.json` (CXL DRAM)
   - `ssd.json` (NVMe SSD)
   - `cold.json` (NVMe Cold)

4. **Run Simulation**:
   Execute the AstraSim simulation across different memory configurations:
   ```bash
   ./venv/bin/python astrasim_integration/astrasim_runner.py
   ```

5. **Analytical Evaluation**:
   Post-process the cycle counts and apply the PSC tiering policy to calculate the final simulated latency.

## Directory Structure
- `chakra_traces/`: Target for converted traces.
- `inputs/`: System and memory configuration files.
- `results/`: Cycle counts and performance logs.
- `chakra_converter.py`: Trace conversion logic.
- `astrasim_runner.py`: Orchestration script for running AstraSim binaries.
