import subprocess
import os
import json
import re

ASTRASIM_BIN = "simulator/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
INPUTS_DIR = "simulator/astrasim_integration/inputs"
RESULTS_DIR = "simulator/astrasim_integration/results"

TIERS = ["hbm", "cxl", "ssd", "cold"]

def run_simulation(workload_trace, tier):
    print(f"--- Running AstraSim for {workload_trace} with {tier} config ---")
    
    # Paths to configs
    workload_cfg = f"simulator/astrasim_integration/chakra_traces/{workload_trace}"
    system_cfg = "simulator/astra-sim/inputs/system/analytical/tpu_v3_8.json"
    network_cfg = "simulator/astra-sim/inputs/network/tpu_v3_8.yml"
    memory_cfg = f"{INPUTS_DIR}/remote_memory/analytical/{tier}.json"
    
    # Check if configs exist
    for cfg in [workload_cfg, memory_cfg]:
        if not os.path.exists(cfg):
            print(f"Error: Config {cfg} missing.")
            return None

    cmd = [
        ASTRASIM_BIN,
        f"--workload-configuration={workload_cfg}",
        f"--system-configuration={system_cfg}",
        f"--network-configuration={network_cfg}",
        f"--remote-memory-configuration={memory_cfg}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Regex to find cycle count or time
        # AstraSim usually prints "Finished simulation at cycle: X"
        match = re.search(r"sys\[0\] finished, (\d+) cycles", output)
        if match:
            cycles = int(match.group(1))
            print(f"Result for {tier}: {cycles} cycles")
            return cycles
        else:
            print("Could not find cycle count in AstraSim output.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"AstraSim failed with error: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"AstraSim binary not found at {ASTRASIM_BIN}")
        return None

def main():
    workloads = ["resnet_et", "bert_et", "unet3d_et"]
    
    for workload in workloads:
        results = {}
        for tier in TIERS:
            cycles = run_simulation(workload, tier)
            if cycles:
                results[tier] = cycles
                
        with open(f"{RESULTS_DIR}/{workload}_cycles.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results for {workload} saved to {RESULTS_DIR}/{workload}_cycles.json")

if __name__ == "__main__":
    main()
