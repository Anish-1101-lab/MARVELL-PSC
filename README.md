# MARVELL-PSC: Phase-Conditioned ML Storage Controller

Predictive storage tiering and prefetching using Phase-Conditioned Machine Learning.

## Project Structure

- `psc/`: Core package.
    - `core/`: Simulator engine, tier configuration, and trace loading.
    - `models/`: ML models (Tiered Predictor, SLM Predictor, RL Controller).
    - `integration/`: Integration scripts for analytical simulators (e.g., AstraSim).
- `scripts/`: CLI tools for running simulations and benchmarks.
- `data/`: Sample trace files and datasets.
- `weights/`: Trained model checkpoints.
- `research/`: Archived development scripts and experimental notes.

## Getting Started

### Installation
```bash
pip install -r simulator/requirements.txt
```

### Running a Simulation
Run the main simulation harness with default settings (Zipfian trace):
```bash
python scripts/main_sim.py
```

Run with trained models:
```bash
python scripts/main_sim.py --lstm weights/phase_classifier.pth --mlp weights/policy_model_conditioned.pth
```

## Features
- **Phase-Conditioned Prediction**: Uses an LSTM to classify I/O phases and an MLP to make tiering decisions.
- **Multi-Tier Support**: Simulates HBM, CXL DRAM, NVMe SSD, and Cold Storage.
- **AstraSim Integration**: Bridges ML-based decisions with cycle-accurate architectural simulation.
