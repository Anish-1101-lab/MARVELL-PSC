#!/bin/bash

# Activate existing virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Installing native OpenMPI via Homebrew (required by mpi4py on macOS)..."
brew install open-mpi > /dev/null 2>&1 || echo "Homebrew installation of open-mpi failed or already exists... proceeding."

echo "Installing dlio-benchmark dependencies natively..."
pip install pyyaml h5py mpi4py numpy omegaconf pandas psutil pydftracer==1.0.2 pytest urllib3 hydra-core
echo "Installing dlio-benchmark (bypassing strict NVIDIA DALI dependency for macOS)..."
pip install --no-deps dlio-benchmark

echo "Creating explicit directories for traces..."
mkdir -p ./real_traces/resnet
mkdir -p ./real_traces/bert
mkdir -p ./real_traces/unet3d

# Common DLIO Dataset configuration overrides
COMMON_ARGS="++workload.dataset.num_files_train=256 ++workload.dataset.num_files_eval=64 ++workload.dataset.record_length=1048576 ++workload.dataset.num_samples_per_file=40 ++workload.workflow.generate_data=True ++workload.workflow.train=True ++workload.framework=pytorch ++workload.reader.data_loader=pytorch ++workload.dataset.format=npz ++workload.train.epochs=1"

echo "========================================="
echo "Generating ResNet50 MLPerf Trace..."
python run_dlio_benchmark.py workload=resnet50_v100 $COMMON_ARGS ++workload.output.folder=./real_traces/resnet/ || echo "DLIO Run Failed. Ensure output directories are correctly configured."
# Some versions of DLIO output to hydra_log natively, so we perform a defensive copy
find . -name "*iostat*.json" -exec cp {} ./real_traces/resnet/ 2>/dev/null \;
find . -name "*.pfw" -exec cp {} ./real_traces/resnet/ 2>/dev/null \;

echo "========================================="
echo "Generating BERT MLPerf Trace..."
python run_dlio_benchmark.py workload=bert_v100 $COMMON_ARGS ++workload.output.folder=./real_traces/bert/ || echo "DLIO Run Failed."
find . -name "*iostat*.json" -exec cp {} ./real_traces/bert/ 2>/dev/null \;
find . -name "*.pfw" -exec cp {} ./real_traces/bert/ 2>/dev/null \;

echo "========================================="
echo "Generating UNet3D MLPerf Trace..."
python run_dlio_benchmark.py workload=unet3d_v100 $COMMON_ARGS ++workload.output.folder=./real_traces/unet3d/ || echo "DLIO Run Failed."
find . -name "*iostat*.json" -exec cp {} ./real_traces/unet3d/ 2>/dev/null \;
find . -name "*.pfw" -exec cp {} ./real_traces/unet3d/ 2>/dev/null \;

echo "========================================="
echo "Summary of Generated Trace Files:"
echo "ResNet50 : $(ls -1 ./real_traces/resnet/*.csv 2>/dev/null | wc -l) trace files"
echo "BERT     : $(ls -1 ./real_traces/bert/*.csv 2>/dev/null | wc -l) trace files"
echo "UNet3D   : $(ls -1 ./real_traces/unet3d/*.csv 2>/dev/null | wc -l) trace files"
echo "Done."
