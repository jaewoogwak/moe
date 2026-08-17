#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Mixtral MoE Profiling Environment Setup
#
# Target:
#   - RTX 4090
#   - CUDA-compatible PyTorch (cu126)
#   - Python 3.11
#   - Hugging Face Transformers / Accelerate
#   - Profiling utilities
# ============================================================

ENV_NAME="${ENV_NAME:-mixtral}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "============================================================"
echo "[1/8] System check"
echo "============================================================"

# ---- Git ----
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is not installed."
    exit 1
fi
echo "[OK] $(git --version)"

# ---- Conda ----
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not installed or not in PATH."
    exit 1
fi
echo "[OK] conda: $(conda --version)"

# ---- NVIDIA GPU ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found."
    exit 1
fi

echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

echo
echo "[PCIe]"
nvidia-smi -q | grep -A 12 "GPU Link Info" || true


echo
echo "============================================================"
echo "[2/8] Initialize conda"
echo "============================================================"

# Allows 'conda activate' inside a shell script
eval "$(conda shell.bash hook)"


echo
echo "============================================================"
echo "[3/8] Create conda environment: ${ENV_NAME}"
echo "============================================================"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists."
else
    conda create -n "${ENV_NAME}" \
        python="${PYTHON_VERSION}" \
        pip \
        -y
fi

conda activate "${ENV_NAME}"

echo "[OK] Python: $(python --version)"
echo "[OK] Python path: $(which python)"


echo
echo "============================================================"
echo "[4/8] Upgrade pip"
echo "============================================================"

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


echo
echo "============================================================"
echo "[5/8] Install PyTorch (CUDA 12.6)"
echo "============================================================"

python -m pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126


echo
echo "============================================================"
echo "[6/8] Install Hugging Face / Mixtral dependencies"
echo "============================================================"

python -m pip install -U \
    transformers \
    accelerate \
    huggingface_hub \
    safetensors \
    sentencepiece \
    protobuf


echo
echo "============================================================"
echo "[7/8] Install profiling / experiment packages"
echo "============================================================"

python -m pip install -U \
    numpy \
    pandas \
    psutil \
    nvidia-ml-py \
    tqdm


echo
echo "============================================================"
echo "[8/8] Verify installation"
echo "============================================================"

python - <<'PY'
import sys
import torch
import transformers
import accelerate
import huggingface_hub
import numpy
import pandas
import psutil
import pynvml

print()
print("=============== Environment ===============")
print(f"Python            : {sys.version.split()[0]}")
print(f"PyTorch           : {torch.__version__}")
print(f"PyTorch CUDA      : {torch.version.cuda}")
print(f"Transformers      : {transformers.__version__}")
print(f"Accelerate        : {accelerate.__version__}")
print(f"HuggingFace Hub   : {huggingface_hub.__version__}")
print(f"NumPy             : {numpy.__version__}")
print(f"Pandas            : {pandas.__version__}")
print()

print("=============== GPU ===============")
print(f"CUDA available    : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

gpu = torch.cuda.get_device_properties(0)

print(f"GPU               : {torch.cuda.get_device_name(0)}")
print(f"VRAM              : {gpu.total_memory / 1024**3:.2f} GiB")
print(f"Compute capability: {gpu.major}.{gpu.minor}")

print()
print("[SUCCESS] Mixtral profiling environment is ready.")
PY


echo
echo "============================================================"
echo "Setup complete"
echo "============================================================"
echo
echo "Activate later with:"
echo
echo "    conda activate ${ENV_NAME}"
echo