#!/usr/bin/env bash

set -euo pipefail

# Recreate the Mixtral profiling environment from the checked-in locks.
#
# The locks were exported from /venv/mixtral on this server. They target
# Linux x86_64 and CUDA 12.6 PyTorch wheels. This script never installs
# unpinned "latest" packages.

ENV_NAME="${ENV_NAME:-mixtral}"
RECREATE="${RECREATE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_LOCK="${SCRIPT_DIR}/requirements/mixtral-conda-linux-64.explicit"
PIP_LOCK="${SCRIPT_DIR}/requirements/mixtral-pip.lock"
CONDA_SH="${CONDA_SH:-/opt/miniforge3/etc/profile.d/conda.sh}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || fail "This lock is for Linux only."
[[ "$(uname -m)" == "x86_64" ]] || fail "This lock is for x86_64 only."
[[ -f "${CONDA_LOCK}" ]] || fail "Missing conda lock: ${CONDA_LOCK}"
[[ -f "${PIP_LOCK}" ]] || fail "Missing pip lock: ${PIP_LOCK}"

# A non-interactive script does not source .bashrc, so initialize conda
# explicitly rather than relying on the caller's interactive shell setup.
if [[ -r "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    fail "Conda is unavailable. Set CONDA_SH to its etc/profile.d/conda.sh path."
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found."
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
echo "Conda: $(conda --version)"

if conda env list | awk -v env_name="${ENV_NAME}" '$1 == env_name { found = 1 } END { exit !found }'; then
    if [[ "${RECREATE}" != "1" ]]; then
        fail "Environment '${ENV_NAME}' already exists. Use a new ENV_NAME, or set RECREATE=1 to replace it."
    fi
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove --name "${ENV_NAME}" --yes
fi

echo "Creating '${ENV_NAME}' from the exact conda package lock"
conda create --name "${ENV_NAME}" --file "${CONDA_LOCK}" --yes
conda activate "${ENV_NAME}"

echo "Installing the exact pip package lock"
python -m pip install --upgrade --force-reinstall --requirement "${PIP_LOCK}"
python -m pip check

echo "Verifying every locked pip package version"
python - "${PIP_LOCK}" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

for raw_line in Path(sys.argv[1]).read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("--"):
        continue
    name, expected = line.split("==", 1)
    try:
        actual = version(name)
    except PackageNotFoundError as exc:
        raise SystemExit(f"missing locked package: {name}") from exc
    if actual != expected:
        raise SystemExit(f"version mismatch for {name}: expected {expected}, got {actual}")
    print(f"[OK] {name}=={actual}")
PY

python - <<'PY'
import sys
import torch
import transformers
import accelerate
import datasets

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Datasets:", datasets.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the recreated environment")
print("GPU:", torch.cuda.get_device_name(0))
print("[SUCCESS] Reproducible Mixtral environment is ready.")
PY

echo
echo "Activate with: conda activate ${ENV_NAME}"
