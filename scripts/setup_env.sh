#!/usr/bin/env bash
# One-shot environment setup for the polymers competition repo
# Usage:
#   bash scripts/setup_env.sh                     # defaults
#   bash scripts/setup_env.sh myenv 3.8.20        # custom name + Python

set -euo pipefail

ENV_NAME="${1:-polymers_comp_env}"
PY_VER="${2:-3.8.20}"

# Target CUDA build for PyTorch / PyG
CUDA_TAG="cu118"
TORCH_VER="2.4.1"
TV_VER="0.19.1"
TA_VER="2.4.1"

# --- ensure conda available & enable 'conda activate' ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Open a conda-enabled shell or add it to PATH."
  exit 1
fi
# (We’ll keep nounset on; no hook sourcing tricks.)
eval "$(conda shell.bash hook)"

# --- create base env (no torch here) ---
echo ">>> Creating base env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"

echo ">>> Activating env"
conda activate "${ENV_NAME}"

# --- QUICK FIX: patch MKL deactivate hook if present ---
PATCH_PATH="$HOME/miniforge3/envs/${ENV_NAME}/etc/conda/deactivate.d/libblas_mkl_deactivate.sh"
if [[ -f "$PATCH_PATH" ]]; then
  echo ">>> Patching MKL deactivate hook at: $PATCH_PATH"
  # GNU sed (Linux)
  if sed --version >/dev/null 2>&1; then
    sed -i 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$PATCH_PATH"
  else
    # BSD sed (macOS)
    sed -i '' 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$PATCH_PATH"
  fi
else
  echo "NOTE: MKL deactivate hook not found yet (will try again after installs)."
fi

# --- install PyTorch + CUDA via conda ---
echo ">>> Installing PyTorch stack (torch=${TORCH_VER}, torchvision=${TV_VER}, torchaudio=${TA_VER}, CUDA=11.8)"
conda install -y \
  "pytorch=${TORCH_VER}" "torchvision=${TV_VER}" "torchaudio=${TA_VER}" \
  pytorch-cuda=11.8 -c pytorch -c nvidia

# --- install PyG wheels matched to torch+CUDA via pip ---
echo ">>> Installing PyG (torch-geometric + extensions) for torch ${TORCH_VER} + ${CUDA_TAG}"
pip install -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" \
  torch-geometric==2.6.1 \
  torch-scatter==2.1.2+pt24${CUDA_TAG} \
  torch-sparse==0.6.18+pt24${CUDA_TAG} \
  torch-cluster==1.6.3+pt24${CUDA_TAG}

# --- update env with the rest of your repo deps (NO torch in environment.yml) ---
if [ -f "environment.yml" ]; then
  echo ">>> Updating env from environment.yml (non-torch deps)"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "NOTE: environment.yml not found; skipping env update."
fi

# --- RE-APPLY QUICK FIX (MKL hook may appear after installs) ---
if [[ -f "$PATCH_PATH" ]]; then
  echo ">>> Ensuring MKL deactivate hook is patched (final pass)"
  if sed --version >/dev/null 2>&1; then
    sed -i 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$PATCH_PATH"
  else
    sed -i '' 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$PATCH_PATH"
  fi
else
  echo "NOTE: MKL deactivate hook still not found at: $PATCH_PATH (nothing to patch)."
fi

# --- sanity checks ---
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| CUDA available:", torch.cuda.is_available())
try:
    import torch_geometric as tg
    print("PyG:", tg.__version__)
except Exception as e:
    print("PyG import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

echo ">>> Done. Activate with: conda activate ${ENV_NAME}"
