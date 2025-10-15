#!/usr/bin/env bash
# One-shot environment setup for the polymers competition repo
# Usage:
#   bash scripts/setup_env.sh                     # default env name + python
#   bash scripts/setup_env.sh myenv 3.8.20        # custom

set -euo pipefail

ENV_NAME="${1:-polymers_comp_env}"
PY_VER="${2:-3.8.20}"       # default to 3.8.20 as requested

# Target CUDA build for PyTorch / PyG
CUDA_TAG="cu118"
TORCH_VER="2.4.1"
TV_VER="0.19.1"
TA_VER="2.4.1"

# --- helpers -----------------------------------------------------------------
patch_mkl_hook() {
  # Patch MKL deactivate hook to tolerate unset CONDA_MKL_INTERFACE_LAYER_BACKUP
  local env_prefix
  env_prefix="$(conda info --envs | awk -v n="$ENV_NAME" '$1==n{print $NF}')"
  if [[ -z "${env_prefix:-}" || ! -d "$env_prefix" ]]; then
    echo "NOTE: cannot locate env prefix to patch MKL hook; skipping."
    return 0
  fi
  local hook="$env_prefix/etc/conda/deactivate.d/libblas_mkl_deactivate.sh"
  if [[ -f "$hook" ]]; then
    echo ">>> Patching MKL deactivate hook to be nounset-safe"
    # make a backup once
    [[ -f "${hook}.bak" ]] || cp -p "$hook" "${hook}.bak"
    # Linux sed is fine; for macOS/BSD sed, use -i '' if needed
    if sed --version >/dev/null 2>&1; then
      sed -i 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$hook"
    else
      sed -i '' 's/\$CONDA_MKL_INTERFACE_LAYER_BACKUP/${CONDA_MKL_INTERFACE_LAYER_BACKUP-}/g' "$hook"
    fi
  fi
}

# --- ensure conda is available & enable 'conda activate' in this shell -------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Open a conda-enabled shell or add it to PATH."
  exit 1
fi

# disable nounset while sourcing conda hooks (prevents unbound var issues)
set +u
eval "$(conda shell.bash hook)"
set -u

# --- create base env (no torch here) -----------------------------------------
echo ">>> Creating base env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"

# activate (wrap with +u to avoid nounset issues inside conda’s hook scripts)
echo ">>> Activating env"
set +u
conda activate "${ENV_NAME}"
set -u

# Immediately patch the MKL hook (if present) to avoid future -u errors
patch_mkl_hook

# --- install PyTorch + CUDA via conda ----------------------------------------
echo ">>> Installing PyTorch stack (torch=${TORCH_VER}, torchvision=${TV_VER}, torchaudio=${TA_VER}, CUDA=11.8)"
conda install -y \
  "pytorch=${TORCH_VER}" "torchvision=${TV_VER}" "torchaudio=${TA_VER}" \
  pytorch-cuda=11.8 -c pytorch -c nvidia

# --- install PyG wheels matched to torch+CUDA via pip ------------------------
echo ">>> Installing PyG (torch-geometric + extensions) for torch ${TORCH_VER} + ${CUDA_TAG}"
pip install -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" \
  torch-geometric==2.6.1 \
  torch-scatter==2.1.2+pt24${CUDA_TAG} \
  torch-sparse==0.6.18+pt24${CUDA_TAG} \
  torch-cluster==1.6.3+pt24${CUDA_TAG}

# --- update env with the rest of your repo deps (NO torch lines in YAML) -----
if [ -f "environment.yml" ]; then
  echo ">>> Updating env from environment.yml (non-torch deps)"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "NOTE: environment.yml not found; skipping env update."
fi

# --- sanity checks ------------------------------------------------------------
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
