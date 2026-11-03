#!/bin/bash
# ============================================================
# One-time environment setup — run on Apocrita LOGIN NODE
# (not as a Slurm job)
#
# Usage:
#   ssh apocrita
#   bash $SCRATCH/simllama/scripts/slurm/setup_env.sh
# ============================================================

set -e

# Locate user-installed Miniforge (could be in $HOME or $SCRATCH)
# non-interactive bash doesn't source ~/.bashrc, so we source conda.sh directly
_CONDA_SH=""
for _candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "${SCRATCH:?}/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$_candidate" ]]; then
        _CONDA_SH="$_candidate"
        echo "Found conda at: $_candidate"
        break
    fi
done
if [[ -z "$_CONDA_SH" ]]; then
    echo "ERROR: Could not find conda.sh. Check where Miniforge was installed."
    echo "  Run: find \$HOME \$SCRATCH -name conda.sh 2>/dev/null | head -5"
    exit 1
fi
source "$_CONDA_SH"

# --- Create conda environment ---
conda create -n moonbeam_train python=3.10 -y

# --- Activate ---
conda activate moonbeam_train

# PyTorch with CUDA -- detect version automatically, fall back to cu118
# Run `nvcc --version` or `module avail cuda` to check manually.
# Matching whl tags: CUDA 11.8 -> cu118 | 12.1 -> cu121 | 12.4 -> cu124
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
if [[ "$CUDA_VER" == "12.4"* ]]; then TORCH_CUDA=cu124
elif [[ "$CUDA_VER" == "12.1"* ]]; then TORCH_CUDA=cu121
elif [[ "$CUDA_VER" == "11.8"* ]]; then TORCH_CUDA=cu118
else
    echo "WARNING: CUDA version '$CUDA_VER' not recognised -- defaulting to cu118."
    echo "         Edit setup_env.sh if your GPU needs a different build."
    TORCH_CUDA=cu118
fi
echo "Installing PyTorch for CUDA $CUDA_VER (whl tag: $TORCH_CUDA)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CUDA

# Project dependencies
pip install numpy pandas mido tqdm matplotlib psutil openpyxl

# --- Verify ---
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Setup complete. Environment: moonbeam_train"
echo "Test with: conda run -n moonbeam_train python -c 'import torch; print(torch.cuda.is_available())'"
