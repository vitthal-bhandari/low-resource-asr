#!/bin/bash
# =============================================================================
# Hyak Setup Script
# Run this once to set up your environment on Hyak
#
# For users WITHOUT a lab: use PROJECT_DIR under scrubbed (21-day purge).
# For users WITH a lab: set ACCOUNT and use PROJECT_DIR under /gscratch/$ACCOUNT/.
# =============================================================================

set -e

echo "=========================================="
echo "Setting up Low-Resource ASR on Hyak"
echo "=========================================="

# Configuration - EDIT THESE
# ACCOUNT: your Hyak account (UW NetID if no lab, e.g. vitthal1; or lab name if in a lab)
ACCOUNT="${HYAK_ACCOUNT:-vitthal1}"

# PROJECT_DIR: no lab = scrubbed (21-day purge); with lab = /gscratch/$ACCOUNT/low-resource-asr
# Set USE_LAB=1 if you have a lab and want project under /gscratch/$ACCOUNT/
USE_LAB="${USE_LAB:-0}"
if [ "$USE_LAB" = "1" ] && [ -n "$ACCOUNT" ]; then
    PROJECT_DIR="/gscratch/$ACCOUNT/low-resource-asr"
else
    PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"
fi

# Optional: Miniconda (only if not using uv)
CONDA_DIR="/gscratch/scrubbed/$USER/miniconda3"
ENV_NAME="low-resource-asr"

echo ""
echo "Configuration:"
echo "  Account: $ACCOUNT"
echo "  Project dir: $PROJECT_DIR"
echo ""

# =============================================================================
# Step 1: Clone or sync project
# =============================================================================

echo "Step 1: Setting up project directory..."

if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/low-resource-asr.git "$PROJECT_DIR"
else
    echo "  Project directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
fi

cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p data/mozilla_speech_data
mkdir -p results/training_logs

echo "  Done."
echo ""

# =============================================================================
# Step 2: Install uv (recommended - matches local Mac workflow and lockfile)
# =============================================================================

echo "Step 2: Setting up uv..."

# Use scratch for uv cache to avoid filling home (10GB quota on Hyak)
export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
echo "  UV cache: $UV_CACHE_DIR"

if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "  uv already installed."
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Use Python 3.12 so torchcodec (datasets[audio]) has wheels; 3.14 is not supported
PYTHON_VERSION="${HYAK_PYTHON_VERSION:-3.12}"
if ! uv python find "$PYTHON_VERSION" &>/dev/null; then
    echo "  Installing Python $PYTHON_VERSION (required for datasets[audio] / torchcodec)..."
    uv python install "$PYTHON_VERSION"
fi
echo "  Using Python $PYTHON_VERSION for .venv"
rm -rf .venv
uv venv --python "$PYTHON_VERSION"
uv sync --no-dev

echo "  Done."
echo ""

# =============================================================================
# Step 3: Verify GPU access (optional; only works on a GPU node)
# =============================================================================

echo "Step 3: Verifying Python environment..."

uv run --no-dev python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
import transformers
print(f'Transformers version: {transformers.__version__}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set ACCOUNT in SLURM scripts: edit ACCOUNT= in scripts/hyak_train_single.slurm and hyak_train_all.slurm"
echo "     (or export HYAK_ACCOUNT=stf before running this script to use scrubbed path)"
echo "  2. Copy your .env file to $PROJECT_DIR/.env"
echo "  3. Upload your data to $PROJECT_DIR/data/mozilla_speech_data/"
echo "  4. Submit jobs: sbatch scripts/hyak_train_single.slurm aln"
echo "                  sbatch scripts/hyak_train_single.slurm sco"
echo ""
echo "To run training in future sessions:"
echo "  cd $PROJECT_DIR"
echo "  uv run python -m src.training.aft_mms <lang> [options]"
echo ""
echo "Optional - Miniconda alternative (if you prefer conda over uv):"
echo "  Install: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
echo "           bash Miniconda3-*.sh -b -p $CONDA_DIR"
echo "  Then: source $CONDA_DIR/etc/profile.d/conda.sh"
echo "        conda create -p /gscratch/scrubbed/\$USER/$ENV_NAME python=3.11 -y"
echo "        conda activate /gscratch/scrubbed/\$USER/$ENV_NAME"
echo "        pip install -e . (or pip install from pyproject.toml)"
echo ""
