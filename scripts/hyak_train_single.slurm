#!/bin/bash
#SBATCH --job-name=mms-aft
#SBATCH --account=YOUR_ACCOUNT      # TODO: Replace with your Hyak account (e.g., stf, escience)
#SBATCH --partition=gpu-rtx6k       # GPU partition (gpu-rtx6k, gpu-a40, or ckpt-gpu)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00             # 12 hours per language
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MMS Adapter Fine-tuning Script for Hyak
# Single language training
#
# Usage:
#   sbatch scripts/hyak_train_single.slurm aln
#   sbatch scripts/hyak_train_single.slurm sco
# =============================================================================

set -e  # Exit on error

# Get language from command line argument
LANG=${1:-"aln"}

echo "=========================================="
echo "MMS Adapter Fine-tuning on Hyak"
echo "=========================================="
echo "Language: $LANG"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# =============================================================================
# Environment Setup
# =============================================================================

# Option 1: Using Miniconda (recommended for development)
# Uncomment and modify the path below if using conda
# source /gscratch/scrubbed/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate low-resource-asr

# Option 2: Using uv (if installed on Hyak)
# Ensure uv is in PATH and venv exists
export PATH="$HOME/.cargo/bin:$PATH"

# Navigate to project directory
cd /gscratch/YOUR_ACCOUNT/low-resource-asr  # TODO: Replace with your project path

# Sync dependencies (if needed)
# uv sync

# =============================================================================
# GPU Check
# =============================================================================

echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training for $LANG..."

uv run python -m src.training.aft_mms "$LANG" \
    --num-epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-3

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
