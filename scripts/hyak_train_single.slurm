#!/bin/bash
#SBATCH --job-name=mms-aft
#SBATCH --account=stf               # From hyakalloc (free student allocation)
#SBATCH --partition=gpu-l40      # stf: try gpu-2080ti if gpu-l40s is crowded; or gpu-140, gpu-140s
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
#   sbatch scripts/hyak_train_single.slurm aln [one|mid|all]
#   sbatch scripts/hyak_train_single.slurm sco one
# =============================================================================

set -e  # Exit on error

# Configuration: set PROJECT_DIR to match hyak_setup.sh
# No lab: /gscratch/scrubbed/$USER/low-resource-asr
# With lab: /gscratch/YOUR_ACCOUNT/low-resource-asr
PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

# Language and split (one=1h, mid=tier 3/5/10h, all=full)
LANG=${1:-"aln"}
SPLIT=${2:-"all"}

echo "=========================================="
echo "MMS Adapter Fine-tuning on Hyak"
echo "=========================================="
echo "Language: $LANG  Split: $SPLIT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# =============================================================================
# Environment Setup (uv)
# =============================================================================

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
cd "$PROJECT_DIR"

# =============================================================================
# GPU Check
# =============================================================================

echo "Checking GPU availability..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training for $LANG (split=$SPLIT)..."

uv run python -m src.training.aft_mms "$LANG" \
    --split "$SPLIT" \
    --num-epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-3

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
