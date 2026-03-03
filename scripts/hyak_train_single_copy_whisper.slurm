#!/bin/bash
#SBATCH --job-name=whisper-aft
#SBATCH --account=stf               # From hyakalloc (free student allocation)
#SBATCH --partition=gpu-l40s      # L40s: 48 GB VRAM; fits whisper-large-v3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                   # Whisper-large-v3 needs more host RAM for optimizer states
#SBATCH --gpus=1
#SBATCH --time=24:00:00             # 24 hours per language (full fine-tuning is slow)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Whisper Large-v3 Fine-tuning Script for Hyak
# Single-language training (one GPU, one language).
#
# Usage:
#   sbatch scripts/hyak_train_single_copy_whisper.slurm aln [one|mid|all]
#   sbatch scripts/hyak_train_single_copy_whisper.slurm sco one
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
echo "Whisper Large-v3 Fine-tuning on Hyak"
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
export HF_CACHE_BASE="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HOME="$HF_CACHE_BASE"
export HF_HUB_CACHE="$HF_CACHE_BASE/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE_BASE/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_BASE/transformers"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
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

uv run python -m src.training.aft_whisper "$LANG" \
    --split "$SPLIT" \
    --num-epochs 5 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --save-to-hf \
    --save-transcriptions \
    --max-audio-sec-from-csv corpus_stats.csv

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
