#!/bin/bash
#SBATCH --job-name=all-models-aft
#SBATCH --account=stf               # From hyakalloc (free student allocation)
#SBATCH --partition=gpu-l40s      # L40s: 48 GB VRAM; fits all 3 models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G                   # Whisper-large-v3 needs ~128G; MMS/XLS-R are fine with less
#SBATCH --gpus=1
#SBATCH --time=6:00:00             # 3 models x 3 splits — allow up to 72 hours
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Combined Fine-tuning Script for Hyak
# Trains ALL 3 models (MMS, XLS-R, Whisper) on ALL 3 splits (one, mid, all)
# for a SINGLE language — 9 training runs total.
#
# Usage:
#   sbatch scripts/hyak_train_single_all_models.slurm aln
#   sbatch scripts/hyak_train_single_all_models.slurm sco
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

LANG=${1:?"Usage: sbatch $0 <lang_code>  (e.g. aln, sco, bxk)"}

SPLITS=("one" "mid" "all")

echo "=========================================="
echo "Combined Fine-tuning (MMS + XLS-R + Whisper)"
echo "=========================================="
echo "Language: $LANG"
echo "Splits: ${SPLITS[*]}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

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
# Training Loop: 3 models x 3 splits = 9 runs
# =============================================================================

for SPLIT in "${SPLITS[@]}"; do

    # -------------------------------------------------------------------------
    # MMS (adapter fine-tuning, 5 epochs, lr=1e-3, +LM)
    # -------------------------------------------------------------------------
    echo "=========================================="
    echo "[MMS] Language=$LANG  Split=$SPLIT  Start: $(date)"
    echo "=========================================="

    uv run python -m src.training.aft_mms "$LANG" \
        --split "$SPLIT" \
        --num-epochs 15 \
        --batch-size 2 \
        --gradient-accumulation-steps 8 \
        --learning-rate 1e-3 \
        --save-to-hf \
        --save-transcriptions \
        --max-audio-sec-from-csv corpus_stats.csv \
        --use-lm

    echo "[MMS] Done: $LANG / $SPLIT  $(date)"
    echo ""

    # -------------------------------------------------------------------------
    # XLS-R (adapter fine-tuning, 5 epochs, lr=1e-3, +LM)
    # -------------------------------------------------------------------------
    echo "=========================================="
    echo "[XLS-R] Language=$LANG  Split=$SPLIT  Start: $(date)"
    echo "=========================================="

    uv run python -m src.training.aft_xlsr "$LANG" \
        --split "$SPLIT" \
        --num-epochs 15 \
        --batch-size 2 \
        --gradient-accumulation-steps 8 \
        --learning-rate 1e-3 \
        --save-to-hf \
        --save-transcriptions \
        --max-audio-sec-from-csv corpus_stats.csv \
        --use-lm

    echo "[XLS-R] Done: $LANG / $SPLIT  $(date)"
    echo ""

    # -------------------------------------------------------------------------
    # Whisper large-v3 (full fine-tuning, 15 epochs, lr=1e-5, no LM)
    # -------------------------------------------------------------------------
    echo "=========================================="
    echo "[Whisper] Language=$LANG  Split=$SPLIT  Start: $(date)"
    echo "=========================================="

    uv run python -m src.training.aft_whisper "$LANG" \
        --split "$SPLIT" \
        --num-epochs 5 \
        --batch-size 2 \
        --gradient-accumulation-steps 8 \
        --learning-rate 1e-5 \
        --save-to-hf \
        --save-transcriptions \
        --max-audio-sec-from-csv corpus_stats.csv

    echo "[Whisper] Done: $LANG / $SPLIT  $(date)"
    echo ""

done

echo "=========================================="
echo "All 9 runs complete for $LANG!"
echo "End time: $(date)"
echo "=========================================="
