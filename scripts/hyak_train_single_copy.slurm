#!/bin/bash
#SBATCH --job-name=mms-aft
#SBATCH --account=myaccount               # From cluster allocation
#SBATCH --partition=gpu-l40      # stf: try gpu-2080ti if gpu-l40s is crowded; or gpu-140, gpu-140s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00             # 12 hours per language
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MMS Adapter Fine-tuning Script for HPC
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
LM_ORDER=${3:-"3"}
LM_ALPHA=${4:-"0.5"}
LM_BETA=${5:-"1.0"}
LM_BEAM_WIDTH=${6:-"100"}

echo "=========================================="
echo "MMS Adapter Fine-tuning on HPC"
echo "=========================================="
echo "Language: $LANG  Split: $SPLIT"
echo "LM params: n=$LM_ORDER alpha=$LM_ALPHA beta=$LM_BETA beam=$LM_BEAM_WIDTH"
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
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
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

uv run python -m src.training.aft_mms "$LANG" \
    --split "$SPLIT" \
    --num-epochs 15 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-3 \
    --save-to-hf \
    --save-transcriptions \
    --max-audio-sec-from-csv corpus_stats.csv \
    --use-lm \
    --lm-order "$LM_ORDER" \
    --lm-alpha "$LM_ALPHA" \
    --lm-beta "$LM_BETA" \
    --beam-width "$LM_BEAM_WIDTH"

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
