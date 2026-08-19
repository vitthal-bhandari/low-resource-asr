#!/bin/bash
#SBATCH --job-name=mms-aft-all
#SBATCH --account=myaccount               # From cluster allocation
#SBATCH --partition=gpu-l40      # stf: try gpu-2080ti if gpu-l40s is crowded; or gpu-140, gpu-140s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00             # 12 hours per job
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-20                # 21 languages (indices 0-20)

# =============================================================================
# MMS Adapter Fine-tuning Script for HPC
# Job Array: Trains ALL 21 languages (one GPU per language).
# Primary differentiator: --split (one | mid | all).
#
# Usage:
#   sbatch scripts/hyak_train_all.slurm [one|mid|all]
#   sbatch scripts/hyak_train_all.slurm one
#   sbatch scripts/hyak_train_all.slurm all
# =============================================================================

set -e  # Exit on error

# Configuration: set PROJECT_DIR to match hyak_setup.sh
# No lab: /gscratch/scrubbed/$USER/low-resource-asr
# With lab: /gscratch/YOUR_ACCOUNT/low-resource-asr
PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

# Split: one (1h), mid (tier 3/5/10h), all (full). Default all.
SPLIT=${1:-"all"}
LM_ORDER=${2:-"3"}
LM_ALPHA=${3:-"0.5"}
LM_BETA=${4:-"1.0"}
LM_BEAM_WIDTH=${5:-"100"}

# All 21 languages
LANGUAGES=(
    "aln"    # Gheg Albanian
    "bew"    # Betawi
    "bxk"    # Bukusu
    "cgg"    # Chiga
    "el-CY"  # Cypriot Greek
    "hch"    # Wixárika
    "kcn"    # Nubi
    "koo"    # Konzo
    "led"    # Lendu
    "lke"    # Kenyi
    "lth"    # Thur
    "meh"    # Southwestern Tlaxiaco Mixtec
    "mmc"    # Michoacán Mazahua
    "pne"    # Western Penan
    "ruc"    # Ruuli
    "rwm"    # Amba
    "sco"    # Scots
    "tob"    # Toba Qom
    "top"    # Papantla Totonac
    "ttj"    # Rutoro
    "ukv"    # Kuku
)

# Get language for this array task
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "MMS Adapter Fine-tuning on HPC"
echo "=========================================="
echo "Language: $LANG  Split: $SPLIT (task $SLURM_ARRAY_TASK_ID of ${#LANGUAGES[@]})"
echo "LM params: n=$LM_ORDER alpha=$LM_ALPHA beta=$LM_BETA beam=$LM_BEAM_WIDTH"
echo "Job ID: $SLURM_JOB_ID (Array Job: $SLURM_ARRAY_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Create logs directory
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

uv run python -m src.training.aft_xlsr "$LANG" \
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
echo "Training complete for $LANG!"
echo "End time: $(date)"
echo "=========================================="
