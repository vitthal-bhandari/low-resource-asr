#!/bin/bash
#SBATCH --job-name=mms-aft-all
#SBATCH --account=YOUR_ACCOUNT      # TODO: Replace with your Hyak account
#SBATCH --partition=gpu-rtx6k       # GPU partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00             # 12 hours per job
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-20                # 21 languages (indices 0-20)

# =============================================================================
# MMS Adapter Fine-tuning Script for Hyak
# Job Array: Trains ALL 21 languages in parallel (one GPU each)
#
# Usage:
#   sbatch scripts/hyak_train_all.slurm
#
# This will submit 21 jobs, each training one language.
# Jobs run in parallel based on available resources.
# =============================================================================

set -e  # Exit on error

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
echo "MMS Adapter Fine-tuning on Hyak"
echo "=========================================="
echo "Language: $LANG (task $SLURM_ARRAY_TASK_ID of ${#LANGUAGES[@]})"
echo "Job ID: $SLURM_JOB_ID (Array Job: $SLURM_ARRAY_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# =============================================================================
# Environment Setup
# =============================================================================

# Option 1: Using Miniconda
# source /gscratch/scrubbed/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate low-resource-asr

# Option 2: Using uv
export PATH="$HOME/.cargo/bin:$PATH"

# Navigate to project directory
cd /gscratch/YOUR_ACCOUNT/low-resource-asr  # TODO: Replace with your project path

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
echo "Training complete for $LANG!"
echo "End time: $(date)"
echo "=========================================="
