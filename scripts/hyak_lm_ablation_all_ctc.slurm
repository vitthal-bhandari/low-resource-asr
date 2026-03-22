#!/bin/bash
#SBATCH --job-name=ctc-lm-all
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-41

# =============================================================================
# Decode-only confirmatory LM ablation: 21 languages x 2 CTC models = 42 tasks
# Uses fixed LM hyperparameters chosen from pilot results.
# Usage:
#   sbatch scripts/hyak_lm_ablation_all_ctc.slurm all 3 0.5 1.0 100
# =============================================================================

set -e

PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

SPLIT=${1:-"all"}
LM_ORDER=${2:-"3"}
LM_ALPHA=${3:-"0.5"}
LM_BETA=${4:-"1.0"}
BEAM_WIDTH=${5:-"100"}

LANGUAGES=(
  "aln" "bew" "bxk" "cgg" "el-CY" "hch" "kcn" "koo" "led" "lke" "lth"
  "meh" "mmc" "pne" "ruc" "rwm" "sco" "tob" "top" "ttj" "ukv"
)
MODELS=("mms" "xlsr")

NUM_LANGS=${#LANGUAGES[@]}
LANG_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LANGS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LANGS))

LANG=${LANGUAGES[$LANG_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

if [[ -z "$LANG" || -z "$MODEL" ]]; then
  echo "Invalid array index mapping: task=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "=========================================="
echo "CTC LM Confirmatory Ablation (decode-only)"
echo "=========================================="
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Language: $LANG"
echo "Split: $SPLIT"
echo "Fixed params: n=$LM_ORDER alpha=$LM_ALPHA beta=$LM_BETA beam=$BEAM_WIDTH"
echo "Start time: $(date)"
echo ""

mkdir -p logs

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

if [[ "$MODEL" == "mms" ]]; then
  CHECKPOINT_DIR="models/mms/$LANG/$SPLIT"
else
  CHECKPOINT_DIR="models/xlsr/$LANG/$SPLIT"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR"
  exit 1
fi

uv run python scripts/run_ctc_lm_ablation.py \
  --model "$MODEL" \
  --lang "$LANG" \
  --split "$SPLIT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --lm-orders "$LM_ORDER" \
  --lm-alphas "$LM_ALPHA" \
  --lm-betas "$LM_BETA" \
  --beam-widths "$BEAM_WIDTH"

echo ""
echo "Completed at: $(date)"
