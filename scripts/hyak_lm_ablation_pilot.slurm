#!/bin/bash
#SBATCH --job-name=ctc-lm-pilot
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# Decode-only pilot LM ablation: 3 pilot languages x 2 CTC models = 6 tasks
# Pulls latest fine-tuned checkpoints from Hugging Face; no retraining.
# =============================================================================

set -e

PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"
SPLIT=${1:-"all"}

LANGUAGES=("sco" "kcn" "ukv")
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
echo "CTC LM Pilot Ablation (decode-only)"
echo "=========================================="
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Language: $LANG"
echo "Split: $SPLIT"
echo "Start time: $(date)"
echo ""

mkdir -p logs

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
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
  HF_REPO_ID="vitthalbhandari/mms-1b-all-aft-${SPLIT}-${LANG}"
else
  HF_REPO_ID="vitthalbhandari/xlsr-1b-aft-${SPLIT}-${LANG}"
fi

uv run python scripts/run_ctc_lm_ablation.py \
  --model "$MODEL" \
  --lang "$LANG" \
  --split "$SPLIT" \
  --hf-repo-id "$HF_REPO_ID" \
  --hf-revision "main" \
  --selection-split "test" \
  --lm-orders "3,4,5" \
  --lm-alphas "0.2,0.5,0.8" \
  --lm-betas "0.0,1.0,2.0" \
  --beam-widths "50,100"

echo ""
echo "Completed at: $(date)"
