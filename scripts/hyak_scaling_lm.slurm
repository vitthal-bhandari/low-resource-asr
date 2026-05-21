#!/bin/bash
#SBATCH --job-name=scaling-lm
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Reuses the same scaling_jobs.txt as training.
# Submit after training jobs complete:
#   sbatch --array=0-$(( $(wc -l < results/splits/scaling_jobs.txt) - 1 )) scripts/hyak_scaling_lm.slurm

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"
fi
cd "$PROJECT_DIR"

export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

module purge
module load gcc
module load cesg/boost/1.76.0
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LMPLZ_MEMORY="1G"

JOBS_FILE="$PROJECT_DIR/results/splits/scaling_jobs.txt"
JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")
LANG="${JOB%%:*}"
SPLIT="${JOB##*:}"

OUT_JSON="$PROJECT_DIR/results/lm_ablation/scaling_exp/mms_${LANG}_${SPLIT}_lm.json"
mkdir -p "$(dirname "$OUT_JSON")"

echo "=========================================="
echo "MMS Scaling LM Decode"
echo "  lang=$LANG  split=$SPLIT"
echo "  output=$OUT_JSON"
echo "=========================================="

# Skip if already done
if [ -f "$OUT_JSON" ]; then
  echo "Output already exists — skipping."; exit 0
fi

uv run python scripts/run_ctc_lm_ablation.py \
    --model mms \
    --lang "$LANG" \
    --split "$SPLIT" \
    --lm-orders 4 \
    --lm-alphas 0.5 \
    --lm-betas 1.0 \
    --beam-widths 100 \
    --require-arpa \
    --output-json "$OUT_JSON"
