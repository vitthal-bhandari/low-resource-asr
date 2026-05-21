#!/bin/bash
#SBATCH --job-name=mms-scaling
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Array size set at submission time:
#   sbatch --array=0-$(( $(wc -l < results/splits/scaling_jobs.txt) - 1 )) scripts/hyak_scaling_exp.slurm

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p logs "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# KenLM (needed for in-training LM build)
module purge
module load gcc
module load cesg/boost/1.76.0
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LMPLZ_MEMORY="1G"

# Resolve lang:split from job list
JOBS_FILE="$PROJECT_DIR/results/splits/scaling_jobs.txt"
JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")
LANG="${JOB%%:*}"
SPLIT="${JOB##*:}"

echo "=========================================="
echo "MMS Scaling Experiment — fresh fine-tune"
echo "  lang=$LANG  split=$SPLIT"
echo "  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

# Sanity: confirm split TSV exists
TRAIN_TSV="$PROJECT_DIR/data/mozilla_speech_data/$LANG/train-${SPLIT}_${LANG}.tsv"
if [ ! -f "$TRAIN_TSV" ]; then
  echo "ERROR: train TSV not found: $TRAIN_TSV"; exit 1
fi
echo "Train TSV: $TRAIN_TSV ($(wc -l < "$TRAIN_TSV") rows)"

uv run python -m src.training.aft_mms "$LANG" \
    --split "$SPLIT" \
    --num-epochs 15 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-3 \
    --early-stopping-patience 3 \
    --max-audio-sec-from-csv corpus_stats.csv \
    --save-transcriptions \
    --save-to-hf
