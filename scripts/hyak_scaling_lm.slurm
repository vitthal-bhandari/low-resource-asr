#!/bin/bash
#SBATCH --job-name=scaling-lm
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
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

# Override with SCALING_JOBS_FILE to decode a subset (e.g. only completed training jobs):
#   SCALING_JOBS_FILE=results/splits/scaling_jobs_lm_ready.txt sbatch --array=0-N ...
JOBS_FILE="${SCALING_JOBS_FILE:-$PROJECT_DIR/results/splits/scaling_jobs.txt}"
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

# Resolve ARPA: reuse the pre-existing full-data ARPA for this language so that
# the LM quality is held constant across all training-data splits. This isolates
# the acoustic model improvement and avoids very sparse LMs for small splits.
# Fallback chain: mms/all → xlsr/all (same transcriptions; handles hch gap) → auto-build.
ARPA_BASE="$PROJECT_DIR/results/lm_ablation/arpa_artifacts"
ARPA_PATH=""
if [ -f "$ARPA_BASE/mms/$LANG/all/lm_4gram.arpa" ]; then
  ARPA_PATH="$ARPA_BASE/mms/$LANG/all/lm_4gram.arpa"
  echo "  ARPA: reusing pre-existing MMS full-data ARPA"
elif [ -f "$ARPA_BASE/xlsr/$LANG/all/lm_4gram.arpa" ]; then
  ARPA_PATH="$ARPA_BASE/xlsr/$LANG/all/lm_4gram.arpa"
  echo "  ARPA: no MMS ARPA found — using XLSR full-data ARPA (same transcriptions)"
else
  echo "  ARPA: no pre-existing ARPA found — will build from training sentences"
fi
echo "  arpa=$ARPA_PATH"

# Resolve checkpoint: prefer local directory (avoids HF push failures on restarted jobs).
# Falls back to HF if local doesn't exist.
LOCAL_CKPT="$PROJECT_DIR/models/mms/$LANG/$SPLIT"
CKPT_ARG=()
if [ -d "$LOCAL_CKPT" ] && [ -f "$LOCAL_CKPT/config.json" ]; then
  CKPT_ARG=(--checkpoint-dir "$LOCAL_CKPT")
  echo "  checkpoint: local ($LOCAL_CKPT)"
else
  echo "  checkpoint: HF (vitthalbhandari/mms-1b-all-aft-${SPLIT}-${LANG})"
fi

# Build the --lm-arpa-path argument conditionally
LM_ARPA_ARG=()
[ -n "$ARPA_PATH" ] && LM_ARPA_ARG=(--lm-arpa-path "$ARPA_PATH")

uv run python scripts/run_ctc_lm_ablation.py \
    --model mms \
    --lang "$LANG" \
    --split "$SPLIT" \
    --lm-orders 4 \
    --lm-alphas 0.5 \
    --lm-betas 1.0 \
    --beam-widths 100 \
    --require-arpa \
    "${CKPT_ARG[@]}" \
    "${LM_ARPA_ARG[@]}" \
    --output-json "$OUT_JSON"
