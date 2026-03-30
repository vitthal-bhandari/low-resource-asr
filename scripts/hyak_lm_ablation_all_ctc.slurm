#!/bin/bash
#SBATCH --job-name=ctc-lm-all
#SBATCH --account=myaccount
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-41

# =============================================================================
# Decode-only confirmatory LM ablation: 21 languages x 2 CTC models = 42 tasks
# Runs two hyperparameter settings in a single task and writes combined summary.
# Usage:
#   sbatch scripts/hyak_lm_ablation_all_ctc.slurm all
#   sbatch scripts/hyak_lm_ablation_all_ctc.slurm all 4 0.5 1.0 100 3 0.2 1.0 50
# =============================================================================

set -e

PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

SPLIT=${1:-"all"}

# Set A defaults (MMS-favored from pilot)
SETA_ORDER=${2:-"4"}
SETA_ALPHA=${3:-"0.5"}
SETA_BETA=${4:-"1.0"}
SETA_BEAM=${5:-"100"}

# Set B defaults (XLS-R-favored from pilot)
SETB_ORDER=${6:-"3"}
SETB_ALPHA=${7:-"0.2"}
SETB_BETA=${8:-"1.0"}
SETB_BEAM=${9:-"50"}

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
echo "Set A: n=$SETA_ORDER alpha=$SETA_ALPHA beta=$SETA_BETA beam=$SETA_BEAM"
echo "Set B: n=$SETB_ORDER alpha=$SETB_ALPHA beta=$SETB_BETA beam=$SETB_BEAM"
echo "Start time: $(date)"
echo ""

mkdir -p logs

module purge
module load gcc
module load cesg/boost/1.76.0

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:$LD_LIBRARY_PATH"
export LMPLZ_MEMORY="512M"
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

echo "Checking lmplz availability..."
which lmplz
echo "quick sanity line for lmplz build test" | lmplz -S 64M -o 3 --discount_fallback > /tmp/lm_sanity_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.arpa
echo "lmplz sanity build OK"

if [[ "$MODEL" == "mms" ]]; then
  HF_REPO_ID="anonymous/mms-1b-all-aft-${SPLIT}-${LANG}"
else
  HF_REPO_ID="anonymous/xlsr-1b-aft-${SPLIT}-${LANG}"
fi

RUN_DIR="results/lm_ablation/two_settings/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"

OUT_A="$RUN_DIR/${MODEL}_${LANG}_${SPLIT}_setA.json"
OUT_B="$RUN_DIR/${MODEL}_${LANG}_${SPLIT}_setB.json"

echo "Running Set A decoding..."
uv run python scripts/run_ctc_lm_ablation.py \
  --model "$MODEL" \
  --lang "$LANG" \
  --split "$SPLIT" \
  --hf-repo-id "$HF_REPO_ID" \
  --hf-revision "main" \
  --selection-split "test" \
  --require-arpa \
  --lm-orders "$SETA_ORDER" \
  --lm-alphas "$SETA_ALPHA" \
  --lm-betas "$SETA_BETA" \
  --beam-widths "$SETA_BEAM" \
  --output-json "$OUT_A"

echo "Running Set B decoding..."
uv run python scripts/run_ctc_lm_ablation.py \
  --model "$MODEL" \
  --lang "$LANG" \
  --split "$SPLIT" \
  --hf-repo-id "$HF_REPO_ID" \
  --hf-revision "main" \
  --selection-split "test" \
  --require-arpa \
  --lm-orders "$SETB_ORDER" \
  --lm-alphas "$SETB_ALPHA" \
  --lm-betas "$SETB_BETA" \
  --beam-widths "$SETB_BEAM" \
  --output-json "$OUT_B"

echo "Refreshing combined test-WER CSV summary..."
uv run python scripts/summarize_two_setting_test_wer.py \
  --run-dir "$RUN_DIR" \
  --split "$SPLIT" \
  --set-a-label "setA" \
  --set-b-label "setB" \
  --output-csv "$RUN_DIR/test_wer_summary_two_settings.csv"

echo ""
echo "Task complete at: $(date)"
echo "Set A JSON: $OUT_A"
echo "Set B JSON: $OUT_B"
echo "Summary CSV (continually refreshed): $RUN_DIR/test_wer_summary_two_settings.csv"
