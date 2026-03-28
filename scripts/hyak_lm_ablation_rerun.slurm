#!/bin/bash
#SBATCH --job-name=ctc-lm-rerun
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Targeted single (model, language) LM ablation rerun.
# Writes setA/setB JSONs into an existing results directory.
# Usage:
#   sbatch scripts/hyak_lm_ablation_rerun.slurm <model> <lang> <run_dir> [split] [hf_revision]
#   sbatch scripts/hyak_lm_ablation_rerun.slurm mms hch results/lm_ablation/two_settings/34052631
#   sbatch scripts/hyak_lm_ablation_rerun.slurm xlsr sco results/lm_ablation/two_settings/34052631 all
#   sbatch scripts/hyak_lm_ablation_rerun.slurm xlsr sco results/lm_ablation/two_settings/recover_sco all 8dad75bbf283ef5976763da84748a2537fe11cbb
# =============================================================================

set -e

PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

MODEL=${1:?"Usage: $0 <model> <lang> <run_dir> [split] [hf_revision]"}
LANG=${2:?"Usage: $0 <model> <lang> <run_dir> [split] [hf_revision]"}
RUN_DIR=${3:?"Usage: $0 <model> <lang> <run_dir> [split] [hf_revision]"}
SPLIT=${4:-"all"}
HF_REVISION=${5:-"main"}

# Set A (MMS-favored from pilot)
SETA_ORDER="4"
SETA_ALPHA="0.5"
SETA_BETA="1.0"
SETA_BEAM="100"

# Set B (XLS-R-favored from pilot)
SETB_ORDER="3"
SETB_ALPHA="0.2"
SETB_BETA="1.0"
SETB_BEAM="50"

echo "=========================================="
echo "CTC LM Ablation Rerun (decode-only)"
echo "=========================================="
echo "Model: $MODEL"
echo "Language: $LANG"
echo "Split: $SPLIT"
echo "Output dir: $RUN_DIR"
echo "HF revision: $HF_REVISION"
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
echo "quick sanity line for lmplz build test" | lmplz -S 64M -o 3 --discount_fallback > /tmp/lm_sanity_${SLURM_JOB_ID}.arpa
echo "lmplz sanity build OK"

if [[ "$MODEL" == "mms" ]]; then
  HF_REPO_ID="vitthalbhandari/mms-1b-all-aft-${SPLIT}-${LANG}"
else
  HF_REPO_ID="vitthalbhandari/xlsr-1b-aft-${SPLIT}-${LANG}"
fi

mkdir -p "$RUN_DIR"

OUT_A="$RUN_DIR/${MODEL}_${LANG}_${SPLIT}_setA.json"
OUT_B="$RUN_DIR/${MODEL}_${LANG}_${SPLIT}_setB.json"

echo "Running Set A decoding..."
uv run python scripts/run_ctc_lm_ablation.py \
  --model "$MODEL" \
  --lang "$LANG" \
  --split "$SPLIT" \
  --hf-repo-id "$HF_REPO_ID" \
  --hf-revision "$HF_REVISION" \
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
  --hf-revision "$HF_REVISION" \
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
echo "Rerun complete at: $(date)"
echo "Set A JSON: $OUT_A"
echo "Set B JSON: $OUT_B"
echo "Summary CSV: $RUN_DIR/test_wer_summary_two_settings.csv"
