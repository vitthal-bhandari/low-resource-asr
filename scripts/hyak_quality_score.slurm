#!/bin/bash
#SBATCH --job-name=quality-score
#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"
fi
cd "$PROJECT_DIR"

export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# KenLM
module purge
module load gcc
module load cesg/boost/1.76.0
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LMPLZ_MEMORY="512M"

mkdir -p logs
which lmplz && echo "lmplz OK"

uv run python scripts/quality_score.py --all
