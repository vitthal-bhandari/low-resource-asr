# Hyak (Klone) Cheat Sheet — Reusable for ML Projects

This is a battle-tested guide for running ML training/inference on the
University of Washington Hyak Klone HPC cluster, distilled from real sessions.
Copy and adapt for new projects. Replace `<placeholders>` with your values.

---

## 0) Mental Model in 10 Seconds

- **Login node** (`ssh <netid>@klone.hyak.uw.edu`): light commands only.
  - `module` does **not** work here.
  - `sbatch`, `salloc`, `squeue`, `hyakalloc`, `git`, `vim`, etc. are fine.
- **Compute node** (via `salloc` or `sbatch`): where `module load`, GPUs, builds, training run.
- Put **everything large** in `/gscratch/...`, **nothing large** in `~/` (10 GB quota).
- `/gscratch` is a symlink to `/mmfs1/gscratch` — same directory, just different paths in tracebacks.
- Your **account** in Slurm is an allocation name (e.g. `stf`), **not your NetID**.
- **`rg` (ripgrep) is not installed** on Hyak — use `grep -r` or load a module if needed.

---

## 1) Storage Layout

| Location | Quota | Purpose | Notes |
|---|---|---|---|
| `~/` (`/mmfs1/home/$USER`) | 10 GB | dotfiles, ssh keys, configs | Backed up. Never put data/caches here. |
| `/gscratch/scrubbed/$USER/` | None | code, data, caches, models, logs | **Files purged after 21 days of inactivity.** |
| `/gscratch/<lab>/` | Per-lab | persistent shared lab storage | Only if you have a lab account. |

Recommended per-project layout:

```bash
PROJECT_DIR=/gscratch/scrubbed/$USER/<project-name>
mkdir -p "$PROJECT_DIR"/{logs,results,data,models,scripts,src}
mkdir -p /gscratch/scrubbed/$USER/.cache/{uv,huggingface,torch,pip}
mkdir -p /gscratch/scrubbed/$USER/tools  # for source-built binaries (e.g. KenLM)
```

> Tip: anything you want to keep beyond 21 days must live elsewhere — Git, Hugging Face Hub, Kopah S3, Lolo archive, or a lab `/gscratch/<lab>/` directory.

---

## 2) Find Your Account and Partitions

Run on login node:

```bash
hyakalloc
```

Example output (the user's actual account is `stf`):

```
Account │       Partition │ CPUs │ Memory │ GPUs │
stf     │         compute │  640 │  2800G │    0 │
stf     │          cpu-g2 │  640 │  4984G │    0 │
stf     │    cpu-g2-mem2x │  192 │  3010G │    0 │
stf     │      gpu-2080ti │   40 │   363G │    8 │
stf     │         gpu-l40 │  128 │  1498G │    8 │
stf     │        gpu-l40s │  160 │  1862G │   10 │
stf     │ compute-hugemem │  120 │  2226G │    0 │
stf     │     interactive │   40 │   175G │    0 │
Checkpoint Resources (preemptible, no allocation needed): ckpt, ckpt-g2
```

Pick a partition by workload:

| Workload | Partition | Why |
|---|---|---|
| Source build (CPU only, e.g. KenLM) | `cpu-g2` or `compute` | Fast queue, no GPU wasted |
| Quick interactive testing | `interactive` | Small/no contention |
| Adapter / LoRA fine-tune (≤1.5B params) | `gpu-l40s` (48 GB VRAM) | Best GPU you have |
| Full fine-tune (Whisper-large-v3) | `gpu-l40s` with `--mem=128G` | Needs more RAM |
| Cheap, preemptible long jobs | `ckpt` / `ckpt-g2` | No allocation required, but jobs can be killed |
| Older GPU is fine | `gpu-2080ti` | Often less queue |

---

## 3) Essential Slurm Commands

```bash
# inspect available accounts/partitions
hyakalloc
sinfo

# interactive session (drops you into a shell on a compute node)
salloc -A stf -p gpu-l40s -N 1 -c 8 --mem=32G --gpus=1 -t 02:00:00

# submit batch job
sbatch scripts/<job>.slurm

# monitor
squeue -u $USER
squeue -j <jobid>
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed

# cancel
scancel <jobid>
```

> Common state codes: `PENDING` (waiting), `RUNNING`, `COMPLETED` (`0:0`), `FAILED`, `OUT_OF_MEMORY` (`0:125`), `TIMEOUT`.

---

## 4) Environment Setup (`uv` + Python)

`uv` is a fast Python package manager that respects a project-local `.venv`.

```bash
# install uv (one-time, in $HOME)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# critical: keep all caches off the 10 GB home quota
export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
mkdir -p "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# persist the cache redirect across sessions
echo 'export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"' >> ~/.bashrc

# pin Python per project (this project uses 3.12)
uv python install 3.12
uv venv --python 3.12
uv sync --no-dev
```

### Python version notes (lessons learned)

- `3.12` is a safe ML default on Hyak: HF `datasets[audio]` (`torchcodec`) has wheels, KenLM still builds.
- `3.13+` may break things like `kenlm` due to CPython API changes.
- Pin via `requires-python = ">=3.11,<3.13"` in `pyproject.toml`.

---

## 5) Minimal Slurm Templates

### 5.1 GPU training template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=stf                  # your allocation, not your NetID
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                      # 128G for Whisper-large-v3
#SBATCH --gpus=1
#SBATCH --time=12:00:00                # 24h+ for full fine-tunes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# uv on PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# resolve project dir robustly (works from sbatch, salloc, or direct call)
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$PWD/pyproject.toml" ]; then
  PROJECT_DIR="$PWD"
else
  PROJECT_DIR="/gscratch/scrubbed/$USER/<project-name>"
fi
cd "$PROJECT_DIR"

# caches off home quota
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

# sanity
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# real work
uv run python -m <your.module> <args>
```

### 5.2 Job arrays (sweeps / multi-language)

```bash
#SBATCH --array=0-20                    # 21 tasks, indices 0..20
#SBATCH --output=logs/%x_%A_%a.out      # %A = array job id, %a = task id
#SBATCH --error=logs/%x_%A_%a.err

ITEMS=(aln bew bxk cgg el-CY hch kcn koo led lke lth meh mmc pne ruc rwm sco tob top ttj ukv)
ITEM=${ITEMS[$SLURM_ARRAY_TASK_ID]}
uv run python -m <module> "$ITEM"
```

---

## 6) Modules (only available inside `salloc`/`sbatch`)

These are the modules actually used in this project. Run inside an allocation:

```bash
module purge
module load gcc                # or gcc/10.2.0 / gcc/11.2.0
module load cmake
module load cesg/boost/1.76.0  # Boost is namespaced under the cesg vendor prefix
module list
module show cesg/boost/1.76.0  # to find $BOOST_ROOT
```

Useful discovery:

```bash
module avail              # list categories
module avail boost        # filter
module spider <name>      # search across all modules
```

> Login-node attempts will print: *"WARNING: The 'module' command is not supported on Klone Login nodes."* That's expected — request an allocation first.

---

## 7) Building Source Tools (KenLM/`lmplz` Worked Example)

Use this exact recipe whenever a project needs a non-Python binary built from source.

```bash
# 1) Clone in scratch (CPU is fine; this is a CPU-only build)
cd /gscratch/scrubbed/$USER/tools
git clone --recursive https://github.com/kpu/kenlm.git
cd kenlm

# 2) Get a CPU allocation (faster queue than GPU for build)
salloc -A stf -p cpu-g2 -N 1 -c 8 --mem=32G -t 02:00:00

# 3) Inside the allocation: load modules
module purge
module load gcc
module load cmake
module load cesg/boost/1.76.0
echo $BOOST_ROOT
ls "$BOOST_ROOT/include/boost" >/dev/null && echo "Boost headers OK"

# 4) Configure with explicit Boost hints (CMake on Hyak doesn't always autodetect)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBOOST_ROOT="$BOOST_ROOT" \
  -DBoost_INCLUDE_DIR="$BOOST_ROOT/include" \
  -DBoost_LIBRARY_DIR="$BOOST_ROOT/lib"

# 5) Build
cmake --build . -j 8

# 6) Verify
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
which lmplz && lmplz --help | head
```

In your **Slurm script**, you must export both PATH and LD_LIBRARY_PATH so the binary can find Boost at runtime:

```bash
module purge
module load gcc
module load cesg/boost/1.76.0
export PATH="/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:${LD_LIBRARY_PATH:-}"
# kenlm-specific: cap memory or it will OOM at 80% of system RAM
export LMPLZ_MEMORY="1G"   # or 512M for small jobs
```

> Common errors & fixes:
> - `cmake: command not found` → you're on the login node. Use `salloc` first.
> - `lmplz: error while loading shared libraries: libboost_program_options.so.X.Y.Z` → forgot `LD_LIBRARY_PATH` (or didn't `module load` Boost in the job).
> - `OUT_OF_MEMORY` (`ExitCode 0:125`) when `lmplz` runs → pass `lmplz -S 512M` to cap memory.

---

## 8) Hugging Face on Hyak

```bash
# .env in project root (loaded by your config)
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx   # if you use W&B
```

```bash
# pre-warm large model downloads from an interactive session
# (avoids burning training wall-time on a 6+ GB download)
salloc -A stf -p gpu-l40s --gpus=1 --mem=32G -t 1:00:00
cd $PROJECT_DIR
uv run python -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
WhisperProcessor.from_pretrained('openai/whisper-large-v3')
"
```

If a download is interrupted, the cache will contain `*.incomplete` blobs and the next job will crash. Clean it manually:

```bash
rm -rf /gscratch/scrubbed/$USER/.cache/huggingface/hub/models--<owner>--<model>
```

Pattern for push-then-reuse from training to evaluation:

```bash
# train pushes checkpoint to HF
uv run python -m <train.module> <lang> --save-to-hf

# later eval/decode pulls back from HF
uv run python scripts/<eval>.py --hf-repo-id "<user>/<model>" --hf-revision main
```

> Tip: log `checkpoint_source`, `checkpoint_ref`, `hf_revision` into result JSONs so you can trace which weights produced each metric.

---

## 9) Persistent Gotchas to Read Before Long Runs

1. **`.venv` is partition-specific.** A venv built on `gpu-l40s` may break on `ckpt-g2` (different system libs / CUDA). If you switch partitions and see `_cuda_bindings_redirector` / `_virtualenv` / `_distutils_hack` errors, rebuild the venv inside the new partition:
   ```bash
   rm -rf .venv && uv venv --python 3.12 && uv sync --no-dev
   ```
2. **`PYTHONPATH` is sometimes stripped by `uv run`.** Set it explicitly in the Slurm script and prefer `python -m src.module`. If your `pyproject.toml` adds a `[build-system]`, double-check that the installed package isn't shadowing your local `src/`. If it is, run:
   ```bash
   uv pip uninstall <project-name>
   ```
3. **`getcwd failed: No such file or directory`** when running `salloc`/`sbatch` → you're sitting in a directory that was deleted. `cd` somewhere valid first.
4. **Old kernel warning** (`Detected kernel version 4.18.0 ... below recommended minimum 5.5.0`) → benign on Hyak compute nodes; do not act on it.
5. **`pyctcdecode` API drift** → newer versions take `kenlm_model_path=` instead of `kenlm_model=`. Code that conditionally tries both keeps you portable.
6. **Whisper-large-v3 dtype mismatch** under AMP → train in pure fp32 (`fp16=False, bf16=False`), or cast `input_features` to `model.dtype` inside the data collator. Bump `--mem` to `128G` and `--time` to `24:00:00` for full fine-tunes.
7. **Whisper label length cap** → tokenizer has a hard 448-token limit; truncate labels in your `prepare_dataset` function.
8. **`gpu-l40s` queue can be slow** even when `hyakalloc` shows free CPUs/GPUs. Free aggregates ≠ free node-fit. Try `gpu-l40` or smaller resource requests if pending.
9. **`ckpt`/`ckpt-g2` are preemptible** → fine for short jobs, dangerous for multi-hour ones without checkpointing.
10. **`/gscratch/scrubbed` is purged after 21 days of inactivity.** Anything important must live in Git, HF, or `/gscratch/<lab>/`.

---

## 10) Pre-Flight Sanity Checklist (Run Before Every Long sbatch)

```bash
# on login node
hyakalloc                                  # confirm allocation/partition still has room
which uv && uv --version                   # uv installed
ls $PROJECT_DIR/pyproject.toml             # right project dir
cat $PROJECT_DIR/.env | grep -E 'HF|WANDB' # secrets set
ls /gscratch/scrubbed/$USER/.cache/uv      # cache redirected
df -h ~                                    # home quota OK

# inside salloc on the partition you'll batch on
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import transformers, datasets; print(transformers.__version__, datasets.__version__)"
which lmplz   # only if your pipeline needs KenLM
```

---

## 11) This Project's Concrete Defaults (Reference)

- Account: `stf` (free UW student allocation)
- Project dir: `/gscratch/scrubbed/$USER/low-resource-asr`
- Python: `3.12` via `uv venv`
- `uv` cache: `/gscratch/scrubbed/$USER/.cache/uv`
- HF cache base: `/gscratch/scrubbed/$USER/.cache/huggingface`
- Torch cache: `/gscratch/scrubbed/$USER/.cache/torch`
- KenLM build: `/gscratch/scrubbed/$USER/tools/kenlm/build/bin`
- Boost module: `cesg/boost/1.76.0`
- Typical adapter run (MMS / XLS-R): `--partition=gpu-l40s --gpus=1 --cpus-per-task=8 --mem=64G --time=12:00:00`
- Typical full fine-tune (Whisper-large-v3): `--partition=gpu-l40s --gpus=1 --cpus-per-task=8 --mem=128G --time=24:00:00`
- HF repo naming: `vitthalbhandari/<model>-aft-<split>-<lang>`
- Logs: `logs/%x_%j.out` (or `%x_%A_%a.out` for arrays)

---

## 12) Copy-Forward Checklist for a New Project

- [ ] Pick `PROJECT_DIR` under `/gscratch/scrubbed/$USER/` (or `/gscratch/<lab>/`).
- [ ] Copy this cheat sheet into the new repo and update Section 11 with project-specific defaults.
- [ ] Pin Python version in `pyproject.toml` (`requires-python`) and `uv venv --python X.Y`.
- [ ] Set all cache env vars (`UV_CACHE_DIR`, `HF_*`, `TORCH_HOME`) to scratch.
- [ ] Add `PROJECT_DIR` resolution + `PYTHONPATH` export in every Slurm script.
- [ ] Confirm account + partition with `hyakalloc` + `sinfo`.
- [ ] Run a 5–10 min interactive smoke test (`salloc`, then `uv run python -c "..."`) before submitting long jobs.
- [ ] Pre-warm large HF model downloads in interactive sessions.
- [ ] If using KenLM/native libs, build once under `tools/`, then load `module`s + export `LD_LIBRARY_PATH` in jobs.
- [ ] If switching partitions, rebuild `.venv` inside the new partition.
- [ ] Push checkpoints to HF (or another persistent store) so `scrubbed` purge can't lose them.
