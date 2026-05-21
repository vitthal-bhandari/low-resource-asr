# Experiment Plan: Data Curation × Scaling × Decoding

**Goal:** Produce three interconnected experiments for the Datology AI interview, demonstrating
how data quality, training data volume, and decoding interact in a truly low-resource ASR setting.

**Core framing:** *What happens to model quality when you have almost no data to curate from?
How do decoding choices and data volume interact with curation quality?*

---

## Overview

| Experiment | Question | New code | Compute |
|---|---|---|---|
| **Exp 1** | Which utterances are low-quality? | `src/data/quality.py`, `scripts/quality_score.py` | CPU-only, ~5 min/lang |
| **Exp 2** | Combined scaling + curation + decoding curve | `scripts/create_curated_splits.py`, `scripts/hyak_scaling_exp.slurm`, `scripts/hyak_scaling_lm.slurm`, `scripts/plot_scaling_curves.py` | GPU (Hyak), ~2 GPU-hours/lang |
| **Exp 3c** | Does n-gram LM compensate for noisy training data? | `scripts/analyze_quality_decoding_correlation.py` | CPU-only, <5 min total |

**Model choice for Exp 2:** MMS-1B-All with bottleneck adapters (fastest to fine-tune, best
aggregate performer in the paper, matches Datology's per-language bespoke conditioning framing).

**Language selection for Exp 2 (6 languages, typologically diverse):**

| Code | Language | Family | Why included |
|---|---|---|---|
| `sco` | Scots | Indo-European (Germanic) | English-proximal; interesting for quality scoring because many utterances look like English |
| `kcn` | Nubi | Creole (Arabic base) | Extreme insertion blowup case; expected largest curation gains |
| `el-CY` | Cypriot Greek | Indo-European (Hellenic) | Best existing n-gram result (27.3% WER reduction); clean orthography |
| `aln` | Gheg Albanian | Indo-European (Albanian) | European; moderate resource |
| `bew` | Betawi | Austronesian | Southeast Asian; distinct phonology and script conventions |
| `hch` | Wixárika | Uto-Aztecan | Mesoamerican indigenous; expected very sparse quality signal |

---

## Phase 0: Extend `corpus_statistics.py` (30 min)

**File to modify:** `scripts/corpus_statistics.py`

Add p5 (5th percentile) to the `corpus_stats()` function and all downstream
table/CSV output. This value is needed by the quality scorer as the lower duration
hard filter threshold.

**Changes:**

1. In `corpus_stats()`, add `"p5_sec": round(sec.quantile(0.05), 2)` to the returned dict.
2. Update the `columns` list in `print_table(...)` to include `("p5_sec", "p5_s")`.
3. Update the CSV column list to include `"p5_sec"` between `"n"` and `"median_sec"`.
4. Update the `--csv` output loop accordingly.

**Run after change:**
```bash
uv run python scripts/corpus_statistics.py --csv > corpus_stats.csv
```

The updated `corpus_stats.csv` at project root is the authoritative per-language duration
threshold table consumed by Exp 1.

---

## Phase 1: Data Quality Scoring

### 1a. Core quality logic — `src/data/quality.py`

New file. Contains all scoring logic; no I/O or CLI. Importable by other scripts.

```python
"""
Utterance-level quality scoring for Mozilla Spontaneous Speech training data.

Two-stage pipeline:
  Stage 1 (hard filters, applied first):
    - Duration filter: drop if duration_sec < p5_sec OR > p95_sec for the language
    - Repetition filter: drop if char_4gram_repetition_ratio > p99 within language
  Stage 2 (soft ranking on survivors):
    - KenLM character 4-gram perplexity trained on all training transcripts
    - Utterances ranked by perplexity percentile (lower perplexity = higher quality)
    - Keep bottom KEEP_FRACTION (default 0.60) by perplexity

Outputs a DataFrame with columns added to the original TSV:
    duration_filtered   bool  — flagged by duration hard filter
    rep_filtered        bool  — flagged by repetition hard filter
    char_4gram_rep      float — character 4-gram repetition ratio (computed for all)
    ppl_score           float — KenLM perplexity (NaN if hard-filtered)
    ppl_pct             float — within-language perplexity percentile (NaN if filtered)
    quality_keep        bool  — True if utterance passes all filters + soft threshold
"""
```

**Functions to implement:**

```python
def char_4gram_repetition_ratio(text: str) -> float:
    """
    Fraction of character 4-grams that are repeated within the utterance.
    Returns 0.0 for texts shorter than 4 characters.
    
    Algorithm:
      grams = [text[i:i+4] for i in range(len(text) - 3)]
      if not grams: return 0.0
      return 1.0 - (len(set(grams)) / len(grams))
    """

def build_kenlm_char_model(
    sentences: list[str],
    output_dir: Path,
    order: int = 4,
) -> Path | None:
    """
    Build a KenLM character-level n-gram ARPA file from a list of sentences.
    
    Process:
      1. Join sentences with newlines; split each sentence into space-separated
         characters (e.g. "hello" → "h e l l o"). This makes KenLM treat each
         character as a token.
      2. Pipe to lmplz: lmplz -o {order} --discount_fallback
      3. Write ARPA to output_dir / f"char_{order}gram.arpa"
    
    Returns path to ARPA file, or None if lmplz not available.
    Raises RuntimeError if lmplz exits non-zero.
    
    Note: Use subprocess.run with text=True, pipe stdin. Check returncode.
    """

def score_perplexity(
    sentences: list[str],
    arpa_path: Path,
) -> list[float]:
    """
    Score each sentence with character-level KenLM perplexity.
    
    Uses kenlm Python bindings:
        import kenlm
        model = kenlm.Model(str(arpa_path))
        for s in sentences:
            char_seq = " ".join(list(s.lower()))
            ppl = model.perplexity(char_seq)
    
    Returns list of float perplexity values, one per sentence.
    High perplexity = atypical character sequence = lower quality.
    """

def compute_quality_scores(
    df: pd.DataFrame,          # full training split DataFrame (from ss-corpus TSV)
    output_dir: Path,          # where to write the KenLM ARPA artifact
    lm_order: int = 4,
    keep_fraction: float = 0.60,
    rep_hard_filter_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Main entry point. Returns df with quality columns added.
    
    Steps:
      1. Compute p5 and p95 of duration_ms within this df (language-specific).
      2. Set duration_filtered = (duration_ms/1000 < p5_sec) | (duration_ms/1000 > p95_sec).
      3. Compute char_4gram_rep for ALL rows (including hard-filtered ones, for diagnostics).
      4. Compute p99 of char_4gram_rep across all rows.
      5. Set rep_filtered = char_4gram_rep > p99_rep (and not already duration_filtered).
         (Only apply rep filter to duration-survivors to avoid double-counting.)
      6. survivors = df[~duration_filtered & ~rep_filtered]
      7. Build KenLM on survivors['transcription'] (write ARPA to output_dir).
      8. Score survivors with perplexity → ppl_score column.
      9. Compute ppl_pct = percentile rank within survivors (0=lowest ppl, 100=highest).
      10. quality_keep = ppl_pct <= (keep_fraction * 100)  # keep bottom X% by ppl
      11. Set quality_keep=False for all hard-filtered rows.
      12. Return augmented df.
    
    Important: build KenLM on survivors only (not the full corpus), so the LM is
    not contaminated by noisy utterances it's supposed to be scoring against.
    """
```

**Dependencies:** `kenlm` (Python bindings, already available for LM ablation), `subprocess`
(for lmplz, same binary used in `_build_arpa_from_text` in `aft_mms.py`).

Check how `aft_mms.py` calls `lmplz` in its `_build_arpa_from_text` function and mirror
that exact subprocess call pattern, but replace word-level tokenization with
character-level space-joining.

---

### 1b. CLI driver — `scripts/quality_score.py`

```
Usage:
    uv run python scripts/quality_score.py --all
    uv run python scripts/quality_score.py sco kcn el-CY
    uv run python scripts/quality_score.py sco --keep-fraction 0.5 --dry-run

Arguments:
    lang [...]         One or more language codes. Omit with --all.
    --all              Run for all 21 languages.
    --keep-fraction    Soft threshold (default: 0.60).
    --lm-order         KenLM order (default: 4).
    --dry-run          Print stats but do not write output TSVs.
    --output-csv       Write summary CSV to results/quality/ (default: True).
```

**Per-language flow:**
1. Load `data/mozilla_speech_data/{lang}/ss-corpus-{lang}.tsv`.
2. Filter to rows where `split == "train"` and `duration_ms > 0` and transcription is non-empty.
   (Quality scoring is only for training data. Val/test are never touched.)
3. Call `compute_quality_scores(train_df, output_dir=results/quality/{lang}/, ...)`.
4. Write augmented TSV to `results/quality/{lang}/quality_scores_{lang}.tsv`.
   This file has all original columns plus: `duration_filtered`, `rep_filtered`,
   `char_4gram_rep`, `ppl_score`, `ppl_pct`, `quality_keep`.
5. Write per-language summary to `results/quality/quality_summary.csv` (append or overwrite).

**Summary CSV columns:**
```
lang, n_train, n_duration_filtered, n_rep_filtered, n_survivors,
n_quality_keep, keep_fraction, mean_ppl_keep, mean_ppl_drop,
median_char_4gram_rep, p99_char_4gram_rep, kenlm_arpa_path
```

**Verification step:** After running, print for each language:
```
sco: 1842 train → 87 duration-filtered → 2 rep-filtered → 921 quality-keep (60.0%)
     ppl survivors: mean=423.1, p50=318.2, p95=1204.7
     ppl quality_keep: mean=289.4 | ppl dropped: mean=632.8
```
This confirms the filter is meaningfully separating high/low quality utterances.

---

## Phase 2: Create Curated + New Hour Splits

### 2a. New script — `scripts/create_curated_splits.py`

```
Usage:
    uv run python scripts/create_curated_splits.py --all
    uv run python scripts/create_curated_splits.py sco kcn el-CY aln bew hch

Arguments:
    lang [...]     Language codes. Omit with --all.
    --all          Run for all 21 languages.
    --dry-run      Print split sizes without writing.
```

**Purpose:** Creates all TSV files needed for Exp 2 training. Specifically:
- New raw hour splits that don't yet exist (2h for all langs; 3h for medium/large langs)
- Curated splits at all hour targets

**Hour targets per tier (matching existing tier logic from `create_splits.py`):**

| Tier | Total train hours | Hour splits to create |
|---|---|---|
| Small (< 7h) | e.g. kcn, hch | 1h (exists), 2h (new), 3h (exists as mid), full (exists) |
| Medium (7–10h) | e.g. sco | 1h (exists), 2h (new), 3h (new), 5h (exists as mid), full (exists) |
| Large (≥ 10h) | e.g. el-CY | 1h (exists), 2h (new), 3h (new), 5h (exists as mid), full (exists) |

**New raw splits to create** (for each of the 6 experiment languages):

Write to `data/mozilla_speech_data/{lang}/`:
- `train-2h_{lang}.tsv` — first 2h by cumulative duration, sorted by audio_file
- `train-3h_{lang}.tsv` — first 3h (medium/large only; for small langs this = train-mid)
- Record filename lists to `results/splits/{lang}_train_2h_filenames.txt` and
  `results/splits/{lang}_train_3h_filenames.txt`.

**Curated splits to create** for each language:

1. Load `results/quality/{lang}/quality_scores_{lang}.tsv`.
2. Extract `quality_keep == True` rows → curated pool.
3. Sort curated pool by `audio_file` (deterministic, mirrors `create_splits.py` convention).
4. Use the same `split_by_duration()` logic as `create_splits.py` to carve out hour splits
   from the curated pool.
5. Write to `data/mozilla_speech_data/{lang}/`:
   - `train-curated-1h_{lang}.tsv`
   - `train-curated-2h_{lang}.tsv`
   - `train-curated-3h_{lang}.tsv` (medium/large only)
   - `train-curated-5h_{lang}.tsv` (medium/large only)
   - `train-curated-all_{lang}.tsv` (all quality_keep rows)
6. Write filename lists to `results/splits/{lang}_train_curated_{hours}_filenames.txt`.

**Important:** If a curated hour split would contain fewer utterances than the equivalent
raw split (because the curated pool is smaller), that is expected and intentional — it is
exactly the point of the experiment. Log a warning if the curated pool has < 30 minutes of
audio total, as training will likely be unstable.

**Verify dry run output should look like:**
```
sco (Medium tier, 8.2h train, 5.0h curated pool):
  raw:     1h=312utt  2h=589utt  3h=843utt  5h=1201utt  all=1842utt
  curated: 1h=298utt  2h=541utt  3h=762utt  5h=1093utt  all=1105utt
```

---

## Phase 3: Training on New Splits

### 3a. New SLURM script — `scripts/hyak_scaling_exp.slurm`

Model: MMS only. 6 languages × (raw: 2h + 3h for medium/large) + (curated: 1h + 2h + 3h for medium/large + 5h for medium/large + all) training jobs.

For raw splits, 1h, 5h, and full already have trained checkpoints on HF (`vitthalbhandari/mms-1b-all-aft-{split}-{lang}`). Only train what doesn't exist:
- Raw 2h: all 6 langs
- Raw 3h: aln, bew, el-CY, sco (medium/large tier only)
- Curated 1h, 2h, all: all 6 langs
- Curated 3h: aln, bew, el-CY, sco
- Curated 5h: aln, bew, el-CY, sco

This is approximately 6 × 2 (raw) + 6 × 3 (curated small-tier) + 4 × 5 (curated med-tier) = ~44 training jobs.

Use `--array` SLURM job array over a fixed list of `lang:split` pairs.

```bash
#!/bin/bash
#SBATCH --job-name=mms-scaling-exp
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-43   # adjust to number of jobs

PROJECT_DIR="/gscratch/scrubbed/$USER/low-resource-asr"

# Define the job matrix as "lang:split" pairs
JOBS=(
  "sco:2h"  "sco:curated-1h"  "sco:curated-2h"  "sco:curated-3h"  "sco:curated-5h"  "sco:curated-all"
  "sco:3h"
  "kcn:2h"  "kcn:curated-1h"  "kcn:curated-2h"  "kcn:curated-all"
  "el-CY:2h" "el-CY:3h" "el-CY:curated-1h" "el-CY:curated-2h" "el-CY:curated-3h" "el-CY:curated-5h" "el-CY:curated-all"
  "aln:2h"  "aln:3h"  "aln:curated-1h"  "aln:curated-2h"  "aln:curated-3h"  "aln:curated-5h"  "aln:curated-all"
  "bew:2h"  "bew:3h"  "bew:curated-1h"  "bew:curated-2h"  "bew:curated-3h"  "bew:curated-5h"  "bew:curated-all"
  "hch:2h"  "hch:curated-1h"  "hch:curated-2h"  "hch:curated-all"
)

JOB=${JOBS[$SLURM_ARRAY_TASK_ID]}
LANG=${JOB%%:*}
SPLIT=${JOB##*:}

cd $PROJECT_DIR
source .venv/bin/activate

uv run python -m src.training.aft_mms $LANG \
    --split $SPLIT \
    --num-epochs 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --save-to-hf
```

**Note on `aft_mms.py` compatibility:** `load_and_preprocess_data(lang, split)` already
constructs `train-{split}_{lang}.tsv` dynamically. Since Phase 2 writes files like
`train-curated-1h_{lang}.tsv`, passing `split="curated-1h"` will resolve correctly
with no changes to `aft_mms.py`. Verify this with a dry-run check before submitting:
```bash
python -c "from src.training.aft_mms import load_and_preprocess_data; load_and_preprocess_data('sco', 'curated-1h')"
```

**Existing checkpoints to reuse (do not retrain):**

| lang | split | HF repo |
|---|---|---|
| sco | one (1h) | `vitthalbhandari/mms-1b-all-aft-one-sco` |
| sco | mid (5h) | `vitthalbhandari/mms-1b-all-aft-mid-sco` |
| sco | all | `vitthalbhandari/mms-1b-all-aft-all-sco` |
| kcn | one, mid, all | same pattern |
| … | … | … |

Before submitting, enumerate existing HF repos and remove those jobs from the array.

---

## Phase 4: LM Decoding on New Checkpoints

### 4a. New SLURM script — `scripts/hyak_scaling_lm.slurm`

For each new checkpoint from Phase 3, run a simplified LM ablation using the best
hyperparameters already identified in `results/lm_ablation/` (no full grid search;
just apply the known-best alpha/beta/beam/order to save compute).

From existing results, best params for MMS are approximately:
- `lm_order=4, alpha=0.5, beta=1.0, beam_width=100` (Set A from hyak_lm_ablation_all_ctc.slurm)

The script should:
1. Iterate over the same `lang:split` pairs as Phase 3 (plus the existing checkpoints for 1h, 5h, all).
2. Call `run_ctc_lm_ablation.py` with `--lm-orders 4 --lm-alphas 0.5 --lm-betas 1.0 --beam-widths 100`.
3. Output JSON to `results/lm_ablation/scaling_exp/{model}_{lang}_{split}_lm.json`.

```bash
#SBATCH --job-name=scaling-lm
#SBATCH --time=4:00:00
#SBATCH --array=0-N  # same job list as Phase 3 + existing checkpoints

# ... setup as above ...

uv run python scripts/run_ctc_lm_ablation.py \
    --model mms \
    --lang $LANG \
    --split $SPLIT \
    --lm-orders 4 \
    --lm-alphas 0.5 \
    --lm-betas 1.0 \
    --beam-widths 100 \
    --require-arpa \
    --output-json results/lm_ablation/scaling_exp/mms_${LANG}_${SPLIT}_lm.json
```

**For existing checkpoints** (1h raw, 5h raw, full raw): reuse or re-run `run_ctc_lm_ablation.py`
on the already-trained HF checkpoints with the same simplified hyperparams for a consistent
comparison. If existing `lm_ablation` JSONs already have 4-gram results with alpha=0.5,
beta=1.0, beam=100, extract those values directly without re-running.

---

## Phase 5: Plotting — Scaling × Curation × Decoding Curves

### 5a. Results aggregation — `scripts/collect_scaling_results.py`

Reads all JSON files from `results/lm_ablation/scaling_exp/` (and existing `results/lm_ablation/`)
and collects WER values into a single CSV for plotting.

**Output: `results/scaling_exp_results.csv`** with columns:
```
lang, split_name, hours_approx, condition, wer_test, cer_test
```

Where:
- `split_name` is e.g. `one`, `2h`, `mid`, `all`, `curated-1h`, `curated-2h`, etc.
- `hours_approx` is a numeric approximation: `one`→1.0, `2h`→2.0, `mid`→3.0 or 5.0
  (read actual hours from the split TSV or from `corpus_stats.csv`).
- `condition` ∈ `{"raw_greedy", "raw_4gram", "curated_greedy", "curated_4gram"}`.

Populate `condition` by inspecting whether the split_name contains `"curated"` and
whether the WER is from `greedy` or `ngram_beam` entries in the JSON.

---

### 5b. Plot script — `scripts/plot_scaling_curves.py`

```
Usage:
    uv run python scripts/plot_scaling_curves.py
    uv run python scripts/plot_scaling_curves.py --langs sco kcn el-CY
    uv run python scripts/plot_scaling_curves.py --metric wer --output-dir results/figures/

Arguments:
    --langs        Subset of languages to plot (default: all 6 experiment languages)
    --metric       wer or cer (default: wer)
    --output-dir   Where to save figures (default: results/figures/)
    --format        png or pdf (default: png, 300 dpi)
```

**Figure layout:** One subplot per language (2×3 grid or 3×2), shared y-axis label "WER".

**Per-language subplot:**
- X-axis: training hours (numeric, log scale if range > 5×)
- Y-axis: WER (0–1)
- Three lines:
  - Line A (gray, dashed): raw data + greedy decoding
  - Line B (blue, solid): curated data + greedy decoding
  - Line C (orange, solid): curated data + 4-gram LM decoding
- Markers at each data point (circles for A, squares for B/C)
- Annotation: gap between A and C at the 1h point, labeled "↓ X.X% WER"
- Legend: shared across subplots (single legend box)

**Implementation notes:**
- Use `matplotlib` with `seaborn` style or plain `matplotlib`.
- Load `results/scaling_exp_results.csv` from Phase 5a.
- Group by `lang` and `condition`, sort by `hours_approx`.
- For languages where 3h or 5h curated data doesn't exist (small tier), simply
  don't plot that point (no interpolation).
- Save one combined figure: `results/figures/scaling_curation_decoding.png`
- Also save individual per-language figures: `results/figures/scaling_{lang}.png`

**Key visual design goal:** The gap between Line A and Line C should be visually largest
at low data volumes (1h, 2h) and narrow at high data volumes (full). This "converging
funnel" shape is the main empirical claim. Annotate this if visible.

---

## Phase 6: Experiment 3c — Decoding as Quality Compensator

### 6a. Script — `scripts/analyze_quality_decoding_correlation.py`

No new training or decoding needed. This is a post-hoc analysis combining:
- Per-language quality scores from `results/quality/quality_summary.csv` (Phase 1)
- Per-language WER improvement from n-gram decoding from existing `results/lm_ablation/*.json`

```
Usage:
    uv run python scripts/analyze_quality_decoding_correlation.py
    uv run python scripts/analyze_quality_decoding_correlation.py --model mms --split all
    uv run python scripts/analyze_quality_decoding_correlation.py --output results/figures/

Arguments:
    --model        mms or xlsr (default: mms)
    --split        Train split to use for lm_ablation JSONs (default: all)
    --output       Output directory for CSV and plot (default: results/quality/)
```

**Step-by-step:**

1. **Load quality summary:** Read `results/quality/quality_summary.csv`.
   Key column: `mean_ppl_keep` (mean perplexity of kept utterances; proxy for corpus quality —
   lower ppl = more internally consistent language = higher quality).
   Also useful: `keep_fraction` (how many survived all filters; for all langs this = 0.60,
   but note what fraction survived hard filters before soft threshold).

2. **Compute hard-filter survival rate per language:**
   From quality_summary: `hard_filter_survival_rate = n_survivors / n_train`
   (n_survivors = n_train - n_duration_filtered - n_rep_filtered).
   This is a cleaner quality proxy: what fraction of utterances were clean enough to even
   reach the soft ranking stage?

3. **Load decoding improvement:** Read existing lm_ablation JSONs from `results/lm_ablation/`.
   For each language with a `{model}_{lang}_{split}_lm_ablation.json`, extract:
   - `greedy_wer` = `results["greedy"]["test"]["wer"]`
   - `best_ngram_wer` = `min(results["ngram_beam"], key=lambda x: x["test"]["wer"])["test"]["wer"]`
   - `wer_reduction_abs` = greedy_wer - best_ngram_wer
   - `wer_reduction_rel` = (greedy_wer - best_ngram_wer) / greedy_wer

4. **Merge** quality summary and decoding improvement on `lang` column.

5. **Compute Pearson r** between:
   - X: `hard_filter_survival_rate` (higher = cleaner corpus)
   - Y: `wer_reduction_rel` (higher = more WER improvement from n-gram LM)
   - Hypothesis: **negative correlation** — cleaner corpora benefit less from n-gram LM
     (the acoustic model already learned clean representations; decoding adds marginal gain).
     Noisier corpora benefit more (n-gram LM compensates for degraded acoustic representations).

6. **Also compute Pearson r** between `mean_ppl_keep` and `wer_reduction_rel`.
   Higher ppl_keep = internally more variable corpus = lower quality. Expect positive r.

7. **Output:**
   - `results/quality/quality_decoding_correlation.csv`:
     ```
     lang, n_train, hard_filter_survival_rate, mean_ppl_keep,
     greedy_wer, best_ngram_wer, wer_reduction_abs, wer_reduction_rel
     ```
   - Print Pearson r values and p-values to stdout.
   - Scatter plot: X=`hard_filter_survival_rate`, Y=`wer_reduction_rel`,
     points labeled by lang code. Draw regression line. Annotate r and p-value.
     Save to `results/figures/quality_decoding_correlation.png`.

**Interpretation guidance for the interview:**
- If r < -0.4 (negative, moderate–strong): the n-gram LM is acting as a data quality
  compensator. Languages with the noisiest training corpora gain the most from decoding.
  This directly parallels Datology's finding that curation matters more in low-resource settings.
- If r ≈ 0: decoding improvement is language-specific and not driven by corpus quality,
  suggesting linguistic structure (e.g. phoneme inventory, morphological complexity) dominates.
  Also a reportable finding.

---

## Data Flow Summary

```
corpus_statistics.py (Phase 0)
    └── corpus_stats.csv (p5_sec added)
           │
           └── quality_score.py (Phase 1)
                   └── results/quality/{lang}/quality_scores_{lang}.tsv
                   └── results/quality/quality_summary.csv
                          │
                          ├── create_curated_splits.py (Phase 2)
                          │       └── data/mozilla_speech_data/{lang}/train-curated-*_{lang}.tsv
                          │       └── data/mozilla_speech_data/{lang}/train-2h_{lang}.tsv  (raw)
                          │       └── data/mozilla_speech_data/{lang}/train-3h_{lang}.tsv  (raw, med/large)
                          │              │
                          │              └── hyak_scaling_exp.slurm (Phase 3)
                          │                      └── HF checkpoints: mms-1b-all-aft-{split}-{lang}
                          │                             │
                          │                             └── hyak_scaling_lm.slurm (Phase 4)
                          │                                     └── results/lm_ablation/scaling_exp/*.json
                          │                                            │
                          │                                            └── collect_scaling_results.py (Phase 5a)
                          │                                                    └── results/scaling_exp_results.csv
                          │                                                           │
                          │                                                           └── plot_scaling_curves.py (Phase 5b)
                          │                                                                   └── results/figures/scaling_*.png
                          │
                          └── analyze_quality_decoding_correlation.py (Phase 6)
                                  (also reads existing results/lm_ablation/*.json)
                                  └── results/quality/quality_decoding_correlation.csv
                                  └── results/figures/quality_decoding_correlation.png
```

---

## Execution Order and Time Budget (3 days)

### Day 1 (CPU work, local machine)
1. Phase 0: Modify `corpus_statistics.py`, regenerate `corpus_stats.csv` (~30 min)
2. Phase 1: Write `src/data/quality.py` and `scripts/quality_score.py` (~3 hrs)
   - Run for all 6 experiment languages; verify output stats
3. Phase 2: Write `scripts/create_curated_splits.py` (~2 hrs)
   - Dry-run first; inspect split sizes before writing
   - Run for all 6 experiment languages
4. Phase 6: Write `scripts/analyze_quality_decoding_correlation.py` (~1.5 hrs)
   - This only uses existing lm_ablation JSONs + Phase 1 outputs
   - Can be completed and finalized on Day 1

### Day 2 (GPU work on Hyak)
5. Phase 3: Submit `hyak_scaling_exp.slurm` — monitor job completion (~8 hrs GPU wall time)
6. While waiting: write `scripts/collect_scaling_results.py` and `scripts/plot_scaling_curves.py`

### Day 3 (GPU work + finalization)
7. Phase 4: Submit `hyak_scaling_lm.slurm` as checkpoints complete from Phase 3 (~4 hrs GPU wall time)
8. Phase 5: Run `collect_scaling_results.py` and `plot_scaling_curves.py` once all LM results are in
9. Produce final figures and verify the empirical story holds

---

## Key Implementation Notes

### On KenLM character-level scoring

The existing `_build_arpa_from_text()` in `aft_mms.py` builds a word-level KenLM. For
quality scoring, we need a character-level KenLM. The only difference is input formatting:
- Word-level: pipe raw sentences to lmplz
- Character-level: pipe space-joined character sequences (e.g. `"hello world"` → `"h e l l o   w o r l d"`)

The lmplz call is identical. Check `aft_mms.py`'s `_build_arpa_from_text()` for the exact
subprocess invocation (it uses `--discount_fallback` and `--arpa` flags). Mirror exactly.

### On the split naming convention

`load_and_preprocess_data(lang, split)` in `aft_mms.py` resolves train TSV as:
```python
train_tsv = lang_dir / f"train-{split}_{lang}.tsv"
```
So split names map directly to filenames. The convention used here:
- `"one"` → `train-one_{lang}.tsv` (existing)
- `"mid"` → `train-mid_{lang}.tsv` (existing)
- `"all"` → `train-all_{lang}.tsv` (existing)
- `"2h"` → `train-2h_{lang}.tsv` (new, Phase 2)
- `"3h"` → `train-3h_{lang}.tsv` (new, Phase 2)
- `"curated-1h"` → `train-curated-1h_{lang}.tsv` (new, Phase 2)
- `"curated-all"` → `train-curated-all_{lang}.tsv` (new, Phase 2)
- etc.

No changes to `aft_mms.py` are required.

### On the `quality_tags` column in the raw TSV

The raw `ss-corpus-{lang}.tsv` has a `quality_tags` column (visible in sco data).
Inspect the values for your experiment languages before building the quality scorer —
if Mozilla already flags some utterances, those flags can serve as a sanity check
(not a replacement) for the KenLM perplexity scores. Add a column
`quality_tags_flagged` to the output of `quality_score.py` for diagnostic comparison.

### On Hyak job submission order

Submit Phase 3 (training) jobs first for the smallest-tier languages (kcn, hch) —
they have the fewest training utterances and will finish first, giving you early LM
ablation results to verify the pipeline before the larger jobs complete.
