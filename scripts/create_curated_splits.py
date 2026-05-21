"""
Create curated and raw hour-based training splits for the scaling experiment.

Raw splits are carved from all valid training rows in ss-corpus-{lang}.tsv.
Curated splits are carved from quality_keep==True rows produced by quality_score.py.

Output TSVs written to data/mozilla_speech_data/{lang}/:
  train-1h_{lang}.tsv
  train-2h_{lang}.tsv
  train-3h_{lang}.tsv           (if raw pool >= 2.55h)
  train-5h_{lang}.tsv           (if raw pool >= 4.25h)
  train-full_{lang}.tsv
  train-curated-1h_{lang}.tsv
  train-curated-2h_{lang}.tsv
  train-curated-3h_{lang}.tsv   (if curated pool >= 2.55h)
  train-curated-5h_{lang}.tsv   (if curated pool >= 4.25h)
  train-curated-full_{lang}.tsv

Also writes results/splits/scaling_jobs.txt with one "lang:split" per line
for use as the SLURM training array input.

Usage:
  uv run python scripts/create_curated_splits.py --langs sco kcn el-CY aln bew hch
  uv run python scripts/create_curated_splits.py --all
  uv run python scripts/create_curated_splits.py --langs sco --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.config import config
from src.data.download import LANGUAGES

QUALITY_DIR = config.results_dir / "lm_ablation" / "arpa_artifacts" / "quality"
JOBS_FILE = config.results_dir / "splits" / "scaling_jobs.txt"
REQUIRED_COLS = ["audio_file", "transcription", "duration_ms", "split"]

MS_PER_HOUR = 3_600_000
HOUR_TARGETS = [1, 2, 3, 5]
# Require at least 85% of target hours in pool before creating a split.
MIN_POOL_FRAC = 0.85


def split_by_duration(df: pd.DataFrame, target_ms: int) -> pd.DataFrame:
    """Return first target_ms worth of audio, sorted deterministically by audio_file."""
    df = df.sort_values("audio_file").reset_index(drop=True)
    return df[df["duration_ms"].cumsum() <= target_ms].copy()


def load_raw_train(lang: str) -> pd.DataFrame | None:
    """Load valid training rows from ss-corpus TSV."""
    path = config.mozilla_data_dir / lang / f"ss-corpus-{lang}.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    df = df[
        (df["split"] == "train")
        & (df["duration_ms"] > 0)
        & df["transcription"].notna()
        & (df["transcription"].str.strip() != "")
    ].copy()
    return df if not df.empty else None


def load_curated_train(lang: str) -> pd.DataFrame | None:
    """Load quality_keep==True rows from quality scoring output."""
    path = QUALITY_DIR / lang / f"quality_scores_{lang}.tsv"
    if not path.exists():
        print(f"  [curated] quality scores not found: {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    curated = df[df["quality_keep"] == True].copy()
    return curated if not curated.empty else None


def create_splits_for_lang(lang: str, dry_run: bool = False) -> list[str]:
    """Create all splits for one language. Returns list of 'lang:split' job strings."""
    raw = load_raw_train(lang)
    curated = load_curated_train(lang)

    if raw is None:
        print(f"[{lang}] SKIP — no raw training data"); return []

    raw_h = raw["duration_ms"].sum() / MS_PER_HOUR
    curated_h = curated["duration_ms"].sum() / MS_PER_HOUR if curated is not None else 0.0
    print(f"[{lang}] raw={raw_h:.2f}h  curated={curated_h:.2f}h")

    lang_dir = config.mozilla_data_dir / lang
    jobs: list[str] = []

    def write_split(df: pd.DataFrame, split_name: str) -> None:
        out = lang_dir / f"train-{split_name}_{lang}.tsv"
        if not dry_run:
            df[REQUIRED_COLS].to_csv(out, sep="\t", index=False)
        jobs.append(f"{lang}:{split_name}")

    # --- Raw splits ---
    for hours in HOUR_TARGETS:
        if raw_h < hours * MIN_POOL_FRAC:
            print(f"  raw-{hours}h: SKIP (pool={raw_h:.2f}h)")
            continue
        sub = split_by_duration(raw, hours * MS_PER_HOUR)
        print(f"  raw-{hours}h: {len(sub)} utt ({sub['duration_ms'].sum()/MS_PER_HOUR:.2f}h)")
        write_split(sub, f"{hours}h")

    # full raw
    print(f"  raw-full: {len(raw)} utt ({raw_h:.2f}h)")
    write_split(raw.sort_values("audio_file").reset_index(drop=True), "full")

    # --- Curated splits ---
    if curated is None:
        print(f"  curated: SKIP — quality scores missing"); return jobs

    for hours in HOUR_TARGETS:
        if curated_h < hours * MIN_POOL_FRAC:
            print(f"  curated-{hours}h: SKIP (pool={curated_h:.2f}h)")
            continue
        sub = split_by_duration(curated, hours * MS_PER_HOUR)
        print(f"  curated-{hours}h: {len(sub)} utt ({sub['duration_ms'].sum()/MS_PER_HOUR:.2f}h)")
        write_split(sub, f"curated-{hours}h")

    # full curated
    print(f"  curated-full: {len(curated)} utt ({curated_h:.2f}h)")
    write_split(curated.sort_values("audio_file").reset_index(drop=True), "curated-full")

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    langs = sorted(LANGUAGES) if args.all else (args.langs or [])
    if not langs:
        parser.error("Provide --langs or --all.")

    all_jobs: list[str] = []
    for lang in langs:
        all_jobs.extend(create_splits_for_lang(lang, dry_run=args.dry_run))

    print(f"\nTotal jobs: {len(all_jobs)}")

    if not args.dry_run:
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(JOBS_FILE, "w") as f:
            f.write("\n".join(all_jobs) + "\n")
        print(f"Job list → {JOBS_FILE}")


if __name__ == "__main__":
    main()
