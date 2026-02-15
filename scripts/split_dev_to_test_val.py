"""
Split the current validation set (dev) into a fixed 45-minute test set and a new validation set.

The resulting TSV files contain only filenames and metadata (no audio). These can be
shared on GitHub; users obtain the actual audio from the official Mozilla download.

Usage:
    uv run python scripts/split_dev_to_test_val.py aln
    uv run python scripts/split_dev_to_test_val.py --all

Output (per language):
    data/mozilla_speech_data/<lang>/validation_<lang>.tsv   # New validation (V - 45 min)
    data/mozilla_speech_data/<lang>/test_<lang>.tsv         # New test (first 45 min of dev)

Training data (train split) is unchanged; it remains in ss-corpus-<lang>.tsv.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import config
from src.data.download import LANGUAGES

# 45 minutes in milliseconds
TEST_DURATION_MS = 45 * 60 * 1000


def split_dev_to_test_val(lang: str, dry_run: bool = False) -> tuple[int, int, int]:
    """
    Split dev set into test (first 45 min) and new validation (remainder).

    Reads ss-corpus-{lang}.tsv, filters split=="dev", sorts by audio_file (deterministic),
    then assigns rows by cumulative duration.

    Args:
        lang: Language ISO code (e.g. 'aln').
        dry_run: If True, only print what would be done.

    Returns:
        (n_test, n_val, total_dev) sample counts.
    """
    lang_dir = config.mozilla_data_dir / lang
    corpus_path = lang_dir / f"ss-corpus-{lang}.tsv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    df = pd.read_csv(corpus_path, sep="\t")
    required = ["audio_file", "transcription", "duration_ms", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {corpus_path}: {missing}")

    dev = df[df["split"] == "dev"].copy()
    if dev.empty:
        raise ValueError(f"No 'dev' split in {corpus_path}")

    dev = dev.sort_values("audio_file").reset_index(drop=True)
    cumsum = dev["duration_ms"].cumsum()
    # First 45 min -> test; rest -> new validation
    test_mask = cumsum <= TEST_DURATION_MS
    test_df = dev[test_mask].copy()
    val_df = dev[~test_mask].copy()
    test_df["split"] = "test"
    val_df["split"] = "dev"

    n_test, n_val = len(test_df), len(val_df)
    total_dev = len(dev)
    test_min = test_df["duration_ms"].sum() / (60 * 1000)
    val_min = val_df["duration_ms"].sum() / (60 * 1000)

    if dry_run:
        print(f"  [dry run] {lang}: dev={total_dev} -> test={n_test} ({test_min:.1f} min), val={n_val} ({val_min:.1f} min)")
        return n_test, n_val, total_dev

    out_test = lang_dir / f"test_{lang}.tsv"
    out_val = lang_dir / f"validation_{lang}.tsv"
    # Keep same columns as original for compatibility
    cols = [c for c in required if c in test_df.columns]
    test_df[cols].to_csv(out_test, sep="\t", index=False)
    val_df[cols].to_csv(out_val, sep="\t", index=False)
    print(f"  {lang}: test -> {out_test} ({n_test} samples, {test_min:.1f} min)")
    print(f"  {lang}: validation -> {out_val} ({n_val} samples, {val_min:.1f} min)")
    return n_test, n_val, total_dev


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dev set into 45-min test and new validation; save filenames to TSV."
    )
    parser.add_argument(
        "lang",
        nargs="?",
        type=str,
        help="Language code (e.g. aln). Omit if using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all 21 languages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be written.",
    )
    args = parser.parse_args()

    if args.all:
        langs = sorted(LANGUAGES.keys())
    elif args.lang and args.lang in LANGUAGES:
        langs = [args.lang]
    else:
        parser.error("Provide a language code or --all. Available: " + ", ".join(sorted(LANGUAGES.keys())))

    print("Splitting dev -> test (45 min) + validation (remainder)")
    print(f"Output: <lang_dir>/test_<lang>.tsv, <lang_dir>/validation_<lang>.tsv")
    print("")

    for lang in langs:
        try:
            split_dev_to_test_val(lang, dry_run=args.dry_run)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {lang}: SKIP - {e}")
    print("Done.")


if __name__ == "__main__":
    main()
