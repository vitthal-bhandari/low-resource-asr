"""
Sum duration_ms for selected languages/splits using filename lists.

By default, this script computes totals for:
  - languages: sco (Scots), kcn (Nubi)
  - splits: one, mid

It uses:
  - filename lists from results/splits/{lang}_train_{split}_filenames.txt
  - corpus metadata from data/mozilla_speech_data/{lang}/ss-corpus-{lang}.tsv

Usage:
  uv run python scripts/sum_split_duration_ms.py
  uv run python scripts/sum_split_duration_ms.py --langs sco kcn --splits one mid
  uv run python scripts/sum_split_duration_ms.py --results-splits-dir /path/to/results/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ensure project root is on path when run as script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import config


def _read_filenames(path: Path) -> list[str]:
    """Read one filename per line from a text file."""
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _sum_duration_ms_for_split(lang: str, split: str, results_splits_dir: Path) -> tuple[int, int]:
    """
    Return (num_utterances, total_duration_ms) for a train split.

    Raises:
        FileNotFoundError: if required files are missing.
        ValueError: if expected columns are missing.
    """
    filenames_path = results_splits_dir / f"{lang}_train_{split}_filenames.txt"
    corpus_path = config.mozilla_data_dir / lang / f"ss-corpus-{lang}.tsv"

    if not filenames_path.exists():
        raise FileNotFoundError(f"Missing filename list: {filenames_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus TSV: {corpus_path}")

    filenames = _read_filenames(filenames_path)
    if not filenames:
        return 0, 0

    df = pd.read_csv(corpus_path, sep="\t")
    required_cols = {"audio_file", "duration_ms"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{corpus_path} missing columns: {sorted(missing_cols)}")

    # Keep only rows in the selected filename list.
    subset = df[df["audio_file"].isin(filenames)]
    total_ms = int(subset["duration_ms"].sum())
    return len(filenames), total_ms


def _format_hours(ms: int) -> str:
    """Format milliseconds as hours with 3 decimals."""
    return f"{ms / (1000 * 60 * 60):.3f}h"


def _iter_pairs(langs: Iterable[str], splits: Iterable[str]) -> Iterable[tuple[str, str]]:
    for lang in langs:
        for split in splits:
            yield lang, split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum duration_ms for filename-list train splits."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["sco", "kcn"],
        help="Language codes to process (default: sco kcn).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["one", "mid"],
        help="Train splits to process (default: one mid).",
    )
    parser.add_argument(
        "--results-splits-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "splits",
        help="Directory containing *_filenames.txt split lists.",
    )
    args = parser.parse_args()

    print("lang\tsplit\tn_utterances\ttotal_duration_ms\ttotal_duration_h")
    for lang, split in _iter_pairs(args.langs, args.splits):
        n, total_ms = _sum_duration_ms_for_split(lang, split, args.results_splits_dir)
        print(f"{lang}\t{split}\t{n}\t{total_ms}\t{_format_hours(total_ms)}")


if __name__ == "__main__":
    main()

