"""
Create deterministic train/validation/test splits and save filenames for reproduction.

1. Dev → test (first 45 min) + validation (remainder). Writes test_{lang}.tsv,
   validation_{lang}.tsv and results/splits/{lang}_test_filenames.txt,
   results/splits/{lang}_validation_filenames.txt.

2. Train → one (1h), mid (tier-dependent: 3h or 5h), all. Tiered by total train hours:
   - Small (< 7h): mid = 3h (1h, 3h, all)
   - Medium (7–10h) and Large (>= 10h): mid = 5h (1h, 5h, all). Large has no 10h checkpoint.
   Writes train-one_{lang}.tsv, train-mid_{lang}.tsv, train-all_{lang}.tsv and
   results/splits/{lang}_train_{one|mid|all}_filenames.txt.

Splits are deterministic (sort by audio_file). Run once; filenames do not change across runs.

Usage:
    uv run python scripts/create_splits.py --all
    uv run python scripts/create_splits.py aln --dry-run
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as script (e.g. python scripts/create_splits.py)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.config import config
from src.data.download import LANGUAGES

# 45 minutes in milliseconds for test set
TEST_DURATION_MS = 45 * 60 * 1000

# 1h, 3h, 5h, 10h in ms
H1_MS = 60 * 60 * 1000
H3_MS = 3 * H1_MS
H5_MS = 5 * H1_MS
H10_MS = 10 * H1_MS

# Tier: Small < 7h -> mid 3h; Medium 7-10h -> mid 5h; Large >= 10h -> mid 5h (same as medium, no 10h)
TIER_SMALL_MAX_H = 7
TIER_MEDIUM_MAX_H = 10


def get_tier_mid_hours(total_train_hours: float) -> int:
    """Return mid split duration in hours: Small=3h, Medium/Large=5h. Large uses 1h, 5h, all (no 10h)."""
    if total_train_hours < TIER_SMALL_MAX_H:
        return 3
    return 5


def split_by_duration(df: pd.DataFrame, duration_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df (sorted by audio_file) into first duration_ms and remainder. Returns (first, rest)."""
    df = df.sort_values("audio_file").reset_index(drop=True)
    cumsum = df["duration_ms"].cumsum()
    mask = cumsum <= duration_ms
    return df[mask].copy(), df[~mask].copy()


def write_tsv_and_filenames(
    df: pd.DataFrame,
    tsv_path: Path,
    filenames_path: Path,
    required_cols: list[str],
) -> None:
    """Write TSV and a one-filename-per-line file for reproduction."""
    cols = [c for c in required_cols if c in df.columns]
    df[cols].to_csv(tsv_path, sep="\t", index=False)
    filenames_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filenames_path, "w") as f:
        for name in df["audio_file"].tolist():
            f.write(name + "\n")


def create_splits_for_lang(lang: str, dry_run: bool = False) -> None:
    """Create dev→test/val and train one/mid/all splits for one language."""
    lang_dir = config.mozilla_data_dir / lang
    corpus_path = lang_dir / f"ss-corpus-{lang}.tsv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    df = pd.read_csv(corpus_path, sep="\t")
    required = ["audio_file", "transcription", "duration_ms", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["duration_ms"] > 0]
    df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]

    splits_dir = config.results_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dev → test (45 min) + validation ----
    dev = df[df["split"] == "dev"].copy()
    if dev.empty:
        raise ValueError(f"No 'dev' split for {lang}")
    test_df, val_df = split_by_duration(dev, TEST_DURATION_MS)
    test_df["split"] = "test"
    val_df["split"] = "dev"

    if not dry_run:
        write_tsv_and_filenames(
            test_df,
            lang_dir / f"test_{lang}.tsv",
            splits_dir / f"{lang}_test_filenames.txt",
            required,
        )
        write_tsv_and_filenames(
            val_df,
            lang_dir / f"validation_{lang}.tsv",
            splits_dir / f"{lang}_validation_filenames.txt",
            required,
        )
    test_min = test_df["duration_ms"].sum() / (60 * 1000)
    val_min = val_df["duration_ms"].sum() / (60 * 1000)
    print(f"  {lang}: test={len(test_df)} ({test_min:.1f} min), val={len(val_df)} ({val_min:.1f} min)")

    # ---- Train → one / mid / all ----
    train_full = df[df["split"] == "train"].copy()
    if train_full.empty:
        raise ValueError(f"No 'train' split for {lang}")
    train_full = train_full.sort_values("audio_file").reset_index(drop=True)
    total_train_ms = train_full["duration_ms"].sum()
    total_train_h = total_train_ms / (60 * 60 * 1000)
    mid_hours = get_tier_mid_hours(total_train_h)
    mid_ms = mid_hours * H1_MS

    train_one_df, _ = split_by_duration(train_full, H1_MS)
    train_mid_df, _ = split_by_duration(train_full, mid_ms)
    train_all_df = train_full

    one_h = train_one_df["duration_ms"].sum() / (60 * 60 * 1000)
    mid_h = train_mid_df["duration_ms"].sum() / (60 * 60 * 1000)
    if not dry_run:
        for name, sub_df in [
            ("one", train_one_df),
            ("mid", train_mid_df),
            ("all", train_all_df),
        ]:
            tsv_path = lang_dir / f"train-{name}_{lang}.tsv"
            fn_path = splits_dir / f"{lang}_train_{name}_filenames.txt"
            write_tsv_and_filenames(sub_df, tsv_path, fn_path, required)
    print(f"  {lang}: train one={len(train_one_df)} ({one_h:.2f}h), mid={len(train_mid_df)} ({mid_h:.2f}h), all={len(train_all_df)} ({total_train_h:.2f}h) tier_mid={mid_hours}h")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits and filename lists.")
    parser.add_argument("lang", nargs="?", type=str, help="Language code. Omit with --all.")
    parser.add_argument("--all", action="store_true", help="Run for all 21 languages.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be written.")
    args = parser.parse_args()

    if args.all:
        langs = sorted(LANGUAGES.keys())
    elif args.lang and args.lang in LANGUAGES:
        langs = [args.lang]
    else:
        parser.error("Provide lang or --all. Available: " + ", ".join(sorted(LANGUAGES.keys())))

    print("Creating splits (dev→test+val; train→one/mid/all by tier)")
    print(f"Filename lists: {config.results_dir / 'splits'}")
    print("")
    for lang in langs:
        try:
            create_splits_for_lang(lang, dry_run=args.dry_run)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {lang}: SKIP - {e}")
    print("Done.")


if __name__ == "__main__":
    main()
