"""
Compute and save utterance-level quality scores for training data.

Reads data/mozilla_speech_data/{lang}/ss-corpus-{lang}.tsv, scores each
training utterance, and writes:
  results/lm_ablation/arpa_artifacts/quality/{lang}/quality_scores_{lang}.tsv
  results/lm_ablation/arpa_artifacts/quality/{lang}/char_{order}gram.arpa
  results/lm_ablation/arpa_artifacts/quality/quality_summary.csv  (appended)

Usage:
  uv run python scripts/quality_score.py --all
  uv run python scripts/quality_score.py sco kcn el-CY
  uv run python scripts/quality_score.py sco --keep-fraction 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.config import config
from src.data.download import LANGUAGES
from src.data.quality import compute_quality_scores

QUALITY_DIR = config.results_dir / "lm_ablation" / "arpa_artifacts" / "quality"
SUMMARY_PATH = QUALITY_DIR / "quality_summary.csv"

SUMMARY_FIELDS = [
    "lang", "n_train", "n_duration_filtered", "n_rep_filtered",
    "n_survivors", "n_quality_keep", "keep_fraction",
    "mean_ppl_keep", "mean_ppl_drop", "hard_filter_survival_rate",
    "kenlm_arpa_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score training utterance quality per language.")
    p.add_argument("langs", nargs="*", help="Language codes (omit with --all).")
    p.add_argument("--all", action="store_true", help="Run for all 21 languages.")
    p.add_argument("--keep-fraction", type=float, default=0.60)
    p.add_argument("--lm-order", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Print stats only; write nothing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    langs = sorted(LANGUAGES) if args.all else args.langs
    if not langs:
        print("Provide language codes or --all."); sys.exit(1)

    summary_rows: list[dict] = []

    for lang in langs:
        corpus_path = config.mozilla_data_dir / lang / f"ss-corpus-{lang}.tsv"
        if not corpus_path.exists():
            print(f"[{lang}] SKIP — corpus TSV not found"); continue

        df = pd.read_csv(corpus_path, sep="\t")
        train = df[
            (df["split"] == "train")
            & (df["duration_ms"] > 0)
            & df["transcription"].notna()
            & (df["transcription"].str.strip() != "")
        ].copy()

        if train.empty:
            print(f"[{lang}] SKIP — no training rows"); continue

        out_dir = QUALITY_DIR / lang
        scored = compute_quality_scores(
            train, out_dir, lm_order=args.lm_order, keep_fraction=args.keep_fraction
        )

        n_dur = int(scored["duration_filtered"].sum())
        n_rep = int(scored["rep_filtered"].sum())
        n_surv = len(scored) - n_dur - n_rep
        n_keep = int(scored["quality_keep"].sum())
        keep_mask = scored["quality_keep"]
        drop_mask = ~keep_mask & ~scored["duration_filtered"] & ~scored["rep_filtered"]
        ppl_keep = scored.loc[keep_mask, "ppl_score"].dropna()
        ppl_drop = scored.loc[drop_mask, "ppl_score"].dropna()

        print(
            f"[{lang}] {len(train)} train → {n_dur} dur-filtered, {n_rep} rep-filtered"
            f" → {n_surv} survivors → {n_keep} quality_keep"
            f" ({100 * n_keep / len(train):.1f}%)"
        )
        if not ppl_keep.empty:
            print(
                f"       ppl  keep={ppl_keep.mean():.1f}  "
                f"drop={ppl_drop.mean():.1f}" if not ppl_drop.empty else
                f"       ppl  keep={ppl_keep.mean():.1f}"
            )

        summary_rows.append({
            "lang": lang,
            "n_train": len(train),
            "n_duration_filtered": n_dur,
            "n_rep_filtered": n_rep,
            "n_survivors": n_surv,
            "n_quality_keep": n_keep,
            "keep_fraction": args.keep_fraction,
            "mean_ppl_keep": round(ppl_keep.mean(), 2) if not ppl_keep.empty else None,
            "mean_ppl_drop": round(ppl_drop.mean(), 2) if not ppl_drop.empty else None,
            "hard_filter_survival_rate": round(n_surv / len(train), 4),
            "kenlm_arpa_path": str(out_dir / f"char_{args.lm_order}gram.arpa"),
        })

        if not args.dry_run:
            out_tsv = out_dir / f"quality_scores_{lang}.tsv"
            scored.to_csv(out_tsv, sep="\t", index=False)
            print(f"       → {out_tsv}")

    if args.dry_run or not summary_rows:
        return

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()
    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
