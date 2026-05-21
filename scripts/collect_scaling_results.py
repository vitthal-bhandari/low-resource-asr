"""
Aggregate greedy-decode and LM-decode results from all scaling experiment runs.

Sources:
  models/mms/{lang}/{split}/eval_results.json
      Written by aft_mms.py after training. Contains lang, split, n_train,
      n_train_hours, greedy val/test WER+CER.
  results/lm_ablation/scaling_exp/mms_{lang}_{split}_lm.json
      Written by run_ctc_lm_ablation.py. Contains ngram_beam[0] test WER+CER.

Output:
  results/scaling_experiment/scaling_results.csv

Usage:
  uv run python scripts/collect_scaling_results.py
  uv run python scripts/collect_scaling_results.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.config import config

MODELS_DIR = config.models_dir / "mms"
LM_JSON_DIR = config.results_dir / "lm_ablation" / "scaling_exp"
OUT_DIR = config.results_dir / "scaling_experiment"
OUT_CSV = OUT_DIR / "scaling_results.csv"

# Splits produced by create_curated_splits.py
HOUR_TARGETS = ["1h", "2h", "3h", "5h", "full"]


def parse_split(split: str) -> tuple[bool, str]:
    """Return (is_curated, hours_label). E.g. 'curated-2h' -> (True, '2h')."""
    if split.startswith("curated-"):
        return True, split[len("curated-"):]
    return False, split


def hours_sort_key(label: str) -> float:
    """Numeric sort key: '1h'->1.0, '2h'->2.0, 'full'->inf."""
    return float("inf") if label == "full" else float(label.rstrip("h"))


def collect_greedy(verbose: bool = False) -> pd.DataFrame:
    """Read every eval_results.json under models/mms/*/*/."""
    rows = []
    for json_path in sorted(MODELS_DIR.glob("*/*/eval_results.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"  [warn] could not read {json_path}: {e}")
            continue

        # Metadata may be absent if job ran before the aft_mms patch — fall back
        # to inferring from the directory structure: models/mms/{lang}/{split}/
        if "lang" not in data:
            data["lang"] = json_path.parts[-3]
        if "split" not in data:
            data["split"] = json_path.parts[-2]

        if verbose:
            print(f"  greedy  {data['lang']:8s}  {data['split']}")
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {
        "eval_wer": "greedy_val_wer",
        "eval_cer": "greedy_val_cer",
        "test_wer": "greedy_test_wer",
        "test_cer": "greedy_test_cer",
    }
    df = df.rename(columns=rename)
    keep = [
        "lang", "split", "n_train", "n_train_hours",
        "greedy_val_wer", "greedy_val_cer",
        "greedy_test_wer", "greedy_test_cer",
    ]
    return df[[c for c in keep if c in df.columns]]


def collect_lm(verbose: bool = False) -> pd.DataFrame:
    """Read every mms_*_*_lm.json under results/lm_ablation/scaling_exp/."""
    rows = []
    for json_path in sorted(LM_JSON_DIR.glob("mms_*_*_lm.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"  [warn] could not read {json_path}: {e}")
            continue

        meta = data.get("meta", {})
        lang = meta.get("lang", "")
        split = meta.get("split", "")

        ngram = data.get("ngram_beam", [])
        if not ngram:
            print(f"  [warn] no ngram_beam entries in {json_path.name}")
            continue

        # We ran with fixed params (order=4, alpha=0.5, beta=1.0, beam=100),
        # so there is exactly one entry. Guard against unexpected multi-entry files
        # by finding the matching entry explicitly.
        entry = next(
            (e for e in ngram
             if e.get("lm_order") == 4
             and e.get("alpha") == 0.5
             and e.get("beta") == 1.0
             and e.get("beam_width") == 100),
            ngram[0],  # fallback to first entry if params differ
        )

        test = entry.get("test", {})
        if not test:
            print(f"  [warn] no test results in {json_path.name}")
            continue

        if verbose:
            print(f"  lm      {lang:8s}  {split}  test_wer={test.get('wer', float('nan')):.4f}")

        rows.append({
            "lang": lang,
            "split": split,
            "lm_test_wer": test.get("wer", float("nan")),
            "lm_test_cer": test.get("cer", float("nan")),
        })

    cols = ["lang", "split", "lm_test_wer", "lm_test_cer"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting greedy results...")
    greedy = collect_greedy(verbose=args.verbose)
    print(f"  {len(greedy)} runs found")

    print("Collecting LM results...")
    lm = collect_lm(verbose=args.verbose)
    print(f"  {len(lm)} runs found")

    if greedy.empty:
        print("No greedy results found — nothing to write.")
        return

    # Left-join: keep all greedy rows; LM may still be running
    df = greedy.merge(lm, on=["lang", "split"], how="left") if not lm.empty else greedy.copy()
    if "lm_test_wer" not in df.columns:
        df["lm_test_wer"] = float("nan")
    if "lm_test_cer" not in df.columns:
        df["lm_test_cer"] = float("nan")

    # Derived columns for plotting convenience
    parsed = df["split"].apply(lambda s: pd.Series(parse_split(s), index=["is_curated", "hours_label"]))
    df = pd.concat([df, parsed], axis=1)
    # Write is_curated as 0/1 so pandas reads it back as int without bool/string ambiguity
    df["is_curated"] = df["is_curated"].astype(int)
    df["hours_sort"] = df["hours_label"].map(hours_sort_key)
    df = df.sort_values(["lang", "is_curated", "hours_sort"]).drop(columns="hours_sort").reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows → {OUT_CSV}")

    # Summary table
    display_cols = ["lang", "split", "n_train_hours", "greedy_test_wer", "lm_test_wer"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
