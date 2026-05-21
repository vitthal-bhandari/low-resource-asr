"""
Exp 3c — Corpus quality vs. LM decoding benefit.

Hypothesis: languages with noisier corpora (lower hard-filter survival rate)
benefit more from n-gram LM decoding because the acoustic model is less
reliable and the language model has more room to correct errors.

Quality proxy:
  hard_filter_survival_rate  — fraction of training utterances that survive
  duration and repetition hard filters (higher = cleaner corpus).

LM benefit (per language, full-data raw split):
  abs_lm_improvement  = greedy_test_wer − lm_test_wer
  rel_lm_improvement  = (greedy_test_wer − lm_test_wer) / greedy_test_wer

Sources:
  results/lm_ablation/arpa_artifacts/quality/quality_summary.csv
  results/scaling_experiment/scaling_results.csv   (run collect_scaling_results.py first)

Writes:
  results/scaling_experiment/quality_decoding_correlation.csv
  results/scaling_experiment/quality_decoding_correlation.png

Usage:
  uv run python scripts/analyze_quality_decoding_correlation.py
  uv run python scripts/analyze_quality_decoding_correlation.py --split full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("[warn] scipy not available — Pearson r computed via numpy, no p-value.")

from src.config import config
from src.data.download import LANGUAGES

QUALITY_CSV = (
    config.results_dir / "lm_ablation" / "arpa_artifacts" / "quality" / "quality_summary.csv"
)
SCALING_CSV = config.results_dir / "scaling_experiment" / "scaling_results.csv"
OUT_DIR = config.results_dir / "scaling_experiment"


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (r, p). p is nan if scipy is unavailable."""
    if _SCIPY and len(x) >= 3:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan")


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    r: float,
    p: float,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    ax.scatter(x, y, color="#2c7bb6", s=80, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(
            row["lang"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
        )

    # Regression line (only meaningful with ≥ 3 points)
    if len(df) >= 3:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, "--", color="#d7191c", linewidth=1.5,
                label=f"r = {r:.3f}" + (f"  p = {p:.3f}" if not np.isnan(p) else ""))
        ax.legend(fontsize=8, framealpha=0.8)

    n_str = f"n = {len(df)}"
    p_str = f"p = {p:.3f}" if not np.isnan(p) else "p = n/a"
    ax.set_title(f"Pearson r = {r:.3f}  ({p_str}, {n_str})", fontsize=9)
    ax.set_xlabel("Hard-filter survival rate  (quality proxy)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="full",
        help="Which training split to use as the representative WER point (default: full)",
    )
    args = parser.parse_args()

    for path in (QUALITY_CSV, SCALING_CSV):
        if not path.exists():
            print(f"ERROR: {path} not found.")
            if path == SCALING_CSV:
                print("  → Run collect_scaling_results.py first.")
            sys.exit(1)

    quality = pd.read_csv(QUALITY_CSV)
    scaling = pd.read_csv(SCALING_CSV)
    # is_curated is written as 0/1 by collect_scaling_results.py
    scaling["is_curated"] = scaling["is_curated"].astype(bool)

    # Representative rows: requested split, raw (not curated)
    rep = scaling[(scaling["split"] == args.split) & (~scaling["is_curated"])].copy()
    if rep.empty:
        print(f"ERROR: No raw '{args.split}' rows found in {SCALING_CSV}.")
        print(f"  Available splits: {sorted(scaling['split'].unique())}")
        sys.exit(1)

    # Merge with quality metadata
    quality_cols = ["lang", "n_train", "n_survivors", "n_quality_keep",
                    "hard_filter_survival_rate", "mean_ppl_keep", "mean_ppl_drop"]
    quality_cols = [c for c in quality_cols if c in quality.columns]
    df = quality[quality_cols].merge(
        rep[["lang", "n_train_hours", "greedy_test_wer", "lm_test_wer"]],
        on="lang",
    ).dropna(subset=["greedy_test_wer", "lm_test_wer"])

    if len(df) < 2:
        print(f"Only {len(df)} language(s) have both quality scores and LM results. "
              "Cannot compute correlation.")
        sys.exit(1)

    # Compute improvement metrics
    df["abs_lm_improvement"] = df["greedy_test_wer"] - df["lm_test_wer"]
    df["rel_lm_improvement"] = df["abs_lm_improvement"] / df["greedy_test_wer"]

    # Correlations
    x = df["hard_filter_survival_rate"].to_numpy(dtype=float)
    r_rel, p_rel = pearson(x, df["rel_lm_improvement"].to_numpy(dtype=float))
    r_abs, p_abs = pearson(x, df["abs_lm_improvement"].to_numpy(dtype=float))

    print(f"\nCorrelation results (split={args.split}, n={len(df)}):")
    print(f"  Relative LM improvement vs. survival rate:  r = {r_rel:.3f}  p = {p_rel:.3f}")
    print(f"  Absolute LM improvement vs. survival rate:  r = {r_abs:.3f}  p = {p_abs:.3f}")
    print()
    display_cols = ["lang", "hard_filter_survival_rate", "greedy_test_wer",
                    "lm_test_wer", "rel_lm_improvement", "abs_lm_improvement"]
    print(df[display_cols].sort_values("rel_lm_improvement", ascending=False)
          .to_string(index=False, float_format="{:.4f}".format))

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Corpus Quality vs. LM Decoding Benefit\n"
        f"(MMS, {args.split} split, raw training data)",
        fontsize=11,
    )

    plot_panel(
        axes[0], df,
        x_col="hard_filter_survival_rate",
        y_col="rel_lm_improvement",
        y_label="Relative WER reduction  (LM − greedy) / greedy",
        r=r_rel, p=p_rel,
    )
    plot_panel(
        axes[1], df,
        x_col="hard_filter_survival_rate",
        y_col="abs_lm_improvement",
        y_label="Absolute WER reduction  greedy − LM",
        r=r_abs, p=p_abs,
    )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "quality_decoding_correlation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.close(fig)

    # ── Output CSV ────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "quality_decoding_correlation.csv"
    save_cols = [c for c in [
        "lang", "hard_filter_survival_rate", "n_train", "n_survivors",
        "n_quality_keep", "mean_ppl_keep", "mean_ppl_drop",
        "n_train_hours", "greedy_test_wer", "lm_test_wer",
        "abs_lm_improvement", "rel_lm_improvement",
    ] if c in df.columns]
    df[save_cols].sort_values("rel_lm_improvement", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
