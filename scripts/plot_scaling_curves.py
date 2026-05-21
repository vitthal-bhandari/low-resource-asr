"""
Plot WER vs. training hours for the scaling experiment.

Reads:  results/scaling_experiment/scaling_results.csv
Output: results/scaling_experiment/scaling_curves.png
        results/scaling_experiment/scaling_curves.pdf

Each panel is one language. Four lines per panel:
  Raw + Greedy     (solid blue)
  Raw + LM         (dashed blue)
  Curated + Greedy (solid orange)
  Curated + LM     (dashed orange)

X-axis: actual training hours (n_train_hours from eval_results.json).
Y-axis: test WER (%).

Usage:
  uv run python scripts/plot_scaling_curves.py
  uv run python scripts/plot_scaling_curves.py --langs sco kcn aln
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")  # no display needed on Hyak or Mac without GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import config
from src.data.download import LANGUAGES  # {lang_code: full_name}

IN_CSV = config.results_dir / "scaling_experiment" / "scaling_results.csv"
OUT_DIR = config.results_dir / "scaling_experiment"

# (is_curated, use_lm) → (legend label, hex color, linestyle, marker)
LINE_STYLES: dict[tuple[bool, bool], tuple[str, str, str, str]] = {
    (False, False): ("Raw + Greedy",    "#1f77b4", "-",  "o"),
    (False, True):  ("Raw + LM",        "#1f77b4", "--", "s"),
    (True,  False): ("Curated + Greedy","#ff7f0e", "-",  "o"),
    (True,  True):  ("Curated + LM",   "#ff7f0e", "--", "s"),
}

NCOLS = 3


def plot_language(ax: plt.Axes, lang_df: pd.DataFrame, lang: str,
                  legend_handles: list, legend_seen: set) -> None:
    """Draw all four lines for one language panel."""
    for (is_curated, use_lm), (label, color, ls, marker) in LINE_STYLES.items():
        subset = lang_df[lang_df["is_curated"] == is_curated].copy()
        if subset.empty:
            continue

        wer_col = "lm_test_wer" if use_lm else "greedy_test_wer"
        if wer_col not in subset.columns:
            continue

        subset = subset.sort_values("n_train_hours")
        x = subset["n_train_hours"].to_numpy(dtype=float)
        y = subset[wer_col].to_numpy(dtype=float) * 100  # → percentage

        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() == 0:
            continue

        line, = ax.plot(
            x[valid], y[valid],
            color=color, linestyle=ls, marker=marker,
            markersize=5, linewidth=1.8, label=label,
        )
        if label not in legend_seen:
            legend_handles.append(line)
            legend_seen.add(label)

    lang_name = LANGUAGES.get(lang, lang)
    ax.set_title(f"{lang}  —  {lang_name}", fontsize=9)
    ax.set_xlabel("Training hours", fontsize=8)
    ax.set_ylabel("Test WER (%)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Subset of language codes to plot (default: all in CSV)")
    args = parser.parse_args()

    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run collect_scaling_results.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    # is_curated is written as 0/1 by collect_scaling_results.py
    df["is_curated"] = df["is_curated"].astype(bool)

    langs = args.langs if args.langs else sorted(df["lang"].unique())
    missing = [l for l in langs if l not in df["lang"].values]
    if missing:
        print(f"[warn] These langs are not in the CSV and will be skipped: {missing}")
        langs = [l for l in langs if l not in missing]
    if not langs:
        print("No languages to plot.")
        sys.exit(1)

    n = len(langs)
    nrows = (n + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(5.2 * NCOLS, 4.0 * nrows),
        squeeze=False,
    )
    fig.suptitle("WER vs. Training Data (MMS Scaling Experiment)", fontsize=13, y=1.01)

    legend_handles: list = []
    legend_seen: set = set()

    for idx, lang in enumerate(langs):
        ax = axes[idx // NCOLS][idx % NCOLS]
        lang_df = df[df["lang"] == lang]
        plot_language(ax, lang_df, lang, legend_handles, legend_seen)

    # Hide unused subplot panels
    for idx in range(n, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    # Shared legend anchored below the figure
    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.03),
            fontsize=8,
            framealpha=0.9,
        )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"scaling_curves.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
