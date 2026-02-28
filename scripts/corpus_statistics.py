"""
Compute and print corpus duration statistics for all 21 languages.

Reads data/mozilla_speech_data/{lang}/ss-corpus-{lang}.tsv for each language,
filters to rows with duration_ms > 0, and reports:
  median, mean, p95 (95th percentile), p97.5 (97.5th percentile), max
all in seconds. Output is a single table suitable for terminal/command line.

Usage:
  uv run python scripts/corpus_stats_all_langs.py
  uv run python scripts/corpus_stats_all_langs.py --csv   # machine-readable
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.config import config
from src.data.download import LANGUAGES


def load_corpus_duration_sec(lang: str) -> pd.Series | None:
    """Load ss-corpus-{lang}.tsv and return duration in seconds (only duration_ms > 0). None if missing."""
    path = config.mozilla_data_dir / lang / f"ss-corpus-{lang}.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if "duration_ms" not in df.columns:
        return None
    df = df[df["duration_ms"] > 0]
    if df.empty:
        return None
    return (df["duration_ms"] / 1000.0)


def corpus_stats(sec: pd.Series) -> dict:
    """Return median, mean, p95, p97.5, max in seconds (rounded to 2 decimals)."""
    if len(sec) == 0:
        return {"median_sec": None, "mean_sec": None, "p95_sec": None, "p97_5_sec": None, "max_sec": None}
    return {
        "median_sec": round(sec.median(), 2),
        "mean_sec": round(sec.mean(), 2),
        "p95_sec": round(sec.quantile(0.95), 2),
        "p97_5_sec": round(sec.quantile(0.975), 2),
        "max_sec": round(sec.max(), 2),
    }


def format_float(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}"


def print_table(rows: list[dict], columns: list[tuple[str, str]], sep: str = "  ") -> None:
    """Print a fixed-width aligned table. columns = [(key, header), ...]."""
    if not rows:
        return
    # Compute column widths: header and max content width
    widths = []
    for key, header in columns:
        content_len = max(len(str(r.get(key, ""))) for r in rows)
        widths.append(max(len(header), content_len))
    # Header
    header_parts = [header.ljust(w) for (_, header), w in zip(columns, widths)]
    print(sep.join(header_parts))
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    for r in rows:
        parts = [str(r.get(key, "")).ljust(w) for (key, _), w in zip(columns, widths)]
        print(sep.join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus duration statistics for all 21 languages (median, mean, p95, p97.5, max)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output CSV instead of a formatted table (lang,name,n,median_sec,mean_sec,p95_sec,p97_5_sec,max_sec).",
    )
    args = parser.parse_args()

    results = []
    for lang in sorted(LANGUAGES.keys()):
        name = LANGUAGES[lang]
        sec = load_corpus_duration_sec(lang)
        if sec is None:
            results.append({
                "lang": lang,
                "name": name,
                "n": 0,
                "median_sec": None,
                "mean_sec": None,
                "p95_sec": None,
                "p97_5_sec": None,
                "max_sec": None,
            })
            continue
        n = len(sec)
        st = corpus_stats(sec)
        results.append({
            "lang": lang,
            "name": name,
            "n": n,
            "median_sec": st["median_sec"],
            "mean_sec": st["mean_sec"],
            "p95_sec": st["p95_sec"],
            "p97_5_sec": st["p97_5_sec"],
            "max_sec": st["max_sec"],
        })

    if args.csv:
        # Machine-readable: one header row, then one row per language
        cols = ["lang", "name", "n", "median_sec", "mean_sec", "p95_sec", "p97_5_sec", "max_sec"]
        print(",".join(cols))
        for r in results:
            print(",".join(str(r[k]) for k in cols))
        return

    # Terminal table: use short headers so it fits
    columns = [
        ("lang", "lang"),
        ("name", "name"),
        ("n", "n"),
        ("median_sec", "median_s"),
        ("mean_sec", "mean_s"),
        ("p95_sec", "p95_s"),
        ("p97_5_sec", "p97.5_s"),
        ("max_sec", "max_s"),
    ]
    # Format floats for display
    for r in results:
        for k in ["median_sec", "mean_sec", "p95_sec", "p97_5_sec", "max_sec"]:
            r[k] = format_float(r[k]) if r.get("n", 0) else "—"
    print_table(results, columns)


if __name__ == "__main__":
    main()
