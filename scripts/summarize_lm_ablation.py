#!/usr/bin/env python3
"""
Summarize decode-only CTC LM ablation JSON outputs into compact CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LM ablation JSON files to CSV.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/lm_ablation",
        help="Directory containing *_lm_ablation.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/lm_ablation",
        help="Directory to write summary CSV files.",
    )
    parser.add_argument(
        "--metric-split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Primary split used for best-entry selection when available.",
    )
    return parser.parse_args()


def _safe_get(dct: dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _best_entry(entries: list[dict[str, Any]], split: str) -> dict[str, Any] | None:
    candidates = [e for e in entries if _safe_get(e, split, "wer") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda e: float(e[split]["wer"]))


def _rel_improvement(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or base == 0:
        return None
    return (base - new) / base


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_dir.rglob("*_lm_ablation.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *_lm_ablation.json files found under {input_dir}")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = _safe_get(data, "meta", "model")
        lang = _safe_get(data, "meta", "lang")
        split = _safe_get(data, "meta", "split")
        greedy = data.get("greedy", {})
        unigram = data.get("unigram_beam", {})
        ngram_entries = data.get("ngram_beam", [])
        best = _best_entry(ngram_entries, args.metric_split)
        if best is None:
            continue

        target_split = args.metric_split
        greedy_wer = _safe_get(greedy, target_split, "wer")
        greedy_cer = _safe_get(greedy, target_split, "cer")
        unigram_wer = _safe_get(unigram, target_split, "wer")
        unigram_cer = _safe_get(unigram, target_split, "cer")
        best_wer = _safe_get(best, target_split, "wer")
        best_cer = _safe_get(best, target_split, "cer")
        best_decode_sec = best.get(f"{target_split}_decode_sec")
        uni_decode_sec = unigram.get(f"{target_split}_decode_sec")

        summary_rows.append(
            {
                "model": model,
                "lang": lang,
                "train_split": split,
                "metric_split": target_split,
                "greedy_wer": greedy_wer,
                "unigram_wer": unigram_wer,
                "best_ngram_wer": best_wer,
                "greedy_cer": greedy_cer,
                "unigram_cer": unigram_cer,
                "best_ngram_cer": best_cer,
                "best_lm_order": best.get("lm_order"),
                "best_alpha": best.get("alpha"),
                "best_beta": best.get("beta"),
                "best_beam_width": best.get("beam_width"),
                "rel_wer_reduction_vs_greedy": _rel_improvement(greedy_wer, best_wer),
                "rel_wer_reduction_vs_unigram": _rel_improvement(unigram_wer, best_wer),
                "unigram_decode_sec": uni_decode_sec,
                "best_ngram_decode_sec": best_decode_sec,
                "source_json": str(path),
            }
        )

        for entry in ngram_entries:
            if _safe_get(entry, args.metric_split, "wer") is None:
                continue
            metric_split = args.metric_split
            detail_rows.append(
                {
                    "model": model,
                    "lang": lang,
                    "train_split": split,
                    "metric_split": metric_split,
                    "lm_order": entry.get("lm_order"),
                    "alpha": entry.get("alpha"),
                    "beta": entry.get("beta"),
                    "beam_width": entry.get("beam_width"),
                    "wer": _safe_get(entry, metric_split, "wer"),
                    "cer": _safe_get(entry, metric_split, "cer"),
                    "decode_sec": entry.get(f"{metric_split}_decode_sec"),
                    "decoder_desc": entry.get("decoder_desc"),
                    "source_json": str(path),
                }
            )

    summary_path = output_dir / "lm_ablation_summary.csv"
    detail_path = output_dir / "lm_ablation_grid.csv"

    if summary_rows:
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if detail_rows:
        with open(detail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    print(f"Processed JSON files: {len(json_paths)}")
    print(f"Summary rows: {len(summary_rows)} -> {summary_path}")
    print(f"Grid rows: {len(detail_rows)} -> {detail_path}")


if __name__ == "__main__":
    main()
