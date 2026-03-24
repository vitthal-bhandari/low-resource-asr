#!/usr/bin/env python3
"""
Summarize two-setting decode runs into a single test-WER CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one CSV across models/languages for setA and setB test WER."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing *_setA.json/*_setB.json")
    parser.add_argument("--output-csv", required=True, help="Path to output summary CSV")
    parser.add_argument("--split", default="all", help="Expected split name (for metadata only)")
    parser.add_argument("--set-a-label", default="setA")
    parser.add_argument("--set-b-label", default="setB")
    return parser.parse_args()


def _safe_get(dct: dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _best_test_entry(data: dict[str, Any]) -> dict[str, Any] | None:
    entries = data.get("ngram_beam", [])
    candidates = [e for e in entries if _safe_get(e, "test", "wer") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda e: float(e["test"]["wer"]))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for json_path in sorted(run_dir.glob("*_setA.json")):
        data = _load_json(json_path)
        model = _safe_get(data, "meta", "model")
        lang = _safe_get(data, "meta", "lang")
        if not model or not lang:
            continue
        key = (model, lang)
        best = _best_test_entry(data)
        row = rows_by_key.setdefault(key, {"model": model, "lang": lang, "split": args.split})
        row["greedy_test_wer"] = _safe_get(data, "greedy", "test", "wer")
        row["unigram_test_wer"] = _safe_get(data, "unigram_beam", "test", "wer")
        if best is not None:
            row[f"{args.set_a_label}_test_wer"] = _safe_get(best, "test", "wer")
            row[f"{args.set_a_label}_n"] = best.get("lm_order")
            row[f"{args.set_a_label}_alpha"] = best.get("alpha")
            row[f"{args.set_a_label}_beta"] = best.get("beta")
            row[f"{args.set_a_label}_beam"] = best.get("beam_width")
            row[f"{args.set_a_label}_decoder_desc"] = best.get("decoder_desc")
        row[f"{args.set_a_label}_json"] = str(json_path)

    for json_path in sorted(run_dir.glob("*_setB.json")):
        data = _load_json(json_path)
        model = _safe_get(data, "meta", "model")
        lang = _safe_get(data, "meta", "lang")
        if not model or not lang:
            continue
        key = (model, lang)
        best = _best_test_entry(data)
        row = rows_by_key.setdefault(key, {"model": model, "lang": lang, "split": args.split})
        if best is not None:
            row[f"{args.set_b_label}_test_wer"] = _safe_get(best, "test", "wer")
            row[f"{args.set_b_label}_n"] = best.get("lm_order")
            row[f"{args.set_b_label}_alpha"] = best.get("alpha")
            row[f"{args.set_b_label}_beta"] = best.get("beta")
            row[f"{args.set_b_label}_beam"] = best.get("beam_width")
            row[f"{args.set_b_label}_decoder_desc"] = best.get("decoder_desc")
        row[f"{args.set_b_label}_json"] = str(json_path)

    rows = [rows_by_key[k] for k in sorted(rows_by_key.keys())]
    if not rows:
        print(f"No setA/setB JSON files found in {run_dir}")
        return

    headers = [
        "model",
        "lang",
        "split",
        "greedy_test_wer",
        "unigram_test_wer",
        f"{args.set_a_label}_test_wer",
        f"{args.set_a_label}_n",
        f"{args.set_a_label}_alpha",
        f"{args.set_a_label}_beta",
        f"{args.set_a_label}_beam",
        f"{args.set_a_label}_decoder_desc",
        f"{args.set_b_label}_test_wer",
        f"{args.set_b_label}_n",
        f"{args.set_b_label}_alpha",
        f"{args.set_b_label}_beta",
        f"{args.set_b_label}_beam",
        f"{args.set_b_label}_decoder_desc",
        f"{args.set_a_label}_json",
        f"{args.set_b_label}_json",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
