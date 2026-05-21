"""
Utterance-level quality scoring for Mozilla Spontaneous Speech training data.

Two-stage pipeline:
  Stage 1 — hard filters (applied first, no ranking):
    - Duration: drop if duration_sec < p5 or > p95 within the language
    - Repetition: drop if char_4gram_repetition_ratio > p99 within the language
  Stage 2 — soft ranking on survivors:
    - Build character-level KenLM (order=4 by default) on survivor transcripts
    - Score each survivor by perplexity; lower = more typical = higher quality
    - Keep bottom `keep_fraction` by perplexity percentile rank

Added columns:
    duration_filtered   bool   flagged by duration hard filter
    rep_filtered        bool   flagged by repetition hard filter
    char_4gram_rep      float  character 4-gram repetition ratio (all rows)
    ppl_score           float  KenLM perplexity (NaN if hard-filtered)
    ppl_pct             float  within-language ppl percentile rank 0-1 (NaN if filtered)
    quality_keep        bool   True iff passes all filters and ppl_pct <= keep_fraction
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import kenlm
import pandas as pd


def char_4gram_repetition_ratio(text: str) -> float:
    """Fraction of character 4-grams that are repeated within the utterance."""
    if len(text) < 4:
        return 0.0
    grams = [text[i : i + 4] for i in range(len(text) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def build_char_kenlm(sentences: list[str], output_dir: Path, order: int = 4) -> Path:
    """
    Build a character-level KenLM ARPA from sentences.

    Each sentence is space-joined into individual characters before piping to
    lmplz, so KenLM treats each character as a token.

    Writes:
        output_dir/char_train.txt        — character-tokenised training text
        output_dir/char_{order}gram.arpa — ARPA file

    Raises RuntimeError if lmplz is not on PATH.
    """
    lmplz_bin = shutil.which("lmplz")
    if lmplz_bin is None:
        raise RuntimeError(
            "lmplz not found on PATH. "
            "On Hyak: module load gcc && module load cesg/boost/1.76.0 && "
            "export PATH=/gscratch/scrubbed/$USER/tools/kenlm/build/bin:$PATH"
        )
    print(f"  lmplz: {lmplz_bin}")

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "char_train.txt"
    arpa_path = output_dir / f"char_{order}gram.arpa"

    with open(text_path, "w", encoding="utf-8") as f:
        for s in sentences:
            chars = " ".join(list(s.lower().strip()))
            if chars:
                f.write(chars + "\n")

    try:
        with open(text_path) as stdin, open(arpa_path, "w") as stdout:
            subprocess.run(
                [lmplz_bin, "-S", "1G", "-o", str(order), "--discount_fallback"],
                stdin=stdin,
                stdout=stdout,
                check=True,
                stderr=subprocess.DEVNULL,
            )
        return arpa_path
    except (subprocess.CalledProcessError, OSError):
        return None


def score_perplexity(sentences: list[str], arpa_path: Path) -> list[float]:
    """Score each sentence with character-level KenLM perplexity."""
    model = kenlm.Model(str(arpa_path))
    scores = []
    for s in sentences:
        char_seq = " ".join(list(s.lower().strip()))
        scores.append(model.perplexity(char_seq) if char_seq else float("inf"))
    return scores


def compute_quality_scores(
    df: pd.DataFrame,
    output_dir: Path,
    lm_order: int = 4,
    keep_fraction: float = 0.60,
) -> pd.DataFrame:
    """
    Add quality columns to a training-split DataFrame. Returns augmented copy.

    Args:
        df:             Training rows from ss-corpus-{lang}.tsv (must have
                        'duration_ms' and 'transcription' columns).
        output_dir:     Where to write KenLM artefacts (char_train.txt, ARPA).
        lm_order:       KenLM n-gram order (default 4).
        keep_fraction:  Fraction of survivors to mark quality_keep=True (default 0.60).
    """
    df = df.copy()

    # --- Stage 1a: duration hard filter ---
    dur_sec = df["duration_ms"] / 1000.0
    p5 = dur_sec.quantile(0.05)
    p95 = dur_sec.quantile(0.95)
    df["duration_filtered"] = (dur_sec < p5) | (dur_sec > p95)

    # --- Stage 1b: repetition hard filter ---
    df["char_4gram_rep"] = df["transcription"].astype(str).map(char_4gram_repetition_ratio)
    p99_rep = df["char_4gram_rep"].quantile(0.99)
    df["rep_filtered"] = (~df["duration_filtered"]) & (df["char_4gram_rep"] > p99_rep)

    # --- Stage 2: KenLM perplexity on survivors ---
    survivor_mask = ~df["duration_filtered"] & ~df["rep_filtered"]
    survivors = df[survivor_mask]

    df["ppl_score"] = float("nan")
    df["ppl_pct"] = float("nan")
    df["quality_keep"] = False

    arpa_path = build_char_kenlm(survivors["transcription"].tolist(), output_dir, order=lm_order)
    ppl = score_perplexity(survivors["transcription"].tolist(), arpa_path)
    df.loc[survivor_mask, "ppl_score"] = ppl
    df.loc[survivor_mask, "ppl_pct"] = df.loc[survivor_mask, "ppl_score"].rank(pct=True)
    df.loc[survivor_mask, "quality_keep"] = df.loc[survivor_mask, "ppl_pct"] <= keep_fraction

    return df
