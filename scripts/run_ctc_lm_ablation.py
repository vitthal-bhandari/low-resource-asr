#!/usr/bin/env python3
"""
Decode-only LM ablation for CTC models (MMS / XLS-R) using saved checkpoints.

This script reuses existing fine-tuned checkpoints and runs:
1) Greedy decoding
2) Beam search with unigrams only (no ARPA)
3) Beam search with n-gram KenLM ARPA (sweep over n/alpha/beta/beam)

It does not retrain models.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor

from src.config import config

try:
    from pyctcdecode import build_ctcdecoder
    from transformers import Wav2Vec2ProcessorWithLM

    _PYCTCDECODE_AVAILABLE = True
except ImportError:
    _PYCTCDECODE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode-only n-gram LM ablations for MMS/XLS-R checkpoints."
    )
    parser.add_argument("--model", choices=["mms", "xlsr"], required=True)
    parser.add_argument("--lang", required=True, help="Language code, e.g. aln")
    parser.add_argument("--split", default="all",
                        help="Training split name used to locate the HF checkpoint (e.g. 1h, curated-2h, full).")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional local checkpoint directory containing model + processor.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Optional Hugging Face repo ID. If omitted, script resolves the default repo ID for model/lang/split and pulls latest from HF.",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Hugging Face revision (branch/tag/commit) for remote loading (default: main).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beam-widths", type=str, default="50,100")
    parser.add_argument("--lm-orders", type=str, default="2,3,4")
    parser.add_argument("--lm-alphas", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--lm-betas", type=str, default="0.0,1.0,2.0")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional explicit output JSON path.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test split evaluation (val only).",
    )
    parser.add_argument(
        "--selection-split",
        choices=["val", "test"],
        default="test",
        help="Split used to select/report the final best LM setting (default: test).",
    )
    parser.add_argument(
        "--allow-val-fallback",
        action="store_true",
        help="If --selection-split=test and test split is unavailable, allow fallback to validation instead of failing.",
    )
    parser.add_argument(
        "--require-arpa",
        action="store_true",
        help="Fail if any n-gram setting falls back to unigram (no ARPA). Recommended for strict LM ablations.",
    )
    parser.add_argument(
        "--lm-arpa-path",
        default=None,
        metavar="PATH",
        help="Path to a pre-built KenLM ARPA file. If omitted, one is built from training sentences.",
    )
    return parser.parse_args()


def _parse_list(raw: str, cast_fn: Callable[[str], Any]) -> list[Any]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one value in comma-separated list.")
    return [cast_fn(v) for v in values]


def _load_impl(model_name: str):
    if model_name == "mms":
        from src.training import aft_mms as impl
    elif model_name == "xlsr":
        from src.training import aft_xlsr as impl
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return impl


def _resolve_checkpoint_ref(args: argparse.Namespace, impl) -> tuple[str, str]:
    """
    Resolve checkpoint reference and source.

    Priority:
      1) Explicit --hf-repo-id
      2) Explicit --checkpoint-dir
      3) Default HF repo from training module helper (latest on --hf-revision)
    """
    if args.hf_repo_id:
        return args.hf_repo_id, "hf"

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        return str(checkpoint_dir), "local"

    default_repo = impl.get_hf_repo_id(args.lang, args.split)
    return default_repo, "hf"


def _prepare_eval_dataset(dataset, prepare_dataset_fn, keep_audio_file: bool):
    keep_cols = ["audio", "sentence"]
    if keep_audio_file:
        keep_cols.append("audio_file")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])
    remove_cols = [c for c in dataset.column_names if c != "audio_file"]
    return dataset.map(
        prepare_dataset_fn,
        remove_columns=remove_cols,
        desc="Processing eval split",
    )


def _greedy_metrics(pred, processor: Wav2Vec2Processor, compute_wer_cer_fn) -> dict[str, Any]:
    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    pred_str = [p.lower() for p in pred_str]
    label_str = [l.lower() for l in label_str]
    metrics = compute_wer_cer_fn(pred_str, label_str)
    return metrics


def _build_unigram_processor(
    processor: Wav2Vec2Processor,
    train_sentences: list[str],
) -> Wav2Vec2ProcessorWithLM:
    if not _PYCTCDECODE_AVAILABLE:
        raise RuntimeError("pyctcdecode is required for unigram and LM decoding.")

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])
    labels: list[str] = []
    for char, _idx in sorted_vocab:
        if char == processor.tokenizer.pad_token:
            labels.append("")
        elif char == processor.tokenizer.word_delimiter_token:
            labels.append(" ")
        else:
            labels.append(char)

    unigrams = list({w for s in train_sentences for w in s.lower().strip().split() if w})
    decoder = build_ctcdecoder(labels=labels, unigrams=unigrams)
    return Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )


def _lm_metrics(pred, processor: Wav2Vec2Processor, lm_processor, decode_fn, metric_fn, beam_width: int):
    t0 = time.perf_counter()
    pred_str, label_str = decode_fn(
        pred.predictions,
        pred.label_ids,
        processor,
        lm_processor,
        beam_width=beam_width,
    )
    decode_sec = time.perf_counter() - t0
    metrics = metric_fn(pred_str, label_str)
    return metrics, decode_sec


def main() -> None:
    args = parse_args()
    if not _PYCTCDECODE_AVAILABLE:
        raise RuntimeError("Install pyctcdecode and kenlm before running LM ablations.")

    lm_orders = _parse_list(args.lm_orders, int)
    lm_alphas = _parse_list(args.lm_alphas, float)
    lm_betas = _parse_list(args.lm_betas, float)
    beam_widths = _parse_list(args.beam_widths, int)

    impl = _load_impl(args.model)
    checkpoint_ref, checkpoint_source = _resolve_checkpoint_ref(args, impl)

    train, val = impl.load_and_preprocess_data(args.lang, args.split)
    train_sentences: list[str] = train["sentence"]

    if checkpoint_source == "hf":
        processor = Wav2Vec2Processor.from_pretrained(
            checkpoint_ref, revision=args.hf_revision
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            checkpoint_ref, revision=args.hf_revision
        )
    else:
        processor = Wav2Vec2Processor.from_pretrained(checkpoint_ref)
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_ref)

    prepare_dataset_fn = impl.create_prepare_dataset_fn(processor)
    val_ds = _prepare_eval_dataset(val, prepare_dataset_fn, keep_audio_file=False)
    test_ds = None
    if not args.skip_test:
        raw_test = impl.load_test_data(args.lang)
        if raw_test is not None and len(raw_test) > 0:
            test_ds = _prepare_eval_dataset(raw_test, prepare_dataset_fn, keep_audio_file=False)
    test_available = test_ds is not None
    if args.selection_split == "test" and not test_available and not args.allow_val_fallback:
        raise ValueError(
            "selection-split=test requires an available test split. "
            "Provide test data for this language/split, or pass --allow-val-fallback."
        )

    eval_tmp_dir = config.results_dir / "lm_ablation" / "tmp_trainer"
    eval_tmp_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(eval_tmp_dir),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        ),
        data_collator=impl.DataCollatorCTCWithPadding(processor=processor, padding=True),
        processing_class=processor.feature_extractor,
    )

    print("Running greedy predictions...")
    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds) if test_ds is not None else None

    results: dict[str, Any] = {
        "meta": {
            "model": args.model,
            "lang": args.lang,
            "split": args.split,
            "checkpoint_source": checkpoint_source,
            "checkpoint_ref": checkpoint_ref,
            "hf_revision": args.hf_revision if checkpoint_source == "hf" else None,
            "selection_split_requested": args.selection_split,
            "selection_split_used": (
                args.selection_split
                if (args.selection_split == "val" or test_available)
                else "val"
            ),
            "lm_orders": lm_orders,
            "lm_alphas": lm_alphas,
            "lm_betas": lm_betas,
            "beam_widths": beam_widths,
        },
        "greedy": {},
        "unigram_beam": {},
        "ngram_beam": [],
    }

    greedy_val = _greedy_metrics(val_pred, processor, impl.compute_wer_cer)
    results["greedy"]["val"] = greedy_val
    if test_pred is not None:
        results["greedy"]["test"] = _greedy_metrics(test_pred, processor, impl.compute_wer_cer)

    print("Running unigram beam baseline...")
    unigram_processor = _build_unigram_processor(processor, train_sentences)
    uni_val_metrics, uni_val_decode_sec = _lm_metrics(
        val_pred, processor, unigram_processor, impl._decode_with_lm, impl.compute_wer_cer, beam_width=100
    )
    results["unigram_beam"]["val"] = uni_val_metrics
    results["unigram_beam"]["val_decode_sec"] = uni_val_decode_sec
    results["unigram_beam"]["beam_width"] = 100
    if test_pred is not None:
        uni_test_metrics, uni_test_decode_sec = _lm_metrics(
            test_pred,
            processor,
            unigram_processor,
            impl._decode_with_lm,
            impl.compute_wer_cer,
            beam_width=100,
        )
        results["unigram_beam"]["test"] = uni_test_metrics
        results["unigram_beam"]["test_decode_sec"] = uni_test_decode_sec

    print("Running n-gram sweep...")
    lm_artifacts_dir = config.results_dir / "lm_ablation" / "arpa_artifacts" / args.model / args.lang / args.split
    lm_artifacts_dir.mkdir(parents=True, exist_ok=True)

    for lm_order, alpha, beta, beam_width in itertools.product(
        lm_orders, lm_alphas, lm_betas, beam_widths
    ):
        lm_processor, lm_desc = impl._build_lm_processor(
            processor=processor,
            sentences=train_sentences,
            output_dir=lm_artifacts_dir,
            arpa_path=args.lm_arpa_path,
            alpha=alpha,
            beta=beta,
            lm_order=lm_order,
        )
        if lm_processor is None:
            raise RuntimeError(f"Failed to build LM processor: {lm_desc}")
        if args.require_arpa and "no ARPA" in lm_desc:
            raise RuntimeError(
                "ARPA LM was required, but decoder fell back to unigram. "
                "Check lmplz availability and KenLM build/runtime dependencies on this node."
            )

        val_metrics, val_decode_sec = _lm_metrics(
            val_pred, processor, lm_processor, impl._decode_with_lm, impl.compute_wer_cer, beam_width=beam_width
        )
        entry: dict[str, Any] = {
            "lm_order": lm_order,
            "alpha": alpha,
            "beta": beta,
            "beam_width": beam_width,
            "decoder_desc": lm_desc,
            "val": val_metrics,
            "val_decode_sec": val_decode_sec,
        }
        if test_pred is not None:
            test_metrics, test_decode_sec = _lm_metrics(
                test_pred,
                processor,
                lm_processor,
                impl._decode_with_lm,
                impl.compute_wer_cer,
                beam_width=beam_width,
            )
            entry["test"] = test_metrics
            entry["test_decode_sec"] = test_decode_sec
        results["ngram_beam"].append(entry)

    out_path = (
        Path(args.output_json)
        if args.output_json
        else config.results_dir
        / "lm_ablation"
        / f"{args.model}_{args.lang}_{args.split}_lm_ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    selection_split = results["meta"]["selection_split_used"]
    best_entry = min(results["ngram_beam"], key=lambda x: x[selection_split]["wer"])
    print("=" * 72)
    print(f"Saved results to: {out_path}")
    print(
        "Checkpoint source: "
        f"{checkpoint_source} ({checkpoint_ref}"
        f"{' @ ' + args.hf_revision if checkpoint_source == 'hf' else ''})"
    )
    print(
        f"Selection split: {selection_split}"
        + (" (fallback from test)" if args.selection_split == "test" and selection_split == "val" else "")
    )
    print(f"Greedy {selection_split} WER:   {results['greedy'][selection_split]['wer']:.4f}")
    print(f"Unigram {selection_split} WER:  {results['unigram_beam'][selection_split]['wer']:.4f}")
    print(f"Best n-gram {selection_split} WER: {best_entry[selection_split]['wer']:.4f}")
    print(
        "Best params: "
        f"n={best_entry['lm_order']}, alpha={best_entry['alpha']}, "
        f"beta={best_entry['beta']}, beam={best_entry['beam_width']}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
