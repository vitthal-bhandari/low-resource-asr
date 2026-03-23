"""
Script for training adapter layers as a means of fine-tuning the MMS model for ASR.

The output of the script is:
- A tokenizer for this specific language
- Adapter layers that are fine-tuned on your data
- (Optional) KenLM ARPA file + Wav2Vec2ProcessorWithLM for LM decoding

Optionally supports LM-decoded evaluation (--use-lm) using pyctcdecode beam
search with an n-gram KenLM language model built from training transcriptions.
LM decoding is applied only to final evaluation (val + test), not during
training, so training speed is unaffected.

For more information about this process and adapter fine tuning, see:
https://huggingface.co/blog/mms_adapters

Usage:
    python -m src.training.aft_mms aln
    python -m src.training.aft_mms aln --num-epochs 10 --batch-size 4
    python -m src.training.aft_mms aln --save-to-hf --hf-repo-id username/mms-aln

    # With LM decoding (auto-builds n-gram LM from training text):
    python -m src.training.aft_mms aln --use-lm
    python -m src.training.aft_mms aln --use-lm --lm-arpa-path /path/to/lm.arpa
    python -m src.training.aft_mms aln --use-lm --beam-width 50 --lm-alpha 0.5 --lm-beta 1.0

Dependencies for LM decoding:
    pip install pyctcdecode kenlm
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import librosa
from datasets import Dataset
from evaluate import load as load_metric
from huggingface_hub import HfApi, login
from safetensors.torch import save_file as safe_save_file
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

from src.config import config
from src.data.download import LANGUAGES

try:
    from pyctcdecode import build_ctcdecoder
    from transformers import Wav2Vec2ProcessorWithLM

    _PYCTCDECODE_AVAILABLE = True
except ImportError:
    _PYCTCDECODE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS model with adapter layers for ASR"
    )
    parser.add_argument(
        "lang",
        type=str,
        help="Language ISO code (e.g., 'aln' for Gheg Albanian)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model and tokenizer (default: models/mms/{lang})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8, effective batch = 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--save-to-hf",
        action="store_true",
        help="Push the fine-tuned model to Hugging Face Hub",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["one", "mid", "all"],
        default="all",
        help="Training data split: one=1h, mid=tier-dependent (3/5/10h), all=full (default: all)",
    )
    parser.add_argument(
        "--save-transcriptions",
        action="store_true",
        help="Save test-set gold and model transcriptions to a TSV file (language, split, datetime in filename).",
    )
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="Drop train/val samples longer than this (seconds) to avoid OOM. Ignored if --max-audio-sec-from-csv is set.",
    )
    parser.add_argument(
        "--max-audio-sec-from-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to corpus_stats CSV with columns lang,p97_5_sec. Use p97.5 for current language as max length (drop longer utterances).",
    )
    parser.add_argument(
        "--use-lm",
        action="store_true",
        help="Enable LM decoding for final evaluation (requires pyctcdecode). Training uses greedy decoding regardless.",
    )
    parser.add_argument(
        "--lm-arpa-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a pre-built KenLM ARPA file. If omitted with --use-lm, one is built from training text (requires lmplz) or beam search with unigrams is used as fallback.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=100,
        help="Beam width for LM decoding (default: 100). Only used with --use-lm.",
    )
    parser.add_argument(
        "--lm-alpha",
        type=float,
        default=0.5,
        help="LM weight (alpha) for beam search (default: 0.5).",
    )
    parser.add_argument(
        "--lm-beta",
        type=float,
        default=1.0,
        help="Word insertion bonus (beta) for beam search (default: 1.0).",
    )
    parser.add_argument(
        "--lm-order",
        type=int,
        default=3,
        help="Order n for auto-built n-gram ARPA LM (default: 3). Only used when --use-lm and --lm-arpa-path is not provided.",
    )
    return parser.parse_args()


# Hugging Face configuration
HF_USERNAME = "vitthalbhandari"


def get_hf_repo_id(lang: str, split: str = "all") -> str:
    """Get the Hugging Face repo ID for a language and split."""
    return f"{HF_USERNAME}/mms-1b-all-aft-{split}-{lang}"


def _max_sec_from_corpus_csv(csv_path: Path, lang: str) -> float | None:
    """
    Read corpus_stats CSV (columns lang, p97_5_sec) and return p97.5 seconds for the given language.
    Returns None if file missing, lang not found, or column missing.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "lang" not in df.columns or "p97_5_sec" not in df.columns:
            return None
        row = df[df["lang"] == lang]
        if row.empty:
            return None
        val = row["p97_5_sec"].iloc[0]
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


################################################################################
# Helper functions for loading and cleaning dataset and defining the vocabulary.
#

# Regex patterns for text cleaning
BRACKETED = re.compile(r"\[[^\]]+\]")
UNINTELL_PAREN = re.compile(r"\(\?+\)")
REPL_PUNC = re.compile('[,?¿¡!";:]+')
MULTISPACE = re.compile("  +")


def clean_transcript(text: str) -> str:
    """
    Clean transcript text by removing annotations and normalizing punctuation.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELL_PAREN, " ", text)
    text = text.replace(" ... ", " ")
    text = text.replace("#x27;", "'")
    text = re.sub(REPL_PUNC, " ", text)
    text = (
        text.replace("...", "!ELLIPSIS!")
        .replace(".", " ")
        .replace("!ELLIPSIS!", "...")
    )
    text = re.sub(MULTISPACE, " ", text)
    return text.strip()


def load_and_preprocess_data(lang: str, split: str = "all") -> tuple[Dataset, Dataset]:
    """
    Load and preprocess train and validation from precomputed split TSVs.
    Run scripts/create_splits.py --all first to generate train-{split}_{lang}.tsv
    and validation_{lang}.tsv.
    """
    lang_dir = config.mozilla_data_dir / lang
    train_tsv = lang_dir / f"train-{split}_{lang}.tsv"
    val_tsv = lang_dir / f"validation_{lang}.tsv"
    audio_dir = config.mozilla_data_dir / "shared_train_validation_audios"

    if not train_tsv.exists():
        raise FileNotFoundError(
            f"Train split TSV not found: {train_tsv}. Run: uv run python scripts/create_splits.py --all"
        )
    if not val_tsv.exists():
        raise FileNotFoundError(
            f"Validation TSV not found: {val_tsv}. Run: uv run python scripts/create_splits.py --all"
        )
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    print(f"\nLoading data for {lang} ({LANGUAGES.get(lang, 'Unknown')}) split={split}...")
    print(f"  Train: {train_tsv}")
    print(f"  Val: {val_tsv}")

    required_cols = ["audio_file", "transcription", "duration_ms", "split"]
    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(val_tsv, sep="\t")
    for name, d in [("train", train_df), ("val", val_df)]:
        missing = [c for c in required_cols if c not in d.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} TSV: {missing}")

    train_df = train_df[train_df["duration_ms"] > 0]
    train_df = train_df[train_df["transcription"].notna() & (train_df["transcription"].str.strip() != "")]
    val_df = val_df[val_df["duration_ms"] > 0]
    val_df = val_df[val_df["transcription"].notna() & (val_df["transcription"].str.strip() != "")]

    train_df["transcription"] = train_df["transcription"].apply(clean_transcript)
    val_df["transcription"] = val_df["transcription"].apply(clean_transcript)
    train_df = train_df[train_df["transcription"].str.strip() != ""]
    val_df = val_df[val_df["transcription"].str.strip() != ""]

    train_h = train_df["duration_ms"].sum() / (1000 * 60 * 60)
    val_h = val_df["duration_ms"].sum() / (1000 * 60 * 60)
    print(f"  Train: {len(train_df)} samples ({train_h:.2f} h)")
    print(f"  Validation: {len(val_df)} samples ({val_h:.2f} h)")

    train_df["audio"] = [{"path": str(audio_dir / x)} for x in train_df["audio_file"]]
    val_df["audio"] = [{"path": str(audio_dir / x)} for x in val_df["audio_file"]]
    train_df = train_df.rename(columns={"transcription": "sentence"})
    val_df = val_df.rename(columns={"transcription": "sentence"})

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    return train_dataset, val_dataset


def load_test_data(lang: str) -> Dataset | None:
    """Load test set from test_{lang}.tsv if present (created by create_splits.py)."""
    lang_dir = config.mozilla_data_dir / lang
    test_tsv = lang_dir / f"test_{lang}.tsv"
    audio_dir = config.mozilla_data_dir / "shared_train_validation_audios"
    if not test_tsv.exists() or not audio_dir.exists():
        return None
    required_cols = ["audio_file", "transcription", "duration_ms", "split"]
    test_df = pd.read_csv(test_tsv, sep="\t")
    if any(c not in test_df.columns for c in required_cols):
        return None
    test_df = test_df[test_df["duration_ms"] > 0]
    test_df = test_df[test_df["transcription"].notna() & (test_df["transcription"].str.strip() != "")]
    test_df["transcription"] = test_df["transcription"].apply(clean_transcript)
    test_df = test_df[test_df["transcription"].str.strip() != ""]
    test_df["audio"] = [{"path": str(audio_dir / x)} for x in test_df["audio_file"]]
    test_df = test_df.rename(columns={"transcription": "sentence"})
    return Dataset.from_pandas(test_df, preserve_index=False)


def extract_all_chars(batch: dict) -> dict:
    """Extract all unique characters from a batch of sentences."""
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def make_vocab(
    train_data: Dataset,
    val_data: Dataset,
    target_lang: str,
    output_dir: Path,
) -> Path:
    """
    Build vocabulary from train and validation sets.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        target_lang: Target language code
        output_dir: Directory to save vocab.json
        
    Returns:
        Path to the saved vocab.json file
    """
    vocab_train = train_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_data.column_names,
    )
    vocab_val = val_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=val_data.column_names,
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Replace space with word delimiter
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    
    # Add special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Wrap in language-specific dict for MMS
    new_vocab_dict = {target_lang: vocab_dict}
    
    # Save vocab file
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as vocab_file:
        json.dump(new_vocab_dict, vocab_file, ensure_ascii=False, indent=2)
    
    print(f"  Vocabulary saved to {vocab_path} ({len(vocab_dict)} tokens)")
    return vocab_path


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    
    Args:
        processor: Wav2Vec2Processor for padding
        padding: Padding strategy (True for dynamic padding)
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def create_prepare_dataset_fn(processor: Wav2Vec2Processor):
    """
    Create a dataset preparation function with the processor in closure.
    Loads audio from path with soundfile/librosa to avoid torchcodec/FFmpeg on HPC.
    """
    TARGET_SR = 16_000

    def prepare_dataset(batch: dict) -> dict:
        path = batch["audio"]["path"]
        array, sr = sf.read(path, dtype="float32")
        if sr != TARGET_SR:
            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        batch["input_values"] = processor(array, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    return prepare_dataset


def create_compute_metrics_fn(processor: Wav2Vec2Processor):
    """
    Create a metrics computation function with the processor in closure.
    
    Args:
        processor: Wav2Vec2Processor for decoding predictions
        
    Returns:
        Function that computes WER metric
    """
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred) -> dict:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # We do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        # Normalize case for fair WER/CER comparison
        pred_str = [p.lower() for p in pred_str]
        label_str = [l.lower() for l in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


################################################################################
# LM decoding helpers (pyctcdecode / KenLM)


def _build_arpa_from_text(
    sentences: list[str], output_dir: Path, order: int = 3
) -> Path | None:
    """
    Build an n-gram ARPA language model from training sentences using KenLM's lmplz.
    Returns the ARPA path on success, None if lmplz is unavailable.
    """
    import shutil
    import subprocess

    lmplz = shutil.which("lmplz")
    if lmplz is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "lm_train.txt"
    arpa_path = output_dir / f"lm_{order}gram.arpa"
    lmplz_memory = os.environ.get("LMPLZ_MEMORY", "1G")

    with open(text_path, "w", encoding="utf-8") as f:
        for s in sentences:
            cleaned = s.lower().strip()
            if cleaned:
                f.write(cleaned + "\n")

    try:
        with open(text_path, "r") as stdin, open(arpa_path, "w") as stdout:
            subprocess.run(
                [lmplz, "-S", lmplz_memory, "-o", str(order), "--discount_fallback"],
                stdin=stdin,
                stdout=stdout,
                check=True,
                capture_output=False,
                stderr=subprocess.DEVNULL,
            )
        print(f"  Built {order}-gram ARPA LM: {arpa_path}")
        return arpa_path
    except (subprocess.CalledProcessError, OSError):
        return None


def _build_lm_processor(
    processor: Wav2Vec2Processor,
    sentences: list[str],
    output_dir: Path,
    arpa_path: str | None = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    lm_order: int = 3,
):
    """
    Build a Wav2Vec2ProcessorWithLM for beam-search decoding.

    Priority: provided ARPA > auto-built ARPA > unigram-only beam search.
    Returns (Wav2Vec2ProcessorWithLM, description_str) or (None, reason).

    Note: MMS tokenizer uses target_lang and a nested vocab dict, but
    get_vocab() returns the flat vocab for the active language, so pyctcdecode
    label extraction works identically to XLS-R.
    """
    if not _PYCTCDECODE_AVAILABLE:
        return None, "pyctcdecode not installed"

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])
    labels = []
    for char, _idx in sorted_vocab:
        if char == processor.tokenizer.pad_token:
            labels.append("")
        elif char == processor.tokenizer.word_delimiter_token:
            labels.append(" ")
        else:
            labels.append(char)

    unigrams = list(
        {w for s in sentences for w in s.lower().strip().split() if w}
    )

    resolved_arpa = None
    if arpa_path is not None:
        p = Path(arpa_path)
        if p.exists():
            resolved_arpa = str(p)
            print(f"  Using provided ARPA: {resolved_arpa}")
        else:
            print(f"  WARNING: --lm-arpa-path {arpa_path} not found, attempting auto-build.")

    if resolved_arpa is None:
        built = _build_arpa_from_text(sentences, output_dir, order=lm_order)
        if built is not None:
            resolved_arpa = str(built)

    if resolved_arpa is not None:
        # pyctcdecode API differs by version:
        # - newer: kenlm_model=
        # - older: kenlm_model_path=
        # Try both for compatibility across environments.
        try:
            decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model=resolved_arpa,
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
        except TypeError:
            decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=resolved_arpa,
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
        desc = f"n-gram LM ({resolved_arpa})"
    else:
        decoder = build_ctcdecoder(
            labels=labels,
            unigrams=unigrams,
        )
        desc = "beam search with unigrams (no ARPA)"

    lm_processor = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )
    return lm_processor, desc


def _decode_with_lm(
    pred_logits: np.ndarray,
    label_ids: np.ndarray,
    processor: Wav2Vec2Processor,
    lm_processor,
    beam_width: int = 100,
) -> tuple[list[str], list[str]]:
    """
    Decode predictions with LM beam search and return (pred_str, label_str).
    """
    pred_str = lm_processor.batch_decode(
        pred_logits, beam_width=beam_width
    ).text
    label_ids_copy = label_ids.copy()
    label_ids_copy[label_ids_copy == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids_copy, group_tokens=False)
    pred_str = [p.lower() for p in pred_str]
    label_str = [l.lower() for l in label_str]
    return pred_str, label_str


def compute_wer_cer(pred_str: list[str], label_str: list[str]) -> dict:
    """Compute WER and CER from pre-decoded prediction and label strings."""
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _model_card_language_metadata(lang: str) -> str:
    """
    Return YAML lines for model card 'language' (and optionally 'language_bcp47').
    HF requires 'language' to be lowercase ISO 639-1/2/3; BCP-47 codes go in language_bcp47 (as array).
    """
    lang_lower = lang.lower()
    # BCP-47 style (e.g. el-CY): use base code for language, full for language_bcp47 (must be array)
    if "-" in lang_lower:
        base = lang_lower.split("-")[0]
        return f"language: {base}\nlanguage_bcp47:\n- {lang_lower}"
    return f"language: {lang_lower}"


def push_to_hub(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    output_dir: Path,
    repo_id: str,
    lang: str,
) -> None:
    """
    Push the fine-tuned model and processor to Hugging Face Hub.
    
    Args:
        model: Fine-tuned model
        processor: Processor (tokenizer + feature extractor)
        output_dir: Local output directory
        repo_id: Hugging Face repo ID (e.g., 'username/mms-aln')
        lang: Language code
    """
    print(f"\nPushing to Hugging Face Hub: {repo_id}")
    
    # Login using token from config or environment
    hf_token = config.hf_token
    if hf_token:
        login(token=hf_token)
    else:
        print("  No HF_TOKEN found in .env, attempting login with cached credentials...")
    
    # Create model card (HF requires lowercase ISO for 'language'; BCP-47 in language_bcp47)
    language_yaml = _model_card_language_metadata(lang)
    model_card = f"""---
{language_yaml}
tags:
- audio
- automatic-speech-recognition
- mms
- adapter
license: cc-by-nc-4.0
datasets:
- mozilla-foundation/common_voice_spontaneous_speech
---

# MMS Adapter Fine-tuned for {LANGUAGES.get(lang, lang)}

This model is a fine-tuned version of [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) 
on the Mozilla Common Voice Spontaneous Speech dataset for {LANGUAGES.get(lang, lang)} ({lang}).

## Training

- Base model: facebook/mms-1b-all
- Fine-tuning method: Adapter layers
- Dataset: Mozilla Common Voice Spontaneous Speech

## Usage

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

processor = Wav2Vec2Processor.from_pretrained("{repo_id}")
model = Wav2Vec2ForCTC.from_pretrained("{repo_id}")

# Load adapter
model.load_adapter("{lang}")

# Transcribe audio
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
```
"""
    
    # Save model card
    model_card_path = output_dir / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    # Push model
    print("  Pushing model...")
    model.push_to_hub(repo_id, token=hf_token)
    
    # Push processor
    print("  Pushing processor...")
    processor.push_to_hub(repo_id, token=hf_token)
    
    # Push adapter file
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(lang)
    adapter_path = output_dir / adapter_file
    if adapter_path.exists():
        print("  Pushing adapter weights...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(adapter_path),
            path_in_repo=adapter_file,
            repo_id=repo_id,
            token=hf_token,
        )
    
    # Push model card
    print("  Pushing model card...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=hf_token,
    )
    
    print(f"  Successfully pushed to: https://huggingface.co/{repo_id}")


def main():
    """Main training function."""
    run_start_time = datetime.now()
    args = parse_args()
    if args.lm_order < 1:
        raise ValueError("--lm-order must be >= 1")
    
    # Validate language code
    lang = args.lang
    if lang not in LANGUAGES:
        print(f"ERROR: Unknown language code '{lang}'")
        print(f"Available languages: {', '.join(sorted(LANGUAGES.keys()))}")
        return
    
    # Setup paths (include split so one/mid/all don't overwrite)
    output_dir = Path(args.output_dir) if args.output_dir else config.models_dir / "mms" / lang / args.split
    training_logs_dir = config.results_dir / "training_logs"
    training_logs_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = training_logs_dir / f"mms_aft_{args.split}_{lang}_{run_start_time:%Y%m%d_%H%M%S}.log"
    
    print("=" * 60)
    print(f"MMS Adapter Fine-tuning for {lang} ({LANGUAGES[lang]}) split={args.split}")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device()}")
    print(f"Push to HF: {args.save_to_hf}")
    if args.save_to_hf:
        print(f"HF Repo: {get_hf_repo_id(lang, args.split)}")
    
    ############################################################################
    # Load and preprocess data (train/val from precomputed split TSVs)
    #
    train, val = load_and_preprocess_data(lang, args.split)
    n_train, n_val = len(train), len(val)

    # Snapshot training sentences for LM building (before columns are removed)
    train_sentences: list[str] = train["sentence"]

    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab_path = make_vocab(train, val, lang, output_dir)
    
    # Create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        str(output_dir),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=lang,
    )
    tokenizer.save_pretrained(str(output_dir))
    
    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    
    # Create processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(output_dir))
    
    # Prepare datasets
    print("\nPreparing datasets...")
    prepare_dataset = create_prepare_dataset_fn(processor)
    
    # Keep only necessary columns for training (audio, sentence needed for prepare_dataset)
    train_columns_to_remove = [c for c in train.column_names if c not in ["audio", "sentence"]]
    val_columns_to_remove = [c for c in val.column_names if c not in ["audio", "sentence"]]
    
    # First remove unnecessary columns, then process
    train = train.remove_columns(train_columns_to_remove)
    val = val.remove_columns(val_columns_to_remove)
    
    train = train.map(
        prepare_dataset,
        remove_columns=train.column_names,
        desc="Processing train",
    )
    
    val = val.map(
        prepare_dataset,
        remove_columns=val.column_names,
        desc="Processing validation",
    )

    # Resolve max length: from CSV (p97.5 per language) or from --max-audio-sec
    max_audio_sec = None
    if args.max_audio_sec_from_csv:
        max_audio_sec = _max_sec_from_corpus_csv(Path(args.max_audio_sec_from_csv), lang)
        if max_audio_sec is not None:
            print(f"  Max audio length from CSV (p97.5 for {lang}): {max_audio_sec}s")
    if max_audio_sec is None and args.max_audio_sec is not None:
        max_audio_sec = args.max_audio_sec

    if max_audio_sec is not None:
        max_samples = int(max_audio_sec * processor.feature_extractor.sampling_rate)
        n_train_before, n_val_before = len(train), len(val)
        train = train.filter(lambda x: x["input_length"] <= max_samples, desc="Filter train by length")
        val = val.filter(lambda x: x["input_length"] <= max_samples, desc="Filter val by length")
        n_drop_train = n_train_before - len(train)
        n_drop_val = n_val_before - len(val)
        if n_drop_train or n_drop_val:
            print(f"  Dropped {n_drop_train} train, {n_drop_val} val samples (over {max_audio_sec}s).")
        if len(train) == 0:
            raise ValueError(f"No training samples left after filtering to max {max_audio_sec}s.")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    ############################################################################
    # Load pretrained MMS model, add adapter layers, and freeze the base model.
    #
    print("\nLoading MMS model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    
    # Initialize and configure adapter layers
    model.init_adapter_layers()
    model.freeze_base_model()
    
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    ############################################################################
    # Training configuration
    #
    
    # Determine fp16/bf16 support
    use_fp16 = torch.cuda.is_available()
    use_bf16 = False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {use_fp16}, BF16: {use_bf16}")
    
    ############################################################################
    # Train
    #
    compute_metrics = create_compute_metrics_fn(processor)
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        processing_class=processor.feature_extractor,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    ############################################################################
    # Save adapter weights locally
    #
    print("\nSaving adapter weights...")
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(lang)
    adapter_file_path = output_dir / adapter_file
    
    safe_save_file(model._get_adapters(), str(adapter_file_path), metadata={"format": "pt"})
    print(f"  Adapter saved to: {adapter_file_path}")
    
    # Save the full model locally
    print("  Saving full model...")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    
    # Final evaluation (validation set — greedy decoding)
    print("\nFinal evaluation (validation, greedy):")
    eval_results = trainer.evaluate()
    eval_wer = eval_results.get("eval_wer", float("nan"))
    eval_cer = eval_results.get("eval_cer", float("nan"))
    print(f"  Greedy WER: {eval_wer:.4f}")
    print(f"  Greedy CER: {eval_cer:.4f}")

    # Evaluate on test set if available (greedy)
    test_wer, test_cer = float("nan"), float("nan")
    test_dataset = load_test_data(lang)
    test_pred = None
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nEvaluating on test set (greedy)...")
        cols_to_keep = ["audio", "sentence", "audio_file"]
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in cols_to_keep]
        )
        test_dataset = test_dataset.map(
            prepare_dataset,
            remove_columns=[c for c in test_dataset.column_names if c != "audio_file"],
            desc="Processing test",
        )
        test_pred = trainer.predict(test_dataset)
        test_metrics = compute_metrics(test_pred)
        test_wer = test_metrics["wer"]
        test_cer = test_metrics["cer"]
        print(f"  Test WER (greedy): {test_wer:.4f}")
        print(f"  Test CER (greedy): {test_cer:.4f}")
        eval_results["test_wer"] = test_wer
        eval_results["test_cer"] = test_cer

        if args.save_transcriptions:
            pred_ids = np.argmax(test_pred.predictions, axis=-1)
            test_pred.label_ids[test_pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(test_pred.label_ids, group_tokens=False)
            pred_str = [p.strip() for p in pred_str]
            label_str = [l.strip() for l in label_str]
            audio_files = test_dataset["audio_file"]
            trans_dir = config.results_dir / "transcriptions"
            trans_dir.mkdir(parents=True, exist_ok=True)
            run_ts = run_start_time.strftime("%Y%m%d_%H%M%S")
            trans_path = trans_dir / f"transcriptions_mms_{lang}_{args.split}_{run_ts}.tsv"
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write("audio_file\treference\thypothesis\n")
                for af, ref, hyp in zip(audio_files, label_str, pred_str):
                    f.write(f"{af}\t{ref}\t{hyp}\n")
            print(f"  Transcriptions saved to: {trans_path}")

    ############################################################################
    # LM-decoded evaluation (only after training, does not affect training loop)
    lm_val_wer, lm_val_cer = float("nan"), float("nan")
    lm_test_wer, lm_test_cer = float("nan"), float("nan")
    lm_desc = "disabled"

    if args.use_lm:
        print("\nBuilding LM decoder...")
        lm_processor, lm_desc = _build_lm_processor(
            processor,
            train_sentences,
            output_dir,
            arpa_path=args.lm_arpa_path,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
            lm_order=args.lm_order,
        )
        if lm_processor is None:
            print(f"  WARNING: Could not build LM processor ({lm_desc}). Skipping LM evaluation.")
        else:
            print(f"  LM decoder: {lm_desc} (beam_width={args.beam_width})")

            # LM eval on validation
            print("\nFinal evaluation (validation, LM decoding)...")
            val_pred = trainer.predict(val)
            lm_pred_str, lm_label_str = _decode_with_lm(
                val_pred.predictions, val_pred.label_ids, processor, lm_processor, args.beam_width
            )
            lm_val_metrics = compute_wer_cer(lm_pred_str, lm_label_str)
            lm_val_wer = lm_val_metrics["wer"]
            lm_val_cer = lm_val_metrics["cer"]
            print(f"  LM WER: {lm_val_wer:.4f}")
            print(f"  LM CER: {lm_val_cer:.4f}")
            eval_results["eval_lm_wer"] = lm_val_wer
            eval_results["eval_lm_cer"] = lm_val_cer

            # LM eval on test
            if test_pred is not None:
                print("\nEvaluating on test set (LM decoding)...")
                lm_test_pred_str, lm_test_label_str = _decode_with_lm(
                    test_pred.predictions, test_pred.label_ids, processor, lm_processor, args.beam_width
                )
                lm_test_metrics = compute_wer_cer(lm_test_pred_str, lm_test_label_str)
                lm_test_wer = lm_test_metrics["wer"]
                lm_test_cer = lm_test_metrics["cer"]
                print(f"  Test LM WER: {lm_test_wer:.4f}")
                print(f"  Test LM CER: {lm_test_cer:.4f}")
                eval_results["test_lm_wer"] = lm_test_wer
                eval_results["test_lm_cer"] = lm_test_cer

                if args.save_transcriptions:
                    lm_pred_str_raw = lm_processor.batch_decode(
                        test_pred.predictions, beam_width=args.beam_width
                    ).text
                    lm_pred_str_raw = [p.strip() for p in lm_pred_str_raw]
                    label_ids_cp = test_pred.label_ids.copy()
                    label_ids_cp[label_ids_cp == -100] = processor.tokenizer.pad_token_id
                    lm_label_str_raw = processor.batch_decode(label_ids_cp, group_tokens=False)
                    lm_label_str_raw = [l.strip() for l in lm_label_str_raw]
                    audio_files = test_dataset["audio_file"]
                    trans_dir = config.results_dir / "transcriptions"
                    trans_dir.mkdir(parents=True, exist_ok=True)
                    run_ts = run_start_time.strftime("%Y%m%d_%H%M%S")
                    lm_trans_path = trans_dir / f"transcriptions_mms_lm_{lang}_{args.split}_{run_ts}.tsv"
                    with open(lm_trans_path, "w", encoding="utf-8") as fh:
                        fh.write("audio_file\treference\thypothesis\n")
                        for af, ref, hyp in zip(audio_files, lm_label_str_raw, lm_pred_str_raw):
                            fh.write(f"{af}\t{ref}\t{hyp}\n")
                    print(f"  LM transcriptions saved to: {lm_trans_path}")

            # Save the LM decoder artifacts alongside the model
            lm_processor.save_pretrained(str(output_dir))

    # Save eval results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Write training log (datetime-named) for future reference
    run_end_time = datetime.now()
    log_lines = [
        f"# MMS Adapter Fine-tuning Run Log",
        f"run_start={run_start_time.isoformat()}",
        f"run_end={run_end_time.isoformat()}",
        f"model=facebook/mms-1b-all",
        f"language={lang}",
        f"split={args.split}",
        f"language_name={LANGUAGES.get(lang, lang)}",
        f"output_dir={output_dir}",
        f"num_train_samples={n_train}",
        f"num_validation_samples={n_val}",
        f"num_epochs={args.num_epochs}",
        f"batch_size={args.batch_size}",
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"effective_batch_size={args.batch_size * args.gradient_accumulation_steps}",
        f"learning_rate={args.learning_rate}",
        f"validation_wer={eval_wer:.6f}",
        f"validation_cer={eval_cer:.6f}",
        f"test_wer={test_wer:.6f}",
        f"test_cer={test_cer:.6f}",
        f"use_lm={args.use_lm}",
        f"lm_order={args.lm_order}",
        f"lm_beam_width={args.beam_width}",
        f"lm_alpha={args.lm_alpha}",
        f"lm_beta={args.lm_beta}",
        f"lm_decoder={lm_desc}",
        f"validation_lm_wer={lm_val_wer:.6f}",
        f"validation_lm_cer={lm_val_cer:.6f}",
        f"test_lm_wer={lm_test_wer:.6f}",
        f"test_lm_cer={lm_test_cer:.6f}",
    ]
    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
        log_lines.append(f"best_validation_wer={trainer.state.best_metric:.6f}")
    log_lines.append("")
    with open(training_log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"  Training log saved to: {training_log_path}")
    
    ############################################################################
    # Push to Hugging Face Hub (if requested)
    #
    if args.save_to_hf:
        repo_id = get_hf_repo_id(lang, args.split)
        push_to_hub(model, processor, output_dir, repo_id, lang)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    if args.save_to_hf:
        print(f"Model pushed to: https://huggingface.co/{get_hf_repo_id(lang, args.split)}")


if __name__ == "__main__":
    main()
