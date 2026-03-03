"""
Script for fine-tuning Whisper large-v3 for ASR on low-resource languages.

Whisper is an encoder-decoder (seq2seq) model that maps log-Mel spectrogram
features to text tokens via cross-entropy loss.  Unlike CTC-based models
(MMS, XLS-R), it has an internal language model via its decoder, so external
LM decoding (pyctcdecode / KenLM) is neither needed nor applicable.

Key differences from the CTC-based aft_mms / aft_xlsr scripts:
  - Uses WhisperForConditionalGeneration (encoder-decoder) instead of
    Wav2Vec2ForCTC (encoder + CTC head).
  - Uses Seq2SeqTrainer with predict_with_generate=True for evaluation.
  - The pre-trained WhisperTokenizer is used directly (no per-language
    vocabulary needs to be built).
  - The feature extractor produces fixed-length 30-second log-Mel
    spectrograms (padding/truncation is handled automatically).
  - Predictions are generated token IDs, decoded with skip_special_tokens.

The output of the script is:
  - A fine-tuned Whisper model checkpoint
  - WhisperProcessor (feature extractor + tokenizer) saved alongside

Usage:
    python -m src.training.aft_whisper aln
    python -m src.training.aft_whisper aln --num-epochs 10 --batch-size 4
    python -m src.training.aft_whisper aln --freeze-encoder
    python -m src.training.aft_whisper aln --save-to-hf
    python -m src.training.aft_whisper aln --whisper-language albanian
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import librosa
from datasets import Dataset
from evaluate import load as load_metric
from huggingface_hub import HfApi, login
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from src.config import config
from src.data.download import LANGUAGES

BASE_MODEL = "openai/whisper-large-v3"

# Best-effort mapping from project language codes to Whisper-supported language
# names.  For languages without a close match the fallback is "english" (the
# model learns to associate the language token with the actual language data
# during fine-tuning, so the specific token matters less than consistency
# between training and inference).
WHISPER_LANGUAGE_MAP: dict[str, str] = {
    "aln": "albanian",
    "el-CY": "greek",
    "sco": "english",
    "bew": "indonesian",
    "pne": "malay",
}

WHISPER_LANGUAGE_FALLBACK = "english"

HF_USERNAME = "vitthalbhandari"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper large-v3 for ASR"
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
        help="Output directory for model (default: models/whisper/{lang}/{split})",
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
        default=1e-5,
        help="Learning rate (default: 1e-5, lower than adapter tuning)",
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
        help="Save test-set gold and model transcriptions to a TSV file.",
    )
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="Drop train/val samples longer than this (seconds). "
             "Whisper truncates to 30s internally; setting <=30 avoids silent truncation. "
             "Ignored if --max-audio-sec-from-csv is set.",
    )
    parser.add_argument(
        "--max-audio-sec-from-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to corpus_stats CSV with columns lang,p97_5_sec.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder and only fine-tune the decoder. "
             "Reduces memory and compute; useful when GPU memory is limited.",
    )
    parser.add_argument(
        "--whisper-language",
        type=str,
        default=None,
        help="Override the Whisper language token (e.g., 'albanian', 'greek'). "
             "Defaults to an auto-mapped value or 'english' for unmapped languages.",
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=225,
        help="Maximum number of tokens to generate during evaluation (default: 225).",
    )
    return parser.parse_args()


def get_hf_repo_id(lang: str, split: str = "all") -> str:
    """Get the Hugging Face repo ID for a language and split."""
    return f"{HF_USERNAME}/whisper-large-v3-aft-{split}-{lang}"


def get_whisper_language(lang: str, override: str | None = None) -> str:
    """
    Resolve the Whisper language token to use for a given project language code.

    Priority: CLI override > WHISPER_LANGUAGE_MAP > fallback ("english").
    """
    if override is not None:
        return override
    return WHISPER_LANGUAGE_MAP.get(lang, WHISPER_LANGUAGE_FALLBACK)


def _max_sec_from_corpus_csv(csv_path: Path, lang: str) -> float | None:
    """
    Read corpus_stats CSV (columns lang, p97_5_sec) and return p97.5 seconds
    for the given language.  Returns None if unavailable.
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
# Data loading and cleaning (shared logic with aft_mms / aft_xlsr)

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
    Run scripts/create_splits.py --all first to generate the TSV files.
    """
    lang_dir = config.mozilla_data_dir / lang
    train_tsv = lang_dir / f"train-{split}_{lang}.tsv"
    val_tsv = lang_dir / f"validation_{lang}.tsv"
    audio_dir = config.mozilla_data_dir / "shared_train_validation_audios"

    if not train_tsv.exists():
        raise FileNotFoundError(
            f"Train split TSV not found: {train_tsv}. "
            "Run: uv run python scripts/create_splits.py --all"
        )
    if not val_tsv.exists():
        raise FileNotFoundError(
            f"Validation TSV not found: {val_tsv}. "
            "Run: uv run python scripts/create_splits.py --all"
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
    train_df = train_df[
        train_df["transcription"].notna() & (train_df["transcription"].str.strip() != "")
    ]
    val_df = val_df[val_df["duration_ms"] > 0]
    val_df = val_df[
        val_df["transcription"].notna() & (val_df["transcription"].str.strip() != "")
    ]

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
    """Load test set from test_{lang}.tsv if present."""
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
    test_df = test_df[
        test_df["transcription"].notna() & (test_df["transcription"].str.strip() != "")
    ]
    test_df["transcription"] = test_df["transcription"].apply(clean_transcript)
    test_df = test_df[test_df["transcription"].str.strip() != ""]
    test_df["audio"] = [{"path": str(audio_dir / x)} for x in test_df["audio_file"]]
    test_df = test_df.rename(columns={"transcription": "sentence"})
    return Dataset.from_pandas(test_df, preserve_index=False)


################################################################################
# Whisper-specific data collator, dataset preparation, and metrics


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper seq2seq training.

    Input features (log-Mel spectrograms) are already fixed-length (30s) so
    only need conversion to tensors.  Labels are variable-length token
    sequences that need padding; padding positions are replaced with -100 so
    they are ignored by the cross-entropy loss.
    """

    processor: Any
    decoder_start_token_id: int
    input_dtype: torch.dtype | None = None

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        if self.input_dtype is not None:
            batch["input_features"] = batch["input_features"].to(dtype=self.input_dtype)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if prepended by tokenizer (it is added by the model)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def create_prepare_dataset_fn(processor: WhisperProcessor):
    """
    Create a dataset preparation function with the processor in closure.

    Loads audio from path with soundfile/librosa (avoids torchcodec/FFmpeg
    issues on HPC).  Produces log-Mel spectrogram input features and
    tokenised label IDs.
    """
    TARGET_SR = 16_000
    # Whisper-large-v3 has a maximum target length (e.g. 448 tokens).  We
    # truncate labels to the tokenizer's configured max length to avoid
    # "Labels' sequence length ... cannot exceed the maximum allowed length"
    # errors on very long transcriptions.
    max_label_length = getattr(processor.tokenizer, "model_max_length", None)
    if max_label_length is None or max_label_length > 1024:
        # Fall back to a safe default if the tokenizer does not define it
        max_label_length = 448

    def prepare_dataset(batch: dict) -> dict:
        path = batch["audio"]["path"]
        array, sr = sf.read(path, dtype="float32")
        if sr != TARGET_SR:
            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=sr
        ).input_features[0]

        # Raw sample count for optional length-based filtering
        batch["input_length"] = len(array)

        tokenized = processor.tokenizer(
            batch["sentence"],
            max_length=max_label_length,
            truncation=True,
        )
        batch["labels"] = tokenized.input_ids
        return batch

    return prepare_dataset


def create_compute_metrics_fn(processor: WhisperProcessor):
    """
    Create a metrics computation function for seq2seq Whisper evaluation.

    With predict_with_generate=True the predictions are already generated
    token IDs (not logits), so we decode them directly.

    Args:
        processor: WhisperProcessor for decoding predictions

    Returns:
        Function that computes WER and CER metrics
    """
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        pred_str = [p.lower() for p in pred_str]
        label_str = [l.lower() for l in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


################################################################################
# Utility helpers


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
    Return YAML lines for model card 'language' field.
    HF requires lowercase ISO 639-1/2/3; BCP-47 codes go in language_bcp47.
    """
    lang_lower = lang.lower()
    if "-" in lang_lower:
        base = lang_lower.split("-")[0]
        return f"language: {base}\nlanguage_bcp47:\n- {lang_lower}"
    return f"language: {lang_lower}"


def push_to_hub(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    output_dir: Path,
    repo_id: str,
    lang: str,
    whisper_language: str,
) -> None:
    """
    Push the fine-tuned Whisper model and processor to Hugging Face Hub.

    Args:
        model: Fine-tuned model
        processor: WhisperProcessor (feature extractor + tokenizer)
        output_dir: Local output directory
        repo_id: Hugging Face repo ID
        lang: Project language code
        whisper_language: Whisper language token used during training
    """
    print(f"\nPushing to Hugging Face Hub: {repo_id}")

    hf_token = config.hf_token
    if hf_token:
        login(token=hf_token)
    else:
        print("  No HF_TOKEN found in .env, attempting login with cached credentials...")

    language_yaml = _model_card_language_metadata(lang)
    model_card = f"""---
{language_yaml}
tags:
- audio
- automatic-speech-recognition
- whisper
license: cc-by-nc-4.0
datasets:
- mozilla-foundation/common_voice_spontaneous_speech
---

# Whisper Large-v3 Fine-tuned for {LANGUAGES.get(lang, lang)}

This model is a fine-tuned version of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
on the Mozilla Common Voice Spontaneous Speech dataset for {LANGUAGES.get(lang, lang)} ({lang}).

## Training

- Base model: {BASE_MODEL}
- Fine-tuning method: Full fine-tuning (seq2seq cross-entropy)
- Whisper language token: {whisper_language}
- Dataset: Mozilla Common Voice Spontaneous Speech

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

processor = WhisperProcessor.from_pretrained("{repo_id}")
model = WhisperForConditionalGeneration.from_pretrained("{repo_id}")

inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(**inputs)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
```
"""

    model_card_path = output_dir / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)

    print("  Pushing model...")
    model.push_to_hub(repo_id, token=hf_token)

    print("  Pushing processor...")
    processor.push_to_hub(repo_id, token=hf_token)

    print("  Pushing model card...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=hf_token,
    )

    print(f"  Successfully pushed to: https://huggingface.co/{repo_id}")


################################################################################
# Main


def main():
    """Main training function."""
    run_start_time = datetime.now()
    args = parse_args()

    lang = args.lang
    if lang not in LANGUAGES:
        print(f"ERROR: Unknown language code '{lang}'")
        print(f"Available languages: {', '.join(sorted(LANGUAGES.keys()))}")
        return

    whisper_language = get_whisper_language(lang, args.whisper_language)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else config.models_dir / "whisper" / lang / args.split
    )
    training_logs_dir = config.results_dir / "training_logs"
    training_logs_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = (
        training_logs_dir
        / f"whisper_aft_{args.split}_{lang}_{run_start_time:%Y%m%d_%H%M%S}.log"
    )

    print("=" * 60)
    print(f"Whisper Large-v3 Fine-tuning for {lang} ({LANGUAGES[lang]}) split={args.split}")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Whisper language token: {whisper_language}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device()}")
    print(f"Push to HF: {args.save_to_hf}")
    if args.save_to_hf:
        print(f"HF Repo: {get_hf_repo_id(lang, args.split)}")

    ############################################################################
    # Load and preprocess data
    train, val = load_and_preprocess_data(lang, args.split)
    n_train, n_val = len(train), len(val)

    ############################################################################
    # Load Whisper processor (feature extractor + tokenizer)
    print(f"\nLoading WhisperProcessor (language={whisper_language})...")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language=whisper_language, task="transcribe"
    )

    ############################################################################
    # Prepare datasets
    print("\nPreparing datasets...")
    prepare_dataset = create_prepare_dataset_fn(processor)

    train_columns_to_remove = [
        c for c in train.column_names if c not in ["audio", "sentence"]
    ]
    val_columns_to_remove = [
        c for c in val.column_names if c not in ["audio", "sentence"]
    ]
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

    # Optional length-based filtering (raw audio samples)
    max_audio_sec = None
    if args.max_audio_sec_from_csv:
        max_audio_sec = _max_sec_from_corpus_csv(
            Path(args.max_audio_sec_from_csv), lang
        )
        if max_audio_sec is not None:
            print(f"  Max audio length from CSV (p97.5 for {lang}): {max_audio_sec}s")
    if max_audio_sec is None and args.max_audio_sec is not None:
        max_audio_sec = args.max_audio_sec

    if max_audio_sec is not None:
        sampling_rate = processor.feature_extractor.sampling_rate
        max_samples = int(max_audio_sec * sampling_rate)
        n_train_before, n_val_before = len(train), len(val)
        train = train.filter(
            lambda x: x["input_length"] <= max_samples,
            desc="Filter train by length",
        )
        val = val.filter(
            lambda x: x["input_length"] <= max_samples,
            desc="Filter val by length",
        )
        n_drop_train = n_train_before - len(train)
        n_drop_val = n_val_before - len(val)
        if n_drop_train or n_drop_val:
            print(
                f"  Dropped {n_drop_train} train, {n_drop_val} val samples "
                f"(over {max_audio_sec}s)."
            )
        if len(train) == 0:
            raise ValueError(
                f"No training samples left after filtering to max {max_audio_sec}s."
            )

    ############################################################################
    # Load pretrained Whisper model in full float32.
    # Mixed precision (fp16/bf16) has caused dtype mismatches between
    # input features and convolution biases on some GPUs, so we keep the
    # training numerically stable and simple by default. If you want to
    # enable bf16/fp16 later, make sure to cast input_features to the
    # same dtype as model.dtype in the data collator.
    print(f"\nLoading Whisper model ({BASE_MODEL}) in float32...")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )

    model.generation_config.language = whisper_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
        print("  Encoder frozen — only decoder parameters are trainable.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    ############################################################################
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_dtype=model.dtype,
    )

    ############################################################################
    # Training configuration (pure float32 to avoid dtype mismatch issues)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
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
    print("  Precision: float32 (no AMP)")
    print(f"  Generation max length: {args.generation_max_length}")

    ############################################################################
    # Train
    compute_metrics = create_compute_metrics_fn(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\nStarting training...")
    trainer.train()

    ############################################################################
    # Save model and processor
    print("\nSaving model and processor...")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"  Model saved to: {output_dir}")

    # Final evaluation (validation set)
    print("\nFinal evaluation (validation):")
    eval_results = trainer.evaluate()
    eval_wer = eval_results.get("eval_wer", float("nan"))
    eval_cer = eval_results.get("eval_cer", float("nan"))
    print(f"  WER: {eval_wer:.4f}")
    print(f"  CER: {eval_cer:.4f}")

    # Evaluate on test set if available
    test_wer, test_cer = float("nan"), float("nan")
    test_dataset = load_test_data(lang)
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nEvaluating on test set...")
        cols_to_keep = ["audio", "sentence", "audio_file"]
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in cols_to_keep]
        )
        test_dataset = test_dataset.map(
            prepare_dataset,
            remove_columns=[
                c for c in test_dataset.column_names if c != "audio_file"
            ],
            desc="Processing test",
        )
        test_pred = trainer.predict(test_dataset)
        test_metrics = compute_metrics(test_pred)
        test_wer = test_metrics["wer"]
        test_cer = test_metrics["cer"]
        print(f"  Test WER: {test_wer:.4f}")
        print(f"  Test CER: {test_cer:.4f}")
        eval_results["test_wer"] = test_wer
        eval_results["test_cer"] = test_cer

        if args.save_transcriptions:
            pred_ids = test_pred.predictions
            label_ids = test_pred.label_ids.copy()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )
            label_str = processor.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )
            pred_str = [p.strip() for p in pred_str]
            label_str = [l.strip() for l in label_str]
            audio_files = test_dataset["audio_file"]
            trans_dir = config.results_dir / "transcriptions"
            trans_dir.mkdir(parents=True, exist_ok=True)
            run_ts = run_start_time.strftime("%Y%m%d_%H%M%S")
            trans_path = (
                trans_dir
                / f"transcriptions_whisper_{lang}_{args.split}_{run_ts}.tsv"
            )
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write("audio_file\treference\thypothesis\n")
                for af, ref, hyp in zip(audio_files, label_str, pred_str):
                    f.write(f"{af}\t{ref}\t{hyp}\n")
            print(f"  Transcriptions saved to: {trans_path}")

    # Save eval results JSON
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # Write training log
    run_end_time = datetime.now()
    log_lines = [
        "# Whisper Large-v3 Fine-tuning Run Log",
        f"run_start={run_start_time.isoformat()}",
        f"run_end={run_end_time.isoformat()}",
        f"model={BASE_MODEL}",
        f"whisper_language={whisper_language}",
        f"freeze_encoder={args.freeze_encoder}",
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
    ]
    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
        log_lines.append(f"best_validation_wer={trainer.state.best_metric:.6f}")
    log_lines.append("")
    with open(training_log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"  Training log saved to: {training_log_path}")

    ############################################################################
    # Push to Hugging Face Hub (if requested)
    if args.save_to_hf:
        repo_id = get_hf_repo_id(lang, args.split)
        push_to_hub(model, processor, output_dir, repo_id, lang, whisper_language)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    if args.save_to_hf:
        print(
            f"Model pushed to: https://huggingface.co/{get_hf_repo_id(lang, args.split)}"
        )


if __name__ == "__main__":
    main()
