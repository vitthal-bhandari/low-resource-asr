"""
Script for training adapter layers as a means of fine-tuning the MMS model for ASR.

The output of the script is:
- A tokenizer for this specific language
- Adapter layers that are fine-tuned on your data

For more information about this process and adapter fine tuning, see:
https://huggingface.co/blog/mms_adapters

Usage:
    python -m src.training.aft_mms aln
    python -m src.training.aft_mms aln --num-epochs 10 --batch-size 4
    python -m src.training.aft_mms aln --save-to-hf --hf-repo-id username/mms-aln
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If set, fixes torch/numpy/python seeds.",
    )
    return parser.parse_args()


# Hugging Face configuration
HF_USERNAME = "vitthalbhandari"


def get_hf_repo_id(lang: str, split: str = "all") -> str:
    """Get the Hugging Face repo ID for a language and split."""
    return f"{HF_USERNAME}/mms-1b-all-aft-{split}-{lang}"


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

        # Ensure attention_mask is present (Wav2Vec2FeatureExtractor with return_attention_mask=True
        # may include it; if not, build from input_length so the model ignores padded positions)
        if "attention_mask" not in batch:
            lengths = [f["input_length"] for f in features]
            max_len = batch["input_values"].shape[1]
            attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)
            for i, L in enumerate(lengths):
                attention_mask[i, :L] = 1
            batch["attention_mask"] = attention_mask

        # One-time print to verify batch contents during a test run
        if not getattr(self, "_logged_batch_keys", False):
            print("  DataCollator batch.keys():", list(batch.keys()))
            self._logged_batch_keys = True

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


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and python."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    run_start_time = datetime.now()
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

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
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    ############################################################################
    # Load pretrained MMS model, add adapter layers, and freeze the base model.
    #
    print("\nLoading MMS model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    
    # Initialize and configure adapter layers
    model.init_adapter_layers()
    model.freeze_base_model()

    # lm_head is randomly reinitialized (ignore_mismatched_sizes=True) and must be trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    # Verify lm_head is trainable (frozen lm_head would prevent learning output distribution)
    print("  lm_head trainable:", all(p.requires_grad for p in model.lm_head.parameters()))

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
        warmup_ratio=0.1,
        weight_decay=0.005,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        seed=args.seed if args.seed is not None else 42,
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
        # Keep audio_file so we can save it in transcriptions TSV when --save-transcriptions
        cols_to_keep = ["audio", "sentence", "audio_file"]
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in cols_to_keep]
        )
        # Remove only columns consumed by prepare_dataset so audio_file remains for export
        test_dataset = test_dataset.map(
            prepare_dataset,
            remove_columns=[c for c in test_dataset.column_names if c != "audio_file"],
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

        # Optionally save gold and model transcriptions for analysis
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
            trans_path = trans_dir / f"transcriptions_{lang}_{args.split}_{run_ts}.tsv"
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write("audio_file\treference\thypothesis\n")
                for af, ref, hyp in zip(audio_files, label_str, pred_str):
                    f.write(f"{af}\t{ref}\t{hyp}\n")
            print(f"  Transcriptions saved to: {trans_path}")
    
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
        f"seed={args.seed if args.seed is not None else 42}",
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
