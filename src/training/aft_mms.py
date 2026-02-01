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
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset
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
    
    return parser.parse_args()


# Hugging Face configuration
HF_USERNAME = "vitthalbhandari"


def get_hf_repo_id(lang: str) -> str:
    """Get the Hugging Face repo ID for a language."""
    return f"{HF_USERNAME}/mms-1b-all-aft-{lang}"


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


def load_and_preprocess_data(lang: str) -> tuple[Dataset, Dataset]:
    """
    Load and preprocess train and validation datasets from TSV files.
    
    Args:
        lang: Language ISO code (e.g., 'aln')
        
    Returns:
        Tuple of (train_dataset, val_dataset) as Hugging Face Datasets
    """
    # Paths
    lang_dir = config.mozilla_data_dir / lang
    tsv_path = lang_dir / f"ss-corpus-{lang}.tsv"
    audio_dir = config.mozilla_data_dir / "shared_train_validation_audios"
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    print(f"\nLoading data for {lang} ({LANGUAGES.get(lang, 'Unknown')})...")
    print(f"  TSV: {tsv_path}")
    print(f"  Audio dir: {audio_dir}")
    
    # Load TSV
    df = pd.read_csv(tsv_path, sep="\t")
    original_count = len(df)
    print(f"  Total rows in TSV: {original_count}")
    
    # Check required columns
    required_cols = ["audio_file", "transcription", "duration_ms", "split"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with duration_ms = 0 or NaN
    df = df[df["duration_ms"] > 0]
    filtered_count = len(df)
    excluded = original_count - filtered_count
    if excluded > 0:
        print(f"  Excluded {excluded} rows with duration_ms = 0")
    
    # Filter out rows with empty transcription
    df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]
    empty_transcript_excluded = filtered_count - len(df)
    if empty_transcript_excluded > 0:
        print(f"  Excluded {empty_transcript_excluded} rows with empty transcription")
    
    # Clean transcriptions
    df["transcription"] = df["transcription"].apply(clean_transcript)
    
    # Filter out rows that became empty after cleaning
    df = df[df["transcription"].str.strip() != ""]
    
    # Print statistics by split
    print(f"\n  Dataset statistics:")
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name]
        total_ms = split_df["duration_ms"].sum()
        total_hours = total_ms / (1000 * 60 * 60)
        print(f"    {split_name}: {len(split_df)} samples, {total_hours:.2f} hours")
    
    # Split into train and validation
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "dev"].copy()
    
    print(f"\n  Final counts:")
    print(f"    Train: {len(train_df)} samples")
    print(f"    Validation: {len(val_df)} samples")
    
    # Add full audio path
    train_df["audio"] = train_df["audio_file"].apply(lambda x: str(audio_dir / x))
    val_df["audio"] = val_df["audio_file"].apply(lambda x: str(audio_dir / x))
    
    # Rename transcription to sentence for consistency
    train_df = train_df.rename(columns={"transcription": "sentence"})
    val_df = val_df.rename(columns={"transcription": "sentence"})
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    
    # Cast audio column to Audio type
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    
    return train_dataset, val_dataset


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
    
    Args:
        processor: Wav2Vec2Processor to use for feature extraction
        
    Returns:
        Function that prepares a single dataset example
    """

    def prepare_dataset(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
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

    def compute_metrics(pred) -> dict:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # We do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        # Normalize case for fair WER comparison
        pred_str = [p.lower() for p in pred_str]
        label_str = [l.lower() for l in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
    
    # Create model card
    model_card = f"""---
language: {lang}
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
    model.push_to_hub(repo_id, use_auth_token=hf_token)
    
    # Push processor
    print("  Pushing processor...")
    processor.push_to_hub(repo_id, use_auth_token=hf_token)
    
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
    args = parse_args()
    
    # Validate language code
    lang = args.lang
    if lang not in LANGUAGES:
        print(f"ERROR: Unknown language code '{lang}'")
        print(f"Available languages: {', '.join(sorted(LANGUAGES.keys()))}")
        return
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else config.models_dir / "mms" / lang
    
    print("=" * 60)
    print(f"MMS Adapter Fine-tuning for {lang} ({LANGUAGES[lang]})")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device()}")
    print(f"Push to HF: {args.save_to_hf}")
    if args.save_to_hf:
        print(f"HF Repo: {get_hf_repo_id(lang)}")
    
    ############################################################################
    # Load and preprocess data
    #
    train, val = load_and_preprocess_data(lang)
    
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
        save_steps=200,
        eval_steps=100,
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        save_total_limit=2,
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
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    print(f"  WER: {eval_results['eval_wer']:.4f}")
    
    # Save eval results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    ############################################################################
    # Push to Hugging Face Hub (if requested)
    #
    if args.save_to_hf:
        repo_id = get_hf_repo_id(lang)
        push_to_hub(model, processor, output_dir, repo_id, lang)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    if args.save_to_hf:
        print(f"Model pushed to: https://huggingface.co/{get_hf_repo_id(lang)}")


if __name__ == "__main__":
    main()
