"""
Script for training adapter layers as a means of fine-tuning the MMS model for ASR.

The output of the script is:
- A tokenizer for this specific language
- Adapter layers that are fine-tuned on your data

The expected input format is a directory with:
    train/
        audios/
        metadata.csv
    validation/
        audios/
        metadata.csv

The metadata.csv should have minimally two columns:
- "file_name" (prepended with "audios/")
- "sentence"

For more information about this process and adapter fine tuning, see:
https://huggingface.co/blog/mms_adapters

Usage:
    python -m src.training.aft_mms <path/to/data/> <lang_code> [--output-dir <dir>]
    
Example:
    python -m src.training.aft_mms data/mozilla_speech_data/aln aln --output-dir models/mms
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Audio, load_dataset
from evaluate import load as load_metric
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS model with adapter layers for ASR"
    )
    parser.add_argument(
        "path_to_data",
        type=str,
        help="Path to the data directory containing train/ and validation/ folders",
    )
    parser.add_argument(
        "target_lang",
        type=str,
        help="Target language code (e.g., 'aln' for Gheg Albanian)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model and tokenizer (default: mms-1b+adapter-ft_{lang})",
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
    return parser.parse_args()


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
    return text.strip().lower()


def preprocess(batch: dict) -> dict:
    """Preprocess a batch by cleaning the transcript."""
    batch["sentence"] = clean_transcript(batch["sentence"])
    return batch


def load_data(path_to_data: str) -> tuple:
    """
    Load and preprocess train and validation datasets.
    
    Args:
        path_to_data: Path to data directory with train/ and validation/ folders
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data = load_dataset("audiofolder", data_dir=path_to_data)
    train = data["train"]
    train = train.map(preprocess)
    val = data["validation"]
    val = val.map(preprocess)
    return train, val


def extract_all_chars(batch: dict) -> dict:
    """Extract all unique characters from a batch of sentences."""
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def make_vocab(
    train_data,
    val_data,
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
        remove_columns=train_data.column_names,  # Fixed: was using global 'train'
    )
    vocab_val = val_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=val_data.column_names,  # Fixed: was using global 'val'
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Replace space with word delimiter
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
    
    print(f"Vocabulary saved to {vocab_path} ({len(vocab_dict)} tokens)")
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

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup paths
    path_to_data = args.path_to_data
    target_lang = args.target_lang
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"mms-1b+adapter-ft_{target_lang}")
    
    print("=" * 60)
    print(f"MMS Adapter Fine-tuning for {target_lang}")
    print("=" * 60)
    print(f"Data path: {path_to_data}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device()}")
    print()
    
    ############################################################################
    # Load data, process it, create tokenizer/processor
    #
    print("Loading data...")
    train, val = load_data(path_to_data)
    print(f"  Train samples: {len(train)}")
    print(f"  Validation samples: {len(val)}")
    
    # Create vocabulary
    print("\nBuilding vocabulary...")
    vocab_path = make_vocab(train, val, target_lang, output_dir)
    
    # Create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        str(output_dir),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=target_lang,
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
    
    train = train.cast_column("audio", Audio(sampling_rate=16_000))
    train = train.map(
        prepare_dataset,
        remove_columns=train.column_names,
        desc="Processing train",
    )
    
    val = val.cast_column("audio", Audio(sampling_rate=16_000))
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
        report_to="none",  # Disable wandb/tensorboard by default
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
    # Save adapter weights
    #
    print("\nSaving adapter weights...")
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    adapter_file = os.path.join(str(output_dir), adapter_file)
    
    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
    print(f"  Adapter saved to: {adapter_file}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    print(f"  WER: {eval_results['eval_wer']:.4f}")
    
    # Save eval results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    print("\nTraining complete!")
    print(f"Model and tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    main()