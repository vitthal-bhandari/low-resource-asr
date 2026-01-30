"""
Script for training adapter layers as a means of fine-tuning the MMS model for ASR. The output of the script is
- a tokenizer for this specific language
- adapter layers that are fine-tuned on your data

The expected input format is a directory with 

- train
- - audios
- - metadata.csv
-validation
- - audios
- - metadata.csv

The metadata.csv should have minimally two columns, "file_name" (prepended with "audios/") and "sentence".


For more information about this process and adapter fine tuning, this blog post is very informative (and a lot of the code below 
was adapted from it): https://huggingface.co/blog/mms_adapters

"""
import os
import sys
import re
import json
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, 
                          Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, 
                          Trainer)
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import torch
from evaluate import load
from safetensors.torch import save_file as safe_save_file

#
# Expect the program to be called as 
# $ python adapter_ft_mms.py <path/to/data/> <lang_code>
#
path_to_data = sys.argv[1]
target_lang = sys.argv[2]
repo_name = f"mms-1b+adapter-ft_{target_lang}"


################################################################################
# helper functions for loading and cleaning dataset and defining the vocabulary.
#
bracketed = re.compile(r"\[[^\]]+\]")
unintell_paren = re.compile(r"\(\?+\)")
repl_punc = re.compile('[,?¿¡!";:]+')
multispace = re.compile("  +")
def clean_transcript(t):
    t = re.sub(bracketed, " ", t)
    t = re.sub(unintell_paren, " ", t)
    t = t.replace(" ... ", " ")
    t = t.replace("#x27;", "'")
    t = re.sub(repl_punc, " ", t)
    t = (t
         .replace("...", "!ELLIPSIS!")
         .replace(".", " ")
         .replace("!ELLIPSIS!", "...")
    )
    t = re.sub(multispace, " ", t)
    return t


def preprocess(batch):
    #
    # Could add more preprocessing steps here...
    #
    batch["sentence"] = clean_transcript(batch["sentence"])
    return batch


def load_data(path_to_data):
    data = load_dataset("audiofolder", data_dir=path_to_data)
    train = data["train"]
    train = train.map(preprocess)
    val = data["validation"]
    val = val.map(preprocess)
    return train, val


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def make_vocab(train_data, val_data):
    """
    builds vocabulary from train and dev sets (maybe its better to exclude 
    dev vocab here but :shrug: we will not have the val vocab when getting 
    eval numbers on val).
    """
    vocab_train = train_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train.column_names
    )
    vocab_val = val_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=val.column_names
    )

    vocab_list = list(
        set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0])
    )
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    new_vocab_dict = {target_lang: vocab_dict}
    with open('vocab.json', 'w') as vocab_file:
        json.dump(new_vocab_dict, vocab_file)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
            self, 
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
            ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} 
                          for feature in features]
        
        label_features = [{"input_ids": feature["labels"]} 
                          for feature in features]
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

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), 
            -100
        )
        batch["labels"] = labels
        return batch

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

wer_metric = load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

if __name__ == "__main__":
    ############################################################################
    # load data, process it, create tokenizer/processor 
    #

    train, val = load_data(path_to_data)
    make_vocab(train, val)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", 
        target_lang=target_lang
    )

    tokenizer.save_pretrained(f'./{repo_name}')

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )

    train = train.cast_column("audio", Audio(sampling_rate=16_000))
    train = train.map(prepare_dataset, remove_columns=train.column_names)

    val = val.cast_column("audio", Audio(sampling_rate=16_000))
    val = val.map(prepare_dataset, remove_columns=val.column_names)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )


    ############################################################################
    # Load pretrained mms model, add adapter layers, and freeze the base model.
    #
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
    model.init_adapter_layers()
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=2,
    eval_strategy="steps",
    num_train_epochs=5,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=200,
    eval_steps=100,
    logging_steps=100,
    learning_rate=1e-3,
    warmup_steps=100,
    save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(), 
                   adapter_file, 
                   metadata={"format": "pt"})