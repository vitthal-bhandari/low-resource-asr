"""
Push already-trained models to Hugging Face Hub.

Loads saved models from models/{model}/{lang}/{split}/ and pushes them to HF.
Useful when training completed without --save-to-hf flag.

Supported --model values: mms, xlsr (whisperv3 etc. can be added when implemented).

Usage:
    # Push one model (default: mms)
    uv run python scripts/push_existing_models.py bew one
    uv run python scripts/push_existing_models.py bew one --model xlsr

    # Push all models for a language (all splits)
    uv run python scripts/push_existing_models.py bew --all-splits --model xlsr

    # Push all languages for a specific split
    uv run python scripts/push_existing_models.py --all-langs --split-arg one --model mms
    uv run python scripts/push_existing_models.py --all-langs --split-arg all --model xlsr

    # Push models from a SLURM job ID (auto-detects lang/split from logs)
    uv run python scripts/push_existing_models.py --job-id 33196711_15 --model mms

    # Push all models (all languages, all splits) - use with caution
    uv run python scripts/push_existing_models.py --all-langs --all-splits --model xlsr
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from src.config import config
from src.data.download import LANGUAGES

# Model name -> (module_path, get_hf_repo_id_name, push_to_hub_name)
# Each module must provide get_hf_repo_id(lang, split) and push_to_hub(model, processor, output_dir, repo_id, lang).
PUSH_REGISTRY = {
    "mms": ("src.training.aft_mms", "get_hf_repo_id", "push_to_hub"),
    "xlsr": ("src.training.aft_xlsr", "get_hf_repo_id", "push_to_hub"),
}


def _get_push_helpers(model_name: str):
    """Resolve get_hf_repo_id and push_to_hub for the given model name."""
    if model_name not in PUSH_REGISTRY:
        supported = ", ".join(sorted(PUSH_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {supported}. "
            "Add support in PUSH_REGISTRY for other models (e.g. whisperv3)."
        )
    mod_path, repo_fn, push_fn = PUSH_REGISTRY[model_name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, repo_fn), getattr(mod, push_fn)


def find_models_from_job_id(job_id: str, model_name: str) -> list[tuple[str, str]]:
    """
    Find (lang, split) pairs from a SLURM job ID by reading SLURM output logs.

    Job ID format: "33196711" (single job) or "33196711_15" (array job ID_task ID).
    Reads SLURM .out logs to extract language and split, then verifies model exists
    under models/{model_name}/{lang}/{split}/.
    """
    models = []
    slurm_logs_dir = Path("logs")
    models_base = config.models_dir / model_name

    if not slurm_logs_dir.exists():
        return []

    # Log filename patterns: mms-aft*, xlsr-aft* (job name from SLURM script)
    job_name_prefix = "mms-aft" if model_name == "mms" else "xlsr-aft"

    # Parse job ID
    if "_" in job_id:
        array_job_id, task_id_str = job_id.rsplit("_", 1)
        try:
            task_id = int(task_id_str)
        except ValueError:
            return []
        patterns = [f"{job_name_prefix}*_{array_job_id}_{task_id}.out"]
    else:
        array_job_id = job_id
        patterns = [
            f"{job_name_prefix}_{array_job_id}.out",
            f"{job_name_prefix}-all_{array_job_id}_*.out",
        ]

    log_files = set()
    for pattern in patterns:
        log_files.update(slurm_logs_dir.glob(pattern))

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()
                lang_match = re.search(r"Language:\s*(\S+)", content)
                split_match = re.search(r"Split:\s*(\S+)", content)
                if not split_match:
                    split_match = re.search(r"split[=:]\s*(\S+)", content, re.IGNORECASE)
                if not lang_match:
                    lang_match = re.search(r"for\s+(\S+)\s+\([^)]+\)\s+split=(\S+)", content)
                    if lang_match:
                        lang = lang_match.group(1).strip()
                        split = lang_match.group(2).strip()
                    else:
                        match = re.search(r"for\s+(\S+)\s+\(split=(\S+)\)", content)
                        if match:
                            lang = match.group(1).strip()
                            split = match.group(2).strip()
                        else:
                            lang = None
                            split = None
                else:
                    lang = lang_match.group(1).strip()
                    split = split_match.group(1).strip() if split_match else "all"

                if lang and lang in LANGUAGES:
                    if not split or split not in ["one", "mid", "all"]:
                        split = "all"
                    model_dir = models_base / lang / split
                    if model_dir.exists() and (model_dir / "config.json").exists():
                        models.append((lang, split))
        except Exception:
            continue

    if not models and models_base.exists():
        training_logs_dir = config.results_dir / "training_logs"
        log_glob = "mms_aft_*.log" if model_name == "mms" else "xlsr_aft_*.log"
        if training_logs_dir.exists():
            for log_file in training_logs_dir.glob(log_glob):
                try:
                    with open(log_file, "r") as f:
                        content = f.read()
                        lang_match = re.search(r"^language=(\S+)", content, re.MULTILINE)
                        split_match = re.search(r"^split=(\S+)", content, re.MULTILINE)
                        if lang_match and split_match:
                            lang = lang_match.group(1)
                            split = split_match.group(1)
                            model_dir = models_base / lang / split
                            if model_dir.exists() and (model_dir / "config.json").exists():
                                models.append((lang, split))
                except Exception:
                    pass

    return list(dict.fromkeys(models))


def _model_weights_exist(model_dir: Path) -> bool:
    """Check that at least one of model.safetensors or pytorch_model.bin exists."""
    return (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()


def push_model(lang: str, split: str, model_name: str) -> bool:
    """Load and push a single model from models/{model_name}/{lang}/{split}/."""
    model_dir = config.models_dir / model_name / lang / split
    if not model_dir.exists():
        print(f"  {model_name}/{lang}/{split}: Model directory not found: {model_dir}")
        return False
    if not _model_weights_exist(model_dir):
        print(f"  {model_name}/{lang}/{split}: Skipping — no model.safetensors or pytorch_model.bin in {model_dir} (training may not have completed).")
        return False

    print(f"\nPushing {model_name}/{lang}/{split}...")
    print(f"  Model dir: {model_dir}")

    try:
        get_hf_repo_id_fn, push_to_hub_fn = _get_push_helpers(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
        processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
        repo_id = get_hf_repo_id_fn(lang, split)
        print(f"  HF Repo: {repo_id}")
        push_to_hub_fn(model, processor, model_dir, repo_id, lang)
        print(f"  ✓ Successfully pushed {model_name}/{lang}/{split}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to push {model_name}/{lang}/{split}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push already-trained models to Hugging Face Hub"
    )
    parser.add_argument(
        "lang",
        nargs="?",
        type=str,
        help="Language code (e.g. bew). Omit with --all-langs or --job-id.",
    )
    parser.add_argument(
        "split",
        nargs="?",
        type=str,
        choices=["one", "mid", "all"],
        help="Split (one/mid/all). Positional argument. Use --split-arg with --all-langs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mms",
        choices=list(PUSH_REGISTRY.keys()),
        help="Model name (e.g. mms, xlsr). Determines models/{model}/{lang}/{split}/ and HF push logic. Default: mms.",
    )
    parser.add_argument(
        "--split-arg",
        type=str,
        dest="split_arg",
        choices=["one", "mid", "all"],
        help="Specify a single split to use (useful with --all-langs). Overrides positional split arg.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="SLURM job ID (e.g. '33196711' or '33196711_15'). Auto-detects lang/split from logs.",
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Push models for all 21 languages.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Push all splits (one, mid, all) for the language(s).",
    )
    args = parser.parse_args()

    # Auto-detect from job ID
    if args.job_id:
        print(f"Finding models for job ID: {args.job_id} (model: {args.model})")
        models = find_models_from_job_id(args.job_id, args.model)
        if models:
            print(f"  Found {len(models)} model(s): {', '.join(f'{l}/{s}' for l, s in models)}")
        else:
            print(f"  ✗ No models found for job {args.job_id}")
            print("  Check that:")
            print("    - SLURM output logs exist in logs/ directory")
            print("    - Training completed successfully")
            print(f"    - Model directories exist in models/{args.model}/{{lang}}/{{split}}/")
            print("  You can also manually specify: python scripts/push_existing_models.py <lang> <split> --model <model>")
            return
        langs = [lang for lang, _ in models]
        splits = [split for _, split in models]
    else:
        # Determine splits: --split-arg takes precedence, then --all-splits, then positional split, then default "all"
        if args.split_arg:  # --split-arg flag (named argument)
            splits = [args.split_arg]
        elif args.all_splits:
            splits = ["one", "mid", "all"]
        elif args.split:  # positional split argument
            splits = [args.split]
        else:
            splits = ["all"]
        
        # Determine languages
        langs = sorted(LANGUAGES.keys()) if args.all_langs else ([args.lang] if args.lang else [])

        if not langs:
            parser.error("Provide lang, --all-langs, or --job-id. Available langs: " + ", ".join(sorted(LANGUAGES.keys())))
        if not splits:
            parser.error("Provide split (one/mid/all), --split, or --all-splits")

    print(f"\nPushing models to Hugging Face Hub (model: {args.model})")
    print(f"Languages: {', '.join(langs)}")
    print(f"Splits: {', '.join(splits)}")
    print("")

    success_count = 0
    total_count = len(langs) * len(splits)

    for lang in langs:
        for split in splits:
            if push_model(lang, split, args.model):
                success_count += 1

    print(f"\n{'='*60}")
    print(f"Push complete: {success_count}/{total_count} models pushed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
