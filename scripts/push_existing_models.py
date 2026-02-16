"""
Push already-trained models to Hugging Face Hub.

Loads saved models from models/mms/{lang}/{split}/ and pushes them to HF.
Useful when training completed without --save-to-hf flag.

Usage:
    # Push one model
    uv run python scripts/push_existing_models.py bew one
    
    # Push all models for a language (all splits)
    uv run python scripts/push_existing_models.py bew --all-splits
    
    # Push all languages for a specific split
    uv run python scripts/push_existing_models.py --all-langs --split-arg one
    uv run python scripts/push_existing_models.py --all-langs --split-arg mid
    uv run python scripts/push_existing_models.py --all-langs --split-arg all
    
    # Push models from a SLURM job ID (auto-detects lang/split from logs)
    uv run python scripts/push_existing_models.py --job-id 33196711_15
    uv run python scripts/push_existing_models.py --job-id 33196711  # array job, all tasks
    
    # Push all models (all languages, all splits) - use with caution
    uv run python scripts/push_existing_models.py --all-langs --all-splits
"""

import argparse
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
from src.training.aft_mms import get_hf_repo_id, push_to_hub


def find_models_from_job_id(job_id: str) -> list[tuple[str, str]]:
    """
    Find (lang, split) pairs from a SLURM job ID by reading SLURM output logs.
    
    Job ID format: "33196711" (single job) or "33196711_15" (array job ID_task ID).
    Reads SLURM .out logs to extract language and split, then verifies model exists.
    """
    models = []
    slurm_logs_dir = Path("logs")
    
    if not slurm_logs_dir.exists():
        return []
    
    # Parse job ID
    if "_" in job_id:
        array_job_id, task_id_str = job_id.rsplit("_", 1)
        try:
            task_id = int(task_id_str)
        except ValueError:
            return []
        # Look for array job logs: mms-aft-all_{array_job_id}_{task_id}.out
        patterns = [f"*_{array_job_id}_{task_id}.out"]
    else:
        array_job_id = job_id
        task_id = None
        # Look for all logs from this job:
        # - Single job: mms-aft_{job_id}.out
        # - Array job (all tasks): mms-aft-all_{array_job_id}_*.out
        patterns = [
            f"*_{array_job_id}.out",      # Single job
            f"*_{array_job_id}_*.out",     # Array job, all tasks
        ]
    
    # Find matching SLURM output logs
    log_files = set()
    for pattern in patterns:
        log_files.update(slurm_logs_dir.glob(pattern))
    
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()
                # Extract language and split from log content
                # Pattern 1: "Language: aln  Split: one" (from SLURM script echo)
                lang_match = re.search(r"Language:\s*(\S+)", content)
                split_match = re.search(r"Split:\s*(\S+)", content)
                
                # Pattern 2: "split=one" or "split=mid" or "split=all" (from training script)
                if not split_match:
                    split_match = re.search(r"split[=:]\s*(\S+)", content, re.IGNORECASE)
                
                # Pattern 3: "MMS Adapter Fine-tuning for aln (Gheg Albanian) split=one"
                if not lang_match:
                    lang_match = re.search(r"for\s+(\S+)\s+\([^)]+\)\s+split=(\S+)", content)
                    if lang_match:
                        lang = lang_match.group(1).strip()
                        split = lang_match.group(2).strip()
                    else:
                        # Pattern 4: "Starting training for aln (split=one)..."
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
                        split = "all"  # Default if not found
                    
                    # Verify model directory exists
                    model_dir = config.models_dir / "mms" / lang / split
                    if model_dir.exists() and (model_dir / "config.json").exists():
                        models.append((lang, split))
        except Exception:
            continue
    
    # If no matches from SLURM logs, check all training logs and match by approximate time
    # (less reliable, but fallback)
    if not models:
        training_logs_dir = config.results_dir / "training_logs"
        if training_logs_dir.exists():
            for log_file in training_logs_dir.glob("mms_aft_*.log"):
                try:
                    with open(log_file, "r") as f:
                        content = f.read()
                        lang_match = re.search(r"^language=(\S+)", content, re.MULTILINE)
                        split_match = re.search(r"^split=(\S+)", content, re.MULTILINE)
                        if lang_match and split_match:
                            lang = lang_match.group(1)
                            split = split_match.group(1)
                            model_dir = config.models_dir / "mms" / lang / split
                            if model_dir.exists() and (model_dir / "config.json").exists():
                                models.append((lang, split))
                except Exception:
                    pass
    
    # Remove duplicates
    return list(dict.fromkeys(models))  # Preserves order, removes duplicates


def push_model(lang: str, split: str) -> bool:
    """Load and push a single model from models/mms/{lang}/{split}/."""
    model_dir = config.models_dir / "mms" / lang / split
    if not model_dir.exists():
        print(f"  {lang}/{split}: Model directory not found: {model_dir}")
        return False
    
    print(f"\nPushing {lang}/{split}...")
    print(f"  Model dir: {model_dir}")
    
    try:
        # Load model and processor
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
        processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
        
        # Get repo ID
        repo_id = get_hf_repo_id(lang, split)
        print(f"  HF Repo: {repo_id}")
        
        # Push to hub
        push_to_hub(model, processor, model_dir, repo_id, lang)
        print(f"  ✓ Successfully pushed {lang}/{split}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to push {lang}/{split}: {e}")
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
        print(f"Finding models for job ID: {args.job_id}")
        models = find_models_from_job_id(args.job_id)
        if models:
            print(f"  Found {len(models)} model(s): {', '.join(f'{l}/{s}' for l, s in models)}")
        else:
            print(f"  ✗ No models found for job {args.job_id}")
            print("  Check that:")
            print("    - SLURM output logs exist in logs/ directory")
            print("    - Training completed successfully")
            print("    - Model directories exist in models/mms/{lang}/{split}/")
            print("  You can also manually specify: python scripts/push_existing_models.py <lang> <split>")
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

    print(f"\nPushing models to Hugging Face Hub")
    print(f"Languages: {', '.join(langs)}")
    print(f"Splits: {', '.join(splits)}")
    print("")

    success_count = 0
    total_count = len(langs) * len(splits)

    for lang in langs:
        for split in splits:
            if push_model(lang, split):
                success_count += 1

    print(f"\n{'='*60}")
    print(f"Push complete: {success_count}/{total_count} models pushed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
