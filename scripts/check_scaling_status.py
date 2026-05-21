"""
Check completion status of all scaling experiment training jobs.

For each lang:split in results/splits/scaling_jobs.txt, reports whether
the job completed (eval_results.json exists), is still running, or failed.

Also scans SLURM .err logs for Python tracebacks so you don't have to.

Usage (run from project root on Hyak):
  uv run python scripts/check_scaling_status.py
  uv run python scripts/check_scaling_status.py --show-errors
  uv run python scripts/check_scaling_status.py --failed-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import config

JOBS_FILE = config.results_dir / "splits" / "scaling_jobs.txt"
MODELS_DIR = config.models_dir / "mms"
LOGS_DIR = _project_root / "logs"

# Patterns that indicate a genuine failure in the .err log
ERROR_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"Error:|"
    r"CUDA out of memory|"
    r"RuntimeError:|"
    r"FileNotFoundError:|"
    r"slurmstepd: error|"
    r"DUE TO TIME LIMIT)",
    re.IGNORECASE,
)


def load_jobs() -> list[tuple[str, str]]:
    if not JOBS_FILE.exists():
        print(f"ERROR: {JOBS_FILE} not found. Run create_curated_splits.py first.")
        sys.exit(1)
    jobs = []
    for line in JOBS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        lang, split = line.split(":", 1)
        jobs.append((lang, split))
    return jobs


def check_job(lang: str, split: str) -> dict:
    result_json = MODELS_DIR / lang / split / "eval_results.json"

    if result_json.exists():
        try:
            data = json.loads(result_json.read_text())
            wer = data.get("greedy_test_wer", data.get("test_wer", None))
            hours = data.get("n_train_hours", None)
            return {
                "status": "completed",
                "wer": wer,
                "hours": hours,
                "error_snippet": None,
            }
        except Exception:
            return {"status": "completed (unreadable json)", "wer": None, "hours": None, "error_snippet": None}

    # Job did not complete — look for error evidence in logs
    # Log pattern: logs/mms-scaling_{array_job_id}_{task_id}.err
    # We can't know the array job ID without sacct, so search all matching .err files
    err_snippet = _find_error_in_logs(lang, split)
    status = "failed" if err_snippet else "missing"
    return {"status": status, "wer": None, "hours": None, "error_snippet": err_snippet}


def _find_error_in_logs(lang: str, split: str) -> str | None:
    """
    Scan all mms-scaling_*.err files for references to this lang/split.
    Returns a short snippet of the first error found, or None.
    """
    if not LOGS_DIR.exists():
        return None

    for err_path in sorted(LOGS_DIR.glob("mms-scaling_*.err"), reverse=True):
        try:
            text = err_path.read_text(errors="replace")
        except Exception:
            continue
        # Check if this log is for our lang/split
        if f"lang={lang}" not in text or f"split={split}" not in text:
            continue
        # Find first error line
        for line in text.splitlines():
            if ERROR_PATTERNS.search(line):
                return line.strip()[:120]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-only", action="store_true",
                        help="Only show jobs that did not complete")
    parser.add_argument("--show-errors", action="store_true",
                        help="Show error snippet for failed jobs")
    args = parser.parse_args()

    jobs = load_jobs()
    results = []
    for lang, split in jobs:
        info = check_job(lang, split)
        info["lang"] = lang
        info["split"] = split
        results.append(info)

    completed = [r for r in results if r["status"] == "completed"]
    failed    = [r for r in results if r["status"] == "failed"]
    missing   = [r for r in results if r["status"] == "missing"]

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*55}")
    print(f"  Scaling experiment job status  ({total} total jobs)")
    print(f"{'='*55}")
    print(f"  Completed : {len(completed):>3}  ✓")
    print(f"  Failed    : {len(failed):>3}  ✗  (error found in logs)")
    print(f"  Missing   : {len(missing):>3}  ?  (still running or not yet submitted)")
    print(f"{'='*55}\n")

    # ── Completed ──────────────────────────────────────────────────────────────
    if completed and not args.failed_only:
        print("COMPLETED:")
        for r in completed:
            wer_str = f"  test_wer={r['wer']:.4f}" if r["wer"] is not None else ""
            hrs_str = f"  ({r['hours']:.2f}h)" if r["hours"] is not None else ""
            print(f"  ✓  {r['lang']:8s}  {r['split']:20s}{hrs_str}{wer_str}")
        print()

    # ── Failed ─────────────────────────────────────────────────────────────────
    if failed:
        print("FAILED (re-submit these before starting LM decode):")
        for r in failed:
            print(f"  ✗  {r['lang']:8s}  {r['split']}")
            if args.show_errors and r["error_snippet"]:
                print(f"       → {r['error_snippet']}")
        print()

    # ── Missing ────────────────────────────────────────────────────────────────
    if missing:
        label = "STILL RUNNING / NOT YET SUBMITTED:"
        print(label)
        for r in missing:
            print(f"  ?  {r['lang']:8s}  {r['split']}")
        print()

    # ── Re-submission helper ───────────────────────────────────────────────────
    needs_rerun = failed + missing
    if needs_rerun:
        print("To resubmit failed/missing jobs, write a new jobs file and submit:")
        print()
        rerun_file = config.results_dir / "splits" / "scaling_jobs_rerun.txt"
        rerun_file.parent.mkdir(parents=True, exist_ok=True)
        rerun_file.write_text(
            "\n".join(f"{r['lang']}:{r['split']}" for r in needs_rerun) + "\n"
        )
        n = len(needs_rerun)
        print(f"  # {rerun_file} written with {n} job(s)")
        print(f"  sbatch --array=0-{n-1} scripts/hyak_scaling_exp.slurm")
        print()
        print("  NOTE: hyak_scaling_exp.slurm reads scaling_jobs.txt by default.")
        print("  Temporarily point it at the rerun file or rename it before submitting.")
    else:
        print("All jobs accounted for. Safe to submit LM decoding array.")


if __name__ == "__main__":
    main()
