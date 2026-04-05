#!/usr/bin/env python3
"""
Run inference on test+val datasets for all (or selected) runs in runs/,
then copy result images to test_results_output/.

Usage:
    python scripts/run_inference_all.py                        # all runs
    python scripts/run_inference_all.py run_20260403_094620   # specific runs
    python scripts/run_inference_all.py run_20260401_* run_20260403_*  # globs

Result files copied per run (whichever exist after inference):
    test_results_regression.png
    test_results_classification.png
    test_results_per_folder_confusion_matrices.png
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "test_results_output"

RESULT_FILES = [
    "test_results_regression.png",
    "test_results_classification.png",
    "test_results_per_folder_confusion_matrices.png",
]


def resolve_runs(names):
    dirs = []
    for name in names:
        path = Path(name)
        if path.is_dir():
            dirs.append(path.resolve())
        elif (RUNS_DIR / name).is_dir():
            dirs.append(RUNS_DIR / name)
        else:
            print(f"WARNING: '{name}' not found, skipping.", file=sys.stderr)
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run names or paths to process (default: all runs in runs/)",
    )
    args = parser.parse_args()

    if args.runs:
        run_dirs = resolve_runs(args.runs)
    else:
        run_dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())

    if not run_dirs:
        print("No runs found. Exiting.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(run_dirs)} run(s) to process.")
    print(f"Output directory: {OUTPUT_DIR}\n")

    passed, failed, skipped = [], [], []

    for run_dir in run_dirs:
        run_name = run_dir.name
        checkpoint = run_dir / "checkpoints" / "checkpoint_best.pt"

        if not checkpoint.exists():
            print(f"[{run_name}] SKIP — no checkpoint_best.pt found")
            skipped.append(run_name)
            continue

        print("=" * 60)
        print(f"[{run_name}] Running inference...")
        print(f"  checkpoint: {checkpoint}\n")

        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"),
             "--mode", "test",
             "--checkpoint", str(checkpoint),
             "--test-on", "both"],
            cwd=str(ROOT),
        )

        print()

        if result.returncode != 0:
            print(f"[{run_name}] FAILED — exit code {result.returncode}")
            failed.append(run_name)
            print()
            continue

        print(f"[{run_name}] Copying results...")
        checkpoint_dir = checkpoint.parent
        copied = 0

        for fname in RESULT_FILES:
            src = checkpoint_dir / fname
            if src.exists():
                dest = OUTPUT_DIR / f"{run_name}_{fname}"
                shutil.copy2(src, dest)
                print(f"  Copied: {fname} -> {dest.name}")
                copied += 1

        if copied == 0:
            print(f"  WARNING: No result images found in {checkpoint_dir}")

        passed.append(run_name)
        print()

    print("=" * 60)
    print("Done.")
    print(f"  Passed  : {len(passed)}")
    print(f"  Skipped : {len(skipped)}")
    print(f"  Failed  : {len(failed)}")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
