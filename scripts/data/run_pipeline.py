r"""
Training data pipeline runner for SimLlama.

Runs the data processing steps in dependency order.
Each script can also be run independently if only one step needs to be re-run.

Step order:
  1. mxl_to_mid.py              -- MXL -> MIDI (run once; requires MuseScore 4)
  2. prepare_condition_csv.py   -- build condition.csv + indexed_tokens_dict.json (run once)
  3. structure_data_preprocess.py -- MIDI + annotation JSON -> [id].npy + [id]_condition.npy
  4. check_anomalies.py         -- QC scan: silence gaps, short/long pieces, condition issues

The wrong-label control (M3) needs no step here: training sets it with --condition_mode m3.

Usage:
    conda run -n moonbeam_test python scripts\data\run_pipeline.py
    conda run -n moonbeam_test python scripts\data\run_pipeline.py --steps 3 4
    conda run -n moonbeam_test python scripts\data\run_pipeline.py --steps 3 4 --dry-run

All four steps run by default, as a fresh download needs. Every step reads its data folder
from the environment variable DATA_ROOT (default data/ninsheetmusic).
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = {
    1: ("mxl_to_mid.py",                 "MXL -> MIDI (requires MuseScore 4)"),
    2: ("prepare_condition_csv.py",       "Build condition.csv + token dict"),
    3: ("structure_data_preprocess.py",   "MIDI + annotation -> .npy training files"),
    4: ("check_anomalies.py",             "QC: anomaly scan on .npy files"),
}

DEFAULT_STEPS = [1, 2, 3, 4]


def run_step(step_num: int, dry_run: bool) -> bool:
    script, desc = STEPS[step_num]
    script_path = SCRIPT_DIR / script

    print(f"\n{'=' * 60}")
    print(f"Step {step_num}: {desc}")
    print(f"  Script: {script_path}")
    print(f"{'=' * 60}")

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    if dry_run:
        print("[DRY RUN] Would run:", sys.executable, str(script_path))
        return True

    result = subprocess.run([sys.executable, str(script_path)], check=False)

    if result.returncode != 0:
        print(f"\n[FAILED] Step {step_num} exited with code {result.returncode}")
        return False

    print(f"\n[OK] Step {step_num} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run SimLlama data processing pipeline.")
    parser.add_argument(
        "--steps", nargs="+", type=int, default=DEFAULT_STEPS,
        metavar="N",
        help=f"Steps to run (default: {DEFAULT_STEPS}). Available: {list(STEPS.keys())}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    invalid = [s for s in args.steps if s not in STEPS]
    if invalid:
        print(f"[ERROR] Unknown step numbers: {invalid}. Valid: {list(STEPS.keys())}")
        sys.exit(1)

    steps_to_run = sorted(set(args.steps))

    print("SimLlama data pipeline")
    print(f"Steps to run: {steps_to_run}")
    if args.dry_run:
        print("(dry run)")

    failed_steps = []
    for step_num in steps_to_run:
        ok = run_step(step_num, args.dry_run)
        if not ok:
            failed_steps.append(step_num)
            print(f"\n[ABORT] Stopping after failed step {step_num}.")
            break

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    ran = [s for s in steps_to_run if s <= (failed_steps[0] if failed_steps else max(steps_to_run))]
    for s in ran:
        status = "[FAILED]" if s in failed_steps else "[OK]"
        print(f"  Step {s}: {status}  {STEPS[s][1]}")
    if failed_steps:
        sys.exit(1)
    print("\nAll steps complete.")


if __name__ == "__main__":
    main()
