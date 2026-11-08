"""
Review Fix Ablation Runs - GPU Required

Runs the two missing ablation experiments needed for the review response:
  Fix #4a: LoRA-only (rank=64, 0 unfrozen blocks) - isolates LoRA contribution
  Fix #4b: Unfrozen-only (8 unfrozen blocks, no LoRA) - isolates unfreezing contribution

Both use seed 42 to match existing results.

Usage:
    python training/run_ablation_fixes.py
    python training/run_ablation_fixes.py --dry_run
    python training/run_ablation_fixes.py --run 4a      # run only Fix #4a
    python training/run_ablation_fixes.py --run 4b      # run only Fix #4b
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ABLATIONS = {
    "4a": {
        "name": "LoRA-only (no unfreezing)",
        "exp_name": "prithvi_v2_lora_only_r64_seed42",
        "lora_rank": 64,
        "unfreeze_blocks": 0,
    },
    "4b": {
        "name": "Unfrozen-only (no LoRA)",
        "exp_name": "prithvi_v2_unfreeze_only_b8_seed42",
        "lora_rank": 0,
        "unfreeze_blocks": 8,
    },
}

SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4


def check_existing(exp_name):
    results_path = PROJECT_ROOT / "results" / exp_name / "test_results.json"
    return results_path.exists()


def run_experiment(cfg):
    import os
    env = os.environ.copy()
    env["LULC_EXP_NAME"] = cfg["exp_name"]

    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_prithvi.py"),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lora_rank", str(cfg["lora_rank"]),
        "--unfreeze_blocks", str(cfg["unfreeze_blocks"]),
    ]

    print(f"  Command: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Review Fix Ablation Runs")
    parser.add_argument("--run", choices=["4a", "4b"], default=None,
                        help="Run only one experiment")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ablations = {args.run: ABLATIONS[args.run]} if args.run else ABLATIONS

    print("=" * 60)
    print("  REVIEW FIX ABLATION RUNS")
    print("=" * 60)

    for key, cfg in ablations.items():
        exists = check_existing(cfg["exp_name"])
        status = "SKIP (exists)" if exists and not args.force else "PENDING"
        print(f"  Fix #{key}: {cfg['name']}")
        print(f"    LoRA rank={cfg['lora_rank']}, unfreeze={cfg['unfreeze_blocks']}")
        print(f"    Experiment: {cfg['exp_name']}")
        print(f"    Status: {status}")

    if args.dry_run:
        print("\n  (dry run)")
        return

    results = []
    for key, cfg in ablations.items():
        if check_existing(cfg["exp_name"]) and not args.force:
            print(f"\n  Skipping Fix #{key} (results exist)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Fix #{key}: {cfg['name']}")
        print(f"{'=' * 60}")

        t0 = time.time()
        result = run_experiment(cfg)
        elapsed = (time.time() - t0) / 60
        ok = result.returncode == 0
        results.append((key, ok, elapsed))
        print(f"  {'OK' if ok else 'FAILED'} in {elapsed:.1f} min")

    if results:
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        for key, ok, elapsed in results:
            print(f"  Fix #{key}: {'OK' if ok else 'FAILED'} in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
