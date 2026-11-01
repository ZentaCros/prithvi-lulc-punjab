"""
LoRA Rank Ablation Study — Prithvi-EO-2.0

Tests LoRA ranks [8, 16, 32, 64] with seed 42 to find the optimal rank.
Rank 64 (the default) should already exist from multi-seed training.

Usage:
    python training/run_lora_ablation.py
    python training/run_lora_ablation.py --dry_run
    python training/run_lora_ablation.py --ranks 8 16
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

RANKS = [8, 16, 32, 64]
SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
UNFREEZE_BLOCKS = 8


def check_existing(rank):
    results_path = PROJECT_ROOT / "results" / f"prithvi_v2_lora_r{rank}_seed{SEED}" / "test_results.json"
    return results_path.exists()


def run_ablation(rank):
    """Train Prithvi with a specific LoRA rank."""
    # Use a distinct experiment name for ablation runs
    exp_name = f"prithvi_v2_lora_r{rank}_seed{SEED}"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_prithvi.py"),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lora_rank", str(rank),
        "--unfreeze_blocks", str(UNFREEZE_BLOCKS),
    ]

    # We need to patch the experiment name. The simplest approach is to
    # modify train_prithvi.py to accept --exp_name, or we accept the default
    # naming (prithvi_v2_seed42) and rename after. For now, use env var approach.
    import os
    env = os.environ.copy()
    env["LULC_EXP_NAME"] = exp_name

    return subprocess.run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="LoRA Rank Ablation Study")
    parser.add_argument("--ranks", nargs="+", type=int, default=None,
                        help=f"LoRA ranks to test (default: {RANKS})")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ranks = args.ranks or RANKS

    print("=" * 60)
    print("  LORA RANK ABLATION STUDY")
    print(f"  Ranks: {ranks} | Seed: {SEED}")
    print("=" * 60)

    runs = []
    for rank in ranks:
        exists = check_existing(rank)
        if exists and not args.force:
            runs.append((rank, "SKIP (exists)"))
        else:
            runs.append((rank, "PENDING"))

    for rank, status in runs:
        print(f"  LoRA r={rank:<4} {status}")

    if args.dry_run:
        print("\n  (dry run)")
        return

    pending = [(r, s) for r, s in runs if s == "PENDING"]
    if not pending:
        print("\n  All ablation runs exist. Use --force to re-run.")
        return

    results = []
    for rank, _ in pending:
        print(f"\n{'=' * 60}")
        print(f"  ABLATION: LoRA rank={rank}")
        print(f"{'=' * 60}")

        t0 = time.time()
        result = run_ablation(rank)
        elapsed = (time.time() - t0) / 60
        ok = result.returncode == 0
        results.append((rank, ok, elapsed))
        print(f"  {'OK' if ok else 'FAILED'} in {elapsed:.1f} min")

    # Summary
    print(f"\n{'=' * 60}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 60}")
    for rank, ok, elapsed in results:
        print(f"  r={rank:<4} {'OK' if ok else 'FAILED':<8} {elapsed:.1f} min")


if __name__ == "__main__":
    main()
