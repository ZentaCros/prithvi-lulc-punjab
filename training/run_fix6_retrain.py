"""
Fix #6: Retrain all Prithvi configurations with fixed spectral band initialization.

Runs 8 training configurations sequentially:
  1. seed 42, r=64, 8 unfrozen (main hybrid)
  2. seed 42, r=32, 8 unfrozen
  3. seed 42, r=16, 8 unfrozen
  4. seed 42, r=8,  8 unfrozen
  5. seed 42, r=64, 0 unfrozen (LoRA-only attribution)
  6. seed 42, r=0,  8 unfrozen (unfrozen-only attribution)
  7. seed 123, r=64, 8 unfrozen (multi-seed)
  8. seed 7,   r=64, 8 unfrozen (multi-seed)

Usage:
    python -I training/run_fix6_retrain.py             # all 8
    python -I training/run_fix6_retrain.py --run 1     # just the main config
    python -I training/run_fix6_retrain.py --run 1-4   # LoRA rank sweep only
"""

import subprocess
import sys
import os
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = r"D:\MY\Envs\shared-ml\Scripts\python.exe"
TRAIN_SCRIPT = str(PROJECT_ROOT / "training" / "train_prithvi.py")

CONFIGS = [
    # (run#, exp_name, seed, lora_rank, unfreeze_blocks, description)
    (1, "prithvi_v2_fix6_seed42",           42,  64, 8, "Main hybrid r=64"),
    (2, "prithvi_v2_fix6_r32_seed42",       42,  32, 8, "Rank sweep r=32"),
    (3, "prithvi_v2_fix6_r16_seed42",       42,  16, 8, "Rank sweep r=16"),
    (4, "prithvi_v2_fix6_r8_seed42",        42,   8, 8, "Rank sweep r=8"),
    (5, "prithvi_v2_fix6_lora_only_seed42", 42,  64, 0, "Attribution: LoRA-only"),
    (6, "prithvi_v2_fix6_unfreeze_only_seed42", 42, 0, 8, "Attribution: unfrozen-only"),
    (7, "prithvi_v2_fix6_seed123",          123, 64, 8, "Multi-seed 123"),
    (8, "prithvi_v2_fix6_seed7",            7,   64, 8, "Multi-seed 7"),
]


def parse_range(s):
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def run_one(run_num, exp_name, seed, lora_rank, unfreeze_blocks, desc):
    results_dir = PROJECT_ROOT / "results" / exp_name
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    results_json = results_dir / "test_results.json"

    if results_json.exists():
        with open(results_json) as f:
            r = json.load(f)
        print(f"  [SKIP] Already done: mIoU={r['miou']:.4f} OA={r['oa']:.4f}")
        return r

    env = os.environ.copy()
    env["LULC_EXP_NAME"] = exp_name

    cmd = [
        PYTHON, "-I", TRAIN_SCRIPT,
        "--epochs", "50",
        "--batch_size", "2",
        "--lr", "1e-4",
        "--seed", str(seed),
        "--patience", "15",
        "--lora_rank", str(lora_rank),
        "--unfreeze_blocks", str(unfreeze_blocks),
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    print(f"  Exp: {exp_name}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] Exit code {result.returncode} after {elapsed:.0f}s", flush=True)
        return None

    print(f"  [OK] Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    if results_json.exists():
        with open(results_json) as f:
            r = json.load(f)
        print(f"  mIoU={r['miou']:.4f}  OA={r['oa']:.4f}  mF1={r['mf1']:.4f}")
        return r
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="all",
                        help="Which runs: 'all', single number, range '1-4', or comma-separated '1,5,6'")
    args = parser.parse_args()

    if args.run == "all":
        run_nums = [c[0] for c in CONFIGS]
    else:
        run_nums = parse_range(args.run)

    configs_to_run = [c for c in CONFIGS if c[0] in run_nums]

    print("=" * 65)
    print("  FIX #6: RETRAIN WITH SPECTRAL BAND INITIALIZATION")
    print("=" * 65)
    print(f"\n  Runs to execute: {len(configs_to_run)} of {len(CONFIGS)}")
    for num, name, seed, rank, unf, desc in configs_to_run:
        print(f"    #{num}: {desc} (seed={seed}, r={rank}, unfreeze={unf})")
    print(flush=True)

    results = {}
    for num, name, seed, rank, unf, desc in configs_to_run:
        print(f"\n{'=' * 65}")
        print(f"  RUN #{num}/{len(CONFIGS)}: {desc}")
        print(f"  seed={seed}  lora_rank={rank}  unfreeze_blocks={unf}")
        print(f"{'=' * 65}", flush=True)

        r = run_one(num, name, seed, rank, unf, desc)
        if r:
            results[name] = r

    # Summary
    print(f"\n\n{'=' * 65}")
    print("  SUMMARY: FIX #6 RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Config':<40} {'mIoU':>8} {'OA':>8} {'mF1':>8}")
    print(f"  {'-' * 66}")
    for num, name, seed, rank, unf, desc in configs_to_run:
        if name in results:
            r = results[name]
            print(f"  {desc:<40} {r['miou']*100:>7.2f}% {r['oa']*100:>7.2f}% {r['mf1']*100:>7.2f}%")
        else:
            print(f"  {desc:<40} {'FAILED':>8}")

    # Save summary
    summary_path = PROJECT_ROOT / "results" / "fix6_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"\n{'=' * 65}")
    print("  ALL DONE")
    print(f"{'=' * 65}", flush=True)


if __name__ == "__main__":
    main()
