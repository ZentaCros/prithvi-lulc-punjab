"""
GPU Work Queue for CAGEO Review Fixes
Run this when GPU is available. Estimated total: ~3-4 hours on RTX 4050.

Tasks (in order of priority):
  1. Fix #4a: LoRA-only ablation (rank=64, 0 unfrozen) -- ~40 min
  2. Fix #4b: Unfrozen-only ablation (0 LoRA, 8 unfrozen) -- ~40 min
  3. Fix #8:  Confusion matrices for all 4 models -- ~5 min
  4. Fix #2:  Prithvi-only change detection TIFs (2018+2026) -- ~20 min
     (requires inference_evaluate.py modification to output per-model TIFs)

Deferred (need data first or full retrain):
  - Fix #6: Band initialization fix + retrain (needs code change + all seeds)
  - Fix #3: Spatial block split (needs retrain all models, ~2 days GPU)

Usage:
    python run_gpu_queue.py --dry_run       # show what would run
    python run_gpu_queue.py                 # run all pending tasks
    python run_gpu_queue.py --task 4a       # run only one task
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_task(name, cmd, env=None):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = (time.time() - t0) / 60
    ok = result.returncode == 0
    print(f"\n  {name}: {'OK' if ok else 'FAILED'} in {elapsed:.1f} min")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(description="GPU Work Queue")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--task", choices=["4a", "4b", "8", "2"], default=None)
    args = parser.parse_args()

    import os
    env = os.environ.copy()

    tasks = {
        "4a": {
            "name": "Fix #4a: LoRA-only ablation",
            "cmd": [sys.executable, str(PROJECT_ROOT / "training" / "run_ablation_fixes.py"), "--run", "4a"],
            "check": PROJECT_ROOT / "results" / "prithvi_v2_lora_only_r64_seed42" / "test_results.json",
        },
        "4b": {
            "name": "Fix #4b: Unfrozen-only ablation",
            "cmd": [sys.executable, str(PROJECT_ROOT / "training" / "run_ablation_fixes.py"), "--run", "4b"],
            "check": PROJECT_ROOT / "results" / "prithvi_v2_unfreeze_only_b8_seed42" / "test_results.json",
        },
        "8": {
            "name": "Fix #8: Confusion matrices (all 4 models)",
            "cmd": [sys.executable, str(PROJECT_ROOT / "evaluation" / "generate_confusion_matrices.py")],
            "check": PROJECT_ROOT / "outputs" / "figures" / "cm_U-Net.png",
        },
        "2": {
            "name": "Fix #2: Prithvi-only change detection TIFs",
            "cmd": None,
            "check": None,
            "note": "Requires inference_evaluate.py modification to save per-model TIFs. Run manually after code update.",
        },
    }

    run_tasks = {args.task: tasks[args.task]} if args.task else tasks

    print("=" * 60)
    print("  GPU WORK QUEUE - CAGEO Review Fixes")
    print("=" * 60)

    for key, t in run_tasks.items():
        exists = t["check"].exists() if t["check"] else False
        status = "DONE" if exists else ("MANUAL" if t.get("note") else "PENDING")
        print(f"\n  [{key}] {t['name']}: {status}")
        if t.get("note"):
            print(f"      Note: {t['note']}")

    if args.dry_run:
        print("\n  (dry run)")
        return

    results = []
    for key, t in run_tasks.items():
        if t["check"] and t["check"].exists():
            print(f"\n  Skipping {t['name']} (already done)")
            continue
        if t["cmd"] is None:
            print(f"\n  Skipping {t['name']} (manual step)")
            continue

        ok, elapsed = run_task(t["name"], t["cmd"])
        results.append((key, t["name"], ok, elapsed))

    if results:
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        total_time = 0
        for key, name, ok, elapsed in results:
            print(f"  [{key}] {name}: {'OK' if ok else 'FAILED'} ({elapsed:.1f} min)")
            total_time += elapsed
        print(f"\n  Total GPU time: {total_time:.1f} min")


if __name__ == "__main__":
    main()
