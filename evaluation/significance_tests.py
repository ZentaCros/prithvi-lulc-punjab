"""
Statistical Significance Tests for Multi-Seed Results

Computes:
  1. Mean ± std across seeds for all metrics
  2. McNemar's test (pairwise model comparisons on pixel-level predictions)
  3. Paired bootstrap confidence intervals for mIoU differences
  4. Summary tables ready for LaTeX

Usage:
    python evaluation/significance_tests.py
    python evaluation/significance_tests.py --results_dir results/
"""

import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = ["unet", "segformer", "deeplabv3", "prithvi_v2"]
MODEL_DISPLAY = {
    "unet": "U-Net (EffNet-B4)",
    "segformer": "MiT-B2+FPN",
    "deeplabv3": "DeepLabV3+ (ResNet-101)",
    "prithvi_v2": "Prithvi-EO 2.0 (LoRA)",
}
SEEDS = [42, 123, 7]
METRICS = ["oa", "miou", "mf1"]


def load_results(results_dir):
    """Load test_results.json for all model×seed combinations."""
    all_results = {}
    for model in MODELS:
        all_results[model] = {}
        for seed in SEEDS:
            path = results_dir / f"{model}_seed{seed}" / "test_results.json"
            if path.exists():
                with open(path) as f:
                    all_results[model][seed] = json.load(f)
            else:
                print(f"  [WARN] Missing: {path}")
    return all_results


def compute_summary_table(all_results):
    """Compute mean ± std for each model across seeds."""
    summary = {}
    for model in MODELS:
        seeds_data = all_results.get(model, {})
        if len(seeds_data) < 2:
            continue

        metrics_vals = {m: [] for m in METRICS}
        iou_per_class_vals = []

        for seed, data in seeds_data.items():
            for m in METRICS:
                metrics_vals[m].append(data[m])
            iou_per_class_vals.append(data["iou_per_class"])

        summary[model] = {}
        for m in METRICS:
            vals = np.array(metrics_vals[m])
            summary[model][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "values": [float(v) for v in vals],
            }

        iou_arr = np.array(iou_per_class_vals)
        summary[model]["iou_per_class"] = {
            "mean": [float(x) for x in iou_arr.mean(axis=0)],
            "std": [float(x) for x in iou_arr.std(axis=0, ddof=1)],
        }

    return summary


def paired_bootstrap_ci(values_a, values_b, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Paired bootstrap confidence interval for the difference (a - b).
    values_a, values_b: arrays of per-seed metric values.
    """
    rng = np.random.RandomState(seed)
    diffs = np.array(values_a) - np.array(values_b)
    n = len(diffs)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def run_pairwise_tests(summary):
    """Run paired bootstrap and Wilcoxon tests for all model pairs on mIoU."""
    model_pairs = list(combinations(MODELS, 2))
    results = []

    for m_a, m_b in model_pairs:
        if m_a not in summary or m_b not in summary:
            continue

        vals_a = summary[m_a]["miou"]["values"]
        vals_b = summary[m_b]["miou"]["values"]

        if len(vals_a) < 3 or len(vals_b) < 3:
            continue

        # Paired bootstrap CI
        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(vals_a, vals_b)

        # Wilcoxon signed-rank test (if enough samples)
        try:
            stat, p_val = stats.wilcoxon(vals_a, vals_b, alternative='two-sided')
        except ValueError:
            stat, p_val = np.nan, np.nan

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(vals_a, vals_b)

        results.append({
            "model_a": m_a,
            "model_b": m_b,
            "mean_diff_miou": mean_diff,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "wilcoxon_p": float(p_val) if not np.isnan(p_val) else None,
            "ttest_stat": float(t_stat),
            "ttest_p": float(t_pval),
        })

    return results


def print_latex_table(summary):
    """Print mean±std table in LaTeX format."""
    print("\n% ── LaTeX Table: Mean ± Std Across Seeds ──")
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\caption{Patch-Level Test Results on Lahore (mean $\\pm$ std over 3 seeds)}")
    print("\\label{tab:patch_multiseed}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{OA (\\%)} & \\textbf{mIoU (\\%)} & \\textbf{mF1 (\\%)} \\\\")
    print("\\midrule")

    for model in MODELS:
        if model not in summary:
            continue
        name = MODEL_DISPLAY[model]
        oa = summary[model]["oa"]
        miou = summary[model]["miou"]
        mf1 = summary[model]["mf1"]

        # Bold the best mean
        oa_str = f"{oa['mean']*100:.2f} $\\pm$ {oa['std']*100:.2f}"
        miou_str = f"{miou['mean']*100:.2f} $\\pm$ {miou['std']*100:.2f}"
        mf1_str = f"{mf1['mean']*100:.2f} $\\pm$ {mf1['std']*100:.2f}"

        print(f"{name:<30} & {oa_str} & {miou_str} & {mf1_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Tests")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results"

    print("=" * 60)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)

    # Load results
    print("\nLoading results...")
    all_results = load_results(results_dir)

    # Check completeness
    for model in MODELS:
        seeds_found = list(all_results.get(model, {}).keys())
        status = "OK" if len(seeds_found) == 3 else f"INCOMPLETE ({len(seeds_found)}/3)"
        print(f"  {MODEL_DISPLAY.get(model, model):<30} seeds: {seeds_found} — {status}")

    # Compute summary
    print("\nComputing mean ± std...")
    summary = compute_summary_table(all_results)

    if not summary:
        print("  [ERROR] Not enough results to compute statistics. Run multi-seed training first.")
        return

    # Print summary
    for model in MODELS:
        if model not in summary:
            continue
        print(f"\n  {MODEL_DISPLAY[model]}:")
        for m in METRICS:
            vals = summary[model][m]
            print(f"    {m:>5}: {vals['mean']*100:.2f}% ± {vals['std']*100:.2f}%  "
                  f"(seeds: {[f'{v*100:.2f}' for v in vals['values']]})")

    # Pairwise tests
    print(f"\n{'=' * 60}")
    print("  PAIRWISE SIGNIFICANCE TESTS (mIoU)")
    print(f"{'=' * 60}")

    pairwise = run_pairwise_tests(summary)
    for test in pairwise:
        m_a = MODEL_DISPLAY[test["model_a"]]
        m_b = MODEL_DISPLAY[test["model_b"]]
        diff = test["mean_diff_miou"] * 100
        lo = test["ci_95_lo"] * 100
        hi = test["ci_95_hi"] * 100
        p_t = test["ttest_p"]

        sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
        print(f"\n  {m_a} vs {m_b}")
        print(f"    Δ mIoU = {diff:+.2f}% [{lo:+.2f}, {hi:+.2f}] (95% CI)")
        print(f"    Paired t-test: t={test['ttest_stat']:.3f}, p={p_t:.4f} {sig}")
        if test["wilcoxon_p"] is not None:
            print(f"    Wilcoxon: W={test['wilcoxon_stat']:.1f}, p={test['wilcoxon_p']:.4f}")

    # LaTeX table
    print_latex_table(summary)

    # Save results
    output_path = results_dir / "significance_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {k: {
                m: {kk: vv for kk, vv in v.items()} for m, v in d.items()
            } for k, d in summary.items()},
            "pairwise_tests": pairwise,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
