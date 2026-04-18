"""
Post-hoc uncertainty quality evaluation.

Reads existing NPZ files (p_test, u_test, y_test) saved during eval runs
and computes uncertainty quality metrics:
  - Selective prediction (rejection curves)
  - AUCO (oracle-normalised)
  - Uncertainty-binned ECE
  - Uncertainty-error Spearman correlation

Also computes calibration metrics (ECE, Brier) for each prediction variant.

Usage:
    python -m thesis_uq.eval.eval_uncertainty --dataset cell2cell
    python -m thesis_uq.eval.eval_uncertainty --dataset telco
    python -m thesis_uq.eval.eval_uncertainty  # runs all datasets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from thesis_uq.metrics.ranking import ece, pr_auc_trapezoid, lift_at_10
from thesis_uq.metrics.uncertainty import (
    selective_prauc,
    selective_lift10,
    auco,
    uncertainty_binned_ece,
    uncertainty_error_correlation,
    uncertainty_report,
)


def guess_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "reports").exists():
            return p
    return Path.cwd()


def load_npz_scores(npz_path: Path) -> dict:
    """Load NPZ and return dict with y_test, p_test, u_test (and valid versions)."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def find_npz_files(uq_dir: Path, dataset: str, method: str, split_seed: int) -> list[Path]:
    """Find all per-seed NPZ files for a dataset/method combo."""
    pattern = f"{dataset}_{method}_eval_split{split_seed}_trainseed*.npz"
    files = sorted(uq_dir.glob(pattern))
    return files


def evaluate_method(
    uq_dir: Path,
    dataset: str,
    method: str,
    split_seed: int,
) -> Optional[dict]:
    """
    Evaluate uncertainty quality for one method across all saved seeds.

    Returns aggregated results (mean ± std) or None if no NPZ files found.
    """
    files = find_npz_files(uq_dir, dataset, method, split_seed)
    if not files:
        return None

    all_results = []

    for npz_path in files:
        scores = load_npz_scores(npz_path)

        y_test = scores["y_test"]
        p_test = scores["p_test"]
        u_test = scores["u_test"]

        seed = int(npz_path.stem.split("trainseed")[-1])

        # Full uncertainty report
        uq_metrics = uncertainty_report(y_test, p_test, u_test)

        # Calibration on the UQ-adjusted probability
        uq_metrics["ece_test"] = ece(y_test, p_test)
        from sklearn.metrics import brier_score_loss
        uq_metrics["brier_test"] = float(brier_score_loss(y_test, p_test))

        # Also get the selective prediction curves (for plotting later)
        uq_metrics["_rejection_curve_prauc"] = selective_prauc(y_test, p_test, u_test)
        uq_metrics["_rejection_curve_lift10"] = selective_lift10(y_test, p_test, u_test)

        uq_metrics["train_seed"] = seed
        all_results.append(uq_metrics)

    return all_results


def aggregate_results(results: list[dict]) -> dict:
    """Compute mean ± std for all numeric keys across seeds."""
    # Collect all scalar keys across all seeds (bin edges can differ)
    all_keys = set()
    for r in results:
        all_keys.update(k for k in r if not k.startswith("_") and k != "train_seed")
    scalar_keys = sorted(all_keys)

    agg = {"n_seeds": len(results)}

    for k in scalar_keys:
        vals = []
        for r in results:
            if k in r and r[k] is not None:
                try:
                    v = float(r[k])
                    if not np.isnan(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

    # Average rejection curves point-by-point
    for curve_key in ["_rejection_curve_prauc", "_rejection_curve_lift10"]:
        curves = [r[curve_key] for r in results if curve_key in r]
        if curves:
            # find common fractions
            all_fracs = set()
            for c in curves:
                all_fracs.update(f for f, _ in c)
            common_fracs = sorted(all_fracs)

            avg_curve = []
            for frac in common_fracs:
                vals = []
                for c in curves:
                    for f, v in c:
                        if abs(f - frac) < 1e-6:
                            vals.append(v)
                if vals:
                    avg_curve.append((frac, float(np.mean(vals)), float(np.std(vals))))
            agg[curve_key.strip("_")] = avg_curve

    return agg


def print_results(dataset: str, method: str, agg: dict):
    """Pretty-print aggregated uncertainty results."""
    n = agg["n_seeds"]
    print(f"\n  {method.upper()} ({n} seeds)")

    # Core uncertainty quality — the real metrics
    print(f"\n    {'--- Uncertainty signal (confidence-controlled) ---':s}")
    keys_signal = [
        ("spearman_unc_error_residual", "Spearman residual (u adds signal)"),
        ("mean_conditional_acc_gap", "Cond. accuracy gap (u adds signal)"),
        ("spearman_unc_error_raw", "Spearman raw (confounded)"),
        ("spearman_confidence_error", "Spearman |p-0.5| vs error (free)"),
    ]
    for key, label in keys_signal:
        mk, sk = f"{key}_mean", f"{key}_std"
        if mk in agg:
            print(f"    {label:42s} = {agg[mk]:+.4f} ± {agg[sk]:.4f}")

    # Calibration
    print(f"\n    {'--- Calibration ---':s}")
    keys_cal = [
        ("ece_test", "ECE"),
        ("brier_test", "Brier"),
    ]
    for key, label in keys_cal:
        mk, sk = f"{key}_mean", f"{key}_std"
        if mk in agg:
            print(f"    {label:42s} = {agg[mk]:.5f} ± {agg[sk]:.5f}")

    # Binned ECE
    print(f"\n    {'--- ECE by uncertainty tercile ---':s}")
    keys_binned = [
        ("ece_low_unc", "ECE (low uncertainty)"),
        ("ece_mid_unc", "ECE (mid uncertainty)"),
        ("ece_high_unc", "ECE (high uncertainty)"),
    ]
    for key, label in keys_binned:
        mk, sk = f"{key}_mean", f"{key}_std"
        if mk in agg:
            print(f"    {label:42s} = {agg[mk]:.5f} ± {agg[sk]:.5f}")

    # Binned accuracy
    print(f"\n    {'--- Accuracy by uncertainty tercile ---':s}")
    keys_acc = [
        ("acc_low_unc", "Accuracy (low uncertainty)"),
        ("acc_mid_unc", "Accuracy (mid uncertainty)"),
        ("acc_high_unc", "Accuracy (high uncertainty)"),
    ]
    for key, label in keys_acc:
        mk, sk = f"{key}_mean", f"{key}_std"
        if mk in agg:
            print(f"    {label:42s} = {agg[mk]:.4f} ± {agg[sk]:.4f}")

    # Rejection curve summary
    curve = agg.get("rejection_curve_prauc", [])
    if curve:
        print(f"\n    {'--- Rejection curve (PR-AUC) ---':s}")
        for frac, mean_val, std_val in curve:
            if frac in [0.0, 0.10, 0.20, 0.30, 0.50]:
                print(f"      remove {frac:4.0%} most uncertain: {mean_val:.4f} ± {std_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc uncertainty quality evaluation")
    parser.add_argument("--dataset", default=None, help="Dataset name (or omit for all)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else guess_repo_root()
    uq_dir = repo_root / "reports" / "uq_scores"
    out_dir = repo_root / "reports" / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["bank", "cell2cell", "telco", "delft", "cdr", "chile"]

    # Methods that save NPZ files with uncertainty
    methods = ["mc_dropout", "laplace", "edl", "cp_single", "cp_cv", "cp_cv_std"]

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"  DATASET: {dataset.upper()} — UNCERTAINTY QUALITY")
        print(f"{'=' * 70}")

        dataset_summary = {}

        for method in methods:
            results = evaluate_method(uq_dir, dataset, method, args.split_seed)

            if results is None:
                print(f"\n  {method.upper()} — no NPZ files found")
                continue

            agg = aggregate_results(results)
            print_results(dataset, method, agg)
            dataset_summary[method] = agg

        # Save summary JSON
        if dataset_summary:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json_path = out_dir / f"{dataset}_uncertainty_split{args.split_seed}.json"

            # Strip non-serialisable curve data for JSON
            serialisable = {}
            for method, agg in dataset_summary.items():
                serialisable[method] = {
                    k: convert(v) for k, v in agg.items()
                    if not isinstance(v, list) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple))
                }
                # Convert tuple curves to list-of-lists for JSON
                for curve_key in ["rejection_curve_prauc", "rejection_curve_lift10"]:
                    if curve_key in agg:
                        serialisable[method][curve_key] = [
                            [f, m, s] for f, m, s in agg[curve_key]
                        ]

            json_path.write_text(json.dumps(serialisable, indent=2, default=convert))
            print(f"\n  ✅ Saved to {json_path}")


if __name__ == "__main__":
    main()