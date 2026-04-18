"""
Conformal prediction evaluation for churn reranking.

Saves NPZ files compatible with eval_uncertainty.py:
  - {dataset}_cp_single_eval_split{split}_trainseed{seed}.npz
  - {dataset}_cp_cv_eval_split{split}_trainseed{seed}.npz
  - {dataset}_cp_cv_std_eval_split{split}_trainseed{seed}.npz

To run Spearman/rejection analysis afterwards, add these methods to
eval_uncertainty.py's methods list: "cp_single", "cp_cv", "cp_cv_std"

Usage:
    for ds in bank cell2cell telco delft cdr chile; do
        python -m thesis_uq.eval.eval_conformal --dataset $ds --seeds 5-15
    done
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.data.registry import load_for_tabnet
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.models.tabnet_conformal import (
    split_conformal_uncertainty,
    split_conformal_uncertainty_loo,
    cv_plus_train_and_predict,
)
from thesis_uq.io import save_uq_scores_npz
from thesis_uq.plots.uq_plots import plot_prob_vs_uncertainty


def parse_seeds(s):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def guess_repo_root():
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "reports").exists():
            return p
    return Path.cwd()


def fit_lr(p, u, y):
    X = np.column_stack([u, p])
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xs, y)
    return scaler, lr


def apply_lr(scaler, lr, p, u):
    X = np.column_stack([u, p])
    return lr.predict_proba(scaler.transform(X))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seeds", default="5-15")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--repo_root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else guess_repo_root()
    dataset = args.dataset
    split_seed = args.split_seed
    seeds = parse_seeds(args.seeds)
    n_folds = args.n_folds

    base_best_file = repo_root / "reports" / "best" / f"{dataset}_baseline_split{split_seed}_trainseeds1-4.json"
    assert base_best_file.exists(), f"Missing: {base_best_file}"
    base_best = json.loads(base_best_file.read_text())
    base_cfg = base_best["config"]

    print(f"=== CONFORMAL PREDICTION EVALUATION: {dataset.upper()} ===")
    print(f"Baseline config: {base_best.get('best_tag', 'unknown')}")
    print(f"Seeds: {seeds}")
    print(f"CV+ folds: {n_folds}")

    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(dataset, repo_root)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(
        X, y, seed=split_seed,
        presplit=(30000, 30000) if dataset == "cdr" else (4939, 1058) if dataset == "chile" else None,
    )
    print(f"Split shapes: {X_train.shape} {X_valid.shape} {X_test.shape}")

    out_dir = repo_root / "reports" / "eval"
    run_dir = out_dir / "runs"
    uq_dir = repo_root / "reports" / "uq_scores"
    plot_dir = repo_root / "reports" / "plots"
    for d in [out_dir, run_dir, uq_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in seeds:
        run_json = run_dir / f"{dataset}_conformal_eval_split{split_seed}_trainseed{seed}.json"
        if run_json.exists():
            row = json.loads(run_json.read_text())
            rows.append(row)
            print(f"⏭️  SKIP seed={seed} (cached)")
            continue

        set_seed(seed)
        print(f"\n=== SEED {seed} ===")

        # ── 1. Train single baseline ─────────────────────────────────
        print("  Training baseline model...")
        baseline_clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list, device_name=args.device,
            n_d=int(base_cfg["n_d"]), n_a=int(base_cfg["n_a"]),
            n_steps=int(base_cfg["n_steps"]), gamma=float(base_cfg["gamma"]),
            mask_type=str(base_cfg["mask_type"]), cat_emb_dim=int(base_cfg["cat_emb_dim"]),
            lr=float(base_cfg["lr"]), weight_decay=float(base_cfg["weight_decay"]),
            max_epochs=int(base_cfg["max_epochs"]), patience=int(base_cfg["patience"]),
            batch_size=int(base_cfg["batch_size"]),
            virtual_batch_size=int(base_cfg["virtual_batch_size"]),
            seed=seed,
        )
        p_base_valid = baseline_clf.predict_proba(X_valid)[:, 1]
        p_base_test = baseline_clf.predict_proba(X_test)[:, 1]
        rep_base = standard_report(y_test, p_base_test)

        # ── 2. Single-model CP ───────────────────────────────────────
        print("  Computing single-model CP...")
        u_cp_test = split_conformal_uncertainty(p_base_valid, y_valid, p_base_test)
        u_cp_valid = split_conformal_uncertainty_loo(p_base_valid, y_valid)

        sc_cp, lr_cp = fit_lr(p_base_valid, u_cp_valid, y_valid)
        p_lr_cp_test = apply_lr(sc_cp, lr_cp, p_base_test, u_cp_test)
        rep_lr_cp = standard_report(y_test, p_lr_cp_test)

        plot_prob_vs_uncertainty(
            y_test, p_base_test, u_cp_test,
            title=f"CP single | P(churn) vs conformal unc | seed={seed}",
            save_path=plot_dir / f"{dataset}_cp_single_scatter_split{split_seed}_seed{seed}.png",
        )
        save_uq_scores_npz(
            uq_dir / f"{dataset}_cp_single_eval_split{split_seed}_trainseed{seed}.npz",
            y_valid=y_valid, p_valid=p_base_valid, u_valid=u_cp_valid,
            y_test=y_test, p_test=p_base_test, u_test=u_cp_test,
        )

        # ── 3. CV+ conformal ─────────────────────────────────────────
        print(f"  Training CV+ ({n_folds} folds)...")
        cv = cv_plus_train_and_predict(
            X_train, y_train, X_valid, X_test,
            cat_idxs, cat_dims_list, base_cfg=base_cfg,
            n_folds=n_folds, device=args.device, seed=seed,
        )

        rep_ens = standard_report(y_test, cv["p_test_ens"])

        # LR(p_ens, u_cp_cv)
        sc_ens_cp, lr_ens_cp = fit_lr(cv["p_valid_ens"], cv["u_valid_cp"], y_valid)
        p_lr_ens_cp_test = apply_lr(sc_ens_cp, lr_ens_cp, cv["p_test_ens"], cv["u_test_cp"])
        rep_lr_ens_cp = standard_report(y_test, p_lr_ens_cp_test)

        # LR(p_ens, u_std)
        sc_ens_std, lr_ens_std = fit_lr(cv["p_valid_ens"], cv["u_valid_std"], y_valid)
        p_lr_ens_std_test = apply_lr(sc_ens_std, lr_ens_std, cv["p_test_ens"], cv["u_test_std"])
        rep_lr_ens_std = standard_report(y_test, p_lr_ens_std_test)

        # LR(p_base, u_cp_cv) — decoupled
        sc_base_cp, lr_base_cp = fit_lr(p_base_valid, cv["u_valid_cp"], y_valid)
        p_lr_base_cp_test = apply_lr(sc_base_cp, lr_base_cp, p_base_test, cv["u_test_cp"])
        rep_lr_base_cp = standard_report(y_test, p_lr_base_cp_test)

        # LR(p_base, u_std) — decoupled
        sc_base_std, lr_base_std = fit_lr(p_base_valid, cv["u_valid_std"], y_valid)
        p_lr_base_std_test = apply_lr(sc_base_std, lr_base_std, p_base_test, cv["u_test_std"])
        rep_lr_base_std = standard_report(y_test, p_lr_base_std_test)

        # Scatter plots: CV+
        plot_prob_vs_uncertainty(
            y_test, cv["p_test_ens"], cv["u_test_cp"],
            title=f"CP CV+ | ens P(churn) vs conformal unc | seed={seed}",
            save_path=plot_dir / f"{dataset}_cp_cv_scatter_split{split_seed}_seed{seed}.png",
        )
        plot_prob_vs_uncertainty(
            y_test, cv["p_test_ens"], cv["u_test_std"],
            title=f"CV+ | ens P(churn) vs fold std | seed={seed}",
            save_path=plot_dir / f"{dataset}_cp_cv_std_scatter_split{split_seed}_seed{seed}.png",
        )

        # NPZ: CV+ CP
        save_uq_scores_npz(
            uq_dir / f"{dataset}_cp_cv_eval_split{split_seed}_trainseed{seed}.npz",
            y_valid=y_valid, p_valid=cv["p_valid_ens"], u_valid=cv["u_valid_cp"],
            y_test=y_test, p_test=cv["p_test_ens"], u_test=cv["u_test_cp"],
        )
        # NPZ: CV+ std
        save_uq_scores_npz(
            uq_dir / f"{dataset}_cp_cv_std_eval_split{split_seed}_trainseed{seed}.npz",
            y_valid=y_valid, p_valid=cv["p_valid_ens"], u_valid=cv["u_valid_std"],
            y_test=y_test, p_test=cv["p_test_ens"], u_test=cv["u_test_std"],
        )

        print(f"    LR(base,cp)    coefs: u={lr_cp.coef_[0][0]:.4f}, p={lr_cp.coef_[0][1]:.4f}")
        print(f"    LR(ens,cp_cv)  coefs: u={lr_ens_cp.coef_[0][0]:.4f}, p={lr_ens_cp.coef_[0][1]:.4f}")
        print(f"    LR(ens,std)    coefs: u={lr_ens_std.coef_[0][0]:.4f}, p={lr_ens_std.coef_[0][1]:.4f}")
        print(f"    LR(base,cp_cv) coefs: u={lr_base_cp.coef_[0][0]:.4f}, p={lr_base_cp.coef_[0][1]:.4f}")
        print(f"    LR(base,std)   coefs: u={lr_base_std.coef_[0][0]:.4f}, p={lr_base_std.coef_[0][1]:.4f}")

        row = {
            "seed": seed,

            "base_auc_pr": rep_base["auc_pr"], "base_lift10": rep_base["lift10"],
            "base_auc_roc": rep_base["auc_roc"], "base_ece": rep_base["ece"], "base_brier": rep_base["brier"],

            "cp_single_auc_pr": rep_lr_cp["auc_pr"], "cp_single_lift10": rep_lr_cp["lift10"],
            "cp_single_auc_roc": rep_lr_cp["auc_roc"], "cp_single_ece": rep_lr_cp["ece"], "cp_single_brier": rep_lr_cp["brier"],
            "cp_single_coef_u": float(lr_cp.coef_[0][0]), "cp_single_coef_p": float(lr_cp.coef_[0][1]),
            "cp_single_intercept": float(lr_cp.intercept_[0]),

            "ens_auc_pr": rep_ens["auc_pr"], "ens_lift10": rep_ens["lift10"],
            "ens_auc_roc": rep_ens["auc_roc"], "ens_ece": rep_ens["ece"], "ens_brier": rep_ens["brier"],

            "ens_cp_auc_pr": rep_lr_ens_cp["auc_pr"], "ens_cp_lift10": rep_lr_ens_cp["lift10"],
            "ens_cp_auc_roc": rep_lr_ens_cp["auc_roc"], "ens_cp_ece": rep_lr_ens_cp["ece"], "ens_cp_brier": rep_lr_ens_cp["brier"],
            "ens_cp_coef_u": float(lr_ens_cp.coef_[0][0]), "ens_cp_coef_p": float(lr_ens_cp.coef_[0][1]),
            "ens_cp_intercept": float(lr_ens_cp.intercept_[0]),

            "ens_std_auc_pr": rep_lr_ens_std["auc_pr"], "ens_std_lift10": rep_lr_ens_std["lift10"],
            "ens_std_auc_roc": rep_lr_ens_std["auc_roc"], "ens_std_ece": rep_lr_ens_std["ece"], "ens_std_brier": rep_lr_ens_std["brier"],
            "ens_std_coef_u": float(lr_ens_std.coef_[0][0]), "ens_std_coef_p": float(lr_ens_std.coef_[0][1]),
            "ens_std_intercept": float(lr_ens_std.intercept_[0]),

            "base_cp_cv_auc_pr": rep_lr_base_cp["auc_pr"], "base_cp_cv_lift10": rep_lr_base_cp["lift10"],
            "base_cp_cv_auc_roc": rep_lr_base_cp["auc_roc"], "base_cp_cv_ece": rep_lr_base_cp["ece"], "base_cp_cv_brier": rep_lr_base_cp["brier"],
            "base_cp_cv_coef_u": float(lr_base_cp.coef_[0][0]), "base_cp_cv_coef_p": float(lr_base_cp.coef_[0][1]),
            "base_cp_cv_intercept": float(lr_base_cp.intercept_[0]),

            "base_std_auc_pr": rep_lr_base_std["auc_pr"], "base_std_lift10": rep_lr_base_std["lift10"],
            "base_std_auc_roc": rep_lr_base_std["auc_roc"], "base_std_ece": rep_lr_base_std["ece"], "base_std_brier": rep_lr_base_std["brier"],
            "base_std_coef_u": float(lr_base_std.coef_[0][0]), "base_std_coef_p": float(lr_base_std.coef_[0][1]),
            "base_std_intercept": float(lr_base_std.intercept_[0]),

            "u_cp_single_mean": float(np.mean(u_cp_test)), "u_cp_single_std": float(np.std(u_cp_test)),
            "u_cp_cv_mean": float(np.mean(cv["u_test_cp"])), "u_cp_cv_std": float(np.std(cv["u_test_cp"])),
            "u_std_mean": float(np.mean(cv["u_test_std"])), "u_std_std": float(np.std(cv["u_test_std"])),
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        print(f"\n  Base:           prauc={rep_base['auc_pr']:.5f}  lift10={rep_base['lift10']:.4f}")
        print(f"  CP single:      prauc={rep_lr_cp['auc_pr']:.5f}  lift10={rep_lr_cp['lift10']:.4f}")
        print(f"  CV+ ensemble:   prauc={rep_ens['auc_pr']:.5f}  lift10={rep_ens['lift10']:.4f}")
        print(f"  CV+ + CP:       prauc={rep_lr_ens_cp['auc_pr']:.5f}  lift10={rep_lr_ens_cp['lift10']:.4f}")
        print(f"  CV+ + std:      prauc={rep_lr_ens_std['auc_pr']:.5f}  lift10={rep_lr_ens_std['lift10']:.4f}")
        print(f"  Base + CV+ CP:  prauc={rep_lr_base_cp['auc_pr']:.5f}  lift10={rep_lr_base_cp['lift10']:.4f}")
        print(f"  Base + CV+ std: prauc={rep_lr_base_std['auc_pr']:.5f}  lift10={rep_lr_base_std['lift10']:.4f}")

    # ── Aggregate ────────────────────────────────────────────────────
    df = pd.DataFrame(rows).set_index("seed").sort_index()
    print("\n=== PER-SEED RESULTS ===")
    print(df.to_string())

    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)

    csv_file = out_dir / f"{dataset}_conformal_eval_split{split_seed}_seeds{seeds[0]}-{seeds[-1]}.csv"
    df.to_csv(csv_file)

    summary = {
        "dataset": dataset, "split_seed": split_seed, "seeds": seeds, "n_folds": n_folds,
        "baseline_config": base_cfg, "baseline_best_tag": base_best.get("best_tag", "unknown"),
        "mean": mean.to_dict(), "std": std.to_dict(),
    }
    json_file = out_dir / f"{dataset}_conformal_eval_split{split_seed}_seeds{seeds[0]}-{seeds[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"CONFORMAL PREDICTION RESULTS: {dataset.upper()}")
    print(f"(mean ± std over {len(seeds)} seeds)")
    print("=" * 70)

    sections = {
        "1. Baseline (single model)": ["base_auc_pr", "base_lift10"],
        "2. Single-model CP: LR(p_base, u_cp)": ["cp_single_auc_pr", "cp_single_lift10"],
        "3. CV+ ensemble (averaging only)": ["ens_auc_pr", "ens_lift10"],
        "4. CV+ + CP: LR(p_ens, u_cp_cv)": ["ens_cp_auc_pr", "ens_cp_lift10"],
        "5. CV+ + std: LR(p_ens, u_std)": ["ens_std_auc_pr", "ens_std_lift10"],
        "6. Decoupled: LR(p_base, u_cp_cv)": ["base_cp_cv_auc_pr", "base_cp_cv_lift10"],
        "7. Decoupled: LR(p_base, u_std)": ["base_std_auc_pr", "base_std_lift10"],
    }
    for name, keys in sections.items():
        print(f"\n  {name}:")
        for k in keys:
            print(f"    {k:25s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  Uncertainty stats:")
    for k in ["u_cp_single_mean", "u_cp_cv_mean", "u_std_mean"]:
        print(f"    {k:25s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  Key comparisons:")
    d1 = mean["cp_single_auc_pr"] - mean["base_auc_pr"]
    d2 = mean["ens_auc_pr"] - mean["base_auc_pr"]
    d3 = mean["ens_cp_auc_pr"] - mean["ens_auc_pr"]
    d4 = mean["ens_std_auc_pr"] - mean["ens_auc_pr"]
    print(f"    Single CP vs Base:       {d1:+.5f}  {'(CP helps)' if d1 > 0.001 else '(CP redundant)'}")
    print(f"    CV+ ens vs Base:         {d2:+.5f}  {'(ensembling helps)' if d2 > 0.001 else '(no ensemble gain)'}")
    print(f"    CV+ CP vs CV+ ens:       {d3:+.5f}  {'(CP adds to ensemble)' if d3 > 0.001 else '(CP redundant on top)'}")
    print(f"    CV+ std vs CV+ ens:      {d4:+.5f}  {'(std adds to ensemble)' if d4 > 0.001 else '(std redundant on top)'}")

    print(f"\n✅ Saved CSV: {csv_file}")
    print(f"✅ Saved JSON: {json_file}")
    print(f"✅ NPZ files in: {uq_dir}")
    print(f"✅ Plots in: {plot_dir}")
    print(f"\nFor Spearman analysis, add 'cp_single', 'cp_cv', 'cp_cv_std' to")
    print(f"methods list in eval_uncertainty.py, then: python -m thesis_uq.eval.eval_uncertainty --dataset {dataset}")


if __name__ == "__main__":
    main()
