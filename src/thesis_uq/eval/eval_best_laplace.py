"""
Evaluation of best Laplace configuration on held-out test set.

For each fresh training seed:
  1. Train TabNet baseline (same backbone as gridsearch)
  2. Fit Laplace posterior with best τ
  3. Predict with three methods:
     - MAP:    deterministic, σ(wᵀh(x)) — no uncertainty
     - Probit: closed-form Bayesian, σ(μ/√(1+π/8·v))
     - MC:     sample w ~ N(w_MAP, Σ), average σ(wᵀh(x))
  4. Fit LR reranker on VALID (using probit predictions + uncertainty)
  5. Report all metrics on TEST

This gives a direct comparison:
  - MAP vs Probit: does the Bayesian correction help discrimination?
  - Probit vs MC: are they consistent? (they should be for well-behaved posteriors)
  - MAP/Probit vs LR: does uncertainty add signal beyond probability?
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.models.tabnet_laplace import (
    LaplaceConfig,
    fit_laplace, laplace_predict,
    posterior_diagnostics,
)
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"
SPLIT_SEED = 42

EVAL_SEEDS = list(range(5, 15))  # seeds 5..14 (never seen during gridsearch)

DEVICE_NAME = "cpu"
MC_SAMPLES_EVAL = 200

BEST_FILE = REPO_ROOT / "reports" / "best" / f"{DATASET}_laplace_split{SPLIT_SEED}_trainseeds1-4.json"


def fit_lr_reranker(p_valid, u_valid, y_valid):
    Xv = np.column_stack([u_valid, p_valid])
    scaler = MinMaxScaler()
    Xv_s = scaler.fit_transform(Xv)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xv_s, y_valid)
    return scaler, lr


def apply_lr_reranker(scaler, lr, p, u):
    Xt = np.column_stack([u, p])
    Xt_s = scaler.transform(Xt)
    return lr.predict_proba(Xt_s)[:, 1]


def main():
    print("=== LAPLACE EVALUATION (unbiased test) ===")
    print("Dataset:", DATASET)
    print("Split seed:", SPLIT_SEED)
    print("Eval seeds:", EVAL_SEEDS)
    print("MC samples:", MC_SAMPLES_EVAL)
    print("Best file:", BEST_FILE)

    best = json.loads(BEST_FILE.read_text())
    CFG = best["config"]
    best_tag = best.get("best_tag", "unknown")
    print("\nBest tag:", best_tag)
    print("Config:", json.dumps(CFG, indent=2))

    tau = float(CFG["prior_precision"])
    print(f"Prior precision τ = {tau}")

    # Configs for both prediction methods
    probit_cfg = LaplaceConfig(prior_precision=tau, pred_method="probit")
    mc_cfg = LaplaceConfig(prior_precision=tau, pred_method="mc", mc_samples=MC_SAMPLES_EVAL)

    out_dir = REPO_ROOT / "reports" / "eval"
    run_dir = out_dir / "runs"
    uq_dir = REPO_ROOT / "reports" / "uq_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    uq_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nSplit shapes:", X_train.shape, X_valid.shape, X_test.shape)

    tabnet_kwargs = dict(
        n_d=int(CFG["n_d"]),
        n_a=int(CFG["n_a"]),
        n_steps=int(CFG["n_steps"]),
        gamma=float(CFG["gamma"]),
        mask_type=str(CFG["mask_type"]),
        cat_emb_dim=int(CFG["cat_emb_dim"]),
    )

    train_kwargs = dict(
        lr=float(CFG["lr"]),
        weight_decay=float(CFG["weight_decay"]),
        max_epochs=int(CFG["max_epochs"]),
        patience=int(CFG["patience"]),
        batch_size=int(CFG["batch_size"]),
        virtual_batch_size=int(CFG["virtual_batch_size"]),
    )

    rows = []

    for seed in EVAL_SEEDS:
        run_json = run_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseed{seed}.json"

        if run_json.exists():
            row = json.loads(run_json.read_text())
            rows.append(row)
            print(f"⏭️  SKIP seed={seed} (cached)")
            continue

        set_seed(seed)
        print(f"\n=== TRAIN SEED {seed} ===")

        # 1. Train baseline
        clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=DEVICE_NAME,
            tabnet_kwargs=tabnet_kwargs,
            train_kwargs=train_kwargs,
            seed=seed,
        )

        # 2. Fit Laplace posterior
        posterior = fit_laplace(clf, X_train, y_train, prior_precision=tau)
        diag = posterior_diagnostics(posterior)
        print(f"  Posterior: cov_trace={diag['cov_trace']:.6f}, "
              f"log_marglik={posterior.log_marginal_likelihood:.1f}")

        # 3a. MAP predictions (deterministic — no Laplace)
        p_map_test = clf.predict_proba(X_test)[:, 1]
        rep_map = standard_report(y_test, p_map_test)

        # 3b. Probit predictions (closed-form Bayesian)
        p_probit_test, u_probit_test = laplace_predict(posterior, clf, X_test, probit_cfg)
        rep_probit = standard_report(y_test, p_probit_test)

        # 3c. MC predictions (sampled Bayesian)
        p_mc_test, u_mc_test = laplace_predict(posterior, clf, X_test, mc_cfg)
        rep_mc = standard_report(y_test, p_mc_test)

        # 4. LR reranker (fit on VALID, apply to TEST)
        p_probit_valid, u_probit_valid = laplace_predict(posterior, clf, X_valid, probit_cfg)
        scaler, lr = fit_lr_reranker(p_probit_valid, u_probit_valid, y_valid)
        p_lr_test = apply_lr_reranker(scaler, lr, p_probit_test, u_probit_test)
        rep_lr = standard_report(y_test, p_lr_test)

        print(f"  LR coefs: u={lr.coef_[0][0]:.4f}, p={lr.coef_[0][1]:.4f}")

        # 5. Collect
        row = {
            "train_seed": seed,

            "map_auc_roc": rep_map["auc_roc"],
            "map_auc_pr": rep_map["auc_pr"],
            "map_lift10": rep_map["lift10"],

            "probit_auc_roc": rep_probit["auc_roc"],
            "probit_auc_pr": rep_probit["auc_pr"],
            "probit_lift10": rep_probit["lift10"],

            "mc_auc_roc": rep_mc["auc_roc"],
            "mc_auc_pr": rep_mc["auc_pr"],
            "mc_lift10": rep_mc["lift10"],

            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_lift10": rep_lr["lift10"],

            "u_mean_probit": float(np.mean(u_probit_test)),
            "u_mean_mc": float(np.mean(u_mc_test)),

            "lr_coef_u": float(lr.coef_[0][0]),
            "lr_coef_p": float(lr.coef_[0][1]),

            "log_marglik": float(posterior.log_marginal_likelihood),
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        # Save NPZ for this seed
        npz_path = uq_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseed{seed}.npz"
        save_uq_scores_npz(npz_path,
                           y_valid=y_valid, p_valid=p_probit_valid, u_valid=u_probit_valid,
                           y_test=y_test, p_test=p_probit_test, u_test=u_probit_test)

        print(f"  MAP:    prauc={rep_map['auc_pr']:.5f}  lift10={rep_map['lift10']:.4f}")
        print(f"  Probit: prauc={rep_probit['auc_pr']:.5f}  lift10={rep_probit['lift10']:.4f}")
        print(f"  MC:     prauc={rep_mc['auc_pr']:.5f}  lift10={rep_mc['lift10']:.4f}")
        print(f"  LR:     prauc={rep_lr['auc_pr']:.5f}  lift10={rep_lr['lift10']:.4f}")

    # Aggregate
    import pandas as pd

    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep.to_string())

    csv_file = out_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseeds{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": EVAL_SEEDS,
        "best_laplace_file": str(BEST_FILE),
        "best_tag": best_tag,
        "config": CFG,
        "mc_samples_eval": MC_SAMPLES_EVAL,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseeds{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("LAPLACE TEST RESULTS (mean ± std over eval seeds)")
    print("=" * 70)

    sections = {
        "MAP (deterministic)": ["map_auc_pr", "map_lift10", "map_auc_roc"],
        "Probit (Bayesian)": ["probit_auc_pr", "probit_lift10", "probit_auc_roc"],
        "MC sampling": ["mc_auc_pr", "mc_lift10", "mc_auc_roc"],
        "LR reranked": ["lr_auc_pr", "lr_lift10", "lr_auc_roc"],
    }

    for section_name, keys in sections.items():
        print(f"\n  {section_name}:")
        for k in keys:
            print(f"    {k:20s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  Uncertainty:")
    print(f"    {'u_mean_probit':20s} = {mean['u_mean_probit']:.5f} ± {std['u_mean_probit']:.5f}")
    print(f"    {'u_mean_mc':20s} = {mean['u_mean_mc']:.5f} ± {std['u_mean_mc']:.5f}")

    print(f"\n  LR coefficients (interpretability):")
    print(f"    {'lr_coef_u':20s} = {mean['lr_coef_u']:.4f} ± {std['lr_coef_u']:.4f}")
    print(f"    {'lr_coef_p':20s} = {mean['lr_coef_p']:.4f} ± {std['lr_coef_p']:.4f}")

    print(f"\n  Prior precision τ = {CFG['prior_precision']}")

    print("\n✅ Saved per-seed CSV to:", csv_file)
    print("✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
