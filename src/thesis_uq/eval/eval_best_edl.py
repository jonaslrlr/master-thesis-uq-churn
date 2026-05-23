"""
Evaluation of best EDL configuration on held-out test set.

For each fresh training seed:
  1. Train TabNet-EDL with best gridsearch config
  2. Train TabNet baseline with best baseline config (same backbone)
  3. Predict: p_edl = alpha_1/S,  u = K/S (vacuity)
  4. Predict: p_base = baseline softmax probability
  5. Fit TWO LR rerankers on VALID → apply to TEST:
     a) LR(p_edl, u):  original — but p and u share S, near-redundant
     b) LR(p_base, u): independent signals — baseline prob has no
        Dirichlet coupling to vacuity 
  6. Report all metrics on TEST

EDL reranking note:
  p_edl and u are linked through S (total evidence): p = alpha_1/S,
  u = K/S. Two samples with the same p can have different S, so they're
  not fully redundant, but the coupling limits what LR can exploit.

  Using p_base (from a standard cross-entropy model) removes this
  coupling entirely — the baseline probability is independent of the
  Dirichlet parameters. This mirrors the Laplace eval design where
  MAP probability + Laplace uncertainty are independent signals.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_edl import (
    EDLConfig, train_tabnet_edl, edl_predict_proba_unc,
)
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.plots.uq_plots import plot_prob_vs_uncertainty
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_SEED = 42
EVAL_SEEDS = list(range(5, 16))  # seeds 5..15 (11 seeds, never seen during gridsearch)

DEVICE_NAME = "cpu"
LAMBDA_SPARSE = 1e-3


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cell2cell",
                        choices=["cell2cell", "telco", "bank", "delft", "cdr", "chile"])
    args = parser.parse_args()

    DATASET = args.dataset
    BEST_FILE = REPO_ROOT / "reports" / "best" / f"{DATASET}_edl_split{SPLIT_SEED}_trainseeds1-4.json"
    BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / f"{DATASET}_baseline_split{SPLIT_SEED}_trainseeds1-4.json"

    print("=== EDL EVALUATION (unbiased test) ===")
    print("Dataset:", DATASET)
    print("Split seed:", SPLIT_SEED)
    print("Eval seeds:", EVAL_SEEDS)
    print("EDL best file:", BEST_FILE)
    print("Baseline best file:", BASELINE_BEST_FILE)

    best = json.loads(BEST_FILE.read_text())
    CFG = best["config"]
    print("\nEDL best tag:", best["best_tag"])
    print("EDL config:", json.dumps(CFG, indent=2))

    # Load baseline config for the independent-probability reranker
    base_best = json.loads(BASELINE_BEST_FILE.read_text())
    BASE_CFG = base_best["config"]
    print("\nBaseline best tag:", base_best["best_tag"])
    print("Baseline config:", json.dumps(BASE_CFG, indent=2))

    out_dir = REPO_ROOT / "reports" / "eval"
    run_dir = out_dir / "runs"
    uq_dir = REPO_ROOT / "reports" / "uq_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    uq_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = REPO_ROOT / "reports" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data + fixed split
    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED, presplit=(30000, 30000) if DATASET == "cdr" else (4939, 1058) if DATASET == "chile" else None)
    print("\nSplit shapes:", X_train.shape, X_valid.shape, X_test.shape)

    # Build EDLConfig from best
    cfg = EDLConfig(
        n_d=CFG["n_d"],
        n_a=CFG["n_a"],
        n_steps=CFG["n_steps"],
        gamma=CFG["gamma"],
        mask_type=CFG["mask_type"],
        cat_emb_dim=CFG["cat_emb_dim"],
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
        max_epochs=CFG["max_epochs"],
        patience=CFG["patience"],
        batch_size=CFG["batch_size"],
        virtual_batch_size=CFG["virtual_batch_size"],
        momentum=CFG.get("momentum", 0.02),
        kl_coef=CFG["kl_coef"],
        anneal_epochs=CFG["anneal_epochs"],
        head_hidden_dim=CFG.get("head_hidden_dim", 0),
    )

    # Baseline training kwargs (same backbone, standard CE loss)
    base_tabnet_kwargs = dict(
        n_d=int(BASE_CFG["n_d"]),
        n_a=int(BASE_CFG["n_a"]),
        n_steps=int(BASE_CFG["n_steps"]),
        gamma=float(BASE_CFG["gamma"]),
        mask_type=str(BASE_CFG["mask_type"]),
        cat_emb_dim=int(BASE_CFG["cat_emb_dim"]),
    )
    base_train_kwargs = dict(
        lr=float(BASE_CFG["lr"]),
        weight_decay=float(BASE_CFG["weight_decay"]),
        max_epochs=int(BASE_CFG["max_epochs"]),
        patience=int(BASE_CFG["patience"]),
        batch_size=int(BASE_CFG["batch_size"]),
        virtual_batch_size=int(BASE_CFG["virtual_batch_size"]),
    )

    rows = []

    for seed in EVAL_SEEDS:
        run_json = run_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseed{seed}.json"

        # resume-safe: check if this run already has the new base_lr fields
        if run_json.exists():
            row = json.loads(run_json.read_text())
            if "base_lr_auc_pr" in row:
                # Already has the new reranker — skip entirely
                rows.append(row)
                print(f"⏭️  SKIP seed={seed} (cached with base_lr)")
                continue
            else:
                # Old format without base_lr — need to re-run
                print(f"🔄 RE-RUN seed={seed} (missing base_lr fields)")

        set_seed(seed)
        print(f"\n=== TRAIN SEED {seed} ===")

        # 1. Train EDL model
        model = train_tabnet_edl(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            cfg=cfg,
            device_name=DEVICE_NAME,
            seed=seed,
            lambda_sparse=LAMBDA_SPARSE,
        )

        # 2. Train baseline model (same seed, standard CE loss)
        baseline_clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=DEVICE_NAME,
            **base_tabnet_kwargs,
            **base_train_kwargs,
            seed=seed,
        )

        # 3. EDL predictions: p = alpha_1/S, u = K/S (vacuity)
        p_valid_edl, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
        p_test_edl,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

        # 4. Baseline predictions (independent of Dirichlet)
        p_valid_base = baseline_clf.predict_proba(X_valid)[:, 1]
        p_test_base  = baseline_clf.predict_proba(X_test)[:, 1]

        plot_prob_vs_uncertainty(
            y_test, p_test_edl, u_test,
            title=f"EDL | P(churn) vs vacuity | test seed={seed}",
            save_path=plot_dir / f"{DATASET}_edl_scatter_split{SPLIT_SEED}_seed{seed}.png",
        )

        # 5. Base EDL metrics
        rep_edl = standard_report(y_test, p_test_edl)

        # 6a. LR reranker: EDL prob + EDL uncertainty (original, coupled)
        scaler_edl, lr_edl = fit_lr_reranker(p_valid_edl, u_valid, y_valid)
        p_lr_test = apply_lr_reranker(scaler_edl, lr_edl, p_test_edl, u_test)
        rep_lr = standard_report(y_test, p_lr_test)

        # 6b. LR reranker: baseline prob + EDL uncertainty (independent, Yameng's suggestion)
        #     Mirrors Laplace eval: MAP prob is independent of posterior uncertainty.
        scaler_base, lr_base = fit_lr_reranker(p_valid_base, u_valid, y_valid)
        p_base_lr_test = apply_lr_reranker(scaler_base, lr_base, p_test_base, u_test)
        rep_base_lr = standard_report(y_test, p_base_lr_test)

        # 7. Also report baseline-only performance for reference
        rep_base = standard_report(y_test, p_test_base)

        print(f"  EDL LR coefs:  u={lr_edl.coef_[0][0]:.4f}, p={lr_edl.coef_[0][1]:.4f}")
        print(f"  Base LR coefs: u={lr_base.coef_[0][0]:.4f}, p={lr_base.coef_[0][1]:.4f}")

        row = {
            "train_seed": seed,

            # EDL standalone
            "edl_auc_roc": rep_edl["auc_roc"],
            "edl_auc_pr": rep_edl["auc_pr"],
            "edl_acc": rep_edl["acc"],
            "edl_lift10": rep_edl["lift10"],

            # Baseline standalone (reference)
            "base_auc_roc": rep_base["auc_roc"],
            "base_auc_pr": rep_base["auc_pr"],
            "base_acc": rep_base["acc"],
            "base_lift10": rep_base["lift10"],

            # LR: EDL prob + EDL uncertainty (original, coupled)
            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_acc": rep_lr["acc"],
            "lr_lift10": rep_lr["lift10"],

            # LR: baseline prob + EDL uncertainty (independent, NEW)
            "base_lr_auc_roc": rep_base_lr["auc_roc"],
            "base_lr_auc_pr": rep_base_lr["auc_pr"],
            "base_lr_acc": rep_base_lr["acc"],
            "base_lr_lift10": rep_base_lr["lift10"],

            # Uncertainty stats
            "u_mean": float(np.mean(u_test)),
            "u_std": float(np.std(u_test)),
            "u_median": float(np.median(u_test)),

            # LR coefficients (EDL-coupled reranker)
            "lr_coef_u": float(lr_edl.coef_[0][0]),
            "lr_coef_p": float(lr_edl.coef_[0][1]),
            "lr_intercept": float(lr_edl.intercept_[0]),

            # LR coefficients (baseline-independent reranker)
            "base_lr_coef_u": float(lr_base.coef_[0][0]),
            "base_lr_coef_p": float(lr_base.coef_[0][1]),
            "base_lr_intercept": float(lr_base.intercept_[0]),
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        # Save NPZ per seed for downstream analysis
        npz_path = uq_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseed{seed}.npz"
        save_uq_scores_npz(npz_path,
                           y_valid=y_valid, p_valid=p_valid_edl, u_valid=u_valid,
                           y_test=y_test,   p_test=p_test_edl,   u_test=u_test)

        print(f"  EDL:      prauc={rep_edl['auc_pr']:.5f}  lift10={rep_edl['lift10']:.4f}")
        print(f"  Baseline: prauc={rep_base['auc_pr']:.5f}  lift10={rep_base['lift10']:.4f}")
        print(f"  LR(edl):  prauc={rep_lr['auc_pr']:.5f}  lift10={rep_lr['lift10']:.4f}")
        print(f"  LR(base): prauc={rep_base_lr['auc_pr']:.5f}  lift10={rep_base_lr['lift10']:.4f}  u_mean={row['u_mean']:.4f}")

    # ── Aggregate ──────────────────────────────────────────────────────
    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep.to_string())

    csv_file = out_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseeds{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": EVAL_SEEDS,
        "best_edl_file": str(BEST_FILE),
        "best_baseline_file": str(BASELINE_BEST_FILE),
        "best_tag": best["best_tag"],
        "edl_config": CFG,
        "baseline_config": BASE_CFG,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseeds{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("EDL TEST RESULTS (mean ± std over eval seeds)")
    print("=" * 70)

    sections = {
        "EDL (base)": ["edl_auc_pr", "edl_lift10", "edl_auc_roc"],
        "Baseline (reference)": ["base_auc_pr", "base_lift10", "base_auc_roc"],
        "LR reranked: p_edl + u (coupled)": ["lr_auc_pr", "lr_lift10", "lr_auc_roc"],
        "LR reranked: p_base + u (independent)": ["base_lr_auc_pr", "base_lr_lift10", "base_lr_auc_roc"],
    }

    for section_name, keys in sections.items():
        print(f"\n  {section_name}:")
        for k in keys:
            print(f"    {k:25s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  Uncertainty (vacuity):")
    print(f"    {'u_mean':25s} = {mean['u_mean']:.5f} ± {std['u_mean']:.5f}")
    print(f"    {'u_median':25s} = {mean['u_median']:.5f} ± {std['u_median']:.5f}")

    print(f"\n  LR coefficients (edl-coupled):")
    print(f"    {'lr_coef_u':25s} = {mean['lr_coef_u']:.4f} ± {std['lr_coef_u']:.4f}")
    print(f"    {'lr_coef_p':25s} = {mean['lr_coef_p']:.4f} ± {std['lr_coef_p']:.4f}")

    print(f"\n  LR coefficients (base-independent):")
    print(f"    {'base_lr_coef_u':25s} = {mean['base_lr_coef_u']:.4f} ± {std['base_lr_coef_u']:.4f}")
    print(f"    {'base_lr_coef_p':25s} = {mean['base_lr_coef_p']:.4f} ± {std['base_lr_coef_p']:.4f}")

    print(f"\n  Config: head_hidden_dim={cfg.head_hidden_dim}, "
          f"kl_coef={cfg.kl_coef}, anneal_epochs={cfg.anneal_epochs}")

    print(f"\n✅ Saved per-seed CSV to: {csv_file}")
    print(f"✅ Saved summary JSON to: {json_file}")


if __name__ == "__main__":
    main()
