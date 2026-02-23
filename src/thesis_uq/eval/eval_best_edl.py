"""
Evaluation of best EDL configuration on held-out test set.

For each fresh training seed:
  1. Train TabNet-EDL with best gridsearch config
  2. Predict: p = alpha_1/S,  u = K/S (vacuity)
  3. Fit LR reranker on VALID using (p, u) → apply to TEST
  4. Report all metrics on TEST

EDL reranking note:
  p and u are linked through S (total evidence) but not redundant —
  two samples can have the same churn probability with very different
  total evidence. LR can exploit this if vacuity varies meaningfully.
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
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
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
                        choices=["cell2cell", "telco"])
    args = parser.parse_args()

    DATASET = args.dataset
    BEST_FILE = REPO_ROOT / "reports" / "best" / f"{DATASET}_edl_split{SPLIT_SEED}_trainseeds1-4.json"

    print("=== EDL EVALUATION (unbiased test) ===")
    print("Dataset:", DATASET)
    print("Split seed:", SPLIT_SEED)
    print("Eval seeds:", EVAL_SEEDS)
    print("Best file:", BEST_FILE)

    best = json.loads(BEST_FILE.read_text())
    CFG = best["config"]
    print("\nBest tag:", best["best_tag"])
    print("Config:", json.dumps(CFG, indent=2))

    out_dir = REPO_ROOT / "reports" / "eval"
    run_dir = out_dir / "runs"
    uq_dir = REPO_ROOT / "reports" / "uq_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    uq_dir.mkdir(parents=True, exist_ok=True)

    # Load data + fixed split
    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
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
        edl_loss=CFG.get("edl_loss", "mse"),
        head_hidden_dim=CFG.get("head_hidden_dim", 0),
    )

    rows = []

    for seed in EVAL_SEEDS:
        run_json = run_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseed{seed}.json"

        # resume-safe
        if run_json.exists():
            row = json.loads(run_json.read_text())
            rows.append(row)
            print(f"⏭️  SKIP seed={seed} (cached)")
            continue

        set_seed(seed)
        print(f"\n=== TRAIN SEED {seed} ===")

        model = train_tabnet_edl(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            cfg=cfg,
            device_name=DEVICE_NAME,
            seed=seed,
            lambda_sparse=LAMBDA_SPARSE,
        )

        # EDL predictions: p = alpha_1/S, u = K/S (vacuity)
        p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
        p_test,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

        # Base EDL metrics
        rep_edl = standard_report(y_test, p_test)

        # LR reranker: (p_edl, u_vacuity) on VALID → apply to TEST
        scaler, lr = fit_lr_reranker(p_valid, u_valid, y_valid)
        p_lr_test = apply_lr_reranker(scaler, lr, p_test, u_test)
        rep_lr = standard_report(y_test, p_lr_test)

        print(f"  LR coefs: u={lr.coef_[0][0]:.4f}, p={lr.coef_[0][1]:.4f}")

        row = {
            "train_seed": seed,

            "edl_auc_roc": rep_edl["auc_roc"],
            "edl_auc_pr": rep_edl["auc_pr"],
            "edl_acc": rep_edl["acc"],
            "edl_lift10": rep_edl["lift10"],

            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_acc": rep_lr["acc"],
            "lr_lift10": rep_lr["lift10"],

            "u_mean": float(np.mean(u_test)),
            "u_std": float(np.std(u_test)),
            "u_median": float(np.median(u_test)),

            "lr_coef_u": float(lr.coef_[0][0]),
            "lr_coef_p": float(lr.coef_[0][1]),
            "lr_intercept": float(lr.intercept_[0]),
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        # Save NPZ per seed for downstream analysis
        npz_path = uq_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseed{seed}.npz"
        save_uq_scores_npz(npz_path,
                           y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                           y_test=y_test,   p_test=p_test,   u_test=u_test)

        print(f"  EDL:  prauc={rep_edl['auc_pr']:.5f}  lift10={rep_edl['lift10']:.4f}  u_mean={row['u_mean']:.4f}")
        print(f"  LR:   prauc={rep_lr['auc_pr']:.5f}  lift10={rep_lr['lift10']:.4f}")

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
        "best_tag": best["best_tag"],
        "config": CFG,
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
        "LR reranked (p+u)": ["lr_auc_pr", "lr_lift10", "lr_auc_roc"],
    }

    for section_name, keys in sections.items():
        print(f"\n  {section_name}:")
        for k in keys:
            print(f"    {k:20s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  Uncertainty (vacuity):")
    print(f"    {'u_mean':20s} = {mean['u_mean']:.5f} ± {std['u_mean']:.5f}")
    print(f"    {'u_median':20s} = {mean['u_median']:.5f} ± {std['u_median']:.5f}")

    print(f"\n  LR coefficients:")
    print(f"    {'lr_coef_u':20s} = {mean['lr_coef_u']:.4f} ± {std['lr_coef_u']:.4f}")
    print(f"    {'lr_coef_p':20s} = {mean['lr_coef_p']:.4f} ± {std['lr_coef_p']:.4f}")

    print(f"\n  Config: edl_loss={cfg.edl_loss}, head_hidden_dim={cfg.head_hidden_dim}, "
          f"kl_coef={cfg.kl_coef}, anneal_epochs={cfg.anneal_epochs}")

    print(f"\n✅ Saved per-seed CSV to: {csv_file}")
    print(f"✅ Saved summary JSON to: {json_file}")


if __name__ == "__main__":
    main()