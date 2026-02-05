from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_mc_dropout import train_tabnet_mc_dropout, mc_predict
from thesis_uq.metrics.ranking import standard_report


REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "telco"

# Fixed split -> test stays constant
SPLIT_SEED = 42

# Evaluation seeds (training randomness only)
TRAIN_SEEDS = list(range(5, 15))  # 5..14 inclusive

DEVICE_NAME = "cpu"

# Best MC-dropout config chosen on trainseeds1-4
BEST_MC_FILE = REPO_ROOT / "reports" / "best" / "telco_mc_dropout_split42_trainseeds1-4.json"

# Use more MC samples for final evaluation
MC_SAMPLES_EVAL = 200


def fit_lr_reranker(p_valid: np.ndarray, u_valid: np.ndarray, y_valid: np.ndarray):
    """
    Fit LR reranker on VALID only (no leakage).
    Features: [u, p]
    """
    Xv = np.column_stack([u_valid, p_valid])
    scaler = MinMaxScaler()
    Xv_s = scaler.fit_transform(Xv)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xv_s, y_valid)
    return scaler, lr


def apply_lr_reranker(scaler, lr, p: np.ndarray, u: np.ndarray) -> np.ndarray:
    Xt = np.column_stack([u, p])
    Xt_s = scaler.transform(Xt)
    return lr.predict_proba(Xt_s)[:, 1]


def main():
    print("Device:", DEVICE_NAME)
    print("Fixed split seed:", SPLIT_SEED)
    print("Eval train seeds:", TRAIN_SEEDS)
    print("Best MC file:", BEST_MC_FILE)
    print("MC samples:", MC_SAMPLES_EVAL)

    best = json.loads(BEST_MC_FILE.read_text())
    CFG = best["config"]
    best_tag = best.get("best_tag", "unknown")

    print("\nUsing best MC config:", best_tag)
    print(CFG)

    # Load data
    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = encode_tabular_for_tabnet(df)

    # Fixed split ONCE
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    # Build kwargs from config
    dropout = float(CFG["dropout"])

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

    for train_seed in TRAIN_SEEDS:
        set_seed(train_seed)
        print(f"\n=== TRAIN SEED {train_seed} ===")

        clf = train_tabnet_mc_dropout(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=DEVICE_NAME,
            dropout=dropout,
            tabnet_kwargs=tabnet_kwargs,
            train_kwargs=train_kwargs,
            seed=train_seed,  # ✅ ensures TabNetClassifier seed changes
        )

        # --- DET mode (dropout off) ---
        p_det = clf.predict_proba(X_test)[:, 1]
        rep_det = standard_report(y_test, p_det)

        # --- MC mode (mean/std) ---
        p_mc, u_mc = mc_predict(clf, X_test, n_samples=MC_SAMPLES_EVAL)
        rep_mc = standard_report(y_test, p_mc)

        # --- LR rerank (fit on VALID using same model’s VALID UQ) ---
        p_valid, u_valid = mc_predict(clf, X_valid, n_samples=MC_SAMPLES_EVAL)
        scaler, lr = fit_lr_reranker(p_valid, u_valid, y_valid)
        p_lr = apply_lr_reranker(scaler, lr, p_mc, u_mc)
        rep_lr = standard_report(y_test, p_lr)

        row = {
            "train_seed": train_seed,

            # DET metrics
            "det_auc_roc": rep_det["auc_roc"],
            "det_auc_pr": rep_det["auc_pr"],
            "det_lift10": rep_det["lift10"],

            # MC metrics
            "mc_auc_roc": rep_mc["auc_roc"],
            "mc_auc_pr": rep_mc["auc_pr"],
            "mc_lift10": rep_mc["lift10"],
            "mc_u_mean": float(np.mean(u_mc)),

            # LR metrics
            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_lift10": rep_lr["lift10"],
        }
        rows.append(row)

        print("DET:", rep_det)
        print("MC :", rep_mc, "| u_mean:", row["mc_u_mean"])
        print("LR :", rep_lr)

    # Summary + save
    import pandas as pd

    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    out_dir = REPO_ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / f"{DATASET}_mc_dropout_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    # mean/std per metric
    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "best_mc_file": str(BEST_MC_FILE),
        "best_tag": best_tag,
        "config": CFG,
        "mc_samples_eval": MC_SAMPLES_EVAL,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_mc_dropout_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    print("\nSaved per-seed CSV to:", csv_file)
    print("Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
