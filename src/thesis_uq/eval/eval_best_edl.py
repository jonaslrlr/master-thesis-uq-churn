from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.metrics.ranking import standard_report

from thesis_uq.models.tabnet_edl import EDLConfig, train_tabnet_edl, edl_predict_proba_unc


REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "telco"

# Fixed split -> test stays constant
SPLIT_SEED = 42

# Evaluation seeds (training randomness only)
TRAIN_SEEDS = list(range(5, 15))  # 5..14

DEVICE_NAME = "cpu"

# Best EDL config chosen on trainseeds1-4
BEST_EDL_FILE = REPO_ROOT / "reports" / "best" / "telco_edl_split42_trainseeds1-4.json"


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
    print("Best EDL file:", BEST_EDL_FILE)

    best = json.loads(BEST_EDL_FILE.read_text())
    best_tag = best.get("best_tag", "unknown")
    CFG_DICT = best["config"]

    print("\nUsing best EDL config:", best_tag)
    print(CFG_DICT)

    cfg = EDLConfig(**CFG_DICT)

    # Load data
    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = encode_tabular_for_tabnet(df)

    # Fixed split ONCE
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    rows = []

    for train_seed in TRAIN_SEEDS:
        set_seed(train_seed)
        print(f"\n=== TRAIN SEED {train_seed} ===")

        model = train_tabnet_edl(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            cfg=cfg,
            device_name=DEVICE_NAME,
            seed=train_seed,
        )

        # --- EDL base ---
        p_test, u_test = edl_predict_proba_unc(model, X_test, device=DEVICE_NAME)
        rep_edl = standard_report(y_test, p_test)

        # --- LR rerank (fit on VALID, apply on TEST) ---
        p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
        scaler, lr = fit_lr_reranker(p_valid, u_valid, y_valid)
        p_lr = apply_lr_reranker(scaler, lr, p_test, u_test)
        rep_lr = standard_report(y_test, p_lr)

        row = {
            "train_seed": train_seed,

            # EDL metrics
            "edl_auc_roc": rep_edl["auc_roc"],
            "edl_auc_pr": rep_edl["auc_pr"],
            "edl_lift10": rep_edl["lift10"],
            "edl_u_mean": float(np.mean(u_test)),

            # LR metrics
            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_lift10": rep_lr["lift10"],
        }
        rows.append(row)

        print("EDL:", rep_edl, "| u_mean:", row["edl_u_mean"])
        print("LR :", rep_lr)

    # Save outputs
    import pandas as pd

    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    out_dir = REPO_ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "best_edl_file": str(BEST_EDL_FILE),
        "best_tag": best_tag,
        "config": CFG_DICT,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    print("\n✅ Saved per-seed CSV to:", csv_file)
    print("✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
