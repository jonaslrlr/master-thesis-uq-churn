from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.metrics.ranking import standard_report

from thesis_uq.models.tabnet_laplace import (
    LaplaceConfig, extract_tabnet_representation, laplace_from_tabnet_features
)

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "telco"

SPLIT_SEED = 42
TRAIN_SEEDS = list(range(5, 15))  # 5..14
DEVICE_NAME = "cpu"

BEST_LAPLACE_FILE = REPO_ROOT / "reports" / "best" / "telco_laplace_split42_trainseeds1-4.json"
BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "telco_baseline_split42_trainseeds1-4.json"


def fit_lr_reranker(p_valid: np.ndarray, u_valid: np.ndarray, y_valid: np.ndarray):
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
    print("Split seed:", SPLIT_SEED)
    print("Train seeds:", TRAIN_SEEDS)
    print("Best Laplace:", BEST_LAPLACE_FILE)
    print("Baseline best:", BASELINE_BEST_FILE)

    lap = json.loads(BEST_LAPLACE_FILE.read_text())
    lap_cfg = LaplaceConfig(**lap["config"])
    print("\nUsing Laplace config:", lap.get("best_tag", "unknown"))
    print(lap_cfg)

    base = json.loads(BASELINE_BEST_FILE.read_text())
    BASE = base["config"]

    tabnet_kwargs = dict(
        n_d=BASE["n_d"], n_a=BASE["n_a"], n_steps=BASE["n_steps"],
        gamma=BASE["gamma"], mask_type=BASE["mask_type"], cat_emb_dim=BASE["cat_emb_dim"],
    )
    train_kwargs = dict(
        lr=BASE["lr"], weight_decay=BASE["weight_decay"],
        max_epochs=BASE["max_epochs"], patience=BASE["patience"],
        batch_size=BASE["batch_size"], virtual_batch_size=BASE["virtual_batch_size"],
    )

    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = encode_tabular_for_tabnet(df)

    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)

    rows = []

    for train_seed in TRAIN_SEEDS:
        set_seed(train_seed)
        print(f"\n=== TRAIN SEED {train_seed} ===")

        clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=DEVICE_NAME,
            dropout=0.0,
            seed=train_seed,
            **tabnet_kwargs,
            **train_kwargs,
        )

        feat_train = extract_tabnet_representation(clf, X_train)
        feat_valid = extract_tabnet_representation(clf, X_valid)
        feat_test  = extract_tabnet_representation(clf, X_test)

        p_map_valid, p_map_test, p_lap_valid_mean, p_lap_valid_std, p_lap_test_mean, p_lap_test_std = (
            laplace_from_tabnet_features(feat_train, y_train, feat_valid, feat_test, lap_cfg, seed=train_seed)
        )

        # MAP head (det)
        rep_map = standard_report(y_test, p_map_test)

        # Laplace mean
        rep_lap = standard_report(y_test, p_lap_test_mean)

        # LR rerank on VALID (Laplace mean + std)
        scaler, lr = fit_lr_reranker(p_lap_valid_mean, p_lap_valid_std, y_valid)
        p_lr = apply_lr_reranker(scaler, lr, p_lap_test_mean, p_lap_test_std)
        rep_lr = standard_report(y_test, p_lr)

        rows.append({
            "train_seed": train_seed,

            "map_auc_roc": rep_map["auc_roc"],
            "map_auc_pr": rep_map["auc_pr"],
            "map_lift10": rep_map["lift10"],

            "lap_auc_roc": rep_lap["auc_roc"],
            "lap_auc_pr": rep_lap["auc_pr"],
            "lap_lift10": rep_lap["lift10"],
            "lap_u_mean": float(np.mean(p_lap_test_std)),

            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_lift10": rep_lr["lift10"],
        })

        print("MAP:", rep_map)
        print("LAP:", rep_lap, "| u_mean:", float(np.mean(p_lap_test_std)))
        print("LR :", rep_lr)

    import pandas as pd
    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    out_dir = REPO_ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "best_laplace_file": str(BEST_LAPLACE_FILE),
        "best_tag": lap.get("best_tag", "unknown"),
        "laplace_config": lap["config"],
        "baseline_backbone_file": str(BASELINE_BEST_FILE),
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_laplace_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    print("\n✅ Saved per-seed CSV to:", csv_file)
    print("✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
