from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.metrics.ranking import standard_report

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "telco"

# Keep test fixed
SPLIT_SEED = 42

# Evaluate robustness to training randomness
TRAIN_SEEDS = list(range(5, 15))  # 5–14

# Use CPU for reproducibility (switch to "mps" later if you want)
DEVICE_NAME = "cpu"

# Load the best config found previously (from selection seeds 1–4)
BEST_FILE = REPO_ROOT / "reports" / "best" / "telco_baseline_split42_trainseeds1-4.json"



def main():
    print("Device:", DEVICE_NAME)
    print("Fixed split seed:", SPLIT_SEED)
    print("Training seeds:", TRAIN_SEEDS)
    print("Best file:", BEST_FILE)

    best = json.loads(BEST_FILE.read_text())
    CFG = best["config"]
    print("Using best config:", best["best_tag"])
    print(CFG)

    # Load data
    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = encode_tabular_for_tabnet(df)

    # Fixed split ONCE (test stays constant)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("Split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    rows = []

    for train_seed in TRAIN_SEEDS:
        set_seed(train_seed)
        print("\n=== TRAIN SEED", train_seed, "===")

        clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=DEVICE_NAME,

            n_d=CFG["n_d"],
            n_a=CFG["n_a"],
            n_steps=CFG["n_steps"],
            gamma=CFG["gamma"],
            cat_emb_dim=CFG["cat_emb_dim"],
            mask_type=CFG["mask_type"],

            lr=CFG["lr"],
            weight_decay=CFG["weight_decay"],
            max_epochs=CFG["max_epochs"],
            patience=CFG["patience"],
            batch_size=CFG["batch_size"],
            virtual_batch_size=CFG["virtual_batch_size"],

            dropout=CFG["dropout"],
            seed=train_seed,
        )

        # Evaluate on fixed test
        proba = clf.predict_proba(X_test)[:, 1]
        rep = standard_report(y_test, proba)
        rep["train_seed"] = train_seed
        rows.append(rep)

        print("Test report:", rep)

    # Summary
    import pandas as pd
    df_rep = pd.DataFrame(rows).set_index("train_seed")
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    # ---- Save results (CSV + JSON summary) ----
    out_dir = REPO_ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / f"{DATASET}_baseline_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.csv"
    df_rep.to_csv(csv_file)

    summary = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "best_file": str(BEST_FILE),
        "best_tag": best["best_tag"],
        "config": CFG,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{DATASET}_baseline_eval_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\ Saved per-seed CSV to:", csv_file)
    print("Saved summary JSON to:", json_file)



if __name__ == "__main__":
    main()
