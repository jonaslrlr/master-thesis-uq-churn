from __future__ import annotations

from pathlib import Path
import itertools
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "telco"

# training randomness only
TRAIN_SEEDS = [1, 2, 3, 4]

# categorical embedding grid
CAT_EMB_GRID = [1, 2, 4]

# FIXED split seed -> fixed train/valid/test (test isolated)
SPLIT_SEED = 42

# Fixed training params
FIXED = dict(
    gamma=1.5,
    lr=2e-2,
    weight_decay=1e-5,
    max_epochs=200,
    patience=30,
    batch_size=1024,
    virtual_batch_size=128,
    dropout=0.0,
)

def main():
    device_name = "cpu"  # reproducibility
    print("Device:", device_name)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("cat_emb_dim grid:", CAT_EMB_GRID)

    # Load data
    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = encode_tabular_for_tabnet(df)

    # Fixed split ONCE
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("Fixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    # Backbone grid
    grid_n = [8, 16, 32, 64]
    grid_steps = [3, 5]
    grid_mask = ["sparsemax", "entmax"]

    results_dir = REPO_ROOT / "reports" / "results"
    best_dir = REPO_ROOT / "reports" / "best"
    results_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)  # (mean_valid_prauc, mean_valid_lift)
    best_cfg = None

    for n_d, n_steps, mask_type, cat_emb_dim in itertools.product(grid_n, grid_steps, grid_mask, CAT_EMB_GRID):
        tag = f"nd{n_d}_steps{n_steps}_{mask_type}_emb{cat_emb_dim}"

        cfg = dict(
            n_d=n_d,
            n_a=n_d,
            n_steps=n_steps,
            mask_type=mask_type,
            cat_emb_dim=cat_emb_dim,
            **FIXED,
        )

        print("\n=== CONFIG", tag, "===")

        valid_praucs, valid_lifts = [], []
        test_praucs, test_lifts = [], []

        for train_seed in TRAIN_SEEDS:
            run_tag = f"{tag}_split{SPLIT_SEED}_trainseed{train_seed}"
            out_json = results_dir / f"{DATASET}_baseline_{run_tag}.json"

            # resume-safe
            if out_json.exists():
                payload = json.loads(out_json.read_text())
                v = payload["metrics"]["valid"]
                t = payload["metrics"]["test"]
                valid_praucs.append(v["auc_pr"])
                valid_lifts.append(v["lift10"])
                test_praucs.append(t["auc_pr"])
                test_lifts.append(t["lift10"])
                print(f"⏭️  SKIP {run_tag}")
                continue

            # training randomness only (global)
            set_seed(train_seed)
            print(f"-> RUN {run_tag}")

            clf = train_tabnet_baseline(
                X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims_list,
                device_name=device_name,
                n_d=cfg["n_d"],
                n_a=cfg["n_a"],
                n_steps=cfg["n_steps"],
                gamma=cfg["gamma"],
                mask_type=cfg["mask_type"],
                cat_emb_dim=cfg["cat_emb_dim"],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                batch_size=cfg["batch_size"],
                virtual_batch_size=cfg["virtual_batch_size"],
                dropout=cfg["dropout"],
                seed=train_seed,  # ✅ CRITICAL: passed into TabNetClassifier(seed=...)
            )

            # VALID
            p_valid = clf.predict_proba(X_valid)[:, 1]
            valid_rep = standard_report(y_valid, p_valid)

            # TEST (report only; do not select by test)
            p_test = clf.predict_proba(X_test)[:, 1]
            test_rep = standard_report(y_test, p_test)

            meta = RunMeta(dataset=DATASET, method="baseline", seed=train_seed, tag=run_tag)
            save_metrics_json(
                out_json,
                meta,
                metrics={"valid": valid_rep, "test": test_rep},
                config={**cfg, "split_seed": SPLIT_SEED, "train_seed": train_seed, "device_name": device_name},
            )

            valid_praucs.append(valid_rep["auc_pr"])
            valid_lifts.append(valid_rep["lift10"])
            test_praucs.append(test_rep["auc_pr"])
            test_lifts.append(test_rep["lift10"])

        # Aggregate across training seeds
        v_mean = float(np.mean(valid_praucs)); v_std = float(np.std(valid_praucs, ddof=0))
        l_mean = float(np.mean(valid_lifts));  l_std = float(np.std(valid_lifts, ddof=0))
        t_mean = float(np.mean(test_praucs));  t_std = float(np.std(test_praucs, ddof=0))
        tl_mean = float(np.mean(test_lifts));  tl_std = float(np.std(test_lifts, ddof=0))

        agg_rows.append((tag, v_mean, v_std, l_mean, l_std, t_mean, t_std, tl_mean, tl_std))

        print(f"VALID PR-AUC mean±std: {v_mean:.4f} ± {v_std:.4f} | Lift mean±std: {l_mean:.4f} ± {l_std:.4f}")
        print(f"TEST  PR-AUC mean±std: {t_mean:.4f} ± {t_std:.4f} | Lift mean±std: {tl_mean:.4f} ± {tl_std:.4f}")

        # Select best by mean valid PR-AUC (tie-breaker: mean valid lift)
        cand = (v_mean, l_mean)
        if cand > best_tuple:
            best_tuple = cand
            best_key = tag
            best_cfg = cfg

    # Print leaderboard
    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)

    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    for tag, v_mean, v_std, l_mean, l_std, *_ in agg_rows:
        print(f"{tag:30s}  prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}")

    # Save best config (include split seed + train seed range)
    best_file = best_dir / f"{DATASET}_baseline_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
    }, indent=2))

    print("\n✅ Saved best config to:", best_file)
    print("Best:", best_key, "with (mean_valid_prauc, mean_valid_lift) =", best_tuple)

if __name__ == "__main__":
    main()
