from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_mc_dropout import train_tabnet_mc_dropout, mc_predict
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.data.registry import load_for_tabnet
from thesis_uq.eval._cli import parse_eval_args

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"
# Fixed split: test is isolated and constant
SPLIT_SEED = 42

# Training randomness only
TRAIN_SEEDS = [1, 2, 3, 4]

# Dropout tuning grid
DROPOUT_GRID = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
MC_SAMPLES = 100

DEVICE_NAME = "cpu"

# Load best baseline backbone from your baseline gridsearch (fixed split protocol)
BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "cell2cell_baseline_split42_trainseeds1-4.json"



def main():
    print("Device:", DEVICE_NAME)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("Dropout grid:", DROPOUT_GRID, "MC samples:", MC_SAMPLES)
    print("Baseline best:", BASELINE_BEST_FILE)

    best_dir = REPO_ROOT / "reports" / "best"
    results_dir = REPO_ROOT / "reports" / "results"
    uq_dir = REPO_ROOT / "reports" / "uq_scores"
    best_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    uq_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASELINE_BEST_FILE.read_text())
    BASE = base["config"]
    base_tag = base.get("best_tag", "unknown")
    print("\nUsing backbone:", base_tag)
    print(BASE)

    # Load data once
    csv_path = REPO_ROOT / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_telco_csv(csv_path)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)

    # Fixed split ONCE
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    # Backbone params (fixed from baseline best)
    tabnet_kwargs = dict(
        n_d=BASE["n_d"],
        n_a=BASE["n_a"],
        n_steps=BASE["n_steps"],
        gamma=BASE["gamma"],
        mask_type=BASE["mask_type"],
        cat_emb_dim=BASE["cat_emb_dim"],
    )

    # Training params (fixed from baseline best)
    train_kwargs = dict(
        lr=BASE["lr"],
        weight_decay=BASE["weight_decay"],
        max_epochs=BASE["max_epochs"],
        patience=BASE["patience"],
        batch_size=BASE["batch_size"],
        virtual_batch_size=BASE["virtual_batch_size"],
    )

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)  # (mean_valid_prauc, mean_valid_lift)
    best_cfg = None

    for drop in DROPOUT_GRID:
        tag = f"drop{drop}"
        cfg = {
            **tabnet_kwargs,
            **train_kwargs,
            "dropout": drop,
            "mc_samples": MC_SAMPLES,
            "split_seed": SPLIT_SEED,
            "train_seeds": TRAIN_SEEDS,
        }

        print("\n=== CONFIG", tag, "===")

        valid_praucs, valid_lifts, valid_u_means = [], [], []
        test_praucs, test_lifts, test_u_means = [], [], []

        for train_seed in TRAIN_SEEDS:
            run_tag = f"mc_{tag}_split{SPLIT_SEED}_trainseed{train_seed}"
            out_json = results_dir / f"{DATASET}_{run_tag}.json"

            # resume-safe
            if out_json.exists():
                payload = json.loads(out_json.read_text())
                v = payload["metrics"]["valid"]
                t = payload["metrics"]["test"]
                valid_praucs.append(v["auc_pr"])
                valid_lifts.append(v["lift10"])
                valid_u_means.append(v["u_mean"])
                test_praucs.append(t["auc_pr"])
                test_lifts.append(t["lift10"])
                test_u_means.append(t["u_mean"])
                print(f"⏭️  SKIP {run_tag}")
                continue

            set_seed(train_seed)
            print(f"-> RUN {run_tag}")

            clf = train_tabnet_mc_dropout(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                device_name=DEVICE_NAME,
                dropout=drop,
                tabnet_kwargs=tabnet_kwargs,
                train_kwargs=train_kwargs,
                seed=train_seed,  # ✅ critical (TabNetClassifier has its own seed param)
            )

            # MC predictions: valid + test
            p_valid, u_valid = mc_predict(clf, X_valid, n_samples=MC_SAMPLES)
            p_test,  u_test  = mc_predict(clf, X_test,  n_samples=MC_SAMPLES)

            valid_rep = standard_report(y_valid, p_valid)
            test_rep = standard_report(y_test, p_test)

            # store mean uncertainty for diagnostics
            valid_rep["u_mean"] = float(np.mean(u_valid))
            test_rep["u_mean"] = float(np.mean(u_test))

            meta = RunMeta(dataset=DATASET, method="mc_dropout", seed=train_seed, tag=run_tag)
            save_metrics_json(out_json, meta, metrics={"valid": valid_rep, "test": test_rep}, config=cfg)

            valid_praucs.append(valid_rep["auc_pr"])
            valid_lifts.append(valid_rep["lift10"])
            valid_u_means.append(valid_rep["u_mean"])

            test_praucs.append(test_rep["auc_pr"])
            test_lifts.append(test_rep["lift10"])
            test_u_means.append(test_rep["u_mean"])

        v_mean = float(np.mean(valid_praucs)); v_std = float(np.std(valid_praucs, ddof=0))
        l_mean = float(np.mean(valid_lifts));  l_std = float(np.std(valid_lifts, ddof=0))
        u_mean = float(np.mean(valid_u_means)); u_std = float(np.std(valid_u_means, ddof=0))

        agg_rows.append((tag, v_mean, v_std, l_mean, l_std, u_mean, u_std))

        print(f"VALID prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}  u_mean={u_mean:.4f}±{u_std:.4f}")

        cand = (v_mean, l_mean)
        if cand > best_tuple:
            best_tuple = cand
            best_key = tag
            best_cfg = cfg

    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)

    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    for tag, v_mean, v_std, l_mean, l_std, u_mean, u_std in agg_rows:
        print(f"{tag:8s}  prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}  u_mean={u_mean:.4f}±{u_std:.4f}")

    # Save best config
    best_file = best_dir / f"{DATASET}_mc_dropout_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "backbone_from": BASELINE_BEST_FILE.name,
    }, indent=2))
    print("\nSaved best MC-dropout config to:", best_file)
    print("Best:", best_key, "with (mean_valid_prauc, mean_valid_lift) =", best_tuple)

    # Optional: save ONE canonical NPZ for reranking/plots (clean)
    canonical_train_seed = TRAIN_SEEDS[0]
    set_seed(canonical_train_seed)

    clf = train_tabnet_mc_dropout(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        device_name=DEVICE_NAME,
        dropout=best_cfg["dropout"],
        tabnet_kwargs=tabnet_kwargs,
        train_kwargs=train_kwargs,
        seed=canonical_train_seed,
    )

    p_valid, u_valid = mc_predict(clf, X_valid, n_samples=MC_SAMPLES)
    p_test,  u_test  = mc_predict(clf, X_test,  n_samples=MC_SAMPLES)

    out_npz = uq_dir / f"{DATASET}_mc_dropout_best_split{SPLIT_SEED}_trainseed{canonical_train_seed}.npz"
    save_uq_scores_npz(out_npz,
                       y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                       y_test=y_test,   p_test=p_test,   u_test=u_test)
    print("Saved best MC UQ scores to:", out_npz)


if __name__ == "__main__":
    main()
