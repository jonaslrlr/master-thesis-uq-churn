from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.models.tabnet_edl import EDLConfig, train_tabnet_edl, edl_predict_proba_unc
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"

# Fixed split (test isolated & constant)
SPLIT_SEED = 42

# Training randomness only
TRAIN_SEEDS = [1, 2, 3, 4]

DEVICE_NAME = "cpu"

# EDL tuning grid
KL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0,3.5,4.0]
ANNEAL_EPOCHS = 50

# Baseline best backbone (fixed-test protocol)
BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "cell2cell_baseline_split42_trainseeds1-4.json"



def main():
    print("Device:", DEVICE_NAME)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("KL grid:", KL_GRID, "anneal_epochs:", ANNEAL_EPOCHS)
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
    X, y, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)

    # Fixed split ONCE
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)  # (mean_valid_prauc, mean_valid_lift)
    best_cfg: EDLConfig | None = None

    for kl in KL_GRID:
        tag = f"kl{kl}"

        cfg = EDLConfig(
            # backbone fixed from baseline best
            n_d=int(BASE["n_d"]),
            n_a=int(BASE["n_a"]),
            n_steps=int(BASE["n_steps"]),
            gamma=float(BASE["gamma"]),
            mask_type=str(BASE["mask_type"]),
            cat_emb_dim=int(BASE["cat_emb_dim"]),
            # training fixed from baseline best
            lr=float(BASE["lr"]),
            weight_decay=float(BASE["weight_decay"]),
            max_epochs=int(BASE["max_epochs"]),
            patience=int(BASE["patience"]),
            batch_size=int(BASE["batch_size"]),
            virtual_batch_size=int(BASE["virtual_batch_size"]),
            # EDL
            kl_coef=float(kl),
            anneal_epochs=int(ANNEAL_EPOCHS),
        )

        print("\n=== CONFIG", tag, "===")

        valid_praucs, valid_lifts, valid_u_means = [], [], []
        test_praucs, test_lifts, test_u_means = [], [], []

        for train_seed in TRAIN_SEEDS:
            run_tag = f"edl_{tag}_split{SPLIT_SEED}_trainseed{train_seed}"
            out_json = results_dir / f"{DATASET}_{run_tag}.json"

            # resume-safe
            if out_json.exists():
                payload = json.loads(out_json.read_text())
                v = payload["metrics"]["valid"]
                t = payload["metrics"]["test"]

                valid_praucs.append(v["auc_pr"])
                valid_lifts.append(v["lift10"])
                valid_u_means.append(v.get("u_mean", np.nan))

                test_praucs.append(t["auc_pr"])
                test_lifts.append(t["lift10"])
                test_u_means.append(t.get("u_mean", np.nan))

                print(f"⏭️  SKIP {run_tag}")
                continue

            print(f"-> RUN {run_tag}")
            # seed controls training randomness in our custom loop
            model = train_tabnet_edl(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                cfg=cfg,
                device_name=DEVICE_NAME,
                seed=train_seed,
            )

            # Predict (prob + uncertainty proxy)
            p_val, u_val = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
            p_te,  u_te  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

            valid_rep = standard_report(y_valid, p_val)
            test_rep  = standard_report(y_test,  p_te)

            valid_rep["u_mean"] = float(np.mean(u_val))
            test_rep["u_mean"]  = float(np.mean(u_te))

            meta = RunMeta(dataset=DATASET, method="edl", seed=train_seed, tag=run_tag)
            save_metrics_json(out_json, meta, metrics={"valid": valid_rep, "test": test_rep}, config={
                **cfg.__dict__,
                "split_seed": SPLIT_SEED,
                "train_seed": train_seed,
                "device_name": DEVICE_NAME,
                "backbone_from": BASELINE_BEST_FILE.name,
            })

            valid_praucs.append(valid_rep["auc_pr"])
            valid_lifts.append(valid_rep["lift10"])
            valid_u_means.append(valid_rep["u_mean"])

            test_praucs.append(test_rep["auc_pr"])
            test_lifts.append(test_rep["lift10"])
            test_u_means.append(test_rep["u_mean"])

        v_mean = float(np.mean(valid_praucs)); v_std = float(np.std(valid_praucs, ddof=0))
        l_mean = float(np.mean(valid_lifts));  l_std = float(np.std(valid_lifts, ddof=0))
        u_mean = float(np.nanmean(valid_u_means)); u_std = float(np.nanstd(valid_u_means, ddof=0))

        agg_rows.append((tag, v_mean, v_std, l_mean, l_std, u_mean, u_std))

        print(f"VALID prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}  u_mean={u_mean:.4f}±{u_std:.4f}")

        cand = (v_mean, l_mean)
        if cand > best_tuple:
            best_tuple = cand
            best_key = tag
            best_cfg = cfg

    # Leaderboard
    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)
    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    for tag, v_mean, v_std, l_mean, l_std, u_mean, u_std in agg_rows:
        print(f"{tag:8s}  prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}  u_mean={u_mean:.4f}±{u_std:.4f}")

    # Save best config JSON
    assert best_cfg is not None
    best_file = best_dir / f"{DATASET}_edl_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg.__dict__,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "backbone_from": BASELINE_BEST_FILE.name,
    }, indent=2))
    print("\n✅ Saved best EDL config to:", best_file)
    print("Best:", best_key, "with (mean_valid_prauc, mean_valid_lift) =", best_tuple)

    # Save ONE canonical NPZ (clean) for reranking/plots
    canonical_seed = TRAIN_SEEDS[0]
    model = train_tabnet_edl(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        cfg=best_cfg,
        device_name=DEVICE_NAME,
        seed=canonical_seed,
    )
    p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
    p_test,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

    out_npz = uq_dir / f"{DATASET}_edl_best_split{SPLIT_SEED}_trainseed{canonical_seed}.npz"
    save_uq_scores_npz(out_npz,
                       y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                       y_test=y_test,   p_test=p_test,   u_test=u_test)
    print("✅ Saved best EDL UQ scores to:", out_npz)


if __name__ == "__main__":
    main()
