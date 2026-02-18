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
    LaplaceConfig, LaplacePosterior,
    fit_laplace, laplace_predict,
    extract_map_weights, posterior_diagnostics,
)
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"
SPLIT_SEED = 42
TRAIN_SEEDS = [1, 2, 3, 4]

# ──────────────────────────────────────────────────────────────────────
# Laplace grid
#
# prior_precision (τ) controls how tight the Gaussian posterior is
# around w_MAP.  We search a broad log-scale range.
#
# Intuition:
#   τ = 1e-4  →  very loose prior, wide posterior, lots of uncertainty
#   τ = 1e+2  →  very tight prior, posterior ≈ delta at MAP, no uncertainty
#
# pred_method is fixed to "probit" for gridsearch speed.
# MC sampling is used at eval time.
# ──────────────────────────────────────────────────────────────────────
PRIOR_PRECISION_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

DEVICE_NAME = "cpu"

BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "cell2cell_baseline_split42_trainseeds1-4.json"


def fit_lr_reranker(p_valid, u_valid, y_valid):
    """Fit LR reranker on valid (no leakage)."""
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
    print("Device:", DEVICE_NAME)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("Prior precision grid:", PRIOR_PRECISION_GRID)
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

    # Load data
    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    tabnet_kwargs = dict(
        n_d=BASE["n_d"],
        n_a=BASE["n_a"],
        n_steps=BASE["n_steps"],
        gamma=BASE["gamma"],
        mask_type=BASE["mask_type"],
        cat_emb_dim=BASE["cat_emb_dim"],
    )

    train_kwargs = dict(
        lr=BASE["lr"],
        weight_decay=BASE["weight_decay"],
        max_epochs=BASE["max_epochs"],
        patience=BASE["patience"],
        batch_size=BASE["batch_size"],
        virtual_batch_size=BASE["virtual_batch_size"],
    )

    total_configs = len(PRIOR_PRECISION_GRID)
    total_runs = total_configs * len(TRAIN_SEEDS)
    print(f"\nTotal configs: {total_configs}, total runs: {total_runs}")
    print("(But TabNet is trained only once per seed — Laplace fitting is fast)\n")

    # Probit prediction config for gridsearch (fast)
    pred_cfg = LaplaceConfig(pred_method="probit")

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)
    best_cfg_dict = None

    for tau in PRIOR_PRECISION_GRID:
        tag = f"tau{tau:.0e}"
        cfg_dict = {
            **tabnet_kwargs,
            **train_kwargs,
            "prior_precision": tau,
            "pred_method": "probit",
            "split_seed": SPLIT_SEED,
            "train_seeds": TRAIN_SEEDS,
        }

        print(f"\n=== CONFIG {tag} (τ={tau}) ===")

        valid_praucs_map, valid_praucs_probit, valid_praucs_lr = [], [], []
        valid_lifts_probit = []
        valid_u_means = []
        log_margliks = []

        for train_seed in TRAIN_SEEDS:
            run_tag = f"laplace_{tag}_split{SPLIT_SEED}_trainseed{train_seed}"
            out_json = results_dir / f"{DATASET}_{run_tag}.json"

            # resume-safe
            if out_json.exists():
                payload = json.loads(out_json.read_text())
                v = payload["metrics"]["valid"]
                valid_praucs_map.append(v["map_auc_pr"])
                valid_praucs_probit.append(v["probit_auc_pr"])
                valid_praucs_lr.append(v["lr_auc_pr"])
                valid_lifts_probit.append(v["probit_lift10"])
                valid_u_means.append(v["u_mean"])
                log_margliks.append(v["log_marglik"])
                print(f"⏭️  SKIP {run_tag}")
                continue

            set_seed(train_seed)
            print(f"-> RUN {run_tag}")

            # Train TabNet baseline (same as baseline gridsearch winner)
            clf = train_tabnet_baseline(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                device_name=DEVICE_NAME,
                tabnet_kwargs=tabnet_kwargs,
                train_kwargs=train_kwargs,
                seed=train_seed,
            )

            # MAP predictions (unchanged across τ)
            p_map_valid = clf.predict_proba(X_valid)[:, 1]
            rep_map_valid = standard_report(y_valid, p_map_valid)

            # Fit Laplace posterior
            posterior = fit_laplace(
                clf, X_train, y_train,
                prior_precision=tau,
            )

            diag = posterior_diagnostics(posterior)
            print(f"  Posterior: cov_trace={diag['cov_trace']:.6f}, "
                  f"eigval_range=[{diag['cov_eigval_min']:.2e}, {diag['cov_eigval_max']:.2e}], "
                  f"log_marglik={posterior.log_marginal_likelihood:.1f}")

            # Probit predictions
            p_probit_valid, u_probit_valid = laplace_predict(
                posterior, clf, X_valid, pred_cfg,
            )
            rep_probit_valid = standard_report(y_valid, p_probit_valid)

            p_probit_test, u_probit_test = laplace_predict(
                posterior, clf, X_test, pred_cfg,
            )
            rep_probit_test = standard_report(y_test, p_probit_test)

            # LR reranker (fit on valid, evaluate on valid via LOO-style — but for
            # simplicity we report in-sample valid LR; test LR is in eval script)
            scaler, lr = fit_lr_reranker(p_probit_valid, u_probit_valid, y_valid)
            p_lr_valid = apply_lr_reranker(scaler, lr, p_probit_valid, u_probit_valid)
            rep_lr_valid = standard_report(y_valid, p_lr_valid)

            # Store metrics
            valid_metrics = {
                "map_auc_pr": rep_map_valid["auc_pr"],
                "map_lift10": rep_map_valid["lift10"],
                "probit_auc_pr": rep_probit_valid["auc_pr"],
                "probit_lift10": rep_probit_valid["lift10"],
                "lr_auc_pr": rep_lr_valid["auc_pr"],
                "lr_lift10": rep_lr_valid["lift10"],
                "u_mean": float(np.mean(u_probit_valid)),
                "u_std": float(np.std(u_probit_valid)),
                "log_marglik": float(posterior.log_marginal_likelihood),
            }

            test_metrics = {
                "probit_auc_pr": rep_probit_test["auc_pr"],
                "probit_lift10": rep_probit_test["lift10"],
                "u_mean": float(np.mean(u_probit_test)),
            }

            meta = RunMeta(dataset=DATASET, method="laplace", seed=train_seed, tag=run_tag)
            save_metrics_json(
                out_json, meta,
                metrics={"valid": valid_metrics, "test": test_metrics},
                config={**cfg_dict, "diagnostics": diag},
            )

            valid_praucs_map.append(valid_metrics["map_auc_pr"])
            valid_praucs_probit.append(valid_metrics["probit_auc_pr"])
            valid_praucs_lr.append(valid_metrics["lr_auc_pr"])
            valid_lifts_probit.append(valid_metrics["probit_lift10"])
            valid_u_means.append(valid_metrics["u_mean"])
            log_margliks.append(valid_metrics["log_marglik"])

        # Aggregate
        vp_mean = float(np.mean(valid_praucs_probit))
        vp_std = float(np.std(valid_praucs_probit, ddof=0))
        vl_mean = float(np.mean(valid_lifts_probit))
        vl_std = float(np.std(valid_lifts_probit, ddof=0))
        vu_mean = float(np.mean(valid_u_means))
        lml_mean = float(np.mean(log_margliks))

        agg_rows.append((tag, vp_mean, vp_std, vl_mean, vl_std, vu_mean, lml_mean))

        vm_mean = float(np.mean(valid_praucs_map))
        vlr_mean = float(np.mean(valid_praucs_lr))

        print(f"VALID  MAP_prauc={vm_mean:.4f}  probit_prauc={vp_mean:.4f}±{vp_std:.4f}  "
              f"LR_prauc={vlr_mean:.4f}  lift10={vl_mean:.4f}±{vl_std:.4f}  "
              f"u_mean={vu_mean:.4f}  log_marglik={lml_mean:.1f}")

        cand = (vp_mean, vl_mean)
        if cand > best_tuple:
            best_tuple = cand
            best_key = tag
            best_cfg_dict = cfg_dict

    # Leaderboard
    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)

    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    print(f"{'tag':15s}  {'probit_prauc':>14s}  {'lift10':>12s}  {'u_mean':>8s}  {'log_marglik':>12s}")
    for tag, vp, vps, vl, vls, vu, lml in agg_rows:
        print(f"{tag:15s}  {vp:.4f}±{vps:.4f}  {vl:.4f}±{vls:.4f}  {vu:.4f}  {lml:12.1f}")

    # Save best
    best_file = best_dir / f"{DATASET}_laplace_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg_dict,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "backbone_from": BASELINE_BEST_FILE.name,
    }, indent=2))
    print("\nSaved best Laplace config to:", best_file)
    print("Best:", best_key, "with (mean_valid_probit_prauc, mean_valid_lift) =", best_tuple)

    # Save canonical NPZ
    canonical_train_seed = TRAIN_SEEDS[0]
    set_seed(canonical_train_seed)

    clf = train_tabnet_baseline(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        device_name=DEVICE_NAME,
        tabnet_kwargs=tabnet_kwargs,
        train_kwargs=train_kwargs,
        seed=canonical_train_seed,
    )

    posterior = fit_laplace(
        clf, X_train, y_train,
        prior_precision=best_cfg_dict["prior_precision"],
    )

    p_valid, u_valid = laplace_predict(posterior, clf, X_valid, pred_cfg)
    p_test, u_test = laplace_predict(posterior, clf, X_test, pred_cfg)

    out_npz = uq_dir / f"{DATASET}_laplace_best_split{SPLIT_SEED}_trainseed{canonical_train_seed}.npz"
    save_uq_scores_npz(out_npz,
                       y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                       y_test=y_test, p_test=p_test, u_test=u_test)
    print("Saved best Laplace UQ scores to:", out_npz)


if __name__ == "__main__":
    main()
