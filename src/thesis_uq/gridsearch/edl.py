from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_edl import (
    EDLConfig, train_tabnet_edl, edl_predict_proba_unc,
)
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json, save_uq_scores_npz
from thesis_uq.data.registry import load_for_tabnet

REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"
SPLIT_SEED = 42
TRAIN_SEEDS = [1, 2, 3, 4]

# ──────────────────────────────────────────────────────────────────────
# EDL tuning grids
#
# kl_coef:          strength of the KL regulariser toward uniform Dirichlet
# anneal_epochs:    epochs over which KL penalty ramps from 0 → kl_coef
# edl_loss:         "mse" = MSE Bayes risk (Sensoy Eq.5)
#                   "ce"  = CE Bayes risk  (Sensoy Eq.4, sharper gradients)
# head_hidden_dim:  0  = single linear head (original)
#                   >0 = 2-layer MLP: Linear(n_d, hidden) → ReLU → Linear(hidden, K)
#
# Ablation groups:
#   A) Original:    edl_loss="mse", head_hidden_dim=0  (baseline reproduction)
#   B) Deep head:   edl_loss="mse", head_hidden_dim>0
#   C) CE loss:     edl_loss="ce",  head_hidden_dim=0
#   D) CE + deep:   edl_loss="ce",  head_hidden_dim>0
# ──────────────────────────────────────────────────────────────────────
KL_COEF_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
ANNEAL_EPOCHS_GRID = [10, 50]
EDL_LOSS_GRID = ["mse", "ce"]
# head_hidden_dim grid is built dynamically as [0, n_d // 2]

DEVICE_NAME = "cpu"
LAMBDA_SPARSE = 1e-3

# Load best baseline backbone
BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "cell2cell_baseline_split42_trainseeds1-4.json"


def main():
    print("Device:", DEVICE_NAME)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("KL coef grid:", KL_COEF_GRID)
    print("Anneal epochs grid:", ANNEAL_EPOCHS_GRID)
    print("EDL loss grid:", EDL_LOSS_GRID)
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

    n_d = BASE["n_d"]
    HEAD_HIDDEN_DIM_GRID = [0, n_d // 2]
    print("Head hidden dim grid:", HEAD_HIDDEN_DIM_GRID)

    # Load data once
    X, y, _, _, _, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    total_configs = (
        len(KL_COEF_GRID) * len(ANNEAL_EPOCHS_GRID)
        * len(EDL_LOSS_GRID) * len(HEAD_HIDDEN_DIM_GRID)
    )
    total_runs = total_configs * len(TRAIN_SEEDS)
    print(f"\nTotal configs: {total_configs}, total runs: {total_runs}")

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)
    best_cfg_dict = None

    for kl_coef in KL_COEF_GRID:
        for anneal_epochs in ANNEAL_EPOCHS_GRID:
            for edl_loss in EDL_LOSS_GRID:
                for head_hidden_dim in HEAD_HIDDEN_DIM_GRID:
                    head_label = "deep" if head_hidden_dim > 0 else "shallow"
                    tag = f"kl{kl_coef}_ann{anneal_epochs}_{edl_loss}_{head_label}"

                    cfg_dict = {
                        # backbone (from baseline)
                        "n_d": BASE["n_d"],
                        "n_a": BASE["n_a"],
                        "n_steps": BASE["n_steps"],
                        "gamma": BASE["gamma"],
                        "mask_type": BASE["mask_type"],
                        "cat_emb_dim": BASE["cat_emb_dim"],
                        # training (from baseline)
                        "lr": BASE["lr"],
                        "weight_decay": BASE["weight_decay"],
                        "max_epochs": BASE["max_epochs"],
                        "patience": BASE["patience"],
                        "batch_size": BASE["batch_size"],
                        "virtual_batch_size": BASE["virtual_batch_size"],
                        # EDL-specific (grid)
                        "kl_coef": kl_coef,
                        "anneal_epochs": anneal_epochs,
                        "edl_loss": edl_loss,
                        "head_hidden_dim": head_hidden_dim,
                        # fixed
                        "split_seed": SPLIT_SEED,
                        "train_seeds": TRAIN_SEEDS,
                    }

                    print(f"\n=== CONFIG {tag} ===")

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
                            valid_u_means.append(v["u_mean"])
                            test_praucs.append(t["auc_pr"])
                            test_lifts.append(t["lift10"])
                            test_u_means.append(t["u_mean"])
                            print(f"⏭️  SKIP {run_tag}")
                            continue

                        set_seed(train_seed)
                        print(f"-> RUN {run_tag}")

                        cfg = EDLConfig(
                            n_d=BASE["n_d"],
                            n_a=BASE["n_a"],
                            n_steps=BASE["n_steps"],
                            gamma=BASE["gamma"],
                            mask_type=BASE["mask_type"],
                            cat_emb_dim=BASE["cat_emb_dim"],
                            lr=BASE["lr"],
                            weight_decay=BASE["weight_decay"],
                            max_epochs=BASE["max_epochs"],
                            patience=BASE["patience"],
                            batch_size=BASE["batch_size"],
                            virtual_batch_size=BASE["virtual_batch_size"],
                            momentum=BASE.get("momentum", 0.02),
                            kl_coef=kl_coef,
                            anneal_epochs=anneal_epochs,
                            edl_loss=edl_loss,
                            head_hidden_dim=head_hidden_dim,
                        )

                        model = train_tabnet_edl(
                            X_train, y_train, X_valid, y_valid,
                            cat_idxs, cat_dims_list,
                            cfg=cfg,
                            device_name=DEVICE_NAME,
                            seed=train_seed,
                            lambda_sparse=LAMBDA_SPARSE,
                        )

                        # Predict
                        p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
                        p_test,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

                        valid_rep = standard_report(y_valid, p_valid)
                        test_rep = standard_report(y_test, p_test)

                        valid_rep["u_mean"] = float(np.mean(u_valid))
                        test_rep["u_mean"] = float(np.mean(u_test))

                        meta = RunMeta(dataset=DATASET, method="edl", seed=train_seed, tag=run_tag)
                        save_metrics_json(out_json, meta, metrics={"valid": valid_rep, "test": test_rep}, config=cfg_dict)

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
                        best_cfg_dict = cfg_dict

    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)

    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    for tag, v_mean, v_std, l_mean, l_std, u_mean, u_std in agg_rows:
        print(f"{tag:40s}  prauc={v_mean:.4f}±{v_std:.4f}  lift10={l_mean:.4f}±{l_std:.4f}  u_mean={u_mean:.4f}±{u_std:.4f}")

    # Save best config
    best_file = best_dir / f"{DATASET}_edl_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg_dict,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "backbone_from": BASELINE_BEST_FILE.name,
    }, indent=2))
    print("\nSaved best EDL config to:", best_file)
    print("Best:", best_key, "with (mean_valid_prauc, mean_valid_lift) =", best_tuple)

    # Save canonical NPZ for reranking/plots
    canonical_train_seed = TRAIN_SEEDS[0]
    set_seed(canonical_train_seed)

    cfg = EDLConfig(
        n_d=best_cfg_dict["n_d"],
        n_a=best_cfg_dict["n_a"],
        n_steps=best_cfg_dict["n_steps"],
        gamma=best_cfg_dict["gamma"],
        mask_type=best_cfg_dict["mask_type"],
        cat_emb_dim=best_cfg_dict["cat_emb_dim"],
        lr=best_cfg_dict["lr"],
        weight_decay=best_cfg_dict["weight_decay"],
        max_epochs=best_cfg_dict["max_epochs"],
        patience=best_cfg_dict["patience"],
        batch_size=best_cfg_dict["batch_size"],
        virtual_batch_size=best_cfg_dict["virtual_batch_size"],
        kl_coef=best_cfg_dict["kl_coef"],
        anneal_epochs=best_cfg_dict["anneal_epochs"],
        edl_loss=best_cfg_dict["edl_loss"],
        head_hidden_dim=best_cfg_dict["head_hidden_dim"],
    )

    model = train_tabnet_edl(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        cfg=cfg,
        device_name=DEVICE_NAME,
        seed=canonical_train_seed,
        lambda_sparse=LAMBDA_SPARSE,
    )

    p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
    p_test,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

    out_npz = uq_dir / f"{DATASET}_edl_best_split{SPLIT_SEED}_trainseed{canonical_train_seed}.npz"
    save_uq_scores_npz(out_npz,
                       y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                       y_test=y_test,   p_test=p_test,   u_test=u_test)
    print("Saved best EDL UQ scores to:", out_npz)


if __name__ == "__main__":
    main()
