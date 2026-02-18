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

# Fresh eval seeds (never seen during gridsearch)
EVAL_SEEDS = list(range(5, 15))  # seeds 5..14

DEVICE_NAME = "cpu"
LAMBDA_SPARSE = 1e-3

# Best EDL config from gridsearch
BEST_FILE = REPO_ROOT / "reports" / "best" / f"{DATASET}_edl_split{SPLIT_SEED}_trainseeds1-4.json"


def main():
    print("=== EDL EVALUATION (unbiased test) ===")
    print("Dataset:", DATASET)
    print("Split seed:", SPLIT_SEED)
    print("Eval seeds:", EVAL_SEEDS)
    print("Best file:", BEST_FILE)

    best = json.loads(BEST_FILE.read_text())
    CFG = best["config"]
    print("\nBest tag:", best["best_tag"])
    print("Config:", json.dumps(CFG, indent=2))

    results_dir = REPO_ROOT / "reports" / "results"
    uq_dir = REPO_ROOT / "reports" / "uq_scores"
    results_dir.mkdir(parents=True, exist_ok=True)
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

    all_test = []

    for seed in EVAL_SEEDS:
        run_tag = f"edl_eval_split{SPLIT_SEED}_trainseed{seed}"
        out_json = results_dir / f"{DATASET}_{run_tag}.json"

        # resume-safe
        if out_json.exists():
            payload = json.loads(out_json.read_text())
            t = payload["metrics"]["test"]
            all_test.append(t)
            print(f"⏭️  SKIP {run_tag}  prauc={t['auc_pr']:.5f}  lift10={t['lift10']:.4f}  u_mean={t['u_mean']:.4f}")
            continue

        set_seed(seed)
        print(f"\n-> RUN {run_tag}")

        model = train_tabnet_edl(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            cfg=cfg,
            device_name=DEVICE_NAME,
            seed=seed,
            lambda_sparse=LAMBDA_SPARSE,
        )

        p_valid, u_valid = edl_predict_proba_unc(model, X_valid, device=DEVICE_NAME)
        p_test,  u_test  = edl_predict_proba_unc(model, X_test,  device=DEVICE_NAME)

        valid_rep = standard_report(y_valid, p_valid)
        test_rep = standard_report(y_test, p_test)

        valid_rep["u_mean"] = float(np.mean(u_valid))
        test_rep["u_mean"] = float(np.mean(u_test))
        test_rep["u_std"] = float(np.std(u_test))
        test_rep["u_median"] = float(np.median(u_test))

        meta = RunMeta(dataset=DATASET, method="edl", seed=seed, tag=run_tag)
        save_metrics_json(out_json, meta, metrics={"valid": valid_rep, "test": test_rep}, config=CFG)

        # Save NPZ per seed for downstream analysis
        npz_path = uq_dir / f"{DATASET}_edl_eval_split{SPLIT_SEED}_trainseed{seed}.npz"
        save_uq_scores_npz(npz_path,
                           y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                           y_test=y_test,   p_test=p_test,   u_test=u_test)

        all_test.append(test_rep)
        print(f"   TEST prauc={test_rep['auc_pr']:.5f}  lift10={test_rep['lift10']:.4f}  u_mean={test_rep['u_mean']:.4f}")

    # Aggregate
    print("\n" + "=" * 70)
    print("EDL TEST RESULTS (mean ± std over eval seeds)")
    print("=" * 70)

    for metric in ["auc_pr", "lift10", "auc_roc"]:
        vals = [t[metric] for t in all_test]
        print(f"  {metric:12s} = {np.mean(vals):.5f} ± {np.std(vals, ddof=0):.5f}")

    u_means = [t["u_mean"] for t in all_test]
    print(f"  {'u_mean':12s} = {np.mean(u_means):.5f} ± {np.std(u_means, ddof=0):.5f}")

    print(f"\nConfig: edl_loss={cfg.edl_loss}, head_hidden_dim={cfg.head_hidden_dim}, "
          f"kl_coef={cfg.kl_coef}, anneal_epochs={cfg.anneal_epochs}")


if __name__ == "__main__":
    main()
