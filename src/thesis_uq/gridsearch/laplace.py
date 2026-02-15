from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.io import RunMeta, save_metrics_json

from thesis_uq.models.tabnet_laplace import (
    LaplaceConfig,
    extract_tabnet_representation,
    laplace_from_tabnet_features,
)

from thesis_uq.data.registry import load_for_tabnet
from thesis_uq.eval._cli import parse_eval_args


REPO_ROOT = Path("/Users/jonaslorler/master-thesis-uq-churn")
DATASET = "cell2cell"

SPLIT_SEED = 42
TRAIN_SEEDS = [1, 2, 3, 4]
DEVICE_NAME = "cpu"

# Tune Laplace prior precision
PRIOR_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]

# Must exist (run baseline gridsearch for cell2cell first)
BASELINE_BEST_FILE = REPO_ROOT / "reports" / "best" / "cell2cell_baseline_split42_trainseeds1-4.json"


def main():
    print("Dataset:", DATASET)
    print("Device:", DEVICE_NAME)
    print("Split seed (fixed):", SPLIT_SEED)
    print("Train seeds (vary):", TRAIN_SEEDS)
    print("Prior grid:", PRIOR_GRID)
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

    # Data (via registry)
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_for_tabnet(DATASET, REPO_ROOT)

    # Fixed split
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=SPLIT_SEED)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    tabnet_kwargs = dict(
        n_d=BASE["n_d"], n_a=BASE["n_a"], n_steps=BASE["n_steps"],
        gamma=BASE["gamma"], mask_type=BASE["mask_type"], cat_emb_dim=BASE["cat_emb_dim"],
    )
    train_kwargs = dict(
        lr=BASE["lr"], weight_decay=BASE["weight_decay"],
        max_epochs=BASE["max_epochs"], patience=BASE["patience"],
        batch_size=BASE["batch_size"], virtual_batch_size=BASE["virtual_batch_size"],
    )

    agg_rows = []
    best_key = None
    best_tuple = (-np.inf, -np.inf)  # (valid prauc mean, valid lift mean)
    best_cfg = None

    for prior_prec in PRIOR_GRID:
        tag = f"prior{prior_prec}"
        cfg = LaplaceConfig(
            prior_prec=float(prior_prec),
            max_iter=200,
            n_samples=300,
            jitter=1e-6,
            standardize=True,
        )

        print("\n=== CONFIG", tag, "===")

        valid_praucs, valid_lifts, valid_u_means = [], [], []

        for train_seed in TRAIN_SEEDS:
            run_tag = f"laplace_{tag}_split{SPLIT_SEED}_trainseed{train_seed}"
            out_json = results_dir / f"{DATASET}_{run_tag}.json"

            # resume-safe
            if out_json.exists():
                payload = json.loads(out_json.read_text())
                v = payload["metrics"]["valid"]
                valid_praucs.append(v["auc_pr"])
                valid_lifts.append(v["lift10"])
                valid_u_means.append(v.get("u_mean", np.nan))
                print("⏭️ SKIP", run_tag)
                continue

            set_seed(train_seed)
            print("-> RUN", run_tag)

            # Train backbone (deterministic dropout=0)
            clf = train_tabnet_baseline(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                device_name=DEVICE_NAME,
                dropout=0.0,
                seed=train_seed,
                **tabnet_kwargs,
                **train_kwargs,
            )

            # Extract TabNet representation
            feat_train = extract_tabnet_representation(clf, X_train)
            feat_valid = extract_tabnet_representation(clf, X_valid)
            feat_test  = extract_tabnet_representation(clf, X_test)

            # Laplace head (old/simple API)
            # returns: p_map_valid, p_map_test, p_lap_valid_mean, u_valid, p_lap_test_mean, u_test
            (
                p_map_valid, p_map_test,
                p_lap_valid, u_valid,
                p_lap_test,  u_test,
            ) = laplace_from_tabnet_features(
                feat_train, y_train,
                feat_valid, feat_test,
                cfg,
                seed=train_seed,
            )

            valid_rep = standard_report(y_valid, p_lap_valid)
            test_rep  = standard_report(y_test,  p_lap_test)

            valid_rep["u_mean"] = float(np.mean(u_valid))
            test_rep["u_mean"]  = float(np.mean(u_test))

            meta = RunMeta(dataset=DATASET, method="laplace", seed=train_seed, tag=run_tag)
            save_metrics_json(
                out_json,
                meta,
                metrics={"valid": valid_rep, "test": test_rep},
                config={
                    "prior_prec": cfg.prior_prec,
                    "max_iter": cfg.max_iter,
                    "n_samples": cfg.n_samples,
                    "standardize": cfg.standardize,
                    "jitter": cfg.jitter,
                    "split_seed": SPLIT_SEED,
                    "train_seed": train_seed,
                    "device_name": DEVICE_NAME,
                    "backbone_from": BASELINE_BEST_FILE.name,
                    **tabnet_kwargs,
                    **train_kwargs,
                },
            )

            valid_praucs.append(valid_rep["auc_pr"])
            valid_lifts.append(valid_rep["lift10"])
            valid_u_means.append(valid_rep["u_mean"])

        v_mean = float(np.mean(valid_praucs)); v_std = float(np.std(valid_praucs, ddof=0))
        l_mean = float(np.mean(valid_lifts));  l_std = float(np.std(valid_lifts, ddof=0))
        u_mean = float(np.mean(valid_u_means)); u_std = float(np.std(valid_u_means, ddof=0))

        agg_rows.append((tag, v_mean, v_std, l_mean, l_std, u_mean, u_std))
        print(
            f"VALID prauc={v_mean:.4f}±{v_std:.4f}  "
            f"lift10={l_mean:.4f}±{l_std:.4f}  "
            f"u_mean={u_mean:.4f}±{u_std:.4f}"
        )

        cand = (v_mean, l_mean)
        if cand > best_tuple:
            best_tuple = cand
            best_key = tag
            best_cfg = cfg

    agg_rows.sort(key=lambda x: (x[1], x[3]), reverse=True)
    print("\n=== LEADERBOARD (VALID mean over training seeds) ===")
    for tag, v_mean, v_std, l_mean, l_std, u_mean, u_std in agg_rows:
        print(
            f"{tag:10s}  prauc={v_mean:.4f}±{v_std:.4f}  "
            f"lift10={l_mean:.4f}±{l_std:.4f}  "
            f"u_mean={u_mean:.4f}±{u_std:.4f}"
        )

    assert best_cfg is not None

    # Save best Laplace config JSON
    best_file = best_dir / f"{DATASET}_laplace_split{SPLIT_SEED}_trainseeds{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}.json"
    best_file.write_text(json.dumps({
        "best_tag": best_key,
        "config": best_cfg.__dict__,
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "backbone_from": BASELINE_BEST_FILE.name,
    }, indent=2))
    print("\n✅ Saved best Laplace config to:", best_file)
    print("Best:", best_key, "with (mean_valid_prauc, mean_valid_lift) =", best_tuple)

    # Save ONE canonical NPZ (seed 1) with STANDARD keys (so other scripts keep working)
    canonical_seed = TRAIN_SEEDS[0]
    set_seed(canonical_seed)

    clf = train_tabnet_baseline(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        device_name=DEVICE_NAME,
        dropout=0.0,
        seed=canonical_seed,
        **tabnet_kwargs,
        **train_kwargs,
    )
    feat_train = extract_tabnet_representation(clf, X_train)
    feat_valid = extract_tabnet_representation(clf, X_valid)
    feat_test  = extract_tabnet_representation(clf, X_test)

    (
        p_map_valid, p_map_test,
        p_lap_valid, u_valid,
        p_lap_test,  u_test,
    ) = laplace_from_tabnet_features(
        feat_train, y_train,
        feat_valid, feat_test,
        best_cfg,
        seed=canonical_seed,
    )

    out_npz = uq_dir / f"{DATASET}_laplace_best_split{SPLIT_SEED}_trainseed{canonical_seed}.npz"
    np.savez(
        out_npz,
        y_valid=y_valid, p_valid=p_lap_valid, u_valid=u_valid,
        y_test=y_test,   p_test=p_lap_test,  u_test=u_test,
        p_map_valid=p_map_valid, p_map_test=p_map_test,
    )
    print("✅ Saved Laplace UQ scores to:", out_npz)


if __name__ == "__main__":
    main()
