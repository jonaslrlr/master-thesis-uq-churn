"""
Nonlinear reranker test with matched p-only baselines.

Tests whether nonlinear models extract more value from uncertainty
than logistic regression. Each model is tested on [u,p] and [p]
alone; the per-seed paired difference delta(u) isolates the
contribution of u.

Models:
  - LR: logistic regression (fixed C=1, no tuning)
  - Spline LR: cubic splines + pairwise interaction terms + LR
    (inner CV over knots and regularisation)
  - XGBoost: gradient boosting (inner CV over depth, n_est, lr, mcw)
  - CatBoost: ordered boosting (inner CV over depth, iterations, lr)

All except LR use inner 3-fold CV on validation for hyperparameter
selection. Final evaluation on held-out test set using thesis
PR-AUC metric (trapezoid over precision-recall curve).

Usage:
    python -m thesis_uq.eval.eval_nonlinear_reranker
    python -m thesis_uq.eval.eval_nonlinear_reranker --dataset bank --seeds 5-7
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    make_scorer,
    precision_recall_curve,
    auc as sk_auc,
)
from sklearn.preprocessing import MinMaxScaler, SplineTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from thesis_uq.metrics.ranking import standard_report

warnings.filterwarnings("ignore")


# ── Consistent PR-AUC scorer (same as thesis metric) ─────────
def _prauc_score(y_true, y_prob):
    if y_true.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(sk_auc(recall, precision))


PRAUC_SCORER = make_scorer(_prauc_score, needs_proba=True)


def guess_repo_root():
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "reports").exists():
            return p
    return Path.cwd()


# ── Candidate builders ───────────────────────────────────────

def make_spline_candidates(n_features):
    """Spline LR candidates. Pairwise interactions only when n_features > 1."""
    configs = []
    for n_knots in [3, 4, 5, 7]:
        for C in [0.01, 0.1, 1.0, 10.0]:
            configs.append((
                f"add_k={n_knots}_C={C}",
                Pipeline([
                    ("scaler", MinMaxScaler()),
                    ("splines", SplineTransformer(
                        n_knots=n_knots, degree=3, include_bias=False
                    )),
                    ("lr", LogisticRegression(max_iter=2000, C=C)),
                ])
            ))
            if n_features > 1:
                configs.append((
                    f"int_k={n_knots}_C={C}",
                    Pipeline([
                        ("scaler", MinMaxScaler()),
                        ("splines", SplineTransformer(
                            n_knots=n_knots, degree=3, include_bias=False
                        )),
                        ("interactions", PolynomialFeatures(
                            degree=2, interaction_only=True,
                            include_bias=False
                        )),
                        ("lr", LogisticRegression(max_iter=2000, C=C)),
                    ])
                ))
    return configs


def make_xgb_candidates():
    """XGBoost candidates with generous depth range."""
    configs = []
    for depth in [3, 5, 7, 10]:
        for n_est in [100, 300, 500]:
            for lr_val in [0.01, 0.05, 0.1]:
                for mcw in [1, 5, 10]:
                    configs.append((
                        f"d={depth}_n={n_est}_lr={lr_val}_mcw={mcw}",
                        XGBClassifier(
                            max_depth=depth,
                            n_estimators=n_est,
                            learning_rate=lr_val,
                            min_child_weight=mcw,
                            subsample=0.8,
                            colsample_bytree=1.0,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric="logloss",
                            verbosity=0,
                        )
                    ))
    return configs


def make_catboost_candidates():
    """CatBoost candidates with generous depth range."""
    configs = []
    for depth in [3, 5, 7, 10]:
        for n_est in [100, 300, 500]:
            for lr_val in [0.01, 0.05, 0.1]:
                configs.append((
                    f"d={depth}_n={n_est}_lr={lr_val}",
                    CatBoostClassifier(
                        depth=depth,
                        iterations=n_est,
                        learning_rate=lr_val,
                        random_seed=42,
                        verbose=0,
                        allow_writing_files=False,
                    )
                ))
    return configs


# ── Model selection and evaluation ───────────────────────────

def select_best(candidates_list, X, y, cv_folds=3):
    """Inner CV to select best model using thesis-consistent PR-AUC."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_score = -np.inf
    best_label = None
    best_model = None

    for label, model in candidates_list:
        try:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=PRAUC_SCORER
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_label = label
                best_model = model
        except Exception:
            continue

    return best_label, best_model, best_score


def eval_model(model, X_train, y_train, X_test, y_test):
    """Fit on train, evaluate on test using thesis metrics."""
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    rep = standard_report(y_test, p_test)
    return rep["auc_pr"], rep["lift10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seeds", default="5-15")
    ap.add_argument("--repo_root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else guess_repo_root()
    uq_dir = repo_root / "reports" / "uq_scores"
    out_dir = repo_root / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_str = args.seeds.strip()
    if "-" in seeds_str:
        a, b = seeds_str.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in seeds_str.split(",")]

    datasets = [args.dataset] if args.dataset else [
        "bank", "cell2cell", "telco", "delft", "cdr", "chile"
    ]

    methods = ["mc_dropout", "laplace", "edl", "cp_single", "cp_cv", "cp_cv_std"]
    model_families = ["LR", "Spline_LR", "XGBoost", "CatBoost"]

    all_results = {}

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"  {ds.upper()}")
        print(f"{'='*70}")

        ds_results = {}

        for method in methods:
            # Per-seed paired data: store both [u,p] and [p] per seed
            seed_data = {
                "base": {"prauc": [], "lift10": []},
            }
            for fam in model_families:
                seed_data[fam] = {
                    "up_prauc": [], "up_lift10": [], "up_cfg": [],
                    "p_prauc": [], "p_lift10": [], "p_cfg": [],
                    "delta_prauc": [], "delta_lift10": [],
                }

            n_found = 0

            for seed in seeds:
                npz_path = uq_dir / f"{ds}_{method}_eval_split{args.split_seed}_trainseed{seed}.npz"
                if not npz_path.exists():
                    continue

                data = np.load(npz_path)
                y_valid = data["y_valid"]
                p_valid = data["p_valid"]
                u_valid = data["u_valid"]
                y_test = data["y_test"]
                p_test = data["p_test"]
                u_test = data["u_test"]

                X_up_val = np.column_stack([u_valid, p_valid])
                X_up_test = np.column_stack([u_test, p_test])
                X_p_val = p_valid.reshape(-1, 1)
                X_p_test = p_test.reshape(-1, 1)

                # Base
                base_rep = standard_report(y_test, p_test)
                seed_data["base"]["prauc"].append(base_rep["auc_pr"])
                seed_data["base"]["lift10"].append(base_rep["lift10"])

                # ── LR (fixed C=1, no tuning) ──
                for fam_name, make_model in [("LR", None)]:
                    pipe_up = Pipeline([
                        ("scaler", MinMaxScaler()),
                        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
                    ])
                    pipe_p = Pipeline([
                        ("scaler", MinMaxScaler()),
                        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
                    ])
                    prauc_up, lift_up = eval_model(
                        pipe_up, X_up_val, y_valid, X_up_test, y_test
                    )
                    prauc_p, lift_p = eval_model(
                        pipe_p, X_p_val, y_valid, X_p_test, y_test
                    )
                    sd = seed_data[fam_name]
                    sd["up_prauc"].append(prauc_up)
                    sd["up_lift10"].append(lift_up)
                    sd["up_cfg"].append("C=1")
                    sd["p_prauc"].append(prauc_p)
                    sd["p_lift10"].append(lift_p)
                    sd["p_cfg"].append("C=1")
                    sd["delta_prauc"].append(prauc_up - prauc_p)
                    sd["delta_lift10"].append(lift_up - lift_p)

                # ── Spline LR (inner CV) ──
                cands_up = make_spline_candidates(n_features=2)
                cands_p = make_spline_candidates(n_features=1)

                lbl_up, pipe_up, _ = select_best(cands_up, X_up_val, y_valid)
                lbl_p, pipe_p, _ = select_best(cands_p, X_p_val, y_valid)

                if pipe_up is not None and pipe_p is not None:
                    prauc_up, lift_up = eval_model(
                        pipe_up, X_up_val, y_valid, X_up_test, y_test
                    )
                    prauc_p, lift_p = eval_model(
                        pipe_p, X_p_val, y_valid, X_p_test, y_test
                    )
                    sd = seed_data["Spline_LR"]
                    sd["up_prauc"].append(prauc_up)
                    sd["up_lift10"].append(lift_up)
                    sd["up_cfg"].append(lbl_up)
                    sd["p_prauc"].append(prauc_p)
                    sd["p_lift10"].append(lift_p)
                    sd["p_cfg"].append(lbl_p)
                    sd["delta_prauc"].append(prauc_up - prauc_p)
                    sd["delta_lift10"].append(lift_up - lift_p)

                # ── XGBoost (inner CV) ──
                xgb_cands = make_xgb_candidates()

                lbl_up, mdl_up, _ = select_best(xgb_cands, X_up_val, y_valid)
                lbl_p, mdl_p, _ = select_best(xgb_cands, X_p_val, y_valid)

                if mdl_up is not None and mdl_p is not None:
                    prauc_up, lift_up = eval_model(
                        mdl_up, X_up_val, y_valid, X_up_test, y_test
                    )
                    prauc_p, lift_p = eval_model(
                        mdl_p, X_p_val, y_valid, X_p_test, y_test
                    )
                    sd = seed_data["XGBoost"]
                    sd["up_prauc"].append(prauc_up)
                    sd["up_lift10"].append(lift_up)
                    sd["up_cfg"].append(lbl_up)
                    sd["p_prauc"].append(prauc_p)
                    sd["p_lift10"].append(lift_p)
                    sd["p_cfg"].append(lbl_p)
                    sd["delta_prauc"].append(prauc_up - prauc_p)
                    sd["delta_lift10"].append(lift_up - lift_p)

                # ── CatBoost (inner CV) ──
                cat_cands = make_catboost_candidates()

                lbl_up, mdl_up, _ = select_best(cat_cands, X_up_val, y_valid)
                lbl_p, mdl_p, _ = select_best(cat_cands, X_p_val, y_valid)

                if mdl_up is not None and mdl_p is not None:
                    prauc_up, lift_up = eval_model(
                        mdl_up, X_up_val, y_valid, X_up_test, y_test
                    )
                    prauc_p, lift_p = eval_model(
                        mdl_p, X_p_val, y_valid, X_p_test, y_test
                    )
                    sd = seed_data["CatBoost"]
                    sd["up_prauc"].append(prauc_up)
                    sd["up_lift10"].append(lift_up)
                    sd["up_cfg"].append(lbl_up)
                    sd["p_prauc"].append(prauc_p)
                    sd["p_lift10"].append(lift_p)
                    sd["p_cfg"].append(lbl_p)
                    sd["delta_prauc"].append(prauc_up - prauc_p)
                    sd["delta_lift10"].append(lift_up - lift_p)

                n_found += 1
                print(f"    seed {seed} done")

            if n_found == 0:
                continue

            # Aggregate with paired deltas
            summary = {"n_seeds_total": n_found}
            summary["base_prauc_mean"] = float(np.mean(seed_data["base"]["prauc"]))
            summary["base_prauc_std"] = float(np.std(seed_data["base"]["prauc"]))
            summary["base_lift10_mean"] = float(np.mean(seed_data["base"]["lift10"]))

            for fam in model_families:
                sd = seed_data[fam]
                n_paired = len(sd["delta_prauc"])
                summary[f"{fam}_n_seeds"] = n_paired

                if n_paired > 0:
                    summary[f"{fam}_up_prauc_mean"] = float(np.mean(sd["up_prauc"]))
                    summary[f"{fam}_up_prauc_std"] = float(np.std(sd["up_prauc"]))
                    summary[f"{fam}_p_prauc_mean"] = float(np.mean(sd["p_prauc"]))
                    summary[f"{fam}_p_prauc_std"] = float(np.std(sd["p_prauc"]))
                    summary[f"{fam}_up_lift10_mean"] = float(np.mean(sd["up_lift10"]))
                    summary[f"{fam}_p_lift10_mean"] = float(np.mean(sd["p_lift10"]))

                    # Paired deltas (mean of per-seed differences)
                    summary[f"{fam}_delta_prauc_mean"] = float(np.mean(sd["delta_prauc"]))
                    summary[f"{fam}_delta_prauc_std"] = float(np.std(sd["delta_prauc"]))
                    summary[f"{fam}_delta_lift10_mean"] = float(np.mean(sd["delta_lift10"]))
                    summary[f"{fam}_delta_lift10_std"] = float(np.std(sd["delta_lift10"]))

                    if sd["up_cfg"]:
                        summary[f"{fam}_up_best_cfg"] = Counter(
                            sd["up_cfg"]
                        ).most_common(1)[0][0]
                    if sd["p_cfg"]:
                        summary[f"{fam}_p_best_cfg"] = Counter(
                            sd["p_cfg"]
                        ).most_common(1)[0][0]

            ds_results[method] = summary

            # Print
            base_m = summary["base_prauc_mean"]
            print(f"\n  {method}:")
            print(f"    {'':16s} {'[u,p]':>10s}  {'[p]':>10s}"
                  f"  {'delta(u)':>12s}  {'n':>3s}  {'config [u,p]'}")
            print(f"    {'Base prob':16s} {base_m:>10.4f}")

            for fam in model_families:
                n = summary.get(f"{fam}_n_seeds", 0)
                if n == 0:
                    print(f"    {fam:16s} {'FAILED':>10s}")
                    continue
                up_m = summary[f"{fam}_up_prauc_mean"]
                p_m = summary[f"{fam}_p_prauc_mean"]
                d_m = summary[f"{fam}_delta_prauc_mean"]
                d_s = summary[f"{fam}_delta_prauc_std"]
                cfg = summary.get(f"{fam}_up_best_cfg", "")
                print(f"    {fam:16s} {up_m:>10.4f}  {p_m:>10.4f}"
                      f"  {d_m:>+7.4f}±{d_s:.4f}  {n:>3d}  {cfg}")

        all_results[ds] = ds_results

    # Save
    out_file = out_dir / "nonlinear_reranker_comparison.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n✅ Saved: {out_file}")

    # Summary table: paired deltas
    print(f"\n{'='*70}")
    print("SUMMARY: mean paired delta(u) ± std")
    print("(positive = uncertainty helps beyond p alone)")
    print(f"{'='*70}")
    header = f"  {'Dataset':10s} {'Method':12s}"
    for fam in model_families:
        header += f"  {fam:>14s}"
    print(header)
    print(f"  {'-'*78}")
    for ds in datasets:
        if ds not in all_results:
            continue
        for method in methods:
            if method not in all_results[ds]:
                continue
            r = all_results[ds][method]
            line = f"  {ds:10s} {method:12s}"
            for fam in model_families:
                d_m = r.get(f"{fam}_delta_prauc_mean", float("nan"))
                d_s = r.get(f"{fam}_delta_prauc_std", float("nan"))
                if np.isnan(d_m):
                    line += f"  {'n/a':>14s}"
                else:
                    line += f"  {d_m:>+.4f}±{d_s:.4f}"
            print(line)


if __name__ == "__main__":
    main()
