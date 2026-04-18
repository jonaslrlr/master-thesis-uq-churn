"""
Conformal prediction for churn reranking.

Two variants:
  1) Split conformal: single model, calibrate on valid
  2) CV+ conformal:   K-fold models on train, OOF calibration, ensemble at test

Both produce a continuous uncertainty score (sum of conformal p-values)
compatible with the LR reranker framework.
"""
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold

from thesis_uq.seed import set_seed
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline


def nonconformity_scores(p, y):
    p = np.asarray(p).reshape(-1)
    y = np.asarray(y).reshape(-1)
    p_true = np.where(y == 1, p, 1 - p)
    return 1.0 - p_true


def conformal_pvalues(s_cal, p_test):
    s_cal = np.asarray(s_cal).reshape(-1)
    p_test = np.asarray(p_test).reshape(-1)
    n_cal = len(s_cal)
    s_if_1 = 1.0 - p_test
    s_if_0 = p_test
    pval_1 = (1 + np.sum(s_cal[:, None] >= s_if_1[None, :], axis=0)) / (1 + n_cal)
    pval_0 = (1 + np.sum(s_cal[:, None] >= s_if_0[None, :], axis=0)) / (1 + n_cal)
    return pval_0, pval_1


def conformal_uncertainty(s_cal, p_test):
    pval_0, pval_1 = conformal_pvalues(s_cal, p_test)
    return pval_0 + pval_1


def split_conformal_uncertainty(p_cal, y_cal, p_test):
    s_cal = nonconformity_scores(p_cal, y_cal)
    return conformal_uncertainty(s_cal, p_test)


def split_conformal_uncertainty_loo(p_cal, y_cal):
    p_cal = np.asarray(p_cal).reshape(-1)
    y_cal = np.asarray(y_cal).reshape(-1)
    s_cal = nonconformity_scores(p_cal, y_cal)
    n = len(s_cal)
    s_if_1 = 1.0 - p_cal
    s_if_0 = p_cal
    count_1 = np.sum(s_cal[:, None] >= s_if_1[None, :], axis=0)
    count_0 = np.sum(s_cal[:, None] >= s_if_0[None, :], axis=0)
    self_geq_1 = (s_cal >= s_if_1).astype(float)
    self_geq_0 = (s_cal >= s_if_0).astype(float)
    pval_1_loo = (1 + count_1 - self_geq_1) / (1 + (n - 1))
    pval_0_loo = (1 + count_0 - self_geq_0) / (1 + (n - 1))
    return pval_0_loo + pval_1_loo


def cv_plus_train_and_predict(
    X_train, y_train, X_valid, X_test,
    cat_idxs, cat_dims_list, base_cfg,
    n_folds=5, device="cpu", seed=0,
):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_train, n_valid, n_test = len(y_train), X_valid.shape[0], X_test.shape[0]

    p_oof = np.zeros(n_train)
    p_valid_folds = np.zeros((n_valid, n_folds))
    p_test_folds = np.zeros((n_test, n_folds))

    for fold_idx, (tr_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"    CV+ fold {fold_idx+1}/{n_folds} "
              f"(train={len(tr_idx)}, oof={len(oof_idx)})")
        fold_seed = seed * 100 + fold_idx
        set_seed(fold_seed)

        clf = train_tabnet_baseline(
            X_train[tr_idx], y_train[tr_idx],
            X_train[oof_idx], y_train[oof_idx],
            cat_idxs, cat_dims_list,
            device_name=device,
            n_d=int(base_cfg["n_d"]), n_a=int(base_cfg["n_a"]),
            n_steps=int(base_cfg["n_steps"]), gamma=float(base_cfg["gamma"]),
            mask_type=str(base_cfg["mask_type"]),
            cat_emb_dim=int(base_cfg["cat_emb_dim"]),
            lr=float(base_cfg["lr"]), weight_decay=float(base_cfg["weight_decay"]),
            max_epochs=int(base_cfg["max_epochs"]), patience=int(base_cfg["patience"]),
            batch_size=int(base_cfg["batch_size"]),
            virtual_batch_size=int(base_cfg["virtual_batch_size"]),
            seed=fold_seed,
        )

        p_oof[oof_idx] = clf.predict_proba(X_train[oof_idx])[:, 1]
        p_valid_folds[:, fold_idx] = clf.predict_proba(X_valid)[:, 1]
        p_test_folds[:, fold_idx] = clf.predict_proba(X_test)[:, 1]

    s_oof = nonconformity_scores(p_oof, y_train)
    p_valid_ens = p_valid_folds.mean(axis=1)
    p_test_ens = p_test_folds.mean(axis=1)
    u_valid_std = p_valid_folds.std(axis=1)
    u_test_std = p_test_folds.std(axis=1)
    u_valid_cp = conformal_uncertainty(s_oof, p_valid_ens)
    u_test_cp = conformal_uncertainty(s_oof, p_test_ens)

    return {
        "p_oof": p_oof, "s_oof": s_oof,
        "p_valid_folds": p_valid_folds, "p_test_folds": p_test_folds,
        "p_valid_ens": p_valid_ens, "p_test_ens": p_test_ens,
        "u_valid_std": u_valid_std, "u_test_std": u_test_std,
        "u_valid_cp": u_valid_cp, "u_test_cp": u_test_cp,
    }
