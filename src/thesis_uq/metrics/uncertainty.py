"""
Uncertainty quality metrics.

These evaluate the uncertainty signal itself — does the model know
when it's wrong, BEYOND what probability confidence (|p-0.5|) already tells you?

Main metrics:
  - selective_prauc:                rejection curve (remove most uncertain, measure PR-AUC)
  - auco:                           area under rejection curve, oracle-normalised
  - uncertainty_binned_ece:         ECE stratified by uncertainty terciles
  - uncertainty_error_correlation:  Spearman raw + residual (after removing |p-0.5| effect)
  - conditional_uncertainty_accuracy: accuracy gap between low/high unc within probability bins
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, auc as sk_auc


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """PR-AUC for a subset. Returns NaN if fewer than 2 positives."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if y.sum() < 2 or len(np.unique(y)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(y, p)
    return float(sk_auc(recall, precision))


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE on a subset."""
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    if len(y_true) == 0:
        return np.nan

    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == bins[-1]:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        total += n_bin * abs(y_true[mask].mean() - prob[mask].mean())
    return float(total / len(y_true))


# ─────────────────────────────────────────────────────────────────────
# Selective prediction (rejection curve)
# ─────────────────────────────────────────────────────────────────────

def selective_prauc(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    fractions: np.ndarray | None = None,
) -> list[Tuple[float, float]]:
    """
    Rejection curve: remove the most uncertain fraction, compute PR-AUC
    on the remaining samples.

    Parameters
    ----------
    y_true : array (N,)
    prob   : array (N,) predicted P(churn)
    unc    : array (N,) uncertainty (higher = more uncertain)
    fractions : array of floats in [0, 1)
        Fraction of samples to REMOVE (most uncertain first).
        Default: 0%, 5%, 10%, ..., 90%.

    Returns
    -------
    List of (fraction_removed, prauc_on_remaining).
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    if fractions is None:
        fractions = np.arange(0.0, 0.95, 0.05)

    # sort by uncertainty: most certain first
    order = np.argsort(unc)

    results = []
    for frac in fractions:
        n_keep = int(len(y_true) * (1.0 - frac))
        if n_keep < 20:
            break
        idx = order[:n_keep]
        val = _pr_auc(y_true[idx], prob[idx])
        if not np.isnan(val):
            results.append((float(frac), val))

    return results


def selective_lift10(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    fractions: np.ndarray | None = None,
) -> list[Tuple[float, float]]:
    """
    Rejection curve using lift@10 instead of PR-AUC.
    Same logic: remove most uncertain, measure lift on remainder.
    """
    import pandas as pd

    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    if fractions is None:
        fractions = np.arange(0.0, 0.95, 0.05)

    order = np.argsort(unc)

    results = []
    for frac in fractions:
        n_keep = int(len(y_true) * (1.0 - frac))
        if n_keep < 20:
            break
        idx = order[:n_keep]
        y_sub = y_true[idx]
        p_sub = prob[idx]

        base = y_sub.mean()
        if base == 0:
            continue
        df = pd.DataFrame({"y": y_sub, "s": p_sub}).sort_values("s", ascending=False)
        top = df.head(len(df) // 10)
        if len(top) == 0:
            continue
        lift = float(top["y"].mean() / base)
        results.append((float(frac), lift))

    return results


# ─────────────────────────────────────────────────────────────────────
# AUCO (Area Under Coverage-accuracy/PRAUC curve, oracle-normalised)
# ─────────────────────────────────────────────────────────────────────

def auco(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    fractions: np.ndarray | None = None,
) -> dict:
    """
    Area under the selective prediction curve.

    Returns the raw area, oracle area (removing actually-wrong predictions
    first), random area (removing predictions randomly), and the normalised
    AUCO = (method - random) / (oracle - random).

    Values:
      1.0 = perfect uncertainty (as good as the oracle)
      0.0 = no better than random
      <0  = worse than random
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    if fractions is None:
        fractions = np.arange(0.0, 0.95, 0.05)

    # Method curve
    method_curve = selective_prauc(y_true, prob, unc, fractions)

    # Oracle: uncertainty = |error|, so wrong predictions removed first
    pred_labels = (prob >= 0.5).astype(int)
    oracle_unc = np.abs(y_true - prob)  # high for wrong predictions
    oracle_curve = selective_prauc(y_true, prob, oracle_unc, fractions)

    # Random: use random "uncertainty"
    rng = np.random.default_rng(42)
    random_unc = rng.random(len(y_true))
    random_curve = selective_prauc(y_true, prob, random_unc, fractions)

    def _area(curve):
        if len(curve) < 2:
            return np.nan
        x = np.array([c[0] for c in curve])
        y = np.array([c[1] for c in curve])
        return float(np.trapz(y, x))

    area_method = _area(method_curve)
    area_oracle = _area(oracle_curve)
    area_random = _area(random_curve)

    denom = area_oracle - area_random
    if abs(denom) < 1e-10:
        normalised = 0.0
    else:
        normalised = (area_method - area_random) / denom

    return {
        "auco_raw": area_method,
        "auco_oracle": area_oracle,
        "auco_random": area_random,
        "auco_normalised": float(normalised),
    }


# ─────────────────────────────────────────────────────────────────────
# Uncertainty-binned ECE
# ─────────────────────────────────────────────────────────────────────

def uncertainty_binned_ece(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    n_bins: int = 3,
    ece_bins: int = 10,
) -> dict:
    """
    Split test set into uncertainty quantiles, compute ECE within each.

    Good uncertainty: low-unc group has low ECE, high-unc group has high ECE.
    Bad uncertainty: ECE is similar across groups.

    Parameters
    ----------
    n_bins : int
        Number of uncertainty groups (default 3 = terciles).
    ece_bins : int
        Number of calibration bins within each group.

    Returns
    -------
    dict with keys like 'ece_low_unc', 'ece_mid_unc', 'ece_high_unc',
    plus 'n_low_unc', 'n_mid_unc', etc. for group sizes.
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    labels = ["low", "mid", "high"] if n_bins == 3 else [f"q{i}" for i in range(n_bins)]

    edges = np.quantile(unc, np.linspace(0, 1, n_bins + 1))
    # ensure last edge includes max
    edges[-1] += 1e-10

    results = {}
    for i in range(n_bins):
        mask = (unc >= edges[i]) & (unc < edges[i + 1])
        label = labels[i] if i < len(labels) else f"q{i}"

        results[f"n_{label}_unc"] = int(mask.sum())
        results[f"ece_{label}_unc"] = _ece(y_true[mask], prob[mask], n_bins=ece_bins)
        results[f"acc_{label}_unc"] = float(
            ((prob[mask] >= 0.5).astype(int) == y_true[mask]).mean()
        ) if mask.sum() > 0 else np.nan
        results[f"churn_rate_{label}_unc"] = float(y_true[mask].mean()) if mask.sum() > 0 else np.nan

    return results


# ─────────────────────────────────────────────────────────────────────
# Uncertainty–error correlation (raw + confidence-controlled)
# ─────────────────────────────────────────────────────────────────────

def uncertainty_error_correlation(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
) -> dict:
    """
    Spearman rank correlation between uncertainty and prediction error,
    both raw and after controlling for |p - 0.5| (probability confidence).

    Raw Spearman is confounded: samples near p=0.5 always have high error
    AND most uncertainty measures are naturally high near the decision
    boundary. So a high raw correlation doesn't mean uncertainty adds
    information beyond what probability already tells you.

    The residual Spearman regresses |p - 0.5| out of both uncertainty
    and error, then correlates what's left. A positive residual means
    uncertainty captures something that probability confidence doesn't.

    Returns
    -------
    dict with:
      spearman_confidence_error : correlation between |p-0.5| and error (free baseline)
      spearman_unc_error_raw    : raw correlation (confounded)
      spearman_unc_error_residual : after removing |p-0.5| effect (the real metric)
    """
    from scipy.stats import spearmanr
    from numpy.polynomial.polynomial import polyfit

    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    error = np.abs(y_true - prob)
    confidence = np.abs(prob - 0.5)  # free "uncertainty" from p alone

    # 1. Baseline: how well does confidence alone predict error?
    corr_conf, _ = spearmanr(confidence, error)

    # 2. Raw: how well does uncertainty predict error? (confounded)
    corr_raw, _ = spearmanr(unc, error)

    # 3. Residual: regress |p-0.5| out of both u and error,
    #    then correlate what's left
    coef_u = polyfit(confidence, unc, 1)
    coef_e = polyfit(confidence, error, 1)

    resid_u = unc - (coef_u[0] + coef_u[1] * confidence)
    resid_e = error - (coef_e[0] + coef_e[1] * confidence)

    corr_resid, pval_resid = spearmanr(resid_u, resid_e)

    return {
        "spearman_confidence_error": float(corr_conf),
        "spearman_unc_error_raw": float(corr_raw),
        "spearman_unc_error_residual": float(corr_resid),
        "spearman_residual_pval": float(pval_resid),
    }


def conditional_uncertainty_accuracy(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    prob_bins: int = 5,
) -> dict:
    """
    Within each probability bin, split by uncertainty (median split)
    and compare accuracy. This controls for the base probability.

    If uncertainty is informative beyond p:
      low-uncertainty accuracy > high-uncertainty accuracy
      within the SAME probability bin.

    Returns
    -------
    dict with per-bin accuracy gaps and a summary mean_conditional_acc_gap.
    Positive mean gap = uncertainty adds signal beyond probability.
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    edges = np.quantile(prob, np.linspace(0, 1, prob_bins + 1))
    edges[-1] += 1e-10

    results = {}
    for i in range(prob_bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if mask.sum() < 20:
            continue

        p_lo = edges[i]
        p_hi = edges[i + 1]
        label = f"p{p_lo:.2f}-{p_hi:.2f}"

        u_sub = unc[mask]
        y_sub = y_true[mask]
        p_sub = prob[mask]

        u_median = np.median(u_sub)
        lo_mask = u_sub <= u_median
        hi_mask = u_sub > u_median

        acc_lo = float(
            ((p_sub[lo_mask] >= 0.5).astype(int) == y_sub[lo_mask]).mean()
        ) if lo_mask.sum() > 0 else np.nan
        acc_hi = float(
            ((p_sub[hi_mask] >= 0.5).astype(int) == y_sub[hi_mask]).mean()
        ) if hi_mask.sum() > 0 else np.nan

        gap = acc_lo - acc_hi if not (np.isnan(acc_lo) or np.isnan(acc_hi)) else np.nan

        results[f"acc_lo_unc_{label}"] = acc_lo
        results[f"acc_hi_unc_{label}"] = acc_hi
        results[f"acc_gap_{label}"] = gap
        results[f"n_{label}"] = int(mask.sum())

    # Summary: average gap across bins (positive = uncertainty helps)
    gaps = [v for k, v in results.items() if "acc_gap" in k and not np.isnan(v)]
    results["mean_conditional_acc_gap"] = float(np.mean(gaps)) if gaps else np.nan

    return results


# ─────────────────────────────────────────────────────────────────────
# Convenience: full uncertainty report
# ─────────────────────────────────────────────────────────────────────

def uncertainty_report(
    y_true: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
) -> dict:
    """
    Compute all uncertainty quality metrics in one call.

    Returns a flat dict with all results.
    """
    results = {}

    # AUCO (includes selective prediction internally)
    results.update(auco(y_true, prob, unc))

    # Binned ECE
    results.update(uncertainty_binned_ece(y_true, prob, unc))

    # Correlation (raw + confidence-controlled)
    results.update(uncertainty_error_correlation(y_true, prob, unc))

    # Conditional accuracy gap (controls for probability)
    results.update(conditional_uncertainty_accuracy(y_true, prob, unc))

    return results
