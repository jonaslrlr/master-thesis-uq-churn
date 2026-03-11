import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc as sk_auc, brier_score_loss


def lift_at_10(y_true, score):
    df = pd.DataFrame({"y": y_true, "s": score}).sort_values("s", ascending=False)
    top = df.head(len(df) // 10)
    base = df["y"].mean()
    return float(top["y"].mean() / base) if base > 0 else np.nan


def pr_auc_trapezoid(y_true, score):
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(sk_auc(recall, precision))


def ece(y_true, prob, n_bins=10):
    """
    Expected Calibration Error.

    Bins predictions by predicted probability, computes the weighted average
    gap between predicted confidence and actual accuracy within each bin.

    Lower is better. 0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)

    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        # include right edge in last bin
        if hi == bins[-1]:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_pred = prob[mask].mean()
        avg_true = y_true[mask].mean()
        total += n_bin * abs(avg_true - avg_pred)

    return float(total / len(y_true))


def standard_report(y_true, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, score),
        "auc_pr": pr_auc_trapezoid(y_true, score),
        "acc": accuracy_score(y_true, pred),
        "lift10": lift_at_10(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "ece": ece(y_true, score),
    }
