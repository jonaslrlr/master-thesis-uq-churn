import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc as sk_auc

def lift_at_10(y_true, score):
    df = pd.DataFrame({"y": y_true, "s": score}).sort_values("s", ascending=False)
    top = df.head(len(df) // 10)
    base = df["y"].mean()
    return float(top["y"].mean() / base) if base > 0 else np.nan

def pr_auc_trapezoid(y_true, score):
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(sk_auc(recall, precision))

def standard_report(y_true, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, score),
        "auc_pr": pr_auc_trapezoid(y_true, score),  #  PR-AUC
        "acc": accuracy_score(y_true, pred),
        "lift10": lift_at_10(y_true, score),
    }
