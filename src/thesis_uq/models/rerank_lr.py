from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


FeatureSpec = Literal["u_p", "p_u", "u_p_up"]  # add more later if needed


@dataclass
class LRReranker:
    scaler: MinMaxScaler
    lr: LogisticRegression
    feature_spec: FeatureSpec

    def _featurize(self, p: np.ndarray, u: np.ndarray) -> np.ndarray:
        p = np.asarray(p).reshape(-1)
        u = np.asarray(u).reshape(-1)

        if self.feature_spec == "u_p":
            X = np.column_stack([u, p])
        elif self.feature_spec == "p_u":
            X = np.column_stack([p, u])
        elif self.feature_spec == "u_p_up":
            X = np.column_stack([u, p, u * p])
        else:
            raise ValueError(f"Unknown feature_spec: {self.feature_spec}")

        return X

    def predict_proba(self, p: np.ndarray, u: np.ndarray) -> np.ndarray:
        X = self._featurize(p, u)
        Xs = self.scaler.transform(X)
        return self.lr.predict_proba(Xs)[:, 1]


def fit_lr_reranker(
    p_valid: np.ndarray,
    u_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_spec: FeatureSpec = "u_p",
    max_iter: int = 2000,
    C: float = 1.0,
    class_weight: Optional[dict] = None,
) -> LRReranker:
    """
    Train LR reranker on VALID only (no leakage).
    Inputs:
      p_valid: base probability score (e.g., mean prob)
      u_valid: uncertainty score (std/vacuity/etc)
      y_valid: true labels
    Returns:
      LRReranker object that can score new points.
    """
    p_valid = np.asarray(p_valid).reshape(-1)
    u_valid = np.asarray(u_valid).reshape(-1)
    y_valid = np.asarray(y_valid).reshape(-1)

    scaler = MinMaxScaler()
    # Build features
    if feature_spec == "u_p":
        X = np.column_stack([u_valid, p_valid])
    elif feature_spec == "p_u":
        X = np.column_stack([p_valid, u_valid])
    elif feature_spec == "u_p_up":
        X = np.column_stack([u_valid, p_valid, u_valid * p_valid])
    else:
        raise ValueError(f"Unknown feature_spec: {feature_spec}")

    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)
    lr.fit(Xs, y_valid)

    return LRReranker(scaler=scaler, lr=lr, feature_spec=feature_spec)


def summarize_lr(r: LRReranker) -> str:
    return f"feature_spec={r.feature_spec} coef={r.lr.coef_} intercept={r.lr.intercept_}"
