from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


@dataclass
class LaplaceConfig:
    prior_prec: float = 1.0          # Gaussian prior precision on weights (L2 strength)
    max_iter: int = 200              # LBFGS steps for MAP
    n_samples: int = 300             # posterior weight samples for predictive mean/std
    jitter: float = 1e-6             # numeric stability added to Hessian
    standardize: bool = True         # standardize TabNet features


@torch.no_grad()
def extract_tabnet_representation(clf, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Extract TabNet decision representation h(x) = sum_t d_t (dim = n_d)
    from a trained TabNetClassifier clf (local fork).
    Assumes deterministic backbone (dropout=0 recommended).
    """
    clf.network.eval()
    X_tensor = torch.from_numpy(X).float()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    feats = []
    for (xb,) in loader:
        xb = xb.to(clf.device).float()
        x_emb = clf.network.embedder(xb)
        steps_output, _ = clf.network.tabnet.encoder(x_emb)
        h = torch.sum(torch.stack(steps_output, dim=0), dim=0)  # (B, n_d)
        feats.append(h.detach().cpu())

    return torch.cat(feats, dim=0).numpy()


def fit_map_logreg_torch(
    Z: np.ndarray,
    y: np.ndarray,
    prior_prec: float,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MAP logistic regression on features Z (N,d) with Gaussian prior precision prior_prec.
    Returns:
      w_map (d+1,) and X_aug (N,d+1) as torch.double tensors.
    Bias is the last element of w_map and is NOT regularized.
    """
    X = torch.from_numpy(Z).double()
    y_t = torch.from_numpy(y).double()

    N, d = X.shape
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=torch.double)], dim=1)  # (N, d+1)

    w = torch.zeros(d + 1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X_aug @ w
        loss_data = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        loss_prior = 0.5 * prior_prec * torch.sum(w[:-1] ** 2)  # no prior on bias
        loss = loss_data + loss_prior
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), X_aug.detach()


@torch.no_grad()
def laplace_posterior_cov(
    X_aug: torch.Tensor,
    w_map: torch.Tensor,
    prior_prec: float,
    jitter: float,
) -> torch.Tensor:
    """
    Full-cov Laplace posterior for logistic regression head:
      Sigma = (X^T R X + prior_prec*I)^(-1)
    where R = diag(p*(1-p)).
    Bias is unregularized (I[-1,-1]=0).
    """
    logits = X_aug @ w_map
    p = torch.sigmoid(logits)
    r = p * (1.0 - p)  # (N,)

    XR = X_aug * r.unsqueeze(1)
    H = X_aug.T @ XR  # (d+1, d+1)

    d1 = X_aug.shape[1]
    I = torch.eye(d1, dtype=torch.double, device=X_aug.device)
    I[-1, -1] = 0.0  # do not regularize bias

    A = H + prior_prec * I + jitter * torch.eye(d1, dtype=torch.double, device=X_aug.device)
    Sigma = torch.linalg.inv(A)
    return Sigma


@torch.no_grad()
def predict_map_probs(Z: np.ndarray, w_map: torch.Tensor) -> np.ndarray:
    """Deterministic MAP head probs for class 1."""
    X = torch.from_numpy(Z).double()
    N = X.shape[0]
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=torch.double)], dim=1)
    logits = X_aug @ w_map
    return torch.sigmoid(logits).cpu().numpy()


@torch.no_grad()
def predict_laplace_probs_std(
    Z: np.ndarray,
    w_map: torch.Tensor,
    Sigma: torch.Tensor,
    n_samples: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample head weights from N(w_map, Sigma) and return predictive mean/std of probs.
    Note: std(prob) can be small due to sigmoid squashing; still useful for ranking/triage.
    """
    torch.manual_seed(seed)

    X = torch.from_numpy(Z).double()
    N = X.shape[0]
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=torch.double)], dim=1)

    # robust cholesky
    # if Sigma is not SPD due to numerics, add diagonal jitter progressively
    diag_jitter = 0.0
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(Sigma + diag_jitter * torch.eye(Sigma.shape[0], dtype=torch.double))
            break
        except RuntimeError:
            diag_jitter = 1e-6 if diag_jitter == 0.0 else diag_jitter * 10.0
    else:
        # last resort: use eigendecomposition (rare)
        evals, evecs = torch.linalg.eigh(Sigma)
        evals = torch.clamp(evals, min=1e-12)
        L = evecs @ torch.diag(torch.sqrt(evals))

    d1 = w_map.shape[0]
    eps = torch.randn(n_samples, d1, dtype=torch.double)
    w_samps = w_map.unsqueeze(0) + eps @ L.T  # (S, d1)

    logits = X_aug @ w_samps.T  # (N, S)
    probs = torch.sigmoid(logits)  # (N, S)

    mean_prob = probs.mean(dim=1).cpu().numpy()
    std_prob = probs.std(dim=1, unbiased=False).cpu().numpy()
    return mean_prob, std_prob


def laplace_from_tabnet_features(
    feat_train: np.ndarray,
    y_train: np.ndarray,
    feat_valid: np.ndarray,
    feat_test: np.ndarray,
    cfg: LaplaceConfig,
    seed: int,
) -> Tuple[
    np.ndarray, np.ndarray,           # p_map_valid, p_map_test
    np.ndarray, np.ndarray,           # p_lap_valid_mean, u_valid_stdprob
    np.ndarray, np.ndarray,           # p_lap_test_mean,  u_test_stdprob
]:
    """
    Fit MAP + Laplace head on TabNet features.
    Returns:
      p_map_valid, p_map_test,
      p_lap_valid_mean, u_valid_stdprob,
      p_lap_test_mean,  u_test_stdprob
    """
    if cfg.standardize:
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(feat_train)
        Z_valid = scaler.transform(feat_valid)
        Z_test  = scaler.transform(feat_test)
    else:
        Z_train, Z_valid, Z_test = feat_train, feat_valid, feat_test

    w_map, X_aug_train = fit_map_logreg_torch(
        Z_train, y_train, prior_prec=cfg.prior_prec, max_iter=cfg.max_iter
    )
    Sigma = laplace_posterior_cov(
        X_aug_train, w_map, prior_prec=cfg.prior_prec, jitter=cfg.jitter
    )

    p_map_valid = predict_map_probs(Z_valid, w_map)
    p_map_test  = predict_map_probs(Z_test,  w_map)

    p_lap_valid_mean, u_valid_stdprob = predict_laplace_probs_std(
        Z_valid, w_map, Sigma, n_samples=cfg.n_samples, seed=seed + 10_000
    )
    p_lap_test_mean, u_test_stdprob = predict_laplace_probs_std(
        Z_test,  w_map, Sigma, n_samples=cfg.n_samples, seed=seed + 20_000
    )

    return (
        p_map_valid, p_map_test,
        p_lap_valid_mean, u_valid_stdprob,
        p_lap_test_mean,  u_test_stdprob,
    )
