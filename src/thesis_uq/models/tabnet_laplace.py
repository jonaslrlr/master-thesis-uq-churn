from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LaplaceConfig:
    """Configuration for the Laplace posterior computation."""
    prior_precision: float = 1e-2       # τ: isotropic Gaussian prior precision
    pred_method: str = "probit"         # "probit" (fast, closed-form) or "mc" (sampling)
    mc_samples: int = 200               # number of posterior samples if pred_method="mc"


# ─────────────────────────────────────────────────────────────────────
# Representation extraction
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_representations(clf, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Extract TabNet encoder representations h(x) ∈ R^{n_d} for all samples.

    Uses a forward hook on final_mapping to capture the input (= representation)
    without modifying the model or re-implementing the forward pass.

    Parameters
    ----------
    clf : TabNetClassifier
        Trained TabNet model (from pytorch_tabnet).
    X : np.ndarray, shape (N, input_dim)
        Feature matrix.

    Returns
    -------
    H : np.ndarray, shape (N, n_d)
        Encoder representations.
    """
    network = clf.network
    network.eval()

    captured = []

    def hook_fn(module, input, output):
        # input is a tuple; input[0] is the representation h ∈ (B, n_d)
        captured.append(input[0].detach().cpu())

    # The nesting is: clf.network (TabNet) → .tabnet (TabNetNoEmbeddings) → .final_mapping
    handle = network.tabnet.final_mapping.register_forward_hook(hook_fn)

    try:
        ds = TensorDataset(torch.from_numpy(X).float())
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for (xb,) in dl:
            xb = xb.to(clf.device)
            network(xb)  # triggers the hook
    finally:
        handle.remove()

    return torch.cat(captured, dim=0).numpy()


# ─────────────────────────────────────────────────────────────────────
# MAP weight extraction
# ─────────────────────────────────────────────────────────────────────

def extract_map_weights(clf) -> np.ndarray:
    """
    Extract the MAP logit-difference weights from TabNet's final_mapping.

    For K=2 softmax, P(y=1|x) = σ((w₁ - w₀)ᵀ h(x)), so the effective
    binary classification weight vector is w_diff = W[1] - W[0].

    Parameters
    ----------
    clf : TabNetClassifier
        Trained TabNet model.

    Returns
    -------
    w_diff : np.ndarray, shape (n_d,)
        MAP logit-difference weights.
    """
    W = clf.network.tabnet.final_mapping.weight.detach().cpu().numpy()  # (2, n_d)
    assert W.shape[0] == 2, f"Expected binary classification (2 classes), got {W.shape[0]}"
    return W[1] - W[0]  # (n_d,)


# ─────────────────────────────────────────────────────────────────────
# Laplace posterior
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LaplacePosterior:
    """
    Stores the Laplace approximation to the posterior over the
    logit-difference weight vector.

    Attributes
    ----------
    w_map : np.ndarray, shape (n_d,)
        MAP logit-difference weights from final_mapping.
    cov : np.ndarray, shape (n_d, n_d)
        Posterior covariance Σ = H⁻¹.
    prior_precision : float
        Prior precision τ used to compute H.
    log_marginal_likelihood : float
        Log evidence approximation: log p(D|τ) ≈ log p(D|w_MAP) - ½ log|H| + (n_d/2) log τ
    """
    w_map: np.ndarray
    cov: np.ndarray
    prior_precision: float
    log_marginal_likelihood: float


def fit_laplace(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    prior_precision: float = 1e-2,
    batch_size: int = 2048,
) -> LaplacePosterior:
    """
    Fit a last-layer Laplace approximation using TabNet's own MAP weights.

    Steps:
    1. Extract encoder representations H = [h(x₁), ..., h(xₙ)] ∈ R^{N × n_d}
    2. Extract MAP weights w = final_mapping.weight[1] - weight[0] ∈ R^{n_d}
    3. Compute logits f = H w, then πᵢ = σ(fᵢ)
    4. Compute GGN Hessian: H = Hᵀ diag(π(1-π)) H + τI
    5. Invert to get posterior covariance Σ = H⁻¹
    6. (Optional) Compute log marginal likelihood for model selection

    Parameters
    ----------
    clf : TabNetClassifier
        Trained model whose final_mapping weights define w_MAP.
    X_train : np.ndarray, shape (N, input_dim)
        Training features (same data the model was trained on).
    y_train : np.ndarray, shape (N,)
        Training labels (for log-likelihood computation).
    prior_precision : float
        τ: isotropic Gaussian prior precision on w.
        Larger τ → tighter prior → less posterior uncertainty.

    Returns
    -------
    LaplacePosterior
    """
    # 1. Representations
    H = extract_representations(clf, X_train, batch_size=batch_size)  # (N, n_d)
    N, n_d = H.shape

    # 2. MAP weights
    w_map = extract_map_weights(clf)  # (n_d,)
    assert w_map.shape == (n_d,), f"Shape mismatch: w_map {w_map.shape} vs n_d={n_d}"

    # 3. MAP logits and probabilities
    logits = H @ w_map  # (N,)
    pi = 1.0 / (1.0 + np.exp(-logits))  # σ(f), (N,)
    pi = np.clip(pi, 1e-8, 1 - 1e-8)   # numerical stability

    # 4. GGN Hessian: H_ggn = Hᵀ diag(π(1-π)) H + τI
    #    For N=40k, n_d=64: (64, N) @ diag @ (N, 64) = (64, 64) — very fast
    D = pi * (1.0 - pi)  # (N,), Hessian of logistic loss w.r.t. logit
    H_weighted = H * np.sqrt(D)[:, None]  # (N, n_d) — equivalent to sqrt(D) H
    hessian = H_weighted.T @ H_weighted + prior_precision * np.eye(n_d)  # (n_d, n_d)

    # 5. Posterior covariance via Cholesky (more stable than direct inverse)
    L = np.linalg.cholesky(hessian)  # H = L Lᵀ
    L_inv = np.linalg.solve_triangular(L, np.eye(n_d), lower=True)
    cov = L_inv.T @ L_inv  # Σ = H⁻¹ = (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹

    # 6. Log marginal likelihood (Laplace evidence approximation)
    #    log p(D|τ) ≈ log p(D|w_MAP) - ½ log|H| + (n_d/2) log(τ)
    #
    #    log p(D|w_MAP) = Σᵢ [yᵢ log πᵢ + (1-yᵢ) log(1-πᵢ)]
    #    log|H| = 2 Σ log(diag(L))
    log_lik = np.sum(y_train * np.log(pi) + (1 - y_train) * np.log(1 - pi))
    log_det_H = 2.0 * np.sum(np.log(np.diag(L)))
    log_prior = -0.5 * prior_precision * np.dot(w_map, w_map)
    log_marglik = log_lik + log_prior - 0.5 * log_det_H + 0.5 * n_d * np.log(prior_precision)

    return LaplacePosterior(
        w_map=w_map,
        cov=cov,
        prior_precision=prior_precision,
        log_marginal_likelihood=log_marglik,
    )


# ─────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────

def _predict_probit(
    posterior: LaplacePosterior,
    H: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probit approximation to the Bayesian predictive distribution.

    P(y=1|x*,D) ≈ σ( μ* / √(1 + π/8 · v*) )

    where μ* = wᵀh(x*), v* = h(x*)ᵀ Σ h(x*).

    Returns (probabilities, uncertainties) where uncertainty = v* (logit variance).
    """
    mu = H @ posterior.w_map  # (N,)

    # Vectorised quadratic form: v_i = hᵢᵀ Σ hᵢ
    # Efficient: (H @ Σ) ⊙ H summed over columns
    H_Sigma = H @ posterior.cov  # (N, n_d)
    v = np.sum(H_Sigma * H, axis=1)  # (N,)
    v = np.maximum(v, 0.0)  # ensure non-negative (numerical)

    # Probit approximation: scale logit by 1/sqrt(1 + π/8 * v)
    kappa = 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * v)
    p = 1.0 / (1.0 + np.exp(-kappa * mu))

    return p, v


def _predict_mc(
    posterior: LaplacePosterior,
    H: np.ndarray,
    n_samples: int = 200,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MC sampling from the Laplace posterior.

    Sample w⁽ˢ⁾ ~ N(w_MAP, Σ), compute σ(w⁽ˢ⁾ᵀ h(x*)) for each sample,
    return mean as probability and std as uncertainty.
    """
    rng = np.random.default_rng(rng_seed)
    n_d = posterior.w_map.shape[0]

    # Sample from posterior via Cholesky of covariance
    L_cov = np.linalg.cholesky(posterior.cov + 1e-10 * np.eye(n_d))
    z = rng.standard_normal((n_samples, n_d))  # (S, n_d)
    w_samples = posterior.w_map[None, :] + z @ L_cov.T  # (S, n_d)

    # Compute logits for all samples × all datapoints: (S, N)
    logits = w_samples @ H.T  # (S, N)
    probs = 1.0 / (1.0 + np.exp(-logits))  # (S, N)

    p_mean = probs.mean(axis=0)  # (N,)
    p_std = probs.std(axis=0)    # (N,)

    return p_mean, p_std


def laplace_predict(
    posterior: LaplacePosterior,
    clf,
    X: np.ndarray,
    cfg: LaplaceConfig,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian predictive distribution using last-layer Laplace.

    Parameters
    ----------
    posterior : LaplacePosterior
        Fitted Laplace posterior.
    clf : TabNetClassifier
        Trained model (for extracting representations).
    X : np.ndarray
        Feature matrix.
    cfg : LaplaceConfig
        Prediction configuration.

    Returns
    -------
    p : np.ndarray, shape (N,)
        Predictive P(y=1|x, D).
    u : np.ndarray, shape (N,)
        Uncertainty measure.
        - If probit: logit variance v* (larger = more uncertain)
        - If mc: predictive std (larger = more uncertain)
    """
    H = extract_representations(clf, X, batch_size=batch_size)

    if cfg.pred_method == "probit":
        return _predict_probit(posterior, H)
    elif cfg.pred_method == "mc":
        return _predict_mc(posterior, H, n_samples=cfg.mc_samples)
    else:
        raise ValueError(f"Unknown pred_method: {cfg.pred_method!r}. Choose 'probit' or 'mc'.")


# ─────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────

def posterior_diagnostics(posterior: LaplacePosterior) -> dict:
    """
    Compute diagnostic statistics about the Laplace posterior.

    Useful for sanity checking: if the posterior is too tight
    (all eigenvalues huge), uncertainty will be near-zero everywhere.
    If too loose, predictions degrade.
    """
    eigvals = np.linalg.eigvalsh(posterior.cov)
    w_norm = np.linalg.norm(posterior.w_map)

    return {
        "n_d": len(posterior.w_map),
        "w_map_norm": float(w_norm),
        "prior_precision": float(posterior.prior_precision),
        "log_marginal_likelihood": float(posterior.log_marginal_likelihood),
        "cov_eigval_min": float(eigvals.min()),
        "cov_eigval_max": float(eigvals.max()),
        "cov_eigval_median": float(np.median(eigvals)),
        "cov_condition_number": float(eigvals.max() / max(eigvals.min(), 1e-15)),
        "cov_trace": float(np.trace(posterior.cov)),
    }
