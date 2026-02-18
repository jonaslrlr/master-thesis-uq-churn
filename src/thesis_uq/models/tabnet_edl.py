from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from thesis_uq.seed import set_seed
from thesis_uq.metrics.ranking import standard_report
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Use your local fork building blocks
from pytorch_tabnet.tab_network import EmbeddingGenerator, TabNetEncoder


@dataclass
class EDLConfig:
    # backbone
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 5
    gamma: float = 1.5
    mask_type: str = "sparsemax"
    cat_emb_dim: int = 1

    # training
    lr: float = 2e-2
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 30
    batch_size: int = 1024
    virtual_batch_size: int = 128
    momentum: float = 0.02

    # EDL
    kl_coef: float = 1.0
    anneal_epochs: int = 50
    edl_loss: str = "mse"           # NEW: "mse" or "ce" (cross-entropy Bayes risk)

    # evidential head architecture
    head_hidden_dim: int = 0        # NEW: 0 = single linear (original), >0 = 2-layer MLP

    # dropout (EDL backbone is typically deterministic, but allow override)
    dropout: float = 0.0
    attn_dropout: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Head builder
# ─────────────────────────────────────────────────────────────────────

def _build_evidence_head(n_d: int, K: int, head_hidden_dim: int) -> torch.nn.Module:
    """
    Build the evidential head mapping TabNet representation h -> K evidence logits.

    Parameters
    ----------
    n_d : int
        Input dimension (TabNet decision dim).
    K : int
        Number of classes.
    head_hidden_dim : int
        If 0: single Linear(n_d, K) — original behaviour.
        If >0: Linear(n_d, head_hidden_dim) -> ReLU -> Linear(head_hidden_dim, K).
        Gives the network a nonlinear knob to modulate evidence magnitude
        independently from class direction.
    """
    if head_hidden_dim <= 0:
        return torch.nn.Linear(n_d, K, bias=True)
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(n_d, head_hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(head_hidden_dim, K, bias=True),
        )


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class TabNetEDL(torch.nn.Module):
    """
    TabNet backbone (embedder + encoder) -> decision representation h (n_d)
    -> evidential head (K logits) -> evidence=softplus -> Dirichlet alpha=evidence+1

    Output:
      probs = alpha / sum(alpha)
      vacuity u = K / sum(alpha)
    """
    def __init__(
        self,
        input_dim: int,
        cat_idxs: list[int],
        cat_dims: list[int],
        cfg: EDLConfig,
        K: int = 2,
    ):
        super().__init__()
        self.K = K
        self.cfg = cfg

        # group matrix: no feature groups, identity
        group_matrix = torch.eye(input_dim).float()

        # fork expects cat_emb_dims as a list for each categorical feature
        cat_emb_dims = [cfg.cat_emb_dim] * len(cat_dims)

        self.embedder = EmbeddingGenerator(
            input_dim=input_dim,
            cat_dims=cat_dims,
            cat_idxs=cat_idxs,
            cat_emb_dims=cat_emb_dims,
            group_matrix=group_matrix,
        )

        # buffer so it moves with model.to(device)
        self.register_buffer("embedding_group_matrix", self.embedder.embedding_group_matrix.float())

        self.encoder = TabNetEncoder(
            input_dim=self.embedder.post_embed_dim,
            output_dim=self.embedder.post_embed_dim,
            n_d=cfg.n_d,
            n_a=cfg.n_a,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=cfg.virtual_batch_size,
            momentum=cfg.momentum,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            mask_type=cfg.mask_type,
            group_attention_matrix=self.embedding_group_matrix,
        )

        # NEW: configurable evidential head
        self.head = _build_evidence_head(cfg.n_d, K, cfg.head_hidden_dim)

    def _sync_group_matrices(self):
        # make sure encoder uses buffer tensor (correct device)
        egm = self.embedding_group_matrix
        self.embedder.embedding_group_matrix = egm
        self.encoder.group_attention_matrix = egm

    def forward(self, x: torch.Tensor):
        self._sync_group_matrices()
        x_emb = self.embedder(x)

        steps_output, M_loss = self.encoder(x_emb)
        h = torch.sum(torch.stack(steps_output, dim=0), dim=0)  # (B, n_d)

        logits = self.head(h)  # (B, K)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        vacuity = (self.K / S).squeeze(1)  # (B,)
        return logits, alpha, probs, vacuity, M_loss


@torch.no_grad()
def edl_predict_proba_unc(model: TabNetEDL, X: np.ndarray, device: str, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      p: P(class=1) from Dirichlet mean
      u: vacuity (uncertainty)
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    ps, us = [], []
    for (xb,) in dl:
        xb = xb.to(device).float()
        _, _, probs, vacuity, _ = model(xb)
        ps.append(probs[:, 1].cpu().numpy())
        us.append(vacuity.cpu().numpy())

    return np.concatenate(ps), np.concatenate(us)


# ─────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────

def _dirichlet_kl(alpha: torch.Tensor, K: int) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) with uniform prior.
    """
    beta = torch.ones((1, K), device=alpha.device)

    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_beta  = torch.lgamma(beta).sum(dim=1, keepdim=True)  - torch.lgamma(S_beta)

    digamma_alpha = torch.digamma(alpha)
    digamma_S = torch.digamma(S_alpha)

    kl = (lnB_beta - lnB_alpha) + ((alpha - beta) * (digamma_alpha - digamma_S)).sum(dim=1, keepdim=True)
    return kl.squeeze(1)


def _kl_term(alpha: torch.Tensor, y_onehot: torch.Tensor, epoch: int, cfg: EDLConfig) -> torch.Tensor:
    """
    Annealed KL regulariser shared by both MSE and CE losses.
    Uses the "removal of non-misleading evidence" trick from Sensoy et al.:
    only penalise evidence for *incorrect* classes.
    """
    K = alpha.shape[1]
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    anneal = min(1.0, epoch / float(cfg.anneal_epochs))
    kl = _dirichlet_kl(alpha_tilde, K).mean()
    return cfg.kl_coef * anneal * kl


def edl_mse_bayes_risk(alpha: torch.Tensor, y_true: torch.Tensor, epoch: int, cfg: EDLConfig) -> torch.Tensor:
    """
    Sensoy et al. Eq. 5: MSE Bayes risk.

    Data-fit = E_Dir[ ||y - p||^2 ] = sum_k (y_k - m_k)^2 + var_k
    where m = alpha/S, var = alpha*(S-alpha)/(S^2*(S+1))
    """
    K = alpha.shape[1]
    y = F.one_hot(y_true, num_classes=K).float()

    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    var = alpha * (S - alpha) / (S * S * (S + 1.0))

    mse = ((y - p) ** 2 + var).sum(dim=1).mean()

    return mse + _kl_term(alpha, y, epoch, cfg)


def edl_ce_bayes_risk(alpha: torch.Tensor, y_true: torch.Tensor, epoch: int, cfg: EDLConfig) -> torch.Tensor:
    """
    Sensoy et al. Eq. 4: Cross-entropy Bayes risk.

    Data-fit = E_Dir[ -sum_k y_k log(p_k) ]
             = sum_k y_k * ( digamma(S) - digamma(alpha_k) )

    Advantages over MSE:
      - Sharper gradients for misclassified examples (does not plateau
        when predictions are already close).
      - Better calibrated evidence in practice (Sensoy §4.2).
    """
    K = alpha.shape[1]
    y = F.one_hot(y_true, num_classes=K).float()

    S = alpha.sum(dim=1, keepdim=True)
    ce = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()

    return ce + _kl_term(alpha, y, epoch, cfg)


def _get_edl_loss_fn(cfg: EDLConfig):
    """Return the appropriate EDL loss based on config."""
    if cfg.edl_loss == "mse":
        return edl_mse_bayes_risk
    elif cfg.edl_loss == "ce":
        return edl_ce_bayes_risk
    else:
        raise ValueError(f"Unknown edl_loss: {cfg.edl_loss!r}. Choose 'mse' or 'ce'.")


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train_tabnet_edl(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray, y_valid: np.ndarray,
    cat_idxs: list[int], cat_dims_list: list[int],
    cfg: EDLConfig,
    device_name: str = "cpu",
    seed: int = 0,
    lambda_sparse: float = 1e-3,
) -> TabNetEDL:
    """
    Train EDL TabNet backbone + evidential head.
    Early stopping on VALID PR-AUC using your standard_report.
    """
    set_seed(seed)
    device = device_name

    model = TabNetEDL(
        input_dim=X_train.shape[1],
        cat_idxs=cat_idxs,
        cat_dims=cat_dims_list,
        cfg=cfg,
        K=2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Select loss function from config
    loss_fn = _get_edl_loss_fn(cfg)

    best_score = -np.inf
    best_state = None
    patience_ctr = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_dl:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            opt.zero_grad()
            _, alpha, _, _, M_loss = model(xb)

            loss = loss_fn(alpha, yb, epoch=epoch, cfg=cfg) + lambda_sparse * M_loss
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # VALID evaluation
        p_val, _ = edl_predict_proba_unc(model, X_valid, device=device, batch_size=2048)
        rep = standard_report(y_valid, p_val)
        score = rep["auc_pr"]

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:03d} | loss {np.mean(losses):.4f} | valid_prauc {score:.5f} | valid_lift {rep['lift10']:.4f}")

        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch}, best_valid_prauc={best_score:.5f}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model
