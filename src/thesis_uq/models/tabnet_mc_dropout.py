import numpy as np
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline


def train_tabnet_mc_dropout(
    X_train, y_train, X_valid, y_valid,
    cat_idxs, cat_dims_list,
    device_name="cpu",
    dropout=0.1,
    attn_dropout=0.0,   # NEW: attention-path dropout
    tabnet_kwargs=None,
    train_kwargs=None,
    seed=0,
):
    """
    Train TabNet with deep dropout enabled for MC dropout experiments.
    This is the baseline trainer with dropout > 0.

    Parameters
    ----------
    dropout : float
        GLU feature-transformer dropout (existing).
    attn_dropout : float
        Attention-path dropout (NEW). Applied after prior scaling,
        before sparsemax/entmax. Enables MC Dropout to capture
        feature-selection uncertainty.
    """
    tabnet_kwargs = tabnet_kwargs or {}
    train_kwargs = train_kwargs or {}

    clf = train_tabnet_baseline(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        device_name=device_name,
        dropout=dropout,
        attn_dropout=attn_dropout,  # NEW
        seed=seed,
        **tabnet_kwargs,
        **train_kwargs,
    )
    return clf


def mc_predict(clf, X, n_samples=50):
    """
    Returns mean_prob (N,), std_prob (N,) for class 1 from MC dropout.
    Requires your patched TabNetClassifier.predict_proba_mc.

    Note: _enable_mc_dropout() sets ALL torch.nn.Dropout modules to
    train mode (including the new attention dropout), while keeping
    BatchNorm/GBN frozen in eval mode.
    """
    mean_proba, std_proba = clf.predict_proba_mc(X, n_samples=n_samples, return_std=True)
    return mean_proba[:, 1], std_proba[:, 1]
