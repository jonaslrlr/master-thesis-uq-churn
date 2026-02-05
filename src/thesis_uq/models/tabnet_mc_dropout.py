import numpy as np
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline


def train_tabnet_mc_dropout(
    X_train, y_train, X_valid, y_valid,
    cat_idxs, cat_dims_list,
    device_name="cpu",
    dropout=0.1,
    tabnet_kwargs=None,
    train_kwargs=None,
    seed=0,  
):
    """
    Train TabNet with deep dropout enabled for MC dropout experiments.
    This is the baseline trainer with dropout > 0.
    """
    tabnet_kwargs = tabnet_kwargs or {}
    train_kwargs = train_kwargs or {}

    clf = train_tabnet_baseline(
        X_train, y_train, X_valid, y_valid,
        cat_idxs, cat_dims_list,
        device_name=device_name,
        dropout=dropout,
        seed=seed,       
        **tabnet_kwargs,
        **train_kwargs,
    )
    return clf


def mc_predict(clf, X, n_samples=50):
    """
    Returns mean_prob (N,), std_prob (N,) for class 1 from MC dropout.
    Requires your patched TabNetClassifier.predict_proba_mc.
    """
    mean_proba, std_proba = clf.predict_proba_mc(X, n_samples=n_samples, return_std=True)
    return mean_proba[:, 1], std_proba[:, 1]
