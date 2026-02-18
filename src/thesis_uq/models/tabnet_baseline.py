import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from thesis_uq.metrics import PRAUC


def train_tabnet_baseline(
    X_train, y_train, X_valid, y_valid,
    cat_idxs, cat_dims_list,
    device_name="cpu",
    n_d=16, n_a=16, n_steps=5, gamma=1.5,
    cat_emb_dim=1,
    mask_type="entmax",
    lr=2e-2, weight_decay=1e-5,
    max_epochs=50, patience=15,
    batch_size=1024, virtual_batch_size=128,
    dropout=0.0,
    attn_dropout=0.0,   # NEW: attention-path dropout
    seed=0
):
    clf = TabNetClassifier(
        n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims_list,
        cat_emb_dim=[cat_emb_dim] * len(cat_dims_list),  # works with your fork
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr, weight_decay=weight_decay),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type=mask_type,
        device_name=device_name,
        verbose=1,
        dropout=dropout,
        attn_dropout=attn_dropout,  # NEW
        seed=seed,
    )

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=[PRAUC],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )
    return clf
