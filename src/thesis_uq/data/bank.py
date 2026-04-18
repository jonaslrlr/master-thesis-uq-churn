"""
Bank Customer Churn dataset loader.

Source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
~10,000 rows, 11 usable features, binary target (Exited).

Domain: banking (account closure = churn).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BankChurnConfig:
    target_col: str = "Exited"
    drop_cols: tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")
    explicit_cat_cols: tuple[str, ...] = ("Geography", "Gender")
    low_card_threshold: int = 20
    fill_cat_value: str = "missing"


def _ensure_csv(local_path: Path) -> Path:
    if local_path.exists():
        return local_path

    print(f"CSV not found at {local_path}, downloading via kagglehub ...")
    import kagglehub
    import shutil

    dl_dir = Path(kagglehub.dataset_download("shrutimechlearn/churn-modelling"))
    candidates = list(dl_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in kagglehub download: {dl_dir}")

    src = candidates[0]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, local_path)
    print(f"Saved to {local_path}")
    return local_path


def load_bank_csv(path) -> pd.DataFrame:
    path = Path(path)
    path = _ensure_csv(path)

    df = pd.read_csv(path)

    col_map = {c: c for c in df.columns}
    for c in df.columns:
        if c.lower() == "customerid":
            col_map[c] = "CustomerId"
    df = df.rename(columns=col_map)

    df = df[df["Exited"].notna()].copy()
    df["Exited"] = df["Exited"].astype(int)

    return df


def encode_tabular_for_tabnet(
    df: pd.DataFrame,
    cfg: Optional[BankChurnConfig] = None,
):
    if cfg is None:
        cfg = BankChurnConfig()

    df = df.copy()

    y = None
    if cfg.target_col in df.columns:
        y = df[cfg.target_col].astype(int).to_numpy()
        df = df.drop(columns=[cfg.target_col])

    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    cat_cols: List[str] = list(cfg.explicit_cat_cols)

    for c in df.columns:
        if c in cat_cols:
            continue
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if nunique <= cfg.low_card_threshold:
                cat_cols.append(c)

    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype(str).fillna(cfg.fill_cat_value)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c] = df[c].fillna(med)

    cat_idxs = []
    cat_dims_list = []
    for i, c in enumerate(df.columns):
        if c in cat_cols:
            df[c] = df[c].astype("category")
            cat_idxs.append(i)
            cat_dims_list.append(int(df[c].cat.categories.size))
            df[c] = df[c].cat.codes.astype(int)

    X = df.to_numpy(dtype=np.float32)
    features = list(df.columns)

    cat_dims = {c: int(df[c].nunique()) for c in cat_cols}

    if y is None:
        return X, None, features, cat_cols, cat_dims, cat_idxs, cat_dims_list

    return X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list
