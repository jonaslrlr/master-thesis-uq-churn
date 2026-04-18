"""
Delft University Telephone Company — data loader for TabNet
============================================================

Source: Duchemin & Matheus (2021), Journal of Supply Chain Management Science
Download: https://www.dropbox.com/s/h40f4y2a5wiue2u/churn.csv?dl=1

20 000 customers, 12 variables (+ 1 index column dropped).
Target: leave (LEAVE / STAY → 1 / 0)
Churn rate: ~49.3 % (nearly balanced).  No temporal component.

Dropped columns:
  - Unnamed: 0  (row index, all unique)

Categorical features (label-encoded for TabNet, 4 cols):
  college (2), reported_satisfaction (5),
  reported_usage_level (5), considering_change_of_plan (5)

Numeric features (7 cols):
  income, overage, leftover, house, handset_price,
  over_15mins_calls_per_month, average_call_duration

No missing values, no duplicates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────
DEFAULT_SUBPATH = "data/raw/delft_churn/churn.csv"

TARGET_COL = "leave"

CAT_COLS = [
    "college",
    "reported_satisfaction",
    "reported_usage_level",
    "considering_change_of_plan",
]


# ── load + clean ─────────────────────────────────────────────────────

def load_delft_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Read the raw Delft churn CSV and return a cleaned DataFrame.

    Drops the index column and maps target to binary 0/1.
    """
    df = pd.read_csv(csv_path)

    # Drop row index
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Map target → binary
    df[TARGET_COL] = df[TARGET_COL].map(
        {"LEAVE": 1, "STAY": 0}
    ).astype(int)

    return df


# ── encode for TabNet ────────────────────────────────────────────────

def encode_tabular_for_tabnet(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict, List[int], List[int]]:
    """
    Label-encode categoricals, return arrays ready for TabNet.

    Returns
    -------
    X             : np.ndarray   (N, D)
    y             : np.ndarray   (N,)
    features      : list[str]    feature names (column order in X)
    cat_cols      : list[str]    names of categorical columns
    cat_dims      : dict         {col_name: n_categories}
    cat_idxs      : list[int]    positional indices of categoricals in X
    cat_dims_list : list[int]    n_categories per categorical (aligned with cat_idxs)
    """
    df = df.copy()

    # Separate target
    y = df.pop(TARGET_COL).values.astype(np.int64)

    # Identify which CAT_COLS are present
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    # Label-encode each categorical → int starting at 0
    cat_dims: dict[str, int] = {}
    for col in cat_cols:
        codes, _uniques = pd.factorize(df[col], sort=True)
        df[col] = codes
        cat_dims[col] = len(_uniques)

    features = list(df.columns)
    X = df.values.astype(np.float32)

    # Build TabNet-style index lists
    cat_idxs = [features.index(c) for c in cat_cols]
    cat_dims_list = [cat_dims[c] for c in cat_cols]

    return X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list


# ── one-call convenience (used by registry.py) ──────────────────────

def load_for_tabnet(
    repo_root: Path | str,
    csv_subpath: str = DEFAULT_SUBPATH,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict, List[int], List[int]]:
    """Read CSV → clean → encode → return TabNet-ready arrays."""
    csv_path = Path(repo_root) / csv_subpath
    df = load_delft_csv(csv_path)
    return encode_tabular_for_tabnet(df)
