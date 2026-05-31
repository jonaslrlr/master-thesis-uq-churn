"""
Chile Postpaid Telecom — data loader for TabNet
=================================================

Source: Confidential dataset provided by Yameng Guo (UGent).
Chilean telecom operator, postpaid subscribers.

7,056 customers, 38 usable features (all numeric after cleaning),
binary target (CHURN).

Domain: telecom (voluntary churn).
Churn rate: ~29.1 %.

Temporal split by START_DATE (subscription start), 70/15/15.
Churn rates: train ~30.9%, val ~25.4%, test ~24.6%.

Dropped columns:
  - ID          (subscriber identifier)
  - START_DATE  (used for temporal ordering, then dropped)
  - END_DATE    (only populated for churners)
  - ACTIVE_WEEKS, ACTIVE_MONTHS  (r=1.0 with ACTIVE_DAYS)
  - AVG_MINUTES_INC_OFFNET_1MONTH (r=1.0 with AVG_INC_OFFNET_1MONTH)
  - AVG_MINUTES_INC_ONNET_1MONTH  (r=1.0 with AVG_INC_ONNET_1MONTH)

Data cleaning:
  - Several numeric columns are stored as strings with Spanish-locale
    thousand separators (e.g. "88.533.333.333" should be 88.533333333).
  - AVG_DATA_3MONTH has one corrupted observation (~1e16); clipped to
    the 99th percentile.

Categorical features: low-cardinality integer columns (<=20 unique).

No missing values after cleaning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────
DEFAULT_SUBPATH = "data/raw/Chile/chile_postpaid.csv"

TARGET_COL = "CHURN"

DROP_COLS = [
    "ID",
    "START_DATE",
    "END_DATE",
    # redundant (r=1.0 with ACTIVE_DAYS)
    "ACTIVE_WEEKS",
    "ACTIVE_MONTHS",
    # duplicate (r=1.0 with AVG_INC_* counterparts)
    "AVG_MINUTES_INC_OFFNET_1MONTH",
    "AVG_MINUTES_INC_ONNET_1MONTH",
]

LOW_CARD_THRESHOLD = 20


# ── helpers ──────────────────────────────────────────────────────────

def _fix_dot_separators(val: str) -> str:
    """
    Fix Spanish-locale thousand separators.
    '88.533.333.333' -> '88.533333333'
    """
    parts = val.split(".")
    if len(parts) <= 2:
        return val
    return parts[0] + "." + "".join(parts[1:])


# ── load + clean ─────────────────────────────────────────────────────

def load_chile_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Read the raw Chile postpaid CSV, fix numeric parsing issues,
    sort temporally by START_DATE, then drop identifier/date/redundant columns.
    """
    df = pd.read_csv(csv_path)

    # Fix dot-separated decimals in object columns (excluding dates)
    obj_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in ("START_DATE", "END_DATE")
    ]
    for col in obj_cols:
        df[col] = df[col].astype(str).apply(_fix_dot_separators)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by subscription start date for temporal split
    df["START_DATE"] = pd.to_datetime(df["START_DATE"], dayfirst=True, errors="coerce")
    df = df.sort_values("START_DATE").reset_index(drop=True)

    # Drop ID, date, and redundant columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Clip AVG_DATA_3MONTH outlier (one observation at ~1e16)
    if "AVG_DATA_3MONTH" in df.columns:
        p99 = df["AVG_DATA_3MONTH"].quantile(0.99)
        df["AVG_DATA_3MONTH"] = df["AVG_DATA_3MONTH"].clip(upper=p99)

    return df


# ── encode for TabNet ────────────────────────────────────────────────

def encode_tabular_for_tabnet(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict, List[int], List[int]]:
    """
    Encode for TabNet. Low-cardinality integer columns are treated as
    categoricals for embedding.

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

    # Identify low-cardinality integer columns as categoricals
    cat_cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            if df[c].nunique(dropna=True) <= LOW_CARD_THRESHOLD:
                cat_cols.append(c)

    # Label-encode categoricals
    cat_dims: dict[str, int] = {}
    for col in cat_cols:
        codes, _uniques = pd.factorize(df[col], sort=True)
        df[col] = codes
        cat_dims[col] = len(_uniques)

    # Fill any numeric NaNs with median
    for col in df.columns:
        if col not in cat_cols and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

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
    """Read CSV -> clean -> sort temporally -> encode -> return TabNet-ready arrays."""
    csv_path = Path(repo_root) / csv_subpath
    df = load_chile_csv(csv_path)
    return encode_tabular_for_tabnet(df)
