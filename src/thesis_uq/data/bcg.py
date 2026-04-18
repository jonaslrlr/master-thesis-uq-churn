"""
BCG Energy — data loader for TabNet
=====================================

Source: BCG x ISCTE Data Science Challenge (energy sector).
14,606 customers, ~20 usable features, binary target (churn).

Domain: energy (contract churn).
Churn rate: ~9.7 % (imbalanced).

Dropped columns:
  - id             (customer identifier)
  - date_activ     (activation date — not used as feature)
  - date_end       (contract end date)
  - date_modif_prod (product modification date)
  - date_renewal   (renewal date)

Categorical features (label-encoded for TabNet, 3 cols):
  channel_sales, has_gas, origin_up

Numeric features (~17 cols):
  cons_12m, cons_gas_12m, cons_last_month,
  forecast_cons_12m, forecast_cons_year, forecast_discount_energy,
  forecast_meter_rent_12m, forecast_price_energy_off_peak,
  forecast_price_energy_peak, forecast_price_pow_off_peak,
  has_gas, imp_cons, margin_gross_pow_ele, margin_net_pow_ele,
  nb_prod_act, net_margin, num_years_antig, pow_max

No missing values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────
DEFAULT_SUBPATH = "data/raw/bcg_energy/client_data.csv"

TARGET_COL = "churn"

DROP_COLS = [
    "id",
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal",
]

CAT_COLS = [
    "channel_sales",
    "has_gas",
    "origin_up",
]


# ── load + clean ─────────────────────────────────────────────────────

def load_bcg_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Read the raw BCG energy client CSV and return a cleaned DataFrame.

    Drops identifier and date columns.
    """
    df = pd.read_csv(csv_path)

    # Drop ID and date columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

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
        df[col] = df[col].fillna("missing")
        codes, _uniques = pd.factorize(df[col], sort=True)
        df[col] = codes
        cat_dims[col] = len(_uniques)

    # Fill any remaining numeric NaNs with median
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
    """Read CSV → clean → encode → return TabNet-ready arrays."""
    csv_path = Path(repo_root) / csv_subpath
    df = load_bcg_csv(csv_path)
    return encode_tabular_for_tabnet(df)
