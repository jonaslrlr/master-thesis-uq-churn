"""
CDR (Call Detail Records) — data loader for TabNet
====================================================

Source: Confidential dataset provided by Yameng Guo (UGent).
Pre-split into train/val/test with proper temporal cohort windows.

90,000 total (30,000 per split), 5 numeric features (pre-scaled), no categoricals.
Target: target (0 / 1)
Churn rate: ~6.7-6.9% across all splits (stable).

Temporal split:
  Train: 30,000 rows, churn ~6.7%
  Valid: 30,000 rows, churn ~6.9%
  Test:  30,000 rows, churn ~6.9%

Dropped columns:
  - Unnamed: 0  (row index / user ID)

Numeric features (5 cols, already scaled 0-1):
  start_call_duration, last_call_duration, count_timestamp,
  min_duration, max_duration

No categorical features.
No missing values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# —— constants ————————————————————————————————————————————

DEFAULT_DIR_SUBPATH = "data/raw/cdr"

TARGET_COL = "target"

DROP_COLS = ["Unnamed: 0"]

FEATURE_COLS = [
    "start_call_duration",
    "last_call_duration",
    "count_timestamp",
    "min_duration",
    "max_duration",
]


# —— load pre-split ———————————————————————————————————————

def _load_single_csv(csv_path: Path | str) -> pd.DataFrame:
    """Read a single CDR CSV, drop ID column."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def load_cdr_combined(cdr_dir: Path | str) -> pd.DataFrame:
    """
    Load all 3 splits and concatenate in order: train, val, test.
    The row order encodes the temporal split boundaries.
    """
    cdr_dir = Path(cdr_dir)
    train = _load_single_csv(cdr_dir / "train.csv")
    val = _load_single_csv(cdr_dir / "val.csv")
    test = _load_single_csv(cdr_dir / "test.csv")
    df = pd.concat([train, val, test], ignore_index=True)
    return df


# —— encode for TabNet ————————————————————————————————————

def encode_tabular_for_tabnet(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict, List[int], List[int]]:
    """
    Return arrays ready for TabNet. No categoricals to encode.

    Returns
    -------
    X             : np.ndarray   (N, D)
    y             : np.ndarray   (N,)
    features      : list[str]    feature names (column order in X)
    cat_cols      : list[str]    [] (no categoricals)
    cat_dims      : dict         {} (no categoricals)
    cat_idxs      : list[int]    [] (no categoricals)
    cat_dims_list : list[int]    [] (no categoricals)
    """
    df = df.copy()

    y = df.pop(TARGET_COL).values.astype(np.int64)

    features = list(df.columns)
    X = df.values.astype(np.float32)

    return X, y, features, [], {}, [], []


# —— one-call convenience (used by registry.py) ——————————

def load_for_tabnet(
    repo_root: Path | str,
    dir_subpath: str = DEFAULT_DIR_SUBPATH,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict, List[int], List[int]]:
    """Read all 3 CSVs → concat → encode → return TabNet-ready arrays."""
    cdr_dir = Path(repo_root) / dir_subpath
    df = load_cdr_combined(cdr_dir)
    return encode_tabular_for_tabnet(df)
