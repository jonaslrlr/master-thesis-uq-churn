from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET_DEFAULT = "Churn"

def load_telco_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Fix TotalCharges + drop ID
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df

def encode_tabular_for_tabnet(
    df: pd.DataFrame,
    target: str = TARGET_DEFAULT,
    cat_unique_threshold: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, int], List[int], List[int]]:
    """
    Returns:
      X (float32), y (int64),
      features (ordered),
      cat_cols (names),
      cat_dims (name -> #categories),
      cat_idxs (indices in features),
      cat_dims_list (dims aligned with cat_idxs)
    """
    df = df.copy()

    # Encode target
    if df[target].dtype == "object":
        df[target] = df[target].map({"Yes": 1, "No": 0}).astype(np.int64)

    nunique = df.nunique(dropna=False)
    types = df.dtypes

    cat_cols: List[str] = []
    cat_dims: Dict[str, int] = {}

    for col in df.columns:
        if col == target:
            continue

        if types[col] == "object" or nunique[col] < cat_unique_threshold:
            df[col] = df[col].fillna("Unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).values).astype(np.int64)
            cat_cols.append(col)
            cat_dims[col] = len(le.classes_)
        else:
            df[col] = df[col].fillna(df[col].mean())

    features = [c for c in df.columns if c != target]
    cat_idxs = [i for i, c in enumerate(features) if c in cat_cols]
    cat_dims_list = [cat_dims[features[i]] for i in cat_idxs]

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.int64)
    return X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list
