from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cell2CellConfig:
    # columns
    id_col: str = "CustomerID"
    target_col: str = "Churn"

    # target mapping
    pos_label: str = "Yes"
    neg_label: str = "No"

    # basic cleaning
    drop_cols: Tuple[str, ...] = ("RVOwner", "Homeownership")  # optional, matches the paper snippet
    fill_cat_value: str = "missing"

    # treat these "Yes/No" cols as binary 0/1
    binary_yesno_cols: Tuple[str, ...] = (
        "ChildrenInHH",
        "HandsetRefurbished",
        "HandsetWebCapable",
        "TruckOwner",
        "BuysViaMailOrder",
        "RespondsToMailOffers",
        "OptOutMailings",
        "NonUSTravel",
        "OwnsComputer",
        "HasCreditCard",
        "NewCellphoneUser",
        "NotNewCellphoneUser",
        "OwnsMotorcycle",
        "MadeCallToRetentionTeam",
    )

    # categorical with small integer mapping
    marital_map: Tuple[Tuple[str, int], ...] = (("Unknown", 0), ("No", 1), ("Yes", 2))

    # special parsing
    creditrating_sep: str = "-"
    handsetprice_unknown: str = "Unknown"


def load_cell2cell_csv(path):
    df = pd.read_csv(path)

    # keep only labeled rows (robust even if file changes)
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    # (optional) match their cleaning a bit
    if "ServiceArea" in df.columns:
        df["ServiceArea"] = df["ServiceArea"].fillna("no info")

    # drop duplicate customers (they did this)
    if "CustomerID" in df.columns:
        df = df.drop_duplicates(subset=["CustomerID"], keep="last")

    return df
    # Drop duplicates on ID if present (keep last like your snippet)
    if cfg.id_col in df.columns:
        df = df.drop_duplicates(subset=[cfg.id_col], keep="last")

    # Drop some columns (only if they exist)
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Fill a known high-cardinality categorical
    if "ServiceArea" in df.columns:
        df["ServiceArea"] = df["ServiceArea"].fillna("no info")

    # Map Yes/No -> 1/0 for chosen columns if they exist
    for c in cfg.binary_yesno_cols:
        if c in df.columns:
            df[c] = df[c].map({"No": 0, "Yes": 1})

    # MaritalStatus mapping (if exists)
    if "MaritalStatus" in df.columns:
        m = dict(cfg.marital_map)
        df["MaritalStatus"] = df["MaritalStatus"].map(m)

    # CreditRating: take prefix before '-' and cast to int (if exists)
    if "CreditRating" in df.columns:
        # some rows can be NaN, handle carefully
        cr = df["CreditRating"].astype(str)
        cr = cr.str.split(cfg.creditrating_sep).str[0]
        df["CreditRating"] = pd.to_numeric(cr, errors="coerce")

    # HandsetPrice: "Unknown" -> 0, cast numeric (if exists)
    if "HandsetPrice" in df.columns:
        hp = df["HandsetPrice"].astype(str).replace(cfg.handsetprice_unknown, "0")
        df["HandsetPrice"] = pd.to_numeric(hp, errors="coerce")

    # Target mapping if present (keep original col name, we’ll map in encoder)
    if cfg.target_col in df.columns:
        # keep as string; encoder will map
        pass

    # IMPORTANT: do NOT dropna() here; handle missing in encoder consistently
    return df


def encode_tabular_for_tabnet(
    df: pd.DataFrame,
    cfg: Optional[Cell2CellConfig] = None,
):
    """
    Encode for TabNet:
      - split y (0/1) if target exists
      - build categorical indices/dims lists
      - return X numeric matrix (float32)

    Returns:
      X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list
    """
    if cfg is None:
        cfg = Cell2CellConfig()

    df = df.copy()

    # Extract y
    y = None
    if cfg.target_col in df.columns:
        y = df[cfg.target_col].map({cfg.neg_label: 0, cfg.pos_label: 1}).astype(int).to_numpy()
        df = df.drop(columns=[cfg.target_col])

    # Drop ID col from features
    if cfg.id_col in df.columns:
        df = df.drop(columns=[cfg.id_col])

    # Identify categorical columns: object + a few known categorical-ish int cols
    cat_cols: List[str] = [c for c in df.columns if df[c].dtype == "object"]

    # ALSO treat low-cardinality integer cols (like already-mapped Yes/No) as categorical
    for c in df.columns:
        if c in cat_cols:
            continue
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if nunique <= 20 and c not in cat_cols:
                # many of these are binary flags; TabNet can treat them as categorical
                cat_cols.append(c)

    # Fill missing: categorical -> "missing", numeric -> median
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype(str).fillna(cfg.fill_cat_value)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # Build category codes for TabNet
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

    # for compatibility with your telco encode signature
    cat_dims = {c: int(df[c].nunique()) for c in cat_cols}

    if y is None:
        # if unlabeled (holdout), return y as None
        return X, None, features, cat_cols, cat_dims, cat_idxs, cat_dims_list

    return X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list
