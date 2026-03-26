"""
Quick EDA for Delft Telephone Company churn dataset.
Run from repo root:  python eda_delft.py
"""
import pandas as pd
import numpy as np

CSV = "data/raw/delft_churn/churn.csv"
df = pd.read_csv(CSV)

print("=" * 70)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Missing values total: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ── Target ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TARGET: Leave")
print(df["leave"].value_counts() if "leave" in df.columns else df["Leave"].value_counts())
target_col = "leave" if "leave" in df.columns else "Leave"
churn_vals = df[target_col].unique()
print(f"Unique values: {list(churn_vals)}")
print(f"Churn rate: {df[target_col].value_counts(normalize=True).to_dict()}")

# ── Dtypes overview ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("COLUMN TYPES")
obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
print(f"  String columns ({len(obj_cols)}): {obj_cols}")
print(f"  Numeric columns ({len(num_cols)}): {num_cols}")

# ── All columns: unique values + dtype ───────────────────────────
print("\n" + "=" * 70)
print("ALL COLUMNS — dtype, nunique, sample values")
for c in df.columns:
    nu = df[c].nunique()
    sample = list(df[c].unique()[:8])
    print(f"  {c:35s}  dtype={str(df[c].dtype):10s}  nunique={nu:5d}  sample={sample}")

# ── String columns: unique values ────────────────────────────────
print("\n" + "=" * 70)
print("STRING COLUMNS — unique values")
for c in obj_cols:
    vals = df[c].unique()
    print(f"  {c:35s} ({len(vals):2d}): {list(vals)}")

# ── Constant / ID columns ────────────────────────────────────────
print("\n" + "=" * 70)
print("CONSTANT / ID COLUMNS (drop candidates)")
found = False
for c in df.columns:
    nu = df[c].nunique()
    if nu <= 1:
        print(f"  {c:35s}  CONSTANT = {df[c].iloc[0]!r}")
        found = True
    elif nu == len(df):
        print(f"  {c:35s}  ALL UNIQUE (likely ID)")
        found = True
if not found:
    print("  None found (good!)")

# ── Missing per column ───────────────────────────────────────────
print("\n" + "=" * 70)
print("MISSING VALUES PER COLUMN")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values!")
else:
    print(missing[missing > 0].to_string())

# ── Numeric summary ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
if num_cols:
    print(df[num_cols].describe().T.to_string())
else:
    print("  No numeric columns found (might all be parsed as object)")

# ── Outlier check (IQR) ─────────────────────────────────────────
print("\n" + "=" * 70)
print("OUTLIER CHECK (IQR method)")
for c in num_cols:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    pct = n_out / len(df) * 100
    if n_out > 0:
        print(f"  {c:35s}  {n_out:5d} outliers ({pct:5.1f}%)")

# ── Skewness ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SKEWNESS (|skew| > 2 flagged)")
for c in num_cols:
    skew = df[c].skew()
    flag = " ⚠️" if abs(skew) > 2 else ""
    print(f"  {c:35s}  skew={skew:+.3f}{flag}")

# ── Correlations with target ─────────────────────────────────────
print("\n" + "=" * 70)
print("POINT-BISERIAL CORRELATION WITH TARGET (numeric cols)")
if target_col in obj_cols:
    # need to encode target first
    y = (df[target_col].str.upper() == "LEAVE").astype(int)
else:
    y = df[target_col]
corrs = {}
for c in num_cols:
    if c != target_col:
        corrs[c] = df[c].corr(y)
if corrs:
    corrs = pd.Series(corrs).sort_values(key=abs, ascending=False)
    print(corrs.to_string())

print("\n" + "=" * 70)
print("Done.")
