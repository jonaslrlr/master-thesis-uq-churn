from __future__ import annotations
from pathlib import Path

from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet as enc_telco
from thesis_uq.data.cell2cell import load_cell2cell_csv, encode_tabular_for_tabnet as enc_cell2cell
from thesis_uq.data.bank import load_bank_csv, encode_tabular_for_tabnet as enc_bank
from thesis_uq.data.delft import load_delft_csv, encode_tabular_for_tabnet as enc_delft
from thesis_uq.data.cdr import load_cdr_combined, encode_tabular_for_tabnet as enc_cdr
from thesis_uq.data.chile import load_chile_csv, encode_tabular_for_tabnet as enc_chile


def load_for_tabnet(dataset: str, repo_root: Path):
    if dataset == "telco":
        csv_path = repo_root / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = load_telco_csv(csv_path)
        return enc_telco(df)

    if dataset == "cell2cell":
        csv_path = repo_root / "data/raw/cell2cell/cell2cell.csv"
        df = load_cell2cell_csv(csv_path)
        return enc_cell2cell(df)

    if dataset == "bank":
        csv_path = repo_root / "data/raw/bank_churn/Churn_Modelling.csv"
        df = load_bank_csv(csv_path)
        return enc_bank(df)

    if dataset == "delft":
        csv_path = repo_root / "data/raw/delft_churn/churn.csv"
        df = load_delft_csv(csv_path)
        return enc_delft(df)

    if dataset == "cdr":
        from thesis_uq.data.cdr import load_for_tabnet as _load_cdr
        return _load_cdr(repo_root)

    if dataset == "chile":
        csv_path = repo_root / "data/raw/Chile/chile_postpaid.csv"
        df = load_chile_csv(csv_path)
        return enc_chile(df)

    raise ValueError(f"Unknown dataset: {dataset}")
