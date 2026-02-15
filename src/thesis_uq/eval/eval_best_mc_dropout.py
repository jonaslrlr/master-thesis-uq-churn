from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.models.tabnet_mc_dropout import train_tabnet_mc_dropout, mc_predict

from thesis_uq.data.registry import load_for_tabnet
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet


def parse_seeds(s: str) -> list[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def guess_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "reports").exists():
            return p
    return Path.cwd()


def load_tabnet_data(dataset: str, repo_root: Path):
    if dataset == "telco":
        csv_path = repo_root / "data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = load_telco_csv(csv_path)
        return encode_tabular_for_tabnet(df)
    return load_for_tabnet(dataset, repo_root)


def fit_lr_reranker(p_valid: np.ndarray, u_valid: np.ndarray, y_valid: np.ndarray):
    Xv = np.column_stack([u_valid, p_valid])
    scaler = MinMaxScaler()
    Xv_s = scaler.fit_transform(Xv)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xv_s, y_valid)
    return scaler, lr


def apply_lr_reranker(scaler, lr, p: np.ndarray, u: np.ndarray) -> np.ndarray:
    Xt = np.column_stack([u, p])
    Xt_s = scaler.transform(Xt)
    return lr.predict_proba(Xt_s)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seeds", default="5-15")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mc_samples", type=int, default=200)

    ap.add_argument("--best_file", default=None)
    ap.add_argument("--repo_root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else guess_repo_root()
    dataset = args.dataset
    split_seed = args.split_seed
    train_seeds = parse_seeds(args.seeds)
    device = args.device
    mc_samples = args.mc_samples

    best_file = Path(args.best_file).expanduser().resolve() if args.best_file else (
        repo_root / "reports" / "best" / f"{dataset}_mc_dropout_split{split_seed}_trainseeds1-4.json"
    )

    print("Repo root:", repo_root)
    print("Dataset:", dataset)
    print("Device:", device)
    print("Fixed split seed:", split_seed)
    print("Eval train seeds:", train_seeds)
    print("Best MC file:", best_file)
    print("MC samples (eval):", mc_samples)

    best = json.loads(best_file.read_text())
    cfg = best["config"]
    best_tag = best.get("best_tag", "unknown")
    print("\nUsing best MC config:", best_tag)
    print(cfg)

    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_tabnet_data(dataset, repo_root)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=split_seed)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    dropout = float(cfg["dropout"])

    tabnet_kwargs = dict(
        n_d=int(cfg["n_d"]),
        n_a=int(cfg["n_a"]),
        n_steps=int(cfg["n_steps"]),
        gamma=float(cfg["gamma"]),
        mask_type=str(cfg["mask_type"]),
        cat_emb_dim=int(cfg["cat_emb_dim"]),
    )
    train_kwargs = dict(
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        batch_size=int(cfg["batch_size"]),
        virtual_batch_size=int(cfg["virtual_batch_size"]),
    )

    out_dir = repo_root / "reports" / "eval"
    run_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for s in train_seeds:
        run_json = run_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseed{s}.json"
        if run_json.exists():
            row = json.loads(run_json.read_text())
            rows.append(row)
            print(f"⏭️  SKIP seed={s} (cached)")
            continue

        set_seed(s)
        print(f"\n=== TRAIN SEED {s} ===")

        clf = train_tabnet_mc_dropout(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=device,
            dropout=dropout,
            tabnet_kwargs=tabnet_kwargs,
            train_kwargs=train_kwargs,
            seed=s,
        )

        # DET (dropout off)
        p_det = clf.predict_proba(X_test)[:, 1]
        rep_det = standard_report(y_test, p_det)

        # MC (mean + uncertainty)
        p_mc, u_mc = mc_predict(clf, X_test, n_samples=mc_samples)
        rep_mc = standard_report(y_test, p_mc)

        # LR rerank (fit on VALID using same model)
        p_valid, u_valid = mc_predict(clf, X_valid, n_samples=mc_samples)
        scaler, lr = fit_lr_reranker(p_valid, u_valid, y_valid)
        p_lr = apply_lr_reranker(scaler, lr, p_mc, u_mc)
        rep_lr = standard_report(y_test, p_lr)

        row = {
            "train_seed": s,

            "det_auc_roc": rep_det["auc_roc"],
            "det_auc_pr": rep_det["auc_pr"],
            "det_lift10": rep_det["lift10"],

            "mc_auc_roc": rep_mc["auc_roc"],
            "mc_auc_pr": rep_mc["auc_pr"],
            "mc_lift10": rep_mc["lift10"],
            "mc_u_mean": float(np.mean(u_mc)),

            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_lift10": rep_lr["lift10"],
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        print("DET:", rep_det)
        print("MC :", rep_mc, "| u_mean:", row["mc_u_mean"])
        print("LR :", rep_lr)

    import pandas as pd

    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    csv_file = out_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseeds{train_seeds[0]}-{train_seeds[-1]}.csv"
    df_rep.to_csv(csv_file)

    summary = {
        "dataset": dataset,
        "split_seed": split_seed,
        "train_seeds": train_seeds,
        "best_mc_file": str(best_file),
        "best_tag": best_tag,
        "config": cfg,
        "mc_samples_eval": mc_samples,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseeds{train_seeds[0]}-{train_seeds[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    print("\n✅ Saved per-seed CSV to:", csv_file)
    print("✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
