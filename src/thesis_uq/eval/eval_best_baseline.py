from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from thesis_uq.seed import set_seed
from thesis_uq.data.splits import train_valid_test_split
from thesis_uq.metrics.ranking import standard_report
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline

# dataset loaders
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
    # everything else through registry (e.g., cell2cell)
    return load_for_tabnet(dataset, repo_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seeds", default="5-15", help='e.g. "5-15" (inclusive) or "5,6,7"')
    ap.add_argument("--device", default="cpu")

    ap.add_argument(
        "--best_file",
        default=None,
        help="Path to reports/best/{dataset}_baseline_split{split}_trainseeds1-4.json",
    )
    ap.add_argument("--repo_root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else guess_repo_root()
    dataset = args.dataset
    split_seed = args.split_seed
    train_seeds = parse_seeds(args.seeds)
    device = args.device

    best_file = Path(args.best_file).expanduser().resolve() if args.best_file else (
        repo_root / "reports" / "best" / f"{dataset}_baseline_split{split_seed}_trainseeds1-4.json"
    )

    print("Repo root:", repo_root)
    print("Dataset:", dataset)
    print("Device:", device)
    print("Fixed split seed:", split_seed)
    print("Eval train seeds:", train_seeds)
    print("Best file:", best_file)

    best = json.loads(best_file.read_text())
    cfg = best["config"]
    best_tag = best.get("best_tag", "unknown")
    print("\nUsing best baseline config:", best_tag)
    print(cfg)

    # load + fixed split once
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_tabnet_data(dataset, repo_root)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=split_seed)
    print("Split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    out_dir = repo_root / "reports" / "eval"
    run_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for s in train_seeds:
        run_json = run_dir / f"{dataset}_baseline_eval_split{split_seed}_trainseed{s}.json"
        if run_json.exists():
            row = json.loads(run_json.read_text())
            rows.append(row)
            print(f"⏭️  SKIP seed={s} (cached)")
            continue

        set_seed(s)
        print(f"\n=== TRAIN SEED {s} ===")

        clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=device,

            n_d=cfg["n_d"],
            n_a=cfg["n_a"],
            n_steps=cfg["n_steps"],
            gamma=cfg["gamma"],
            cat_emb_dim=cfg["cat_emb_dim"],
            mask_type=cfg["mask_type"],

            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            batch_size=cfg["batch_size"],
            virtual_batch_size=cfg["virtual_batch_size"],

            dropout=cfg.get("dropout", 0.0),
            seed=s,
        )

        p_test = clf.predict_proba(X_test)[:, 1]
        rep = standard_report(y_test, p_test)
        row = {"train_seed": s, **rep}
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))
        print("Test report:", rep)

    import pandas as pd

    df_rep = pd.DataFrame(rows).set_index("train_seed").sort_index()
    print("\n=== PER-SEED TEST RESULTS ===")
    print(df_rep)

    mean = df_rep.mean(numeric_only=True)
    std = df_rep.std(numeric_only=True)

    print("\n=== MEAN ± STD (TEST, fixed split) ===")
    for k in mean.index:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    csv_file = out_dir / f"{dataset}_baseline_eval_split{split_seed}_trainseeds{train_seeds[0]}-{train_seeds[-1]}.csv"
    df_rep.to_csv(csv_file)

    summary = {
        "dataset": dataset,
        "split_seed": split_seed,
        "train_seeds": train_seeds,
        "best_file": str(best_file),
        "best_tag": best_tag,
        "config": cfg,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{dataset}_baseline_eval_split{split_seed}_trainseeds{train_seeds[0]}-{train_seeds[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n✅ Saved per-seed CSV to:", csv_file)
    print("✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
