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
from thesis_uq.models.tabnet_baseline import train_tabnet_baseline          # NEW

# dataset loaders
from thesis_uq.data.registry import load_for_tabnet
from thesis_uq.data.telco import load_telco_csv, encode_tabular_for_tabnet
from thesis_uq.plots.uq_plots import plot_prob_vs_uncertainty
from thesis_uq.io import save_uq_scores_npz


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
    """Fit LR reranker on VALID only (no leakage). Features: [u, p]."""
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
    ap.add_argument("--seeds", default="5-15", help='e.g. "5-15" (inclusive) or "5,6,7"')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mc_samples", type=int, default=200, help="MC samples for evaluation (test + valid UQ)")

    ap.add_argument(
        "--best_file",
        default=None,
        help="Path to reports/best/{dataset}_mc_dropout_split{split}_trainseeds1-4.json",
    )
    ap.add_argument(                                                         # NEW
        "--baseline_best_file",                                              # NEW
        default=None,                                                        # NEW
        help="Path to reports/best/{dataset}_baseline_split{split}_trainseeds1-4.json",
    )                                                                        # NEW
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
    baseline_best_file = Path(args.baseline_best_file).expanduser().resolve() if args.baseline_best_file else (
        repo_root / "reports" / "best" / f"{dataset}_baseline_split{split_seed}_trainseeds1-4.json"
    )                                                                        # NEW

    print("Repo root:", repo_root)
    print("Dataset:", dataset)
    print("Device:", device)
    print("Fixed split seed:", split_seed)
    print("Eval train seeds:", train_seeds)
    print("Best MC file:", best_file)
    print("Best baseline file:", baseline_best_file)                         # NEW
    print("MC samples (eval):", mc_samples)

    if not best_file.exists():
        raise FileNotFoundError(
            f"Best-file not found for dataset={dataset!r}.\nExpected:\n  {best_file}\n"
            f"Create it first (gridsearch/best) or pass --best_file explicitly."
        )
    if not best_file.name.startswith(f"{dataset}_"):
        raise ValueError(f"Best-file name {best_file.name!r} does not match dataset={dataset!r}")
    if not baseline_best_file.exists():                                      # NEW
        raise FileNotFoundError(                                             # NEW
            f"Baseline best-file not found.\nExpected:\n  {baseline_best_file}"
        )                                                                    # NEW

    best = json.loads(best_file.read_text())
    cfg = best["config"]
    best_tag = best.get("best_tag", "unknown")
    print("\nUsing best MC config:", best_tag)
    print(cfg)

    # ── NEW: load baseline config ────────────────────────────────────
    base_best = json.loads(baseline_best_file.read_text())
    base_cfg = base_best["config"]
    print("\nBaseline config:", base_best.get("best_tag", "unknown"))
    print(json.dumps(base_cfg, indent=2))
    # ─────────────────────────────────────────────────────────────────

    # load + fixed split once
    X, y, features, cat_cols, cat_dims, cat_idxs, cat_dims_list = load_tabnet_data(dataset, repo_root)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y, seed=split_seed, presplit=(30000, 30000) if dataset == "cdr" else (4939, 1058) if dataset == "chile" else None)
    print("\nFixed split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    dropout = float(cfg["dropout"])
    attn_dropout = float(cfg.get("attn_dropout", 0.0))

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

    uq_dir = repo_root / "reports" / "uq_scores"
    uq_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = repo_root / "reports" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for s in train_seeds:
        run_json = run_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseed{s}.json"

        # ── NEW: cache check — skip only if decoupled fields exist ───
        if run_json.exists():
            row = json.loads(run_json.read_text())
            if "det_lr_auc_pr" in row and "base_lr_auc_pr" in row:
                rows.append(row)
                print(f"⏭️  SKIP seed={s} (cached with decoupled rerankers)")
                continue
            else:
                print(f"🔄 RE-RUN seed={s} (missing decoupled reranker fields)")
        # ─────────────────────────────────────────────────────────────

        set_seed(s)
        print(f"\n=== TRAIN SEED {s} ===")

        # Train MCD model
        try:
            clf = train_tabnet_mc_dropout(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                device_name=device,
                dropout=dropout,
                attn_dropout=attn_dropout,
                tabnet_kwargs=tabnet_kwargs,
                train_kwargs=train_kwargs,
                seed=s,
            )
        except TypeError:
            clf = train_tabnet_mc_dropout(
                X_train, y_train, X_valid, y_valid,
                cat_idxs, cat_dims_list,
                device_name=device,
                dropout=dropout,
                tabnet_kwargs=tabnet_kwargs,
                train_kwargs=train_kwargs,
                seed=s,
            )

        # ── NEW: train baseline model (no dropout, standard CE) ──────
        baseline_clf = train_tabnet_baseline(
            X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims_list,
            device_name=device,
            n_d=int(base_cfg["n_d"]), n_a=int(base_cfg["n_a"]),
            n_steps=int(base_cfg["n_steps"]), gamma=float(base_cfg["gamma"]),
            mask_type=str(base_cfg["mask_type"]),
            cat_emb_dim=int(base_cfg["cat_emb_dim"]),
            lr=float(base_cfg["lr"]), weight_decay=float(base_cfg["weight_decay"]),
            max_epochs=int(base_cfg["max_epochs"]), patience=int(base_cfg["patience"]),
            batch_size=int(base_cfg["batch_size"]),
            virtual_batch_size=int(base_cfg["virtual_batch_size"]),
            seed=s,
        )
        # ─────────────────────────────────────────────────────────────

        # DET: MCD model, dropout OFF
        p_det_test = clf.predict_proba(X_test)[:, 1]
        p_det_valid = clf.predict_proba(X_valid)[:, 1]                       # NEW
        rep_det = standard_report(y_test, p_det_test)

        # MC: MCD model, dropout ON, T forward passes
        p_mc_test, u_mc_test = mc_predict(clf, X_test, n_samples=mc_samples)
        p_mc_valid, u_mc_valid = mc_predict(clf, X_valid, n_samples=mc_samples)
        rep_mc = standard_report(y_test, p_mc_test)

        # ── NEW: baseline predictions ────────────────────────────────
        p_base_test = baseline_clf.predict_proba(X_test)[:, 1]
        p_base_valid = baseline_clf.predict_proba(X_valid)[:, 1]
        rep_base = standard_report(y_test, p_base_test)
        # ─────────────────────────────────────────────────────────────

        plot_prob_vs_uncertainty(
            y_test, p_mc_test, u_mc_test,
            title=f"MC Dropout | P(churn) vs std | test seed={s}",
            save_path=plot_dir / f"{dataset}_mc_dropout_scatter_split{split_seed}_seed{s}.png",
        )

        save_uq_scores_npz(
            uq_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseed{s}.npz",
            y_valid=y_valid, p_valid=p_mc_valid, u_valid=u_mc_valid,
            y_test=y_test, p_test=p_mc_test, u_test=u_mc_test)

        # ── Reranker A: LR(p_mc, u_mc) — original, coupled ──────────
        scaler, lr = fit_lr_reranker(p_mc_valid, u_mc_valid, y_valid)
        p_lr = apply_lr_reranker(scaler, lr, p_mc_test, u_mc_test)
        rep_lr = standard_report(y_test, p_lr)

        # ── NEW: Reranker B: LR(p_det, u_mc) — same-model decoupled ─
        sc_det, lr_det = fit_lr_reranker(p_det_valid, u_mc_valid, y_valid)
        p_lr_det = apply_lr_reranker(sc_det, lr_det, p_det_test, u_mc_test)
        rep_lr_det = standard_report(y_test, p_lr_det)

        # ── NEW: Reranker C: LR(p_base, u_mc) — fully decoupled ─────
        sc_base, lr_base = fit_lr_reranker(p_base_valid, u_mc_valid, y_valid)
        p_lr_base = apply_lr_reranker(sc_base, lr_base, p_base_test, u_mc_test)
        rep_lr_base = standard_report(y_test, p_lr_base)
        # ─────────────────────────────────────────────────────────────

        print(f"  LR(mc)   coefs: u={lr.coef_[0][0]:.4f}, p={lr.coef_[0][1]:.4f}")
        print(f"  LR(det)  coefs: u={lr_det.coef_[0][0]:.4f}, p={lr_det.coef_[0][1]:.4f}")
        print(f"  LR(base) coefs: u={lr_base.coef_[0][0]:.4f}, p={lr_base.coef_[0][1]:.4f}")

        row = {
            "train_seed": s,

            # Baseline standalone (reference)
            "base_auc_roc": rep_base["auc_roc"],
            "base_auc_pr": rep_base["auc_pr"],
            "base_acc": rep_base["acc"],
            "base_lift10": rep_base["lift10"],
            "base_ece": rep_base["ece"],
            "base_brier": rep_base["brier"],

            # DET: MCD model, dropout OFF
            "det_auc_roc": rep_det["auc_roc"],
            "det_auc_pr": rep_det["auc_pr"],
            "det_acc": rep_det["acc"],
            "det_lift10": rep_det["lift10"],
            "det_ece": rep_det["ece"],
            "det_brier": rep_det["brier"],

            # MC: MCD model, dropout ON, averaged
            "mc_auc_roc": rep_mc["auc_roc"],
            "mc_auc_pr": rep_mc["auc_pr"],
            "mc_acc": rep_mc["acc"],
            "mc_lift10": rep_mc["lift10"],
            "mc_u_mean": float(np.mean(u_mc_test)),
            "mc_ece": rep_mc["ece"],
            "mc_brier": rep_mc["brier"],

            # Reranker A: LR(p_mc, u_mc) — original, coupled
            "lr_auc_roc": rep_lr["auc_roc"],
            "lr_auc_pr": rep_lr["auc_pr"],
            "lr_acc": rep_lr["acc"],
            "lr_lift10": rep_lr["lift10"],
            "lr_ece": rep_lr["ece"],
            "lr_brier": rep_lr["brier"],
            "lr_coef_u": float(lr.coef_[0][0]),
            "lr_coef_p": float(lr.coef_[0][1]),
            "lr_intercept": float(lr.intercept_[0]),

            # Reranker B: LR(p_det, u_mc) — same-model decoupled
            "det_lr_auc_roc": rep_lr_det["auc_roc"],
            "det_lr_auc_pr": rep_lr_det["auc_pr"],
            "det_lr_acc": rep_lr_det["acc"],
            "det_lr_lift10": rep_lr_det["lift10"],
            "det_lr_ece": rep_lr_det["ece"],
            "det_lr_brier": rep_lr_det["brier"],
            "det_lr_coef_u": float(lr_det.coef_[0][0]),
            "det_lr_coef_p": float(lr_det.coef_[0][1]),
            "det_lr_intercept": float(lr_det.intercept_[0]),

            # Reranker C: LR(p_base, u_mc) — fully decoupled
            "base_lr_auc_roc": rep_lr_base["auc_roc"],
            "base_lr_auc_pr": rep_lr_base["auc_pr"],
            "base_lr_acc": rep_lr_base["acc"],
            "base_lr_lift10": rep_lr_base["lift10"],
            "base_lr_ece": rep_lr_base["ece"],
            "base_lr_brier": rep_lr_base["brier"],
            "base_lr_coef_u": float(lr_base.coef_[0][0]),
            "base_lr_coef_p": float(lr_base.coef_[0][1]),
            "base_lr_intercept": float(lr_base.intercept_[0]),
        }
        rows.append(row)
        run_json.write_text(json.dumps(row, indent=2))

        print(f"  Base:      prauc={rep_base['auc_pr']:.5f}")
        print(f"  DET:       prauc={rep_det['auc_pr']:.5f}")
        print(f"  MC:        prauc={rep_mc['auc_pr']:.5f}  u_mean={row['mc_u_mean']:.4f}")
        print(f"  LR(mc):    prauc={rep_lr['auc_pr']:.5f}")
        print(f"  LR(det):   prauc={rep_lr_det['auc_pr']:.5f}")
        print(f"  LR(base):  prauc={rep_lr_base['auc_pr']:.5f}")

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
        "best_baseline_file": str(baseline_best_file),
        "best_tag": best_tag,
        "config": cfg,
        "baseline_config": base_cfg,
        "mc_samples_eval": mc_samples,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    json_file = out_dir / f"{dataset}_mc_dropout_eval_split{split_seed}_trainseeds{train_seeds[0]}-{train_seeds[-1]}.json"
    json_file.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("MC DROPOUT TEST RESULTS (mean ± std over eval seeds)")
    print("=" * 70)

    sections = {
        "Baseline (no dropout, reference)": ["base_auc_pr", "base_lift10"],
        "DET (MCD model, dropout OFF)": ["det_auc_pr", "det_lift10"],
        "MC mean (MCD model, averaged)": ["mc_auc_pr", "mc_lift10"],
        "LR: p_mc + u_mc (coupled)": ["lr_auc_pr", "lr_lift10"],
        "LR: p_det + u_mc (same-model decoupled)": ["det_lr_auc_pr", "det_lr_lift10"],
        "LR: p_base + u_mc (fully decoupled)": ["base_lr_auc_pr", "base_lr_lift10"],
    }

    for section_name, keys in sections.items():
        print(f"\n  {section_name}:")
        for k in keys:
            print(f"    {k:25s} = {mean[k]:.5f} ± {std[k]:.5f}")

    print(f"\n  LR coefficients (coupled: p_mc + u_mc):")
    print(f"    {'lr_coef_u':25s} = {mean['lr_coef_u']:.4f} ± {std['lr_coef_u']:.4f}")
    print(f"    {'lr_coef_p':25s} = {mean['lr_coef_p']:.4f} ± {std['lr_coef_p']:.4f}")

    print(f"\n  LR coefficients (same-model: p_det + u_mc):")
    print(f"    {'det_lr_coef_u':25s} = {mean['det_lr_coef_u']:.4f} ± {std['det_lr_coef_u']:.4f}")
    print(f"    {'det_lr_coef_p':25s} = {mean['det_lr_coef_p']:.4f} ± {std['det_lr_coef_p']:.4f}")

    print(f"\n  LR coefficients (fully decoupled: p_base + u_mc):")
    print(f"    {'base_lr_coef_u':25s} = {mean['base_lr_coef_u']:.4f} ± {std['base_lr_coef_u']:.4f}")
    print(f"    {'base_lr_coef_p':25s} = {mean['base_lr_coef_p']:.4f} ± {std['base_lr_coef_p']:.4f}")

    print(f"\n✅ Saved per-seed CSV to:", csv_file)
    print(f"✅ Saved summary JSON to:", json_file)


if __name__ == "__main__":
    main()
