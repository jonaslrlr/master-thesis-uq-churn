"""
Reproduce all thesis table values from saved eval JSONs and NPZ files.
Outputs: discrimination, reranking, lift, LR coefficients, residual Spearman, CIs.

Usage:
    python -m thesis_uq.eval.compute_thesis_tables
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
from numpy.polynomial.polynomial import polyfit


def guess_repo_root():
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "reports").exists():
            return p
    return Path.cwd()


def load_jsons(ds, eval_dir):
    b = json.load(open(eval_dir / f"{ds}_baseline_eval_split42_trainseeds5-15.json"))
    mc = json.load(open(eval_dir / f"{ds}_mc_dropout_eval_split42_trainseeds5-15.json"))
    la = json.load(open(eval_dir / f"{ds}_laplace_eval_split42_trainseeds5-15.json"))
    edl = json.load(open(eval_dir / f"{ds}_edl_eval_split42_trainseeds5-15.json"))
    cp = json.load(open(eval_dir / f"{ds}_conformal_eval_split42_seeds5-15.json"))
    return b, mc, la, edl, cp


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def fmt2(m, s):
    return f"{m:.2f}±{s:.2f}"


def ci95(m, s, n=11):
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * s / np.sqrt(n)
    return f"{m:.3f} [{m - margin:.3f}, {m + margin:.3f}]"


def residual_spearman(u, p, y):
    error = np.abs(y - p)
    conf = np.abs(p - 0.5)
    cu = polyfit(conf, u, 1)
    ce = polyfit(conf, error, 1)
    u_r = u - (cu[0] + cu[1] * conf)
    e_r = error - (ce[0] + ce[1] * conf)
    return spearmanr(u_r, e_r).correlation


def main():
    repo_root = guess_repo_root()
    eval_dir = repo_root / "reports" / "eval"
    uq_dir = repo_root / "reports" / "uq_scores"

    datasets = ["bank", "cell2cell", "telco", "delft", "cdr", "chile"]
    seeds = list(range(5, 16))

    for ds in datasets:
        b, mc, la, edl, cp = load_jsons(ds, eval_dir)
        M, S = "mean", "std"

        print(f"\n{'='*70}")
        print(f"  {ds.upper()}")
        print(f"{'='*70}")

        # --- Discrimination PR-AUC ---
        print("\n  --- Discrimination PR-AUC ---")
        rows = [
            ("Baseline", b, "auc_pr"),
            ("MCD (det)", mc, "det_auc_pr"),
            ("MCD (mc)", mc, "mc_auc_pr"),
            ("Laplace (MAP)", la, "map_auc_pr"),
            ("Laplace (probit)", la, "probit_auc_pr"),
            ("EDL", edl, "edl_auc_pr"),
            ("CV+ ensemble", cp, "ens_auc_pr"),
        ]
        for name, d, k in rows:
            print(f"  {name:22s} {fmt(d[M][k], d[S][k])}")

        # --- Discrimination Lift ---
        print("\n  --- Discrimination Lift@10 ---")
        rows_lift = [
            ("Baseline", b, "lift10"),
            ("MCD (det)", mc, "det_lift10"),
            ("MCD (mc)", mc, "mc_lift10"),
            ("Laplace (MAP)", la, "map_lift10"),
            ("Laplace (probit)", la, "probit_lift10"),
            ("EDL", edl, "edl_lift10"),
            ("CV+ ensemble", cp, "ens_lift10"),
        ]
        for name, d, k in rows_lift:
            print(f"  {name:22s} {fmt(d[M][k], d[S][k])}")

        # --- Reranking PR-AUC ---
        print("\n  --- Reranking PR-AUC ---")
        rows_re = [
            ("MCD LR (coupled)", mc, "lr_auc_pr"),
            ("MCD LR (det)", mc, "det_lr_auc_pr"),
            ("MCD LR (base)", mc, "base_lr_auc_pr"),
            ("Laplace LR", la, "lr_auc_pr"),
            ("EDL LR (coupled)", edl, "lr_auc_pr"),
            ("EDL LR (base)", edl, "base_lr_auc_pr"),
            ("CP single LR", cp, "cp_single_auc_pr"),
            ("CV+ + CP LR", cp, "ens_cp_auc_pr"),
            ("CV+ + std LR", cp, "ens_std_auc_pr"),
            ("Decoupled CP", cp, "base_cp_cv_auc_pr"),
            ("Decoupled std", cp, "base_std_auc_pr"),
        ]
        for name, d, k in rows_re:
            print(f"  {name:22s} {fmt(d[M][k], d[S][k])}")

        # --- Reranking Lift ---
        print("\n  --- Reranking Lift@10 ---")
        rows_re_lift = [
            ("MCD LR (coupled)", mc, "lr_lift10"),
            ("MCD LR (det)", mc, "det_lr_lift10"),
            ("MCD LR (base)", mc, "base_lr_lift10"),
            ("Laplace LR", la, "lr_lift10"),
            ("EDL LR (coupled)", edl, "lr_lift10"),
            ("EDL LR (base)", edl, "base_lr_lift10"),
            ("CP single LR", cp, "cp_single_lift10"),
            ("CV+ + CP LR", cp, "ens_cp_lift10"),
            ("CV+ + std LR", cp, "ens_std_lift10"),
            ("Decoupled CP", cp, "base_cp_cv_lift10"),
            ("Decoupled std", cp, "base_std_lift10"),
        ]
        for name, d, k in rows_re_lift:
            print(f"  {name:22s} {fmt(d[M][k], d[S][k])}")

        # --- LR Coefficients ---
        print("\n  --- LR Coefficients ---")
        coef_rows = [
            ("MCD", mc, "lr_coef_u", "lr_coef_p"),
            ("Laplace", la, "lr_coef_u", "lr_coef_p"),
            ("EDL (coupled)", edl, "lr_coef_u", "lr_coef_p"),
            ("EDL (base)", edl, "base_lr_coef_u", "base_lr_coef_p"),
            ("CP single", cp, "cp_single_coef_u", "cp_single_coef_p"),
            ("CP cv conf", cp, "ens_cp_coef_u", "ens_cp_coef_p"),
            ("CP cv std", cp, "ens_std_coef_u", "ens_std_coef_p"),
        ]
        for name, d, ku, kp in coef_rows:
            print(f"  {name:18s} bu={d[M][ku]:+.2f}±{d[S][ku]:.2f}  bp={d[M][kp]:+.2f}±{d[S][kp]:.2f}")

        # --- Confidence Intervals ---
        print("\n  --- 95% CIs (PR-AUC) ---")
        all_ci_rows = rows + rows_re
        for name, d, k in all_ci_rows:
            print(f"  {name:22s} {ci95(d[M][k], d[S][k])}")

        # --- Residual Spearman ---
        print("\n  --- Residual Spearman ---")
        for method in ["mc_dropout", "laplace", "edl", "cp_single", "cp_cv", "cp_cv_std"]:
            vals = []
            for seed in seeds:
                f = uq_dir / f"{ds}_{method}_eval_split42_trainseed{seed}.npz"
                if not f.exists():
                    continue
                try:
                    d = np.load(f)
                    rs = residual_spearman(d["u_test"], d["p_test"], d["y_test"])
                    vals.append(rs)
                except Exception:
                    continue
            if vals:
                print(f"  {method:14s} {np.mean(vals):+.4f}±{np.std(vals):.4f}")


if __name__ == "__main__":
    main()
