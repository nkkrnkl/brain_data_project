from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import build_graphs as bg

METRICS = ["clustering_w", "modularity_q", "char_path_length",
           "small_worldness", "mean_betweenness"]

KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
FD_GRID = [0.3, 0.5, 0.9]
PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
N_PERM = 1000


def _load_metrics_with_manifest(fd: float, kappa: float,
                                cache_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Inner-join Layer-4 metrics with the build manifest's phenotypic columns."""
    metrics_path = cache_dir / "layer4" / f"metrics_FD{fd}_kappa{kappa}.csv"
    manifest_path = results_dir / f"manifest_FD{fd}_kappa{kappa}.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(manifest_path)
    keep_cols = ["subject_id", "age", "age_group", "sex", "mean_FD"]
    df = metrics.merge(manifest[keep_cols], on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)
    return df


def group_comparison(df: pd.DataFrame, metrics: list[str] = METRICS) -> pd.DataFrame:
    """Mann-Whitney U child-vs-adult on each metric; FDR across the panel."""
    rows = []
    for m in metrics:
        child = df.loc[df["age_group"] == "child", m].dropna().to_numpy()
        adult = df.loc[df["age_group"] == "adult", m].dropna().to_numpy()
        if len(child) < 3 or len(adult) < 3:
            rows.append({"metric": m, "n_child": len(child), "n_adult": len(adult)})
            continue
        U, p = stats.mannwhitneyu(child, adult, alternative="two-sided")
        n1, n2 = len(child), len(adult)
        r_eff = 1.0 - 2.0 * U / (n1 * n2)
        rows.append({
            "metric": m,
            "n_child": n1, "n_adult": n2,
            "child_mean": float(np.mean(child)), "child_sd": float(np.std(child, ddof=1)),
            "adult_mean": float(np.mean(adult)), "adult_sd": float(np.std(adult, ddof=1)),
            "U": float(U), "p": float(p), "effect_size_r": float(r_eff),
        })
    out = pd.DataFrame(rows)
    if "p" in out.columns and out["p"].notna().any():
        mask = out["p"].notna()
        _, q, _, _ = multipletests(out.loc[mask, "p"].to_numpy(), method="fdr_bh")
        out.loc[mask, "FDR_q"] = q
    return out


def age_regression(df: pd.DataFrame, metrics: list[str] = METRICS,
                   run_permutation: bool = False, seed: int = 20260507) -> pd.DataFrame:
    """OLS metric ~ age + sex + mean_FD; FDR across the metric panel."""
    rows = []
    for m in metrics:
        sub = df[[m, "age", "sex_num", "mean_FD"]].dropna()
        if len(sub) < 10:
            rows.append({"metric": m, "n": len(sub)})
            continue
        X = sm.add_constant(sub[["age", "sex_num", "mean_FD"]])
        model = sm.OLS(sub[m], X).fit()
        beta_age = float(model.params["age"])
        se_age = float(model.bse["age"])
        p_age = float(model.pvalues["age"])
        row = {
            "metric": m, "n": len(sub),
            "beta_age": beta_age, "se_age": se_age,
            "t_age": beta_age / se_age if se_age > 0 else np.nan,
            "p_age": p_age,
            "r2": float(model.rsquared),
        }
        if run_permutation and p_age < 0.05:
            rng = np.random.default_rng(seed)
            null_betas = np.empty(N_PERM)
            y = sub[m].to_numpy()
            X_arr = X.to_numpy()
            age_idx = list(X.columns).index("age")
            for k in range(N_PERM):
                Xp = X_arr.copy()
                Xp[:, age_idx] = rng.permutation(Xp[:, age_idx])
                bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
                null_betas[k] = bp[age_idx]
            p_perm = float(np.mean(np.abs(null_betas) >= np.abs(beta_age)))
            row["p_perm"] = p_perm
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p_age" in out.columns and out["p_age"].notna().any():
        mask = out["p_age"].notna()
        _, q, _, _ = multipletests(out.loc[mask, "p_age"].to_numpy(), method="fdr_bh")
        out.loc[mask, "FDR_q"] = q
    return out


def fd_age_residual_flag(df: pd.DataFrame) -> bool:
    """Return True if mean_FD correlates with age at |r|>0.2 — used to gate
    the permutation test."""
    if len(df) < 5:
        return False
    r, _ = stats.pearsonr(df["age"].to_numpy(), df["mean_FD"].to_numpy())
    return abs(r) > 0.2


def run_one(fd: float, kappa: float, cache_dir: Path, results_dir: Path,
            run_permutation_if_flagged: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_metrics_with_manifest(fd, kappa, cache_dir, results_dir)
    flagged = fd_age_residual_flag(df)
    gc = group_comparison(df)
    reg = age_regression(df, run_permutation=(run_permutation_if_flagged and flagged))

    gc_path = results_dir / f"group_comparison_FD{fd}_kappa{kappa}.csv"
    reg_path = results_dir / f"age_regression_FD{fd}_kappa{kappa}.csv"
    gc.to_csv(gc_path, index=False)
    reg.to_csv(reg_path, index=False)
    return gc, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["primary", "kappa", "fd", "all"])
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    combos = []
    if args.mode in ("primary", "all"):
        combos.append((PRIMARY_FD, PRIMARY_KAPPA))
    if args.mode in ("kappa", "all"):
        combos.extend((PRIMARY_FD, k) for k in KAPPA_GRID if k != PRIMARY_KAPPA)
    if args.mode in ("fd", "all"):
        combos.extend((fd, PRIMARY_KAPPA) for fd in FD_GRID if fd != PRIMARY_FD)

    for fd, kappa in combos:
        try:
            gc, reg = run_one(fd, kappa, args.cache_dir, args.results_dir)
            print(f"\n=== FD={fd}, κ={kappa} ===")
            print("group comparison:"); print(gc.to_string(index=False))
            print("\nage regression:"); print(reg.to_string(index=False))
        except FileNotFoundError as e:
            print(f"skip FD={fd} κ={kappa}: missing input ({e})")


if __name__ == "__main__":
    main()
