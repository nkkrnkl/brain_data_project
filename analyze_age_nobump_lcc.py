from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


METRICS = ["clustering_w", "modularity_q", "char_path_length",
           "small_worldness", "mean_betweenness"]


def _load_metrics_with_manifest(fd: float, kappa: float, cache_dir: Path, results_dir: Path) -> pd.DataFrame:
    metrics_path = cache_dir / "layer4_nobump" / f"metrics_nobump_FD{fd}_kappa{kappa}.csv"
    manifest_path = results_dir / f"manifest_nobump_FD{fd}_kappa{kappa}.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(manifest_path)

    keep_cols = ["subject_id", "age", "age_group", "sex", "mean_FD",
                 "connected_at_requested_kappa", "largest_cc_n_nodes", "largest_cc_frac_nodes"]
    df = metrics.merge(manifest[keep_cols], on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)
    return df


def group_comparison(df: pd.DataFrame, metrics: list[str] = METRICS) -> pd.DataFrame:
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


def age_regression(df: pd.DataFrame, metrics: list[str] = METRICS) -> pd.DataFrame:
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
        rows.append({
            "metric": m, "n": len(sub),
            "beta_age": beta_age, "se_age": se_age,
            "t_age": beta_age / se_age if se_age > 0 else np.nan,
            "p_age": p_age,
            "r2": float(model.rsquared),
        })
    out = pd.DataFrame(rows)
    if "p_age" in out.columns and out["p_age"].notna().any():
        mask = out["p_age"].notna()
        _, q, _, _ = multipletests(out.loc[mask, "p_age"].to_numpy(), method="fdr_bh")
        out.loc[mask, "FDR_q"] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = _load_metrics_with_manifest(args.fd, args.kappa, args.cache_dir, args.results_dir)

    gc = group_comparison(df)
    reg = age_regression(df)

    gc_path = args.results_dir / f"group_comparison_nobump_FD{args.fd}_kappa{args.kappa}.csv"
    reg_path = args.results_dir / f"age_regression_nobump_FD{args.fd}_kappa{args.kappa}.csv"
    gc.to_csv(gc_path, index=False)
    reg.to_csv(reg_path, index=False)

    print(f"wrote: {gc_path}")
    print(f"wrote: {reg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

