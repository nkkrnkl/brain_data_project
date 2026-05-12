from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
PANEL = ["clustering_w", "modularity_q", "char_path_length",
         "small_worldness", "mean_betweenness"]


def _fit(df: pd.DataFrame, m: str, covs: tuple[str, ...]) -> dict:
    sub = df[[m, *covs]].dropna()
    if len(sub) < 10:
        return {"metric": m, "n": len(sub), "covariates": "+".join(covs)}
    X = sm.add_constant(sub[list(covs)])
    model = sm.OLS(sub[m], X).fit()
    out = {
        "metric": m, "n": len(sub),
        "covariates": "+".join(covs),
        "beta_age": float(model.params["age"]),
        "se_age": float(model.bse["age"]),
        "p_age": float(model.pvalues["age"]),
        "r2": float(model.rsquared),
    }
    if "median_graph_freq" in covs:
        out["beta_gft"] = float(model.params["median_graph_freq"])
        out["se_gft"] = float(model.bse["median_graph_freq"])
        out["p_gft"] = float(model.pvalues["median_graph_freq"])
    return out


def main():
    cache_dir = Path("cache"); results_dir = Path("results")
    metrics = pd.read_csv(cache_dir / "layer4" /
                          f"metrics_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    manifest = pd.read_csv(results_dir /
                           f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    gft = pd.read_csv(results_dir / "gft_per_subject.csv")

    df = metrics.merge(
        manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]],
        on="subject_id", how="inner")
    df = df.merge(gft[["subject_id", "median_graph_freq",
                       "energy_low", "energy_mid", "energy_high"]],
                  on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)
    print(f"merged N = {len(df)}")

    rows = []
    for m in PANEL:
        rows.append({**_fit(df, m, ("age", "sex_num", "mean_FD")), "model": "base"})
        rows.append({**_fit(df, m, ("age", "sex_num", "mean_FD", "median_graph_freq")),
                     "model": "+gft"})
    out = pd.DataFrame(rows)

    for label in ("base", "+gft"):
        sub = out[out["model"] == label]
        ps = sub["p_age"].to_numpy()
        mask = np.isfinite(ps)
        if mask.any():
            _, q, _, _ = multipletests(ps[mask], method="fdr_bh")
            qs = np.full_like(ps, np.nan, dtype=float)
            qs[mask] = q
            out.loc[sub.index, "FDR_q"] = qs

    path = results_dir / f"gft_crosstest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv"
    out.to_csv(path, index=False)
    print("\n=== GFT-as-covariate cross-test ===")
    show = ["metric", "model", "n", "beta_age", "se_age", "p_age", "FDR_q", "r2"]
    print(out[show].to_string(index=False))
    print("\n=== beta_gft / p_gft when present ===")
    print(out[out["model"] == "+gft"][["metric", "beta_gft", "p_gft"]].to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
