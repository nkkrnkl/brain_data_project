from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import build_graphs as bg
import analyze_absolute_threshold as az

PRIMARY_FD = 0.5
KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
PANEL = ["clustering_w", "modularity_q", "char_path_length",
         "small_worldness", "mean_betweenness"]
METRIC_LABELS = {
    "clustering_w":     "Clustering (Onnela)",
    "modularity_q":     "Modularity Q",
    "char_path_length": "Characteristic path L",
    "small_worldness":  "Small-worldness σ",
    "mean_betweenness": "Mean betweenness",
}


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
    if "mean_z_upper" in covs:
        out["beta_meanz"] = float(model.params["mean_z_upper"])
        out["p_meanz"] = float(model.pvalues["mean_z_upper"])
    return out


def run_one_kappa(fd: float, kappa: float, cache_dir: Path,
                  results_dir: Path) -> pd.DataFrame:
    metrics_path = cache_dir / "layer4" / f"metrics_FD{fd}_kappa{kappa}.csv"
    manifest_path = results_dir / f"manifest_FD{fd}_kappa{kappa}.csv"
    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(manifest_path)

    df = metrics.merge(
        manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]],
        on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)

    cfg = bg.Config(fd_threshold=fd, kappa=kappa, cache_dir=cache_dir)
    conf_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()
    df["mean_z_upper"] = [
        az._mean_z_upper(np.load(az._layer2_path(cache_dir, sid, fd, conf_h, edge_h)))
        if az._layer2_path(cache_dir, sid, fd, conf_h, edge_h).exists() else np.nan
        for sid in df["subject_id"]
    ]

    rows = []
    for m in PANEL:
        rows.append({**_fit(df, m, ("age", "sex_num", "mean_FD")),
                     "model": "base", "kappa": kappa})
        rows.append({**_fit(df, m, ("age", "sex_num", "mean_FD", "mean_z_upper")),
                     "model": "+meanz", "kappa": kappa})
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for i, m in enumerate(PANEL):
        ax = axes_flat[i]
        sub = df[df["metric"] == m]
        for label, color, marker in [("base", "#4c78a8", "o"),
                                     ("+meanz", "#e45756", "s")]:
            d = sub[sub["model"] == label].sort_values("kappa")
            ax.errorbar(d["kappa"], d["beta_age"], yerr=d["se_age"],
                        fmt=f"{marker}-", color=color, capsize=4, lw=1.5,
                        label=label, markersize=7)
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_title(METRIC_LABELS[m], fontsize=10)
        ax.set_xlabel("κ (edge density)")
        ax.set_ylabel("β_age")
        ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle("κ-sensitivity of age effect, with vs without mean-Fisher-z covariate "
                 f"(FD={PRIMARY_FD})", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    cache_dir = Path("cache"); results_dir = Path("results")
    all_rows = []
    for k in KAPPA_GRID:
        try:
            df = run_one_kappa(PRIMARY_FD, k, cache_dir, results_dir)
        except FileNotFoundError as e:
            print(f"skip κ={k}: {e}")
            continue
        all_rows.append(df)
        print(f"κ={k}: rows={len(df)}")
    if not all_rows:
        raise SystemExit("no κ cells available")
    out = pd.concat(all_rows, ignore_index=True)
    path = results_dir / "meanz_kappa_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("\n=== clustering_w β_age across κ ===")
    cw = out[out["metric"] == "clustering_w"][["kappa", "model", "beta_age", "p_age", "r2"]]
    print(cw.sort_values(["kappa", "model"]).to_string(index=False))
    make_figure(out, results_dir / "figures" / "fig9_meanz_kappa_sensitivity.png")
    print("\nfigure: results/figures/fig9_meanz_kappa_sensitivity.png")


if __name__ == "__main__":
    main()
