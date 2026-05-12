from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import build_graphs as bg

PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
Z_THRESHOLDS = [0.3, 0.5]


def _layer2_path(cache_dir: Path, sid: str, fd: float, conf_h: str, edge_h: str) -> Path:
    return cache_dir / "layer2" / f"{sid}_FD{fd}_conf{conf_h}_edge{edge_h}.npy"


def _abs_threshold_graph(z: np.ndarray, z_thresh: float) -> nx.Graph:
    """Edges where z > z_thresh; weighted graph, weights = z values."""
    n = z.shape[0]
    G = nx.empty_graph(n)
    iu = np.triu_indices(n, k=1)
    w = z[iu]
    keep = w > z_thresh
    rows = iu[0][keep]
    cols = iu[1][keep]
    weights = w[keep]
    for r, c, ww in zip(rows, cols, weights):
        G.add_edge(int(r), int(c), weight=float(ww))
    return G


def _clustering_w(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    return float(np.mean(list(nx.clustering(G, weight="weight").values())))


def _mean_z_upper(z: np.ndarray) -> float:
    iu = np.triu_indices(z.shape[0], k=1)
    return float(np.mean(z[iu]))


def _largest_cc_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.connected_components(G)), default=0)


def compute_per_subject(manifest: pd.DataFrame, cache_dir: Path,
                        z_thresh: float, fd: float) -> pd.DataFrame:
    cfg = bg.Config(fd_threshold=fd, kappa=PRIMARY_KAPPA, cache_dir=cache_dir)
    conf_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()

    rows = []
    for _, r in manifest.iterrows():
        sid = r["subject_id"]
        p = _layer2_path(cache_dir, sid, fd, conf_h, edge_h)
        if not p.exists():
            continue
        Z = np.load(p)
        G = _abs_threshold_graph(Z, z_thresh)
        rows.append({
            "subject_id": sid,
            "z_thresh": z_thresh,
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "lcc_size": _largest_cc_size(G),
            "is_connected": (G.number_of_edges() > 0
                             and nx.is_connected(G)),
            "mean_z_upper": _mean_z_upper(Z),
            "clustering_w": _clustering_w(G),
        })
    return pd.DataFrame(rows)


def age_regression(df: pd.DataFrame, metric: str = "clustering_w",
                   covariates: tuple[str, ...] = ("age", "sex_num", "mean_FD"),
                   ) -> dict:
    sub = df[[metric, *covariates]].dropna()
    if len(sub) < 10:
        return {"metric": metric, "n": len(sub), "covariates": "+".join(covariates)}
    X = sm.add_constant(sub[list(covariates)])
    model = sm.OLS(sub[metric], X).fit()
    out = {
        "metric": metric, "n": len(sub),
        "covariates": "+".join(covariates),
        "beta_age": float(model.params["age"]),
        "se_age": float(model.bse["age"]),
        "t_age": float(model.tvalues["age"]),
        "p_age": float(model.pvalues["age"]),
        "r2": float(model.rsquared),
    }
    if "mean_z_upper" in covariates:
        out["beta_meanz"] = float(model.params["mean_z_upper"])
        out["se_meanz"] = float(model.bse["mean_z_upper"])
        out["p_meanz"] = float(model.pvalues["mean_z_upper"])
    return out


def group_comparison(df: pd.DataFrame, metric: str = "clustering_w") -> dict:
    child = df.loc[df["age_group"] == "child", metric].dropna().to_numpy()
    adult = df.loc[df["age_group"] == "adult", metric].dropna().to_numpy()
    if len(child) < 3 or len(adult) < 3:
        return {"metric": metric, "n_child": len(child), "n_adult": len(adult)}
    U, p = stats.mannwhitneyu(child, adult, alternative="two-sided")
    n1, n2 = len(child), len(adult)
    return {
        "metric": metric,
        "n_child": n1, "n_adult": n2,
        "child_mean": float(np.mean(child)), "child_sd": float(np.std(child, ddof=1)),
        "adult_mean": float(np.mean(adult)), "adult_sd": float(np.std(adult, ddof=1)),
        "U": float(U), "p": float(p),
        "effect_size_r": float(1.0 - 2.0 * U / (n1 * n2)),
    }


def _phenotype_join(per_subject: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    pheno = manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]]
    df = per_subject.merge(pheno, on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)
    return df


def run_absolute_threshold(z_thresh: float, manifest: pd.DataFrame,
                           cache_dir: Path, results_dir: Path,
                           fd: float = PRIMARY_FD) -> tuple[pd.DataFrame, dict, dict]:
    per = compute_per_subject(manifest, cache_dir, z_thresh, fd)
    df = _phenotype_join(per, manifest)

    reg_base = age_regression(df, "clustering_w",
                              covariates=("age", "sex_num", "mean_FD"))
    reg_meanz = age_regression(df, "clustering_w",
                               covariates=("age", "sex_num", "mean_FD", "mean_z_upper"))
    gc = group_comparison(df, "clustering_w")

    per_path = results_dir / f"absthresh_per_subject_z{z_thresh}.csv"
    df.to_csv(per_path, index=False)

    reg_rows = pd.DataFrame([reg_base, reg_meanz])
    reg_path = results_dir / f"absthresh_age_regression_z{z_thresh}.csv"
    reg_rows.to_csv(reg_path, index=False)

    gc_path = results_dir / f"absthresh_group_comparison_z{z_thresh}.csv"
    pd.DataFrame([gc]).to_csv(gc_path, index=False)

    return df, reg_base, reg_meanz, gc


def run_meanz_covariate_primary(cache_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Re-run the primary κ=0.10 metric panel adding mean Fisher-z as a covariate.

    Loads layer 2 z-matrices (for mean_z_upper) and the layer 4 metric panel
    (for the actual metric values); refits OLS metric ~ age + sex + mean_FD
    with and without mean_z_upper across all five metrics.
    """
    metrics_path = cache_dir / "layer4" / f"metrics_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv"
    manifest_path = results_dir / f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv"
    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(manifest_path)
    df = metrics.merge(
        manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]],
        on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)

    cfg = bg.Config(fd_threshold=PRIMARY_FD, kappa=PRIMARY_KAPPA, cache_dir=cache_dir)
    conf_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()

    mean_z = []
    for sid in df["subject_id"]:
        p = _layer2_path(cache_dir, sid, PRIMARY_FD, conf_h, edge_h)
        mean_z.append(_mean_z_upper(np.load(p)) if p.exists() else np.nan)
    df["mean_z_upper"] = mean_z

    panel = ["clustering_w", "modularity_q", "char_path_length",
             "small_worldness", "mean_betweenness"]

    rows = []
    for m in panel:
        rows.append({**age_regression(df, m, ("age", "sex_num", "mean_FD")),
                     "model": "base"})
        rows.append({**age_regression(df, m, ("age", "sex_num", "mean_FD", "mean_z_upper")),
                     "model": "+meanz"})
    out = pd.DataFrame(rows)

    for label in ("base", "+meanz"):
        sub = out[out["model"] == label]
        ps = sub["p_age"].to_numpy()
        mask = np.isfinite(ps)
        if mask.any():
            _, q, _, _ = multipletests(ps[mask], method="fdr_bh")
            qs = np.full_like(ps, np.nan, dtype=float)
            qs[mask] = q
            out.loc[sub.index, "FDR_q"] = qs

    path = results_dir / f"absthresh_meanz_covariate_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv"
    out.to_csv(path, index=False)
    return out


def make_figure(results_dir: Path, out_path: Path):
    """Side-by-side scatter of clustering vs age at z>0.3 and z>0.5."""
    fig, axes = plt.subplots(1, len(Z_THRESHOLDS), figsize=(5.5 * len(Z_THRESHOLDS), 5),
                             sharey=False)
    if len(Z_THRESHOLDS) == 1:
        axes = [axes]

    for ax, zt in zip(axes, Z_THRESHOLDS):
        per_path = results_dir / f"absthresh_per_subject_z{zt}.csv"
        reg_path = results_dir / f"absthresh_age_regression_z{zt}.csv"
        if not per_path.exists():
            ax.set_title(f"z > {zt} (no data)")
            continue
        df = pd.read_csv(per_path)
        reg = pd.read_csv(reg_path)
        for grp, sub in df.groupby("age_group"):
            ax.scatter(sub["age"], sub["clustering_w"], alpha=0.7, label=grp,
                       s=30, edgecolor="none")
        x = df["age"].to_numpy(); y = df["clustering_w"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xr, slope * xr + intercept, "k--", lw=1.5, alpha=0.7)
        base = reg[reg["covariates"] == "age+sex_num+mean_FD"].iloc[0]
        meanz = reg[reg["covariates"] == "age+sex_num+mean_FD+mean_z_upper"].iloc[0]
        ax.set_title(
            f"Clustering, edges where z > {zt}  (n={int(base['n'])})\n"
            f"base:    β={base['beta_age']:.4f}  p={base['p_age']:.3g}\n"
            f"+meanz:  β={meanz['beta_age']:.4f}  p={meanz['p_age']:.3g}",
            fontsize=10)
        ax.set_xlabel("age (yr)"); ax.set_ylabel("clustering (Onnela)")
        ax.legend(fontsize=8)

    fig.suptitle("Absolute-threshold replication  ·  edge weight = Fisher z, "
                 "no top-κ rule, no κ-bump", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--fd", type=float, default=PRIMARY_FD)
    args = ap.parse_args()

    manifest_path = args.results_dir / f"manifest_FD{args.fd}_kappa{PRIMARY_KAPPA}.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest_full = pd.read_csv(manifest_path)
    manifest = manifest_full[manifest_full["included"]].copy()
    print(f"loaded manifest: {len(manifest)} included subjects at FD={args.fd}")

    for zt in Z_THRESHOLDS:
        df, reg_base, reg_meanz, gc = run_absolute_threshold(
            zt, manifest, args.cache_dir, args.results_dir, fd=args.fd)
        n_disc = int((~df["is_connected"]).sum())
        print(f"\n=== z > {zt} ===  N={len(df)}, disconnected={n_disc}")
        print(f"  density (mean): {df['density'].mean():.4f}  "
              f"LCC size (mean): {df['lcc_size'].mean():.1f}/{100}")
        print(f"  base   : β_age={reg_base['beta_age']:.5f}  p={reg_base['p_age']:.3g}  "
              f"R²={reg_base['r2']:.3f}")
        print(f"  +meanz : β_age={reg_meanz['beta_age']:.5f}  p={reg_meanz['p_age']:.3g}  "
              f"R²={reg_meanz['r2']:.3f}  "
              f"β_meanz={reg_meanz['beta_meanz']:.4f} p={reg_meanz['p_meanz']:.3g}")
        print(f"  group  : child={gc.get('child_mean', float('nan')):.4f}  "
              f"adult={gc.get('adult_mean', float('nan')):.4f}  "
              f"U={gc.get('U', float('nan')):.0f}  p={gc.get('p', float('nan')):.3g}")

    print("\n=== mean-z covariate sweep over primary panel ===")
    out = run_meanz_covariate_primary(args.cache_dir, args.results_dir)
    print(out[["metric", "model", "n", "beta_age", "p_age", "FDR_q"]].to_string(index=False))

    make_figure(args.results_dir, args.results_dir / "figures" / "fig7_absthresh.png")
    print("\nfigure: results/figures/fig7_absthresh.png")


if __name__ == "__main__":
    main()
