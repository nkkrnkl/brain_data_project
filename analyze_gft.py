from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import build_graphs as bg

PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
N_ROIS = 100
EPS = 1e-12

BAND_LABELS = ["low (smooth)", "mid", "high (segregated)"]


def _layer1_path(cache_dir: Path, sid: str, fd: float, conf_h: str) -> Path:
    return cache_dir / "layer1" / f"{sid}_FD{fd}_conf{conf_h}.npz"


def _layer2_path(cache_dir: Path, sid: str, fd: float, conf_h: str, edge_h: str) -> Path:
    return cache_dir / "layer2" / f"{sid}_FD{fd}_conf{conf_h}_edge{edge_h}.npy"


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2}.

    Isolated nodes (degree 0) contribute a row/column of zeros so the
    eigendecomposition stays well-defined; they show up as eigenvalue-0
    components and contribute no projected energy because the eigenvector is
    a singleton and U^T x picks up only that node's mean (which is ~0 after
    confound regression).
    """
    deg = W.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > EPS
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dinv = np.diag(d_inv_sqrt)
    L = np.eye(W.shape[0]) - Dinv @ W @ Dinv
    return (L + L.T) / 2.0


def gft_energy(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Per-eigenvalue energy: e_k = Σ_t (U^T x_t)_k² normalized to sum 1.

    X is (T, N); U is (N, N) with columns = eigenvectors.
    Returns vector of length N.
    """
    Xhat = X @ U
    e = np.sum(Xhat ** 2, axis=0)
    s = e.sum()
    return e / s if s > 0 else e


def band_fractions(eigvals: np.ndarray, energy: np.ndarray) -> tuple[float, float, float]:
    """Split eigenvalues into thirds by *index* (rank), sum normalized energy in each.

    Index-thirds rather than value-thirds because the sym-norm Laplacian
    eigenvalue density is heavily front-loaded near 0 — value-thirds would
    put almost every eigenvalue in the low band for some subjects and
    very few for others, making bands incomparable across subjects.
    """
    n = len(eigvals)
    order = np.argsort(eigvals)
    ev_s = eigvals[order]
    en_s = energy[order]
    third = n // 3
    low = en_s[:third].sum()
    mid = en_s[third:2 * third].sum()
    high = en_s[2 * third:].sum()
    return float(low), float(mid), float(high)


def median_graph_freq(eigvals: np.ndarray, energy: np.ndarray) -> float:
    """Energy-weighted median eigenvalue.

    Sort by eigenvalue ascending, take the eigenvalue where cumulative
    normalized energy first crosses 0.5.
    """
    order = np.argsort(eigvals)
    ev_s = eigvals[order]
    en_s = energy[order]
    cdf = np.cumsum(en_s)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    idx = int(np.searchsorted(cdf, 0.5))
    return float(ev_s[min(idx, len(ev_s) - 1)])


def per_subject_row(sid: str, cache_dir: Path, fd: float,
                    conf_h: str, edge_h: str) -> dict | None:
    p1 = _layer1_path(cache_dir, sid, fd, conf_h)
    p2 = _layer2_path(cache_dir, sid, fd, conf_h, edge_h)
    if not (p1.exists() and p2.exists()):
        return None
    with np.load(p1, allow_pickle=False) as f:
        X = np.asarray(f["clean_ts"], dtype=float)
    W = np.asarray(np.load(p2), dtype=float)
    np.fill_diagonal(W, 0.0)
    W = np.where(W > 0, W, 0.0)

    L = normalized_laplacian(W)
    eigvals, U = np.linalg.eigh(L)
    energy = gft_energy(X, U)

    low, mid, high = band_fractions(eigvals, energy)
    med = median_graph_freq(eigvals, energy)

    return {
        "subject_id": sid,
        "n_volumes": int(X.shape[0]),
        "n_isolated_nodes": int((W.sum(axis=1) <= EPS).sum()),
        "lambda_min": float(eigvals.min()),
        "lambda_max": float(eigvals.max()),
        "energy_low": low,
        "energy_mid": mid,
        "energy_high": high,
        "median_graph_freq": med,
    }


def compute_per_subject_table(manifest: pd.DataFrame, cache_dir: Path,
                              fd: float = PRIMARY_FD) -> pd.DataFrame:
    cfg = bg.Config(fd_threshold=fd, kappa=PRIMARY_KAPPA, cache_dir=cache_dir)
    conf_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()

    rows = []
    for i, (_, r) in enumerate(manifest.iterrows(), 1):
        sid = r["subject_id"]
        out = per_subject_row(sid, cache_dir, fd, conf_h, edge_h)
        if out is None:
            print(f"  [{i}/{len(manifest)}] {sid}: missing cache, skipped")
            continue
        rows.append(out)
        if i % 25 == 0:
            print(f"  [{i}/{len(manifest)}] {sid}: λ∈[{out['lambda_min']:.3f},"
                  f"{out['lambda_max']:.3f}]  med={out['median_graph_freq']:.3f}")
    return pd.DataFrame(rows)


def band_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in zip(["energy_low", "energy_mid", "energy_high"], BAND_LABELS):
        child = df.loc[df["age_group"] == "child", col].dropna().to_numpy()
        adult = df.loc[df["age_group"] == "adult", col].dropna().to_numpy()
        if len(child) < 3 or len(adult) < 3:
            rows.append({"band": label, "metric": col,
                         "n_child": len(child), "n_adult": len(adult)})
            continue
        U, p = stats.mannwhitneyu(child, adult, alternative="two-sided")
        n1, n2 = len(child), len(adult)
        rows.append({
            "band": label, "metric": col,
            "n_child": n1, "n_adult": n2,
            "child_mean": float(np.mean(child)), "child_sd": float(np.std(child, ddof=1)),
            "adult_mean": float(np.mean(adult)), "adult_sd": float(np.std(adult, ddof=1)),
            "U": float(U), "p": float(p),
            "effect_size_r": float(1.0 - 2.0 * U / (n1 * n2)),
        })
    out = pd.DataFrame(rows)
    if "p" in out.columns and out["p"].notna().any():
        mask = out["p"].notna()
        _, q, _, _ = multipletests(out.loc[mask, "p"].to_numpy(), method="fdr_bh")
        out.loc[mask, "FDR_q"] = q
    return out


def median_freq_regression(df: pd.DataFrame) -> dict:
    sub = df[["median_graph_freq", "age", "sex_num", "mean_FD"]].dropna()
    if len(sub) < 10:
        return {"n": len(sub)}
    X = sm.add_constant(sub[["age", "sex_num", "mean_FD"]])
    model = sm.OLS(sub["median_graph_freq"], X).fit()
    return {
        "n": len(sub),
        "beta_age": float(model.params["age"]),
        "se_age": float(model.bse["age"]),
        "t_age": float(model.tvalues["age"]),
        "p_age": float(model.pvalues["age"]),
        "beta_sex": float(model.params["sex_num"]),
        "p_sex": float(model.pvalues["sex_num"]),
        "beta_meanFD": float(model.params["mean_FD"]),
        "p_meanFD": float(model.pvalues["mean_FD"]),
        "r2": float(model.rsquared),
    }


def make_figure(per_subject: pd.DataFrame, band_df: pd.DataFrame,
                reg: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    melt = per_subject.melt(
        id_vars=["subject_id", "age_group"],
        value_vars=["energy_low", "energy_mid", "energy_high"],
        var_name="band_col", value_name="energy_frac")
    melt["band"] = melt["band_col"].map(dict(zip(
        ["energy_low", "energy_mid", "energy_high"], BAND_LABELS)))

    bands = BAND_LABELS
    groups = ["child", "adult"]
    width = 0.35
    x = np.arange(len(bands))
    colors = {"child": "#4c78a8", "adult": "#f58518"}
    for i, g in enumerate(groups):
        means = [melt[(melt["band"] == b) & (melt["age_group"] == g)]["energy_frac"].mean()
                 for b in bands]
        sds = [melt[(melt["band"] == b) & (melt["age_group"] == g)]["energy_frac"].std(ddof=1)
               for b in bands]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sds, capsize=3,
               label=g, color=colors[g], edgecolor="black", linewidth=0.5)
    rng = np.random.default_rng(20260507)
    for j, b in enumerate(bands):
        for i, g in enumerate(groups):
            vals = melt[(melt["band"] == b) & (melt["age_group"] == g)]["energy_frac"].to_numpy()
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full_like(vals, x[j] + (i - 0.5) * width) + jitter, vals,
                       s=10, alpha=0.5, color="black")
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("fraction of GFT energy")
    ax.set_title("Spectral energy by band, child vs adult")
    ax.legend()
    annot = []
    for _, r in band_df.iterrows():
        if "p" in r and pd.notna(r.get("p", np.nan)):
            annot.append(f"{r['band']}: p={r['p']:.3g}, q={r['FDR_q']:.3g}")
    if annot:
        ax.text(0.02, 0.98, "\n".join(annot), transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax = axes[1]
    for grp, sub in per_subject.groupby("age_group"):
        ax.scatter(sub["age"], sub["median_graph_freq"], alpha=0.7,
                   label=grp, s=30, edgecolor="none", color=colors.get(grp))
    x = per_subject["age"].to_numpy(); y = per_subject["median_graph_freq"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 5:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        xr = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xr, slope * xr + intercept, "k--", lw=1.5, alpha=0.7)
    ax.set_xlabel("age (yr)")
    ax.set_ylabel("median graph frequency (eigenvalue at energy CDF=0.5)")
    title = "Median graph frequency vs age"
    if reg.get("n"):
        title += (f"  ·  β={reg['beta_age']:.4f}  "
                  f"p={reg['p_age']:.3g}  R²={reg['r2']:.3f}  n={reg['n']}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("GFT of cleaned BOLD on subject-specific symmetric normalized Laplacian "
                 f"(FD={PRIMARY_FD}, full Fisher-z adjacency)", fontsize=11)
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

    print("computing GFT per subject…")
    per = compute_per_subject_table(manifest, args.cache_dir, fd=args.fd)
    if per.empty:
        raise SystemExit("no GFT rows produced — is the layer1/layer2 cache populated?")

    pheno = manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]]
    df = per.merge(pheno, on="subject_id", how="inner")
    df["sex_num"] = (df["sex"].str.upper() == "M").astype(int)

    per_path = args.results_dir / "gft_per_subject.csv"
    df.to_csv(per_path, index=False)
    print(f"wrote {per_path}  ({len(df)} subjects)")

    band_df = band_comparison(df)
    band_path = args.results_dir / "gft_band_comparison.csv"
    band_df.to_csv(band_path, index=False)
    print("\n=== band comparison (child vs adult) ===")
    print(band_df.to_string(index=False))

    reg = median_freq_regression(df)
    reg_path = args.results_dir / "gft_median_freq_regression.csv"
    pd.DataFrame([reg]).to_csv(reg_path, index=False)
    print("\n=== median graph frequency ~ age + sex + mean_FD ===")
    print(pd.DataFrame([reg]).to_string(index=False))

    make_figure(df, band_df, reg, args.results_dir / "figures" / "fig8_gft.png")
    print("\nfigure: results/figures/fig8_gft.png")


if __name__ == "__main__":
    main()
