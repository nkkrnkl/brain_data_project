"""Phase-4 deliverable: writeup-ready figures.

Outputs (under results/figures/):
    fig1_cohort_flow.png             — exclusion funnel
    fig2_group_avg_zmatrix.png       — child / adult / (adult-child) Z-matrices
    fig3_age_regression_panels.png   — per-metric scatter at primary settings
    fig6_kappa_sensitivity.png       — β_age vs κ, panel per metric
    fig_connectedness.png            — κ_final histogram + bumping by age group
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import build_graphs as bg

KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
FD_GRID = [0.3, 0.5, 0.9]
PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
METRICS = ["clustering_w", "modularity_q", "char_path_length",
           "small_worldness", "mean_betweenness"]
METRIC_LABELS = {
    "clustering_w":     "Clustering coefficient (Onnela)",
    "modularity_q":     "Modularity Q",
    "char_path_length": "Characteristic path length L",
    "small_worldness":  "Small-worldness σ",
    "mean_betweenness": "Mean betweenness centrality",
}


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig 1 — cohort flow (exclusion funnel)
# --------------------------------------------------------------------------

def fig1_cohort_flow(results_dir: Path, out_path: Path):
    excl = pd.read_csv(results_dir / "exclusion_table.csv")
    primary = excl[excl["fd"] == PRIMARY_FD]

    n_pheno = 155
    n_bucket = int(primary["n_in_cohort"].sum())
    n_inc = int(primary["n_included"].sum())
    n_min_ret = int(primary["n_min_retained_excluded"].sum())
    n_kmax = int(primary["n_kappa_max_excluded"].sum())

    stages = [
        (f"participants.tsv\nn = {n_pheno}", "#dde7f0"),
        (f"BOLD in bucket\nn = {n_bucket}\n(−{n_pheno - n_bucket} no BOLD)", "#cfe2cb"),
        (f"≥4 min retained\nn = {n_bucket - n_min_ret}\n(−{n_min_ret} FD-scrubbed)", "#f7d8a8"),
        (f"connected at κ ≤ 0.5\nn = {n_inc}\n(−{n_kmax} κ_max)", "#a8d3a0"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1)
    ax.axis("off")
    box_w, box_h = 0.85, 0.6
    centers = []
    for i, (label, color) in enumerate(stages):
        x = i + (1 - box_w) / 2
        cx = x + box_w / 2
        centers.append(cx)
        rect = mpatches.FancyBboxPatch(
            (x, 0.2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(cx, 0.5, label, ha="center", va="center", fontsize=10)
    for i in range(len(stages) - 1):
        ax.annotate("", xy=(centers[i + 1] - box_w / 2, 0.5),
                    xytext=(centers[i] + box_w / 2, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    by_group = primary.set_index("age_group")["n_included"].to_dict()
    fig.suptitle(f"Cohort flow at primary (FD={PRIMARY_FD}, κ={PRIMARY_KAPPA})  ·  "
                 f"final: child n={by_group.get('child', 0)}, adult n={by_group.get('adult', 0)}",
                 fontsize=11)
    _save(fig, out_path)


# --------------------------------------------------------------------------
# fig 2 — child vs adult mean z-matrices + difference
# --------------------------------------------------------------------------

def fig2_group_avg_zmatrix(cache_dir: Path, results_dir: Path, out_path: Path):
    cfg = bg.Config(fd_threshold=PRIMARY_FD, kappa=PRIMARY_KAPPA, cache_dir=cache_dir)
    edge_h = cfg.edge_measure_hash()
    conf_h = cfg.confound_hash()

    manifest = pd.read_csv(results_dir / f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    inc = manifest[manifest["included"]].copy()

    def avg_z(group_df: pd.DataFrame) -> np.ndarray:
        zs = []
        for sid in group_df["subject_id"]:
            p = cache_dir / "layer2" / f"{sid}_FD{PRIMARY_FD}_conf{conf_h}_edge{edge_h}.npy"
            if p.exists():
                zs.append(np.load(p))
        return np.mean(zs, axis=0) if zs else np.zeros((100, 100))

    Z_child = avg_z(inc[inc["age_group"] == "child"])
    Z_adult = avg_z(inc[inc["age_group"] == "adult"])
    Z_diff = Z_adult - Z_child

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = max(Z_child.max(), Z_adult.max())
    for ax, M, title in zip(axes[:2], [Z_child, Z_adult], ["child mean", "adult mean"]):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"{title} (n={int((inc['age_group']=='child').sum() if title.startswith('child') else (inc['age_group']=='adult').sum())})")
        ax.set_xlabel("ROI"); ax.set_ylabel("ROI")
        plt.colorbar(im, ax=ax, fraction=0.046)
    vmax_d = np.abs(Z_diff).max()
    im = axes[2].imshow(Z_diff, cmap="seismic", vmin=-vmax_d, vmax=vmax_d, aspect="equal")
    axes[2].set_title("(adult − child)")
    axes[2].set_xlabel("ROI"); axes[2].set_ylabel("ROI")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Group-average Fisher-z connectivity at FD={PRIMARY_FD}, κ={PRIMARY_KAPPA}")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------
# fig 3 — per-metric scatter at primary settings
# --------------------------------------------------------------------------

def fig3_age_regression_panels(cache_dir: Path, results_dir: Path, out_path: Path):
    metrics_path = cache_dir / "layer4" / f"metrics_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv"
    manifest = pd.read_csv(results_dir / f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    df = pd.read_csv(metrics_path).merge(
        manifest[["subject_id", "age", "age_group", "sex", "mean_FD"]],
        on="subject_id")
    reg = pd.read_csv(results_dir / f"age_regression_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for i, m in enumerate(METRICS):
        ax = axes_flat[i]
        for grp, sub in df.groupby("age_group"):
            ax.scatter(sub["age"], sub[m], alpha=0.7, label=grp,
                       edgecolor="none", s=30)
        # Fit line over the full age range
        if df[m].notna().sum() >= 5:
            x = df["age"].to_numpy(); y = df[m].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                xr = np.linspace(x[mask].min(), x[mask].max(), 50)
                ax.plot(xr, slope * xr + intercept, "k--", lw=1.5, alpha=0.7)
        rec = reg[reg["metric"] == m]
        if not rec.empty:
            r = rec.iloc[0]
            beta = r.get("beta_age", float("nan"))
            p = r.get("p_age", float("nan"))
            q = r.get("FDR_q", float("nan"))
            ax.set_title(f"{METRIC_LABELS[m]}\nβ={beta:.4f}  p={p:.3g}  q={q:.3g}",
                         fontsize=10)
        ax.set_xlabel("age (yr)")
        ax.set_ylabel(m)
        ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle(f"Age regression at primary (FD={PRIMARY_FD}, κ={PRIMARY_KAPPA}, n={len(df)})  ·  "
                 f"OLS metric ~ age + sex + mean_FD", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------
# fig 6 — κ sensitivity (β_age vs κ, panel per metric)
# --------------------------------------------------------------------------

def fig6_kappa_sensitivity(results_dir: Path, out_path: Path):
    rows = []
    for k in KAPPA_GRID:
        path = results_dir / f"age_regression_FD{PRIMARY_FD}_kappa{k}.csv"
        if not path.exists():
            continue
        reg = pd.read_csv(path)
        for _, r in reg.iterrows():
            rows.append({
                "metric": r["metric"], "kappa": k,
                "beta_age": r.get("beta_age"),
                "se_age": r.get("se_age"),
                "FDR_q": r.get("FDR_q"),
            })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for i, m in enumerate(METRICS):
        ax = axes_flat[i]
        sub = df[df["metric"] == m].sort_values("kappa")
        if sub.empty:
            ax.set_title(f"{METRIC_LABELS[m]} (no data)")
            continue
        ax.errorbar(sub["kappa"], sub["beta_age"], yerr=sub["se_age"],
                    fmt="o-", capsize=4, lw=1.5)
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        # Star points where FDR_q < .05
        sig = sub[sub["FDR_q"] < 0.05]
        if not sig.empty:
            ax.scatter(sig["kappa"], sig["beta_age"], s=120, marker="*",
                       facecolor="gold", edgecolor="black", zorder=5,
                       label="FDR-q<0.05")
            ax.legend(fontsize=8)
        ax.set_title(METRIC_LABELS[m], fontsize=10)
        ax.set_xlabel("κ (edge density)")
        ax.set_ylabel("β_age")
    axes_flat[-1].axis("off")
    fig.suptitle(f"κ sensitivity at FD={PRIMARY_FD}: age-effect across the κ grid",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------
# fig connectedness
# --------------------------------------------------------------------------

def fig_connectedness(results_dir: Path, out_path: Path):
    per_sub = pd.read_csv(results_dir / "connectedness_report.csv")
    by_grp = pd.read_csv(results_dir / "connectedness_summary_by_group.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for grp, sub in per_sub.groupby("age_group"):
        axes[0].hist(sub["kappa_final"], bins=20, alpha=0.55, label=grp,
                     edgecolor="black")
    axes[0].axvline(PRIMARY_KAPPA, color="red", ls="--", lw=1,
                    label=f"primary κ={PRIMARY_KAPPA}")
    axes[0].set_xlabel("final κ (after per-subject bump)")
    axes[0].set_ylabel("# subjects")
    axes[0].set_title("Distribution of κ_final by age group")
    axes[0].legend()

    axes[1].bar(by_grp["age_group"], by_grp["pct_bumped"],
                edgecolor="black")
    for i, (grp, pct) in enumerate(zip(by_grp["age_group"], by_grp["pct_bumped"])):
        axes[1].text(i, pct + 1, f"{pct:.0f}%", ha="center", fontsize=10)
    axes[1].set_ylabel("% of group needing κ-bump")
    axes[1].set_title("κ-bumping rate by age group")
    axes[1].set_ylim(0, 105)

    fig.suptitle(f"Connectedness rule diagnostic (FD={PRIMARY_FD}, κ_req={PRIMARY_KAPPA})",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


def main():
    results_dir = Path("results")
    cache_dir = Path("cache")
    figs_dir = results_dir / "figures"

    fig1_cohort_flow(results_dir, figs_dir / "fig1_cohort_flow.png")
    fig2_group_avg_zmatrix(cache_dir, results_dir, figs_dir / "fig2_group_avg_zmatrix.png")
    fig3_age_regression_panels(cache_dir, results_dir, figs_dir / "fig3_age_regression_panels.png")
    fig6_kappa_sensitivity(results_dir, figs_dir / "fig6_kappa_sensitivity.png")
    fig_connectedness(results_dir, figs_dir / "fig_connectedness.png")
    print("figures written under results/figures/")


if __name__ == "__main__":
    main()
