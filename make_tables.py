"""Phase-5 deliverable: writeup-ready tables.

Outputs (all under results/):
    exclusion_table.csv          — N per (age_group × FD) × exclusion_reason
    connectedness_report.csv     — per-subject and per-group κ-bumping summary
    fd_sensitivity_table.csv     — β_age × FD × metric
    kappa_sensitivity_table.csv  — β_age × κ × metric
    prediction_vs_observed.csv   — Gopnik explore-exploit prediction vs observed
    limitations_numbers.csv      — single-column numbers for the Discussion
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
FD_GRID = [0.3, 0.5, 0.9]
PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10
METRICS = ["clustering_w", "modularity_q", "char_path_length",
           "small_worldness", "mean_betweenness"]


# --------------------------------------------------------------------------
# Predictions hand-coded from Gopnik explore-exploit + Richardson 2018 framing
# --------------------------------------------------------------------------
# Children explore (broad, integrated, less segregated brains) → with age
# the network becomes more segregated and more efficient locally.
PREDICTED_SIGN_WITH_AGE = {
    "clustering_w":      "+",   # local segregation increases with age
    "modularity_q":      "+",   # community structure sharpens with age
    "char_path_length":  "+",   # less integrated → longer paths with age
    "small_worldness":   "+",   # higher σ as the network matures
    "mean_betweenness":  "-",   # fewer cross-network bridges as integration drops
}
PREDICTED_RATIONALE = {
    "clustering_w":     "explore→exploit: local clustering increases with age",
    "modularity_q":     "communities sharpen with age (Gu et al 2015)",
    "char_path_length": "decreased global integration → longer paths",
    "small_worldness":  "more small-world with developmental segregation",
    "mean_betweenness": "fewer hub-mediated cross-network shortcuts",
}


def _sign(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return "+" if x > 0 else ("-" if x < 0 else "0")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def exclusion_table(results_dir: Path) -> pd.DataFrame:
    """Cross-tab: rows = (age_group × FD), cols = exclusion reason counts."""
    rows = []
    for fd in FD_GRID:
        m = pd.read_csv(results_dir / f"manifest_FD{fd}_kappa{PRIMARY_KAPPA}.csv")
        for grp, g in m.groupby("age_group"):
            counts = g["exclusion_reason"].fillna("").value_counts().to_dict()
            rows.append({
                "age_group": grp,
                "fd": fd,
                "n_in_cohort": len(g),
                "n_included": int(g["included"].sum()),
                "n_min_retained_excluded": int(counts.get("min_retained_volumes", 0)),
                "n_kappa_max_excluded": int(counts.get("kappa_max_disconnected", 0)),
                "n_other_excluded": int(sum(v for k, v in counts.items()
                                             if k not in ("", "min_retained_volumes",
                                                          "kappa_max_disconnected"))),
            })
    return pd.DataFrame(rows)


def connectedness_report(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-subject and per-group κ-bump report at primary settings."""
    m = pd.read_csv(results_dir / f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    inc = m[m["included"]].copy()
    per_subject = inc[["subject_id", "age", "age_group", "kappa_requested",
                       "kappa_final", "connected_at_requested_kappa"]].copy()
    per_subject["bumped"] = ~per_subject["connected_at_requested_kappa"]

    by_group = (
        per_subject.groupby("age_group")
        .agg(n=("subject_id", "count"),
             n_bumped=("bumped", "sum"),
             mean_kappa_final=("kappa_final", "mean"),
             max_kappa_final=("kappa_final", "max"))
        .reset_index()
    )
    by_group["pct_bumped"] = (100 * by_group["n_bumped"] / by_group["n"]).round(1)
    return per_subject, by_group


def kappa_sensitivity_table(results_dir: Path) -> pd.DataFrame:
    """β_age (q) per metric across the κ grid at FD=0.5."""
    rows = []
    for k in KAPPA_GRID:
        path = results_dir / f"age_regression_FD{PRIMARY_FD}_kappa{k}.csv"
        if not path.exists():
            continue
        reg = pd.read_csv(path)
        for _, r in reg.iterrows():
            rows.append({
                "metric": r["metric"], "kappa": k,
                "beta_age": r.get("beta_age"), "se_age": r.get("se_age"),
                "p_age": r.get("p_age"), "FDR_q": r.get("FDR_q"),
                "n": r.get("n"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot(index="metric", columns="kappa",
                     values=["beta_age", "FDR_q", "n"])
    pivot.columns = [f"{a}_kappa{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    # Stability: sign of β_age consistent across all κ?
    beta_cols = [c for c in pivot.columns if c.startswith("beta_age_")]
    sign_consistent = pivot[beta_cols].apply(
        lambda row: len(set(_sign(v) for v in row if pd.notna(v))) == 1, axis=1)
    pivot["sign_consistent_across_kappa"] = sign_consistent
    return pivot


def fd_sensitivity_table(results_dir: Path) -> pd.DataFrame:
    """β_age (q, N) per metric across the FD grid at κ=0.10."""
    rows = []
    for fd in FD_GRID:
        path = results_dir / f"age_regression_FD{fd}_kappa{PRIMARY_KAPPA}.csv"
        if not path.exists():
            continue
        reg = pd.read_csv(path)
        for _, r in reg.iterrows():
            rows.append({
                "metric": r["metric"], "fd": fd,
                "beta_age": r.get("beta_age"), "p_age": r.get("p_age"),
                "FDR_q": r.get("FDR_q"), "n": r.get("n"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot(index="metric", columns="fd",
                     values=["beta_age", "FDR_q", "n"])
    pivot.columns = [f"{a}_FD{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    beta_cols = [c for c in pivot.columns if c.startswith("beta_age_FD")]
    sign_consistent = pivot[beta_cols].apply(
        lambda row: len(set(_sign(v) for v in row if pd.notna(v))) == 1, axis=1)
    pivot["sign_consistent_across_fd"] = sign_consistent
    return pivot


def prediction_vs_observed(results_dir: Path) -> pd.DataFrame:
    """Predicted (theory) vs observed (primary) age-effect direction per metric."""
    reg = pd.read_csv(results_dir / f"age_regression_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    rows = []
    for m in METRICS:
        rec = reg[reg["metric"] == m]
        if rec.empty:
            continue
        rec = rec.iloc[0]
        observed = _sign(rec.get("beta_age"))
        predicted = PREDICTED_SIGN_WITH_AGE[m]
        rows.append({
            "metric": m,
            "predicted_sign_with_age": predicted,
            "observed_sign_with_age": observed,
            "beta_age_primary": rec.get("beta_age"),
            "p_age_primary": rec.get("p_age"),
            "FDR_q_primary": rec.get("FDR_q"),
            "theory_consistent": predicted == observed,
            "rationale": PREDICTED_RATIONALE[m],
        })
    return pd.DataFrame(rows)


def limitations_numbers(results_dir: Path) -> pd.DataFrame:
    """Single-column table of named numbers the Discussion section cites verbatim."""
    rows = []

    primary = pd.read_csv(results_dir / f"manifest_FD{PRIMARY_FD}_kappa{PRIMARY_KAPPA}.csv")
    n_total = len(primary)
    n_inc = int(primary["included"].sum())
    n_min_retained = int((primary["exclusion_reason"] == "min_retained_volumes").sum())
    n_kmax = int((primary["exclusion_reason"] == "kappa_max_disconnected").sum())
    n_bumped = int((~primary.loc[primary["included"], "connected_at_requested_kappa"]).sum())
    mean_k_final = float(primary.loc[primary["included"], "kappa_final"].mean())

    rows += [
        ("n_in_participants_tsv", 155),
        ("n_with_bold_in_bucket", n_total),
        ("n_excluded_data_availability", 155 - n_total),
        ("pct_excluded_data_availability",
         round(100 * (155 - n_total) / 155, 1)),
        ("n_included_at_primary", n_inc),
        ("n_excluded_min_retained_at_primary", n_min_retained),
        ("n_excluded_kappa_max_at_primary", n_kmax),
        ("pct_excluded_at_primary", round(100 * (n_total - n_inc) / n_total, 1)),
        ("n_kappa_bumped_at_primary", n_bumped),
        ("pct_kappa_bumped_at_primary", round(100 * n_bumped / max(n_inc, 1), 1)),
        ("mean_kappa_final_at_primary", round(mean_k_final, 3)),
    ]

    # Residual FD-vs-age correlations per FD condition
    from scipy.stats import pearsonr  # local import keeps module imports light
    for fd in FD_GRID:
        m = pd.read_csv(results_dir / f"manifest_FD{fd}_kappa{PRIMARY_KAPPA}.csv")
        inc = m[m["included"] & m["age"].notna() & m["mean_FD"].notna()]
        if len(inc) >= 3:
            r, _ = pearsonr(inc["age"], inc["mean_FD"])
            rows.append((f"residual_fd_age_r_at_FD{fd}", round(float(r), 3)))
            rows.append((f"n_included_at_FD{fd}", int(inc.shape[0])))

    return pd.DataFrame(rows, columns=["name", "value"])


def main():
    results_dir = Path("results")

    print("exclusion_table.csv …")
    exclusion_table(results_dir).to_csv(results_dir / "exclusion_table.csv", index=False)

    print("connectedness_report.csv …")
    per_sub, by_grp = connectedness_report(results_dir)
    per_sub.to_csv(results_dir / "connectedness_report.csv", index=False)
    by_grp.to_csv(results_dir / "connectedness_summary_by_group.csv", index=False)

    print("kappa_sensitivity_table.csv …")
    kappa_sensitivity_table(results_dir).to_csv(
        results_dir / "kappa_sensitivity_table.csv", index=False)

    print("fd_sensitivity_table.csv …")
    fd_sensitivity_table(results_dir).to_csv(
        results_dir / "fd_sensitivity_table.csv", index=False)

    print("prediction_vs_observed.csv …")
    prediction_vs_observed(results_dir).to_csv(
        results_dir / "prediction_vs_observed.csv", index=False)

    print("limitations_numbers.csv …")
    limitations_numbers(results_dir).to_csv(
        results_dir / "limitations_numbers.csv", index=False)

    print("done.")


if __name__ == "__main__":
    main()
