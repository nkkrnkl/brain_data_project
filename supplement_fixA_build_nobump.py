"""supplement_fixA_build_nobump.py — supplementary Fix A (no κ-bumping) build.

Goal: keep *all* existing code/analysis intact and add a parallel build that:
  - uses the same Layer 1 (scrub+regress) and Layer 2 (z-matrix) caches
  - builds Layer 3 graphs at the *requested* κ only (no per-subject bumping)
  - writes outputs to separate paths so nothing overwrites the main analysis

Outputs (per FD, κ):
  - results/manifest_nobump_FD{fd}_kappa{kappa}.csv
  - results/qc_nobump_FD{fd}_kappa{kappa}/qc_nobump_FD{fd}_kappa{kappa}.png
  - cache/layer3_nobump/*.graphml (+ .json metadata)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import build_graphs as bg


MANIFEST_NOBUMP_COLUMNS = [
    "subject_id", "age", "age_group", "sex",
    "fd_threshold", "kappa_requested", "kappa_final",
    "n_volumes_retained", "mean_FD",
    "n_edges", "density", "mean_weight",
    "connected_at_requested_kappa",
    "largest_cc_n_nodes", "largest_cc_frac_nodes",
    "layer1_cache", "layer2_cache", "layer3_cache",
    "included", "exclusion_reason",
]


def _layer3_nobump_path(cfg: bg.Config, subject_id: str) -> Path:
    null_h = cfg.null_model_hash()
    return (
        cfg.cache_dir
        / "layer3_nobump"
        / f"{subject_id}_FD{cfg.fd_threshold}_kappa{cfg.kappa}_null{null_h}.graphml"
    )


def _largest_cc_stats(G: nx.Graph) -> tuple[int, float]:
    """Return (n_nodes_in_lcc, fraction_of_total_nodes)."""
    n = G.number_of_nodes()
    if n == 0:
        return 0, float("nan")
    if G.number_of_edges() == 0:
        # Components are singletons; LCC size is 1 by construction.
        return 1, 1.0 / float(n)
    comps = list(nx.connected_components(G))
    if not comps:
        return 0, float("nan")
    lcc = max(comps, key=len)
    return int(len(lcc)), float(len(lcc) / float(n))


def build_subject_graph_nobump(
    subject_id: str,
    cfg: bg.Config,
    stats: Optional[bg.CacheStats] = None,
    masker=None,
) -> dict:
    """Build subject graph at requested κ only; reuses Layer1/2 caches."""
    confound_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()
    fd = cfg.fd_threshold
    kappa = cfg.kappa

    # ---- Layer 1 (reuse bg's implementation + cache) ----
    layer1_path = cfg.cache_dir / "layer1" / f"{subject_id}_FD{fd}_conf{confound_h}.npz"

    def _compute_layer1():
        raw = bg.load_subject_raw(subject_id)
        result = bg.scrub_and_regress(raw["bold_img"], raw["confounds"], cfg, masker=masker)
        if result is None:
            return {
                "excluded": True,
                "exclusion_reason": "min_retained_volumes",
                "clean_ts": np.zeros((0, cfg.n_rois), dtype=np.float32),
                "n_volumes_retained": 0,
                "mean_FD_retained": float("nan"),
            }
        return {**result, "excluded": False, "exclusion_reason": ""}

    layer1, hit1 = bg.load_or_compute(layer1_path, _compute_layer1, bg._layer1_save, bg._layer1_load)
    if stats is not None:
        stats.record(1, hit1)

    if bool(layer1["excluded"]):
        return {
            "subject_id": subject_id,
            "included": False,
            "exclusion_reason": str(layer1["exclusion_reason"]),
            "n_volumes_retained": int(layer1["n_volumes_retained"]),
            "mean_FD_retained": float(layer1["mean_FD_retained"]),
            "layer1_cache": str(layer1_path),
            "layer2_cache": "",
            "layer3_cache": "",
            "graph": None,
            "kappa_final": float("nan"),
            "connected_at_requested_kappa": False,
            "n_edges": 0,
            "density": 0.0,
            "mean_weight": 0.0,
            "largest_cc_n_nodes": 0,
            "largest_cc_frac_nodes": float("nan"),
        }

    # ---- Layer 2 (reuse bg's implementation + cache) ----
    layer2_path = cfg.cache_dir / "layer2" / f"{subject_id}_FD{fd}_conf{confound_h}_edge{edge_h}.npy"

    def _compute_layer2():
        return bg.compute_z_matrix(np.asarray(layer1["clean_ts"]))

    Z, hit2 = bg.load_or_compute(layer2_path, _compute_layer2, bg._layer2_save, bg._layer2_load)
    if stats is not None:
        stats.record(2, hit2)

    # ---- Layer 3 (no bumping; separate cache dir) ----
    layer3_path = _layer3_nobump_path(cfg, subject_id)

    def _compute_layer3():
        G = bg.threshold_to_density(Z, kappa)
        connected = G.number_of_nodes() > 0 and nx.is_connected(G)
        lcc_n, lcc_frac = _largest_cc_stats(G)
        return {
            "graph": G,
            "kappa_final": float(kappa),  # by definition: no bumping
            "connected_at_requested_kappa": bool(connected),
            "largest_cc_n_nodes": int(lcc_n),
            "largest_cc_frac_nodes": float(lcc_frac),
        }

    layer3, hit3 = bg.load_or_compute(layer3_path, _compute_layer3, bg._layer3_save, bg._layer3_load)
    if stats is not None:
        stats.record(3, hit3)

    G = layer3["graph"]
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    return {
        "subject_id": subject_id,
        "included": True,
        "exclusion_reason": "",
        "n_volumes_retained": int(layer1["n_volumes_retained"]),
        "mean_FD_retained": float(layer1["mean_FD_retained"]),
        "layer1_cache": str(layer1_path),
        "layer2_cache": str(layer2_path),
        "layer3_cache": str(layer3_path),
        "graph": G,
        "kappa_final": float(layer3["kappa_final"]),
        "connected_at_requested_kappa": bool(layer3["connected_at_requested_kappa"]),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
        "largest_cc_n_nodes": int(layer3["largest_cc_n_nodes"]),
        "largest_cc_frac_nodes": float(layer3["largest_cc_frac_nodes"]),
    }


def build_all_nobump(
    subject_ids: list[str],
    cfg: bg.Config,
    manifest_path: Path,
    stats: Optional[bg.CacheStats] = None,
) -> pd.DataFrame:
    """Cohort build for the no-bump supplementary analysis."""
    pheno = bg._phenotype_lookup()
    masker = bg._get_schaefer_masker(cfg)

    rows = []
    for i, sid in enumerate(subject_ids):
        bg.logger.info("[nobump %d/%d] %s  (FD=%s, κ=%s)", i + 1, len(subject_ids), sid,
                       cfg.fd_threshold, cfg.kappa)
        try:
            res = build_subject_graph_nobump(sid, cfg, stats=stats, masker=masker)
        except Exception as e:
            bg.logger.exception("No-bump build failed for %s", sid)
            res = {
                "subject_id": sid, "included": False,
                "exclusion_reason": f"build_error:{type(e).__name__}",
                "n_volumes_retained": 0, "mean_FD_retained": float("nan"),
                "layer1_cache": "", "layer2_cache": "", "layer3_cache": "",
                "graph": None, "kappa_final": float("nan"),
                "connected_at_requested_kappa": False,
                "n_edges": 0, "density": 0.0, "mean_weight": 0.0,
                "largest_cc_n_nodes": 0, "largest_cc_frac_nodes": float("nan"),
            }

        p = pheno.loc[sid] if sid in pheno.index else None
        rows.append({
            "subject_id": sid,
            "age": float(p["age"]) if p is not None and pd.notna(p["age"]) else float("nan"),
            "age_group": str(p["age_group"]) if p is not None else "",
            "sex": str(p["sex"]) if p is not None else "",
            "fd_threshold": cfg.fd_threshold,
            "kappa_requested": cfg.kappa,
            "kappa_final": res["kappa_final"],
            "n_volumes_retained": res["n_volumes_retained"],
            "mean_FD": res["mean_FD_retained"],
            "n_edges": res["n_edges"],
            "density": res["density"],
            "mean_weight": res["mean_weight"],
            "connected_at_requested_kappa": res["connected_at_requested_kappa"],
            "largest_cc_n_nodes": res["largest_cc_n_nodes"],
            "largest_cc_frac_nodes": res["largest_cc_frac_nodes"],
            "layer1_cache": res["layer1_cache"],
            "layer2_cache": res["layer2_cache"],
            "layer3_cache": res["layer3_cache"],
            "included": res["included"],
            "exclusion_reason": res["exclusion_reason"],
        })

    df = pd.DataFrame(rows, columns=MANIFEST_NOBUMP_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


def qc_report_nobump(manifest: pd.DataFrame, output_dir: Path) -> dict:
    """QC wrapper that reuses bg.qc_report output naming conventions."""
    # We reuse bg.qc_report directly; it only needs the manifest schema columns it uses,
    # which are present here. The only mismatch is that panel 6 "κ-bumped" becomes
    # "not connected at requested κ" in this analysis, which is exactly what we want.
    flags = bg.qc_report(manifest.rename(columns={"kappa_requested": "kappa_requested"}), output_dir)
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--cohort-from", default=None,
                    help="Path to text file with one subject_id per line. Skips bucket listing.")
    ap.add_argument("--cohort-from-manifest", default=None,
                    help="Path to an existing manifest CSV to reuse its subject_id column (skips bucket listing).")
    ap.add_argument("--offline", action="store_true",
                    help="Offline mode: do not attempt any GCS reads. Requires Layer 2 cache to exist; "
                         "subjects missing Layer 2 will be marked excluded with exclusion_reason=missing_layer2_cache.")
    args = ap.parse_args()

    manifest_seed: Optional[pd.DataFrame] = None

    if args.cohort_from is not None and args.cohort_from:
        cohort = [ln.strip() for ln in Path(args.cohort_from).read_text().splitlines() if ln.strip()]
    elif args.cohort_from_manifest is not None and args.cohort_from_manifest:
        manifest_seed = pd.read_csv(Path(args.cohort_from_manifest))
        cohort = [str(s) for s in manifest_seed["subject_id"].astype(str).tolist()]
    else:
        cohort = bg.complete_cohort()

    cfg = bg.Config(fd_threshold=args.fd, kappa=args.kappa,
                    cache_dir=args.cache_dir, results_dir=args.results_dir)

    stats = bg.CacheStats()
    manifest_path = args.results_dir / f"manifest_nobump_FD{args.fd}_kappa{args.kappa}.csv"
    if args.offline:
        # Offline mode: only build Layer 3 if Layer 2 exists; never touch GCS.
        # Prefer phenotypic columns from an existing manifest if provided.
        seed_lookup = {}
        if manifest_seed is not None:
            for _, r in manifest_seed.iterrows():
                seed_lookup[str(r["subject_id"])] = r.to_dict()

        # Otherwise, try the bucket phenotype table (may fail offline).
        pheno = None
        if manifest_seed is None:
            try:
                pheno = bg._phenotype_lookup()
            except Exception:
                pheno = None
        rows = []
        confound_h = cfg.confound_hash()
        edge_h = cfg.edge_measure_hash()
        for sid in cohort:
            # If we were given a seed manifest, prefer its layer2_cache pointer.
            seed = seed_lookup.get(str(sid), {})
            seed_layer2 = seed.get("layer2_cache", "")
            seed_layer2 = "" if (seed_layer2 is None or (isinstance(seed_layer2, float) and not np.isfinite(seed_layer2))) else seed_layer2
            seed_layer2 = str(seed_layer2).strip()
            layer2_path = Path(seed_layer2) if seed_layer2 else (
                cfg.cache_dir / "layer2" / f"{sid}_FD{cfg.fd_threshold}_conf{confound_h}_edge{edge_h}.npy"
            )
            layer1_path = cfg.cache_dir / "layer1" / f"{sid}_FD{cfg.fd_threshold}_conf{confound_h}.npz"
            if not layer2_path.exists():
                # phenotype from seed manifest if available
                age = seed.get("age", float("nan"))
                age_group = seed.get("age_group", "")
                sex = seed.get("sex", "")
                if manifest_seed is None:
                    p = pheno.loc[sid] if pheno is not None and sid in pheno.index else None
                    age = float(p["age"]) if p is not None and pd.notna(p["age"]) else float("nan")
                    age_group = str(p["age_group"]) if p is not None else ""
                    sex = str(p["sex"]) if p is not None else ""
                rows.append({
                    "subject_id": sid,
                    "age": float(age) if pd.notna(age) else float("nan"),
                    "age_group": str(age_group),
                    "sex": str(sex),
                    "fd_threshold": cfg.fd_threshold,
                    "kappa_requested": cfg.kappa,
                    "kappa_final": float("nan"),
                    "n_volumes_retained": 0,
                    "mean_FD": float("nan"),
                    "n_edges": 0,
                    "density": 0.0,
                    "mean_weight": 0.0,
                    "connected_at_requested_kappa": False,
                    "largest_cc_n_nodes": 0,
                    "largest_cc_frac_nodes": float("nan"),
                    "layer1_cache": str(layer1_path) if layer1_path.exists() else "",
                    "layer2_cache": "",
                    "layer3_cache": "",
                    "included": False,
                    "exclusion_reason": "missing_layer2_cache",
                })
                continue

            Z = np.load(layer2_path)
            G = bg.threshold_to_density(Z, cfg.kappa)
            connected = G.number_of_nodes() > 0 and nx.is_connected(G)
            lcc_n, lcc_frac = _largest_cc_stats(G)
            layer3_path = _layer3_nobump_path(cfg, sid)
            bg._layer3_save(layer3_path, {
                "graph": G,
                "kappa_final": float(cfg.kappa),
                "connected_at_requested_kappa": bool(connected),
                "largest_cc_n_nodes": int(lcc_n),
                "largest_cc_frac_nodes": float(lcc_frac),
            })
            weights = [d["weight"] for _, _, d in G.edges(data=True)]
            age = seed.get("age", float("nan"))
            age_group = seed.get("age_group", "")
            sex = seed.get("sex", "")
            if manifest_seed is None:
                p = pheno.loc[sid] if pheno is not None and sid in pheno.index else None
                age = float(p["age"]) if p is not None and pd.notna(p["age"]) else float("nan")
                age_group = str(p["age_group"]) if p is not None else ""
                sex = str(p["sex"]) if p is not None else ""
            rows.append({
                "subject_id": sid,
                "age": float(age) if pd.notna(age) else float("nan"),
                "age_group": str(age_group),
                "sex": str(sex),
                "fd_threshold": cfg.fd_threshold,
                "kappa_requested": cfg.kappa,
                "kappa_final": float(cfg.kappa),
                "n_volumes_retained": 0,
                "mean_FD": float("nan"),
                "n_edges": G.number_of_edges(),
                "density": nx.density(G),
                "mean_weight": float(np.mean(weights)) if weights else 0.0,
                "connected_at_requested_kappa": bool(connected),
                "largest_cc_n_nodes": int(lcc_n),
                "largest_cc_frac_nodes": float(lcc_frac),
                "layer1_cache": str(layer1_path) if layer1_path.exists() else "",
                "layer2_cache": str(layer2_path),
                "layer3_cache": str(layer3_path),
                "included": True,
                "exclusion_reason": "",
            })

        manifest = pd.DataFrame(rows, columns=MANIFEST_NOBUMP_COLUMNS)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    else:
        manifest = build_all_nobump(cohort, cfg, manifest_path, stats=stats)

    qc_dir = args.results_dir / f"qc_nobump_FD{args.fd}_kappa{args.kappa}"
    flags = qc_report_nobump(manifest, qc_dir)

    print(f"built manifest: {manifest_path}")
    print(f"qc: {flags.get('png')}")
    print(f"cache stats: L1 hits/miss={stats.layer1_hits}/{stats.layer1_misses}  "
          f"L2 hits/miss={stats.layer2_hits}/{stats.layer2_misses}  "
          f"L3 hits/miss={stats.layer3_hits}/{stats.layer3_misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

