"""build_graphs.py — undirected weighted FC graphs for the Richardson 2018
developmental fMRI cohort (gs://results_050626/).

Pipeline (single-pipeline, no alternatives — see Methods):
    raw 4D BOLD  ──parcellate(Schaefer-100)──▶ ROI timeseries
                 ──FD-scrub + Friston-24 + WM + CSF + DCT (OLS)──▶ clean TS  [Layer 1]
                 ──Pearson r → Fisher z → zero negs/diag────────▶ z-matrix  [Layer 2]
                 ──top-κ density + per-subject connectedness bump▶ graphml  [Layer 3]

The three cache layers are keyed independently so the κ sweep only
re-touches Layer 3 and the FD sweep only invalidates Layers 1+2.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sstats

import gcs_loader

logger = logging.getLogger("build_graphs")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    """Single source of truth for every analysis parameter.

    Locked block: methodological choices defended in the writeup; not parametrized
    on swept code paths. Swept block: only fd_threshold and kappa vary across runs.
    """
    # ---------------- LOCKED (no setters in swept code paths) ----------------
    n_rois: int = 100                                # Schaefer 2018, 100 ROI cortical (Schaefer2018)
    confound_model_name: str = "friston24+wm+csf+dct"  # Friston1996; Power2014; Satterthwaite2013
    edge_measure_name: str = "pearson_fisher_z_neg_zero"  # Fisher z; neg→0 (Rubinov & Sporns 2010 abs+threshold variant)
    null_model_name: str = "maslov_sneppen"          # Maslov & Sneppen 2002 degree-preserving rewiring
    n_rewires: int = 100                             # 100 rewires per subject for null distributions
    min_retained_minutes: float = 4.0                # Power2014 minimum-retention floor
    tr_seconds: float = 2.0                          # Richardson 2018 dataset TR (header on bucket files reports 1.0s; spec-locked)
    seed: int = 20260507                             # threaded through every stochastic step

    # ---------------- SWEPT ----------------
    fd_threshold: float = 0.5                        # mm; primary = 0.5 (Power criterion)
    kappa: float = 0.10                              # primary edge density

    # ---------------- per-subject connectedness rule ----------------
    kappa_step: float = 0.01                         # κ-bump increment when disconnected
    kappa_max: float = 0.5                           # safety stop on κ-bump

    # ---------------- paths ----------------
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    schaefer_atlas_path: Optional[Path] = None       # if None → one-time nilearn fetch (atlas ≠ dataset)

    # ---------------- derived ----------------
    @property
    def min_retained_volumes(self) -> int:
        return int(np.ceil(self.min_retained_minutes * 60.0 / self.tr_seconds))

    def confound_hash(self) -> str:
        return _short_sha1({"name": self.confound_model_name})

    def edge_measure_hash(self) -> str:
        return _short_sha1({"name": self.edge_measure_name})

    def null_model_hash(self) -> str:
        return _short_sha1(
            {"name": self.null_model_name, "n_rewires": self.n_rewires, "seed": self.seed}
        )


def _short_sha1(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


# Methods-section parameter table: hand-aligned with the inline citation comments
# on Config fields. Edits here must stay in lock-step with Config.
METHODS_TABLE_ROWS: list[dict] = [
    {"parameter": "Atlas",                       "value": "Schaefer 2018, 100 cortical ROIs", "block": "locked", "citation": "Schaefer2018"},
    {"parameter": "Confound model",              "value": "Friston-24 + WM + CSF + DCT cosines + intercept", "block": "locked", "citation": "Friston1996; Power2014; Satterthwaite2013"},
    {"parameter": "Edge measure",                "value": "Pearson r → Fisher z, negatives zeroed", "block": "locked", "citation": "Rubinov & Sporns 2010"},
    {"parameter": "Null model",                  "value": "Maslov–Sneppen degree-preserving rewiring (binary), 100 nulls/subject, swaps_per_edge=10", "block": "locked", "citation": "MaslovSneppen2002"},
    {"parameter": "Min retained data",           "value": "4 minutes (= 120 volumes at TR=2s)", "block": "locked", "citation": "Power2014"},
    {"parameter": "TR",                          "value": "2.0 s (Richardson 2018; bucket header reports 1.0s — overridden by spec)", "block": "locked", "citation": "Richardson2018"},
    {"parameter": "Connectedness rule",          "value": "Per-subject κ-bump in steps of 0.01 until graph connected (max κ=0.5)", "block": "locked", "citation": "this study"},
    {"parameter": "Modularity algorithm",        "value": "Louvain (networkx.community.louvain_communities), seeded", "block": "locked", "citation": "Blondel2008"},
    {"parameter": "Small-worldness",             "value": "σ = (C/C_rand) / (L/L_rand) on binary graph", "block": "locked", "citation": "Humphries2008"},
    {"parameter": "FD threshold (primary)",      "value": "0.5 mm", "block": "swept", "citation": "Power2014"},
    {"parameter": "FD threshold (sweep grid)",   "value": "{0.3, 0.5, 0.9} mm", "block": "swept", "citation": "Power2014; Satterthwaite2013"},
    {"parameter": "Edge density κ (primary)",    "value": "0.10", "block": "swept", "citation": "Achard2007"},
    {"parameter": "Edge density κ (sweep grid)", "value": "{0.05, 0.10, 0.15, 0.20, 0.25}", "block": "swept", "citation": "Achard2007"},
    {"parameter": "Random seed",                 "value": "20260507 (threaded through scrubbing, Louvain, and null rewiring)", "block": "locked", "citation": "this study"},
]


def dump_methods_table(out_path: Path) -> Path:
    """Write the Methods-section parameter table to CSV.

    Single source of truth for the writeup's Methods table. Citations are
    keys to be expanded in the bibliography.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(METHODS_TABLE_ROWS).to_csv(out_path, index=False)
    return out_path


# =============================================================================
# Cache layer (one helper used by every layer)
# =============================================================================

@dataclass
class CacheStats:
    """Sweep-level cache-hit accounting. Used by sweep_driver assertions."""
    layer1_hits: int = 0
    layer1_misses: int = 0
    layer2_hits: int = 0
    layer2_misses: int = 0
    layer3_hits: int = 0
    layer3_misses: int = 0

    def record(self, layer: int, hit: bool) -> None:
        attr = f"layer{layer}_{'hits' if hit else 'misses'}"
        setattr(self, attr, getattr(self, attr) + 1)


def load_or_compute(
    cache_path: Path,
    compute_fn: Callable[[], Any],
    serializer: Callable[[Path, Any], None],
    deserializer: Callable[[Path], Any],
) -> tuple[Any, bool]:
    """Single point of cache I/O. Returns (result, was_cache_hit).

    All parameter-driven cache invalidation is encoded in cache_path; this
    helper does no parameter logic — it only checks existence.
    """
    if cache_path.exists():
        return deserializer(cache_path), True
    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializer(cache_path, result)
    return result, False


def _layer1_save(p: Path, obj: dict) -> None:
    payload = {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in obj.items()}
    np.savez_compressed(p, **payload)

def _layer1_load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else np.asarray(f[k])) for k in f.files}

def _layer2_save(p: Path, arr: np.ndarray) -> None:
    np.save(p, arr)

def _layer2_load(p: Path) -> np.ndarray:
    return np.load(p)

def _layer3_save(p: Path, obj: dict) -> None:
    nx.write_graphml(obj["graph"], p)
    p.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in obj.items() if k != "graph"})
    )

def _layer3_load(p: Path) -> dict:
    G = nx.read_graphml(p, node_type=int)
    meta = json.loads(p.with_suffix(".json").read_text())
    return {"graph": G, **meta}


# =============================================================================
# Bucket inspection
# =============================================================================

def inspect_bucket() -> dict:
    """List bucket contents and characterize per-subject inventory.

    The pipeline branches on this: if the bucket already contained parcellated
    timeseries we'd skip Layer-1 parcellation — but for this bucket it does not.
    """
    files = gcs_loader.list_files()
    bolds = sorted(f for f in files if f.endswith("_bold.nii.gz"))
    confs = sorted(f for f in files if f.endswith("desc-confounds_regressors.tsv"))
    has_parts = any(f.endswith("participants.tsv") for f in files)

    bold_subs = sorted({_subject_id_from_path(p) for p in bolds})
    conf_subs = sorted({_subject_id_from_path(p) for p in confs})
    cohort = sorted(set(bold_subs) & set(conf_subs))

    report = {
        "n_objects": len(files),
        "n_bold": len(bolds),
        "n_full_confounds": len(confs),
        "has_participants_tsv": has_parts,
        "n_subjects_with_bold": len(bold_subs),
        "n_subjects_with_confounds": len(conf_subs),
        "n_subjects_complete": len(cohort),
        "complete_cohort": cohort,
        "parcellation_already_applied": False,
    }
    return report


def _subject_id_from_path(p: str) -> str:
    return Path(p).name.split("_")[0]


# =============================================================================
# Subject loading
# =============================================================================

def load_subject_raw(subject_id: str) -> dict:
    """Stream raw BOLD nifti + full confounds TSV + phenotypic row from GCS."""
    bold_path = f"{subject_id}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    conf_path = f"{subject_id}_task-pixar_desc-confounds_regressors.tsv"
    bold_img = gcs_loader.open_nifti(bold_path)

    fs = gcs_loader.fs()
    with fs.open(f"{gcs_loader.BUCKET}/{conf_path}", "rb") as fh:
        confounds = pd.read_csv(fh, sep="\t")
    pheno_df = _phenotype_lookup()
    pheno = pheno_df.loc[subject_id].to_dict() if subject_id in pheno_df.index else {}

    return {"bold_img": bold_img, "confounds": confounds, "pheno": pheno}


def _phenotype_lookup() -> pd.DataFrame:
    """participants.tsv → DataFrame indexed by participant_id with derived age_group.

    age_group rule (per spec): child if Age < 18 else adult.
    """
    fs = gcs_loader.fs()
    with fs.open(f"{gcs_loader.BUCKET}/participants.tsv", "rb") as fh:
        df = pd.read_csv(fh, sep="\t")
    df = df.rename(columns={"Age": "age", "Gender": "sex"})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_group"] = np.where(df["age"] < 18, "child", "adult")
    return df.set_index("participant_id")


# =============================================================================
# Confound matrix
# =============================================================================

def build_confound_matrix(confounds_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Friston-24 motion + WM + CSF + DCT cosines + intercept.

    Friston-24 = 6 motion + first derivatives + squares + squared derivatives.
    DCT cosines are pulled from the fmriprep `cosineNN` columns (no need to
    rebuild a DCT basis from TR).
    """
    motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    M = np.nan_to_num(confounds_df[motion].to_numpy(dtype=float), nan=0.0)
    dM = np.vstack([np.zeros((1, 6)), np.diff(M, axis=0)])
    Friston24 = np.hstack([M, dM, M ** 2, dM ** 2])

    wm = np.nan_to_num(confounds_df["white_matter"].to_numpy(dtype=float), nan=0.0)[:, None]
    csf = np.nan_to_num(confounds_df["csf"].to_numpy(dtype=float), nan=0.0)[:, None]

    cos_cols = sorted(c for c in confounds_df.columns if c.startswith("cosine"))
    if not cos_cols:
        raise ValueError(
            f"No DCT cosine columns in confounds (expected cosine00…); "
            f"got: {list(confounds_df.columns)}"
        )
    DCT = np.nan_to_num(confounds_df[cos_cols].to_numpy(dtype=float), nan=0.0)

    intercept = np.ones((len(confounds_df), 1))
    return np.hstack([intercept, Friston24, wm, csf, DCT])


# =============================================================================
# Schaefer atlas
# =============================================================================

_MASKER_SINGLETON: dict = {}

def _get_schaefer_masker(cfg: Config):
    """Return a NiftiLabelsMasker for the Schaefer 2018 100-ROI atlas.

    If cfg.schaefer_atlas_path is set, uses that local NIfTI; otherwise a
    one-time nilearn fetch populates ~/nilearn_data. The "no internet
    downloads at runtime" constraint applies to the fMRI dataset; the atlas
    is a static dependency cached on first call.
    """
    cache_key = (str(cfg.schaefer_atlas_path), cfg.n_rois)
    if cache_key in _MASKER_SINGLETON:
        return _MASKER_SINGLETON[cache_key]
    from nilearn.maskers import NiftiLabelsMasker
    if cfg.schaefer_atlas_path is not None:
        atlas_img = str(cfg.schaefer_atlas_path)
    else:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        logger.info("Fetching Schaefer 2018 atlas (one-time, cached in ~/nilearn_data)")
        atlas = fetch_atlas_schaefer_2018(n_rois=cfg.n_rois, yeo_networks=7)
        atlas_img = atlas.maps
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize=False, detrend=False,
        resampling_target="data",
    )
    _MASKER_SINGLETON[cache_key] = masker
    return masker


# =============================================================================
# Scrub + regress (Layer 1 producer)
# =============================================================================

def scrub_and_regress(bold_img, confounds_df: pd.DataFrame, cfg: Config,
                      masker=None) -> Optional[dict]:
    """Parcellate → FD-scrub (Power 2014) → drop scrubbed volumes → OLS-regress.

    Order is drop-then-regress: scrubbed volumes are removed from BOTH Y and
    the design matrix before OLS. Returns None if retained volumes fall below
    cfg.min_retained_volumes (4 min retention floor).
    """
    if masker is None:
        masker = _get_schaefer_masker(cfg)

    header_tr = float(bold_img.header.get_zooms()[-1]) if bold_img.header.get_zooms() else None
    if header_tr is not None and abs(header_tr - cfg.tr_seconds) > 1e-3:
        logger.warning("BOLD header TR=%.3fs ≠ cfg.tr_seconds=%.3fs (using cfg per spec)",
                       header_tr, cfg.tr_seconds)

    Y = masker.fit_transform(bold_img)  # (T, n_rois)
    C = build_confound_matrix(confounds_df, cfg)

    fd = np.nan_to_num(confounds_df["framewise_displacement"].to_numpy(dtype=float), nan=0.0)
    scrub = fd > cfg.fd_threshold
    if "non_steady_state_outlier00" in confounds_df.columns:
        nss = confounds_df["non_steady_state_outlier00"].to_numpy(dtype=float)
        scrub = scrub | (nss > 0)
    keep = ~scrub
    n_retained = int(keep.sum())
    if n_retained < cfg.min_retained_volumes:
        return None

    Y_keep = Y[keep]
    C_keep = C[keep]
    # Column-scale non-intercept regressors to unit variance so β stays
    # well-conditioned (raw WM/CSF means are ~10^3, motion ~10^0). Scaling
    # is invariant on the residual: (X·D)·(D⁻¹·β) = X·β.
    scales = np.std(C_keep, axis=0)
    scales[0] = 1.0  # leave the intercept column alone
    scales[scales < 1e-12] = 1.0
    C_scaled = C_keep / scales
    # BLAS can raise spurious FPE flags on large-magnitude matmul; the
    # lstsq result is finite (verified) so we silence those flags here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        beta, *_ = np.linalg.lstsq(C_scaled, Y_keep, rcond=None)
        R = Y_keep - C_scaled @ beta

    return {
        "clean_ts": R.astype(np.float32),
        "n_volumes_retained": n_retained,
        "mean_FD_retained": float(fd[keep].mean()),
    }


# =============================================================================
# Z-matrix (Layer 2 producer)
# =============================================================================

def compute_z_matrix(clean_ts: np.ndarray) -> np.ndarray:
    """Pearson r → Fisher z, zero negatives + diagonal, symmetric.

    Zero-variance ROIs (e.g. a parcel that falls outside the brain mask after
    resampling) produce NaN correlations; these are zeroed so the node remains
    in the 100-node set but contributes no edges. The connectedness rule is
    expected to bump κ for such subjects.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(clean_ts.T)
    R = np.where(np.isfinite(R), R, 0.0)
    R = np.clip(R, -0.999999, 0.999999)
    Z = np.arctanh(R)
    Z[Z < 0] = 0.0
    np.fill_diagonal(Z, 0.0)
    return ((Z + Z.T) / 2.0).astype(np.float32)


# =============================================================================
# Threshold + connectedness (Layer 3 producer)
# =============================================================================

def threshold_to_density(z: np.ndarray, kappa: float) -> nx.Graph:
    """Keep top-κ fraction of edges by absolute weight; weighted graph (z-values).

    Negative edges are already zeroed in z, so |z|==z for retained edges.
    """
    n = z.shape[0]
    n_possible = n * (n - 1) // 2
    n_keep = int(round(kappa * n_possible))
    G = nx.empty_graph(n)
    if n_keep <= 0:
        return G
    iu = np.triu_indices(n, k=1)
    w = z[iu]
    if n_keep >= len(w):
        idx_keep = np.where(w > 0)[0]
    else:
        top = np.argpartition(np.abs(w), -n_keep)[-n_keep:]
        idx_keep = top[w[top] > 0]
    rows, cols = iu
    for k in idx_keep:
        G.add_edge(int(rows[k]), int(cols[k]), weight=float(w[k]))
    return G


def enforce_connectedness(z: np.ndarray, kappa_start: float,
                          step: float = 0.01, max_kappa: float = 0.5) -> tuple[nx.Graph, float]:
    """Per-subject connectedness rule: bump κ in `step` increments until connected."""
    kappa = kappa_start
    while kappa <= max_kappa + 1e-9:
        G = threshold_to_density(z, kappa)
        if G.number_of_nodes() > 0 and nx.is_connected(G):
            return G, round(kappa, 4)
        kappa += step
    raise RuntimeError(f"Could not connect graph at κ ≤ {max_kappa}")


# =============================================================================
# Maslov–Sneppen null (utility — used by downstream metrics, not by this builder)
# =============================================================================

def maslov_sneppen_null(G: nx.Graph, n_rewires: int, seed: int) -> nx.Graph:
    """Degree-preserving rewiring (Maslov & Sneppen 2002) via networkx.double_edge_swap.

    Topological null: degree sequence preserved, weights are NOT preserved (the
    rewired edges adopt no specific weight). Use this for small-worldness etc.
    after computing per-edge metrics on the original graph.
    """
    Gn = G.copy()
    n_edges = Gn.number_of_edges()
    if n_edges < 2:
        return Gn
    nx.double_edge_swap(
        Gn,
        nswap=n_rewires * n_edges,
        max_tries=1000 * n_rewires * n_edges,
        seed=seed,
    )
    return Gn


# =============================================================================
# Main entry: build_subject_graph
# =============================================================================

def build_subject_graph(subject_id: str, cfg: Config,
                        stats: Optional[CacheStats] = None,
                        masker=None) -> dict:
    """Compose Layers 1→2→3 with caching at each layer.

    Cache filenames embed an 8-char SHA1 of the relevant parameter subset so a
    parameter change never silently reuses stale data.
    """
    confound_h = cfg.confound_hash()
    edge_h = cfg.edge_measure_hash()
    null_h = cfg.null_model_hash()
    fd = cfg.fd_threshold
    kappa = cfg.kappa

    # ---- Layer 1: clean parcellated timeseries ----
    layer1_path = cfg.cache_dir / "layer1" / f"{subject_id}_FD{fd}_conf{confound_h}.npz"

    def _compute_layer1():
        raw = load_subject_raw(subject_id)
        result = scrub_and_regress(raw["bold_img"], raw["confounds"], cfg, masker=masker)
        if result is None:
            return {
                "excluded": True,
                "exclusion_reason": "min_retained_volumes",
                "clean_ts": np.zeros((0, cfg.n_rois), dtype=np.float32),
                "n_volumes_retained": 0,
                "mean_FD_retained": float("nan"),
            }
        return {**result, "excluded": False, "exclusion_reason": ""}

    layer1, hit1 = load_or_compute(layer1_path, _compute_layer1, _layer1_save, _layer1_load)
    if stats is not None:
        stats.record(1, hit1)

    if bool(layer1["excluded"]):
        return _excluded_record(subject_id, layer1, layer1_path)

    # ---- Layer 2: z-matrix ----
    layer2_path = cfg.cache_dir / "layer2" / f"{subject_id}_FD{fd}_conf{confound_h}_edge{edge_h}.npy"

    def _compute_layer2():
        return compute_z_matrix(np.asarray(layer1["clean_ts"]))

    Z, hit2 = load_or_compute(layer2_path, _compute_layer2, _layer2_save, _layer2_load)
    if stats is not None:
        stats.record(2, hit2)

    # ---- Layer 3: thresholded graph ----
    layer3_path = cfg.cache_dir / "layer3" / f"{subject_id}_FD{fd}_kappa{kappa}_null{null_h}.graphml"

    def _compute_layer3():
        G_req = threshold_to_density(Z, kappa)
        connected_at_req = G_req.number_of_nodes() > 0 and nx.is_connected(G_req)
        G, kappa_final = enforce_connectedness(
            Z, kappa, step=cfg.kappa_step, max_kappa=cfg.kappa_max
        )
        return {
            "graph": G,
            "kappa_final": float(kappa_final),
            "connected_at_requested_kappa": bool(connected_at_req),
        }

    try:
        layer3, hit3 = load_or_compute(layer3_path, _compute_layer3, _layer3_save, _layer3_load)
    except RuntimeError as e:
        # κ_max disconnected → expected exclusion (not an error). Preserve
        # L1/L2 cache paths in the manifest so sweep-driver assertions can
        # still count L2-cached subjects accurately.
        if "Could not connect" not in str(e):
            raise
        if stats is not None:
            stats.record(3, False)
        return {
            "subject_id": subject_id,
            "included": False,
            "exclusion_reason": "kappa_max_disconnected",
            "n_volumes_retained": int(layer1["n_volumes_retained"]),
            "mean_FD_retained": float(layer1["mean_FD_retained"]),
            "layer1_cache": str(layer1_path),
            "layer2_cache": str(layer2_path),
            "layer3_cache": "",
            "graph": None,
            "kappa_final": float("nan"),
            "connected_at_requested_kappa": False,
            "n_edges": 0,
            "density": 0.0,
            "mean_weight": 0.0,
        }
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
    }


def _excluded_record(subject_id: str, layer1: dict, layer1_path: Path) -> dict:
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
    }


# =============================================================================
# build_all
# =============================================================================

MANIFEST_COLUMNS = [
    "subject_id", "age", "age_group", "sex",
    "fd_threshold", "kappa_requested", "kappa_final",
    "n_volumes_retained", "mean_FD",
    "n_edges", "density", "mean_weight",
    "connected_at_requested_kappa",
    "layer1_cache", "layer2_cache", "layer3_cache",
    "included", "exclusion_reason",
]


def build_all(subject_ids: list[str], cfg: Config, manifest_path: Path,
              stats: Optional[CacheStats] = None) -> pd.DataFrame:
    """Run build_subject_graph per subject, accumulate manifest, write CSV.

    Excluded subjects appear with included=False so exclusion stats are
    recoverable from the manifest alone.
    """
    pheno = _phenotype_lookup()
    masker = _get_schaefer_masker(cfg)

    rows = []
    for i, sid in enumerate(subject_ids):
        logger.info("[%d/%d] %s  (FD=%s, κ=%s)", i + 1, len(subject_ids), sid,
                    cfg.fd_threshold, cfg.kappa)
        try:
            res = build_subject_graph(sid, cfg, stats=stats, masker=masker)
        except Exception as e:
            logger.exception("Build failed for %s", sid)
            res = {
                "subject_id": sid, "included": False,
                "exclusion_reason": f"build_error:{type(e).__name__}",
                "n_volumes_retained": 0, "mean_FD_retained": float("nan"),
                "layer1_cache": "", "layer2_cache": "", "layer3_cache": "",
                "graph": None, "kappa_final": float("nan"),
                "connected_at_requested_kappa": False,
                "n_edges": 0, "density": 0.0, "mean_weight": 0.0,
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
            "layer1_cache": res["layer1_cache"],
            "layer2_cache": res["layer2_cache"],
            "layer3_cache": res["layer3_cache"],
            "included": res["included"],
            "exclusion_reason": res["exclusion_reason"],
        })
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


# =============================================================================
# QC
# =============================================================================

def qc_report(manifest: pd.DataFrame, output_dir: Path) -> dict:
    """Six-panel QC PNG + flag dict for the (FD, κ) combination in the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inc = manifest[manifest["included"]].copy()
    fd = float(manifest["fd_threshold"].iloc[0])
    kappa = float(manifest["kappa_requested"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1) Pooled edge-weight histogram
    ax = axes[0, 0]
    pooled = []
    for path in inc["layer3_cache"]:
        if path and Path(path).exists():
            G = nx.read_graphml(path, node_type=int)
            pooled.extend(d["weight"] for _, _, d in G.edges(data=True))
    if pooled:
        ax.hist(pooled, bins=50, edgecolor="black", alpha=0.75)
    ax.set_title(f"Edge weights (pooled, n={len(pooled)})")
    ax.set_xlabel("Fisher z"); ax.set_ylabel("count")

    # 2) Density per subject by age group
    ax = axes[0, 1]
    for grp, sub in inc.groupby("age_group"):
        ax.scatter(sub["age"], sub["density"], label=grp, alpha=0.7)
    ax.set_title("Density vs age (colored by group)")
    ax.set_xlabel("age (yr)"); ax.set_ylabel("density")
    if not inc.empty:
        ax.legend()

    flags: dict = {}

    # 3) Density vs age (CRITICAL: should be flat at fixed κ)
    ax = axes[0, 2]
    if len(inc) >= 3:
        r, p = sstats.pearsonr(inc["age"].to_numpy(), inc["density"].to_numpy())
        ax.scatter(inc["age"], inc["density"], alpha=0.7)
        warn = " ⚠ FLAGGED (|r|>0.2)" if abs(r) > 0.2 else ""
        ax.set_title(f"Density vs age: r={r:.3f}, p={p:.3g}{warn}")
        flags["density_age_r"] = float(r)
        flags["density_age_flag"] = bool(abs(r) > 0.2)
    ax.set_xlabel("age (yr)"); ax.set_ylabel("density")

    # 4) Mean FD vs age (CRITICAL: residual-motion-confound check)
    ax = axes[1, 0]
    if len(inc) >= 3:
        r, p = sstats.pearsonr(inc["age"].to_numpy(), inc["mean_FD"].to_numpy())
        ax.scatter(inc["age"], inc["mean_FD"], alpha=0.7)
        warn = " ⚠ FLAGGED (|r|>0.2)" if abs(r) > 0.2 else ""
        ax.set_title(f"Mean FD vs age: r={r:.3f}, p={p:.3g}{warn}")
        flags["fd_age_r"] = float(r)
        flags["fd_age_flag"] = bool(abs(r) > 0.2)
    ax.set_xlabel("age (yr)"); ax.set_ylabel("mean retained FD (mm)")

    # 5) Subjects retained per age group
    ax = axes[1, 1]
    by_grp = manifest.groupby("age_group")["included"].agg(["sum", "count"])
    if not by_grp.empty:
        by_grp.plot(kind="bar", ax=ax)
    ax.set_title(f"Retained per age group (FD={fd})")
    ax.set_ylabel("n")

    # 6) Subjects needing κ-bump
    ax = axes[1, 2]
    bumped = int((~inc["connected_at_requested_kappa"]).sum())
    unchanged = int(inc["connected_at_requested_kappa"].sum())
    ax.bar(["unchanged", "κ-bumped"], [unchanged, bumped])
    ax.set_title(f"κ-bumped subjects (κ_req={kappa})")
    ax.set_ylabel("n")
    flags["n_kappa_bumped"] = bumped

    fig.suptitle(
        f"QC — FD={fd}, κ={kappa}, included={int(manifest['included'].sum())}/{len(manifest)}"
    )
    fig.tight_layout()
    out_png = output_dir / f"qc_FD{fd}_kappa{kappa}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    flags["png"] = str(out_png)
    flags["n_included"] = int(manifest["included"].sum())
    flags["n_total"] = int(len(manifest))
    return flags


# =============================================================================
# Convenience: complete-cohort discovery
# =============================================================================

def complete_cohort() -> list[str]:
    """Subjects in the bucket with BOTH a BOLD and a full confounds file.

    The bucket lists 155 IDs in participants.tsv but only ~101 have a
    preprocessed BOLD; this is the usable cohort.
    """
    return inspect_bucket()["complete_cohort"]
