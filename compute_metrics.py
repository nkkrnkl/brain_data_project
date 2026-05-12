from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import build_graphs as bg

logger = logging.getLogger("compute_metrics")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
FD_GRID = [0.3, 0.5, 0.9]
PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10

METRIC_COLS = [
    "subject_id", "fd", "kappa", "kappa_final",
    "clustering_w", "modularity_q", "char_path_length", "small_worldness",
    "mean_betweenness", "n_nulls_used",
]


def _binarize(G: nx.Graph) -> nx.Graph:
    """Return an unweighted copy of G (any edge with weight>0 → edge=1)."""
    Gb = nx.Graph()
    Gb.add_nodes_from(G.nodes())
    Gb.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0)
    return Gb


def clustering_weighted(G: nx.Graph) -> float:
    """Mean Onnela weighted clustering (nx default for weighted graphs)."""
    if G.number_of_edges() == 0:
        return 0.0
    return float(np.mean(list(nx.clustering(G, weight="weight").values())))


def modularity_louvain(G: nx.Graph, seed: int) -> float:
    """Louvain modularity Q on weighted graph; deterministic with seed."""
    if G.number_of_edges() == 0:
        return 0.0
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return float(nx.community.modularity(G, comms, weight="weight"))


def char_path_length_binary(Gb: nx.Graph) -> float:
    """Mean shortest-path length on the binary graph; assumes connected."""
    if Gb.number_of_nodes() < 2:
        return 0.0
    return float(nx.average_shortest_path_length(Gb))


def clustering_binary(Gb: nx.Graph) -> float:
    """Mean clustering coefficient on the binary graph (used in σ)."""
    if Gb.number_of_edges() == 0:
        return 0.0
    return float(np.mean(list(nx.clustering(Gb).values())))


def mean_betweenness_binary(Gb: nx.Graph) -> float:
    """Mean (normalized) betweenness centrality on the binary graph."""
    if Gb.number_of_edges() == 0:
        return 0.0
    return float(np.mean(list(nx.betweenness_centrality(Gb, normalized=True).values())))


def maslov_sneppen_null_binary(Gb: nx.Graph, swaps_per_edge: int, seed: int) -> nx.Graph:
    """One Maslov–Sneppen-rewired null preserving the binary degree sequence.

    Number of swap attempts = swaps_per_edge * |E|.
    """
    Gn = Gb.copy()
    n_edges = Gn.number_of_edges()
    if n_edges < 2:
        return Gn
    nx.double_edge_swap(
        Gn,
        nswap=swaps_per_edge * n_edges,
        max_tries=100 * swaps_per_edge * n_edges,
        seed=seed,
    )
    return Gn


def small_worldness(G: nx.Graph, n_nulls: int, seed: int,
                    swaps_per_edge: int = 10) -> tuple[float, int]:
    """σ = (C/C_rand)/(L/L_rand), binary, over n_nulls Maslov–Sneppen rewirings.

    Disconnected nulls are skipped (would yield infinite L). Returns
    (sigma, n_nulls_used). n_nulls_used < n_nulls indicates some nulls were
    discarded.
    """
    Gb = _binarize(G)
    if Gb.number_of_edges() < 2 or not nx.is_connected(Gb):
        return float("nan"), 0

    C = clustering_binary(Gb)
    L = char_path_length_binary(Gb)

    rng = np.random.default_rng(seed)
    null_seeds = rng.integers(0, 2**31 - 1, size=n_nulls)

    Cs, Ls = [], []
    for s in null_seeds:
        Gn = maslov_sneppen_null_binary(Gb, swaps_per_edge=swaps_per_edge, seed=int(s))
        if not nx.is_connected(Gn):
            continue
        Cs.append(clustering_binary(Gn))
        Ls.append(char_path_length_binary(Gn))

    if not Cs:
        return float("nan"), 0
    C_rand = float(np.mean(Cs))
    L_rand = float(np.mean(Ls))
    if C_rand <= 0 or L_rand <= 0:
        return float("nan"), len(Cs)
    return float((C / C_rand) / (L / L_rand)), len(Cs)


def compute_metrics_for_subject(graph_path: Path, cfg: bg.Config) -> dict:
    """Return one dict of metrics for the cached graphml at graph_path."""
    G = nx.read_graphml(str(graph_path), node_type=int)
    Gb = _binarize(G)

    clust_w = clustering_weighted(G)
    mod_q = modularity_louvain(G, seed=cfg.seed)
    L = char_path_length_binary(Gb) if nx.is_connected(Gb) else float("nan")
    sigma, n_nulls = small_worldness(G, n_nulls=cfg.n_rewires, seed=cfg.seed)
    btw = mean_betweenness_binary(Gb)

    return {
        "clustering_w": clust_w,
        "modularity_q": mod_q,
        "char_path_length": L,
        "small_worldness": sigma,
        "mean_betweenness": btw,
        "n_nulls_used": n_nulls,
    }


def _layer4_path(cfg: bg.Config) -> Path:
    return cfg.cache_dir / "layer4" / f"metrics_FD{cfg.fd_threshold}_kappa{cfg.kappa}.csv"


def _load_existing(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return pd.DataFrame(columns=METRIC_COLS)


def compute_metrics_all(cfg: bg.Config, manifest_path: Path,
                        verbose: bool = True) -> pd.DataFrame:
    """Compute the metric panel for every included subject in the manifest.

    Row-level cache: rows already in the Layer-4 CSV are not recomputed.
    Excluded subjects (no Layer 3 graph) are skipped silently — they are
    already in the build manifest and will be merged downstream by Phase 3.
    """
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    included = manifest[manifest["included"]].copy()

    cache_path = _layer4_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(cache_path)
    done_ids = set(existing["subject_id"].astype(str))

    rows = existing.to_dict("records")
    n_to_do = sum(1 for sid in included["subject_id"] if sid not in done_ids)
    logger.info("FD=%s κ=%s : %d included, %d to compute, %d cached",
                cfg.fd_threshold, cfg.kappa, len(included), n_to_do, len(done_ids))

    for i, (_, row) in enumerate(included.iterrows(), 1):
        sid = row["subject_id"]
        if sid in done_ids:
            continue
        graph_path = Path(row["layer3_cache"])
        if not graph_path.exists():
            logger.warning("missing graphml for %s: %s", sid, graph_path)
            continue
        t0 = time.time()
        metrics = compute_metrics_for_subject(graph_path, cfg)
        dt = time.time() - t0
        if verbose:
            logger.info("[%d/%d] %s  σ=%.3f  Q=%.3f  L=%.3f  C_w=%.3f  B=%.4f  (%.1fs)",
                        i, len(included), sid, metrics["small_worldness"],
                        metrics["modularity_q"], metrics["char_path_length"],
                        metrics["clustering_w"], metrics["mean_betweenness"], dt)
        rows.append({
            "subject_id": sid,
            "fd": cfg.fd_threshold,
            "kappa": cfg.kappa,
            "kappa_final": float(row["kappa_final"]),
            **metrics,
        })
        if i % 10 == 0:
            pd.DataFrame(rows, columns=METRIC_COLS).to_csv(cache_path, index=False)

    df = pd.DataFrame(rows, columns=METRIC_COLS)
    df.to_csv(cache_path, index=False)
    return df


def _manifest_for(cfg: bg.Config) -> Path:
    return cfg.results_dir / f"manifest_FD{cfg.fd_threshold}_kappa{cfg.kappa}.csv"


def _run(fd: float, kappa: float, results_dir: Path, cache_dir: Path,
         smoke: bool = False, smoke_n: int = 5) -> pd.DataFrame:
    cfg = bg.Config(fd_threshold=fd, kappa=kappa,
                    cache_dir=cache_dir, results_dir=results_dir)
    manifest_path = _manifest_for(cfg)
    if smoke:
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["included"]].head(smoke_n)
        smoke_path = manifest_path.parent / f"_smoke_manifest_FD{fd}_kappa{kappa}.csv"
        manifest.to_csv(smoke_path, index=False)
        return compute_metrics_all(cfg, smoke_path)
    return compute_metrics_all(cfg, manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="primary",
                    choices=["primary", "kappa", "fd", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if args.smoke:
        logger.info("=== SMOKE: 5 subjects at FD=%s, κ=%s ===", PRIMARY_FD, PRIMARY_KAPPA)
        df = _run(PRIMARY_FD, PRIMARY_KAPPA, args.results_dir, args.cache_dir,
                  smoke=True, smoke_n=5)
        print(df[["subject_id", "clustering_w", "modularity_q",
                  "char_path_length", "small_worldness",
                  "mean_betweenness", "n_nulls_used"]].to_string(index=False))
        return

    if args.mode in ("primary", "all"):
        logger.info("=== PRIMARY: FD=%s, κ=%s ===", PRIMARY_FD, PRIMARY_KAPPA)
        _run(PRIMARY_FD, PRIMARY_KAPPA, args.results_dir, args.cache_dir)

    if args.mode in ("kappa", "all"):
        logger.info("=== κ SWEEP: FD=%s, κ ∈ %s ===", PRIMARY_FD, KAPPA_GRID)
        for k in KAPPA_GRID:
            _run(PRIMARY_FD, k, args.results_dir, args.cache_dir)

    if args.mode in ("fd", "all"):
        logger.info("=== FD SWEEP: FD ∈ %s, κ=%s ===", FD_GRID, PRIMARY_KAPPA)
        for fd in FD_GRID:
            _run(fd, PRIMARY_KAPPA, args.results_dir, args.cache_dir)


if __name__ == "__main__":
    main()
