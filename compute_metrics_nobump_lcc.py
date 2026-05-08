"""compute_metrics_nobump_lcc.py — supplementary Fix A metrics on LCC.

Reads the no-bump manifest produced by supplement_fixA_build_nobump.py and
computes the same metric panel as compute_metrics.py, but on the *largest
connected component* (LCC) when the requested-κ graph is disconnected.

Outputs:
  - cache/layer4_nobump/metrics_nobump_FD{fd}_kappa{kappa}.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import build_graphs as bg
import compute_metrics as cm

logger = logging.getLogger("compute_metrics_nobump_lcc")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


METRIC_COLS = [
    "subject_id", "fd", "kappa",
    "connected_at_requested_kappa",
    "lcc_n_nodes", "lcc_frac_nodes",
    "clustering_w", "modularity_q", "char_path_length", "small_worldness",
    "mean_betweenness", "n_nulls_used",
]


def _largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if G.number_of_edges() == 0:
        # Keep one node so downstream metrics are well-defined / return 0 or nan.
        H = nx.Graph()
        H.add_node(next(iter(G.nodes())))
        return H
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()


def _layer4_path(cfg: bg.Config) -> Path:
    return cfg.cache_dir / "layer4_nobump" / f"metrics_nobump_FD{cfg.fd_threshold}_kappa{cfg.kappa}.csv"


def compute_metrics_for_subject_lcc(graph_path: Path, cfg: bg.Config) -> dict:
    G = nx.read_graphml(str(graph_path), node_type=int)
    connected = (G.number_of_nodes() > 0) and nx.is_connected(G)
    H = _largest_connected_component(G)
    lcc_n = H.number_of_nodes()
    lcc_frac = (lcc_n / G.number_of_nodes()) if G.number_of_nodes() else float("nan")

    # Weighted metrics on LCC
    clust_w = cm.clustering_weighted(H)
    mod_q = cm.modularity_louvain(H, seed=cfg.seed)

    # Binary metrics on LCC
    Hb = cm._binarize(H)
    L = cm.char_path_length_binary(Hb) if (Hb.number_of_nodes() >= 2 and nx.is_connected(Hb)) else float("nan")
    sigma, n_nulls = cm.small_worldness(H, n_nulls=cfg.n_rewires, seed=cfg.seed)
    btw = cm.mean_betweenness_binary(Hb)

    return {
        "connected_at_requested_kappa": bool(connected),
        "lcc_n_nodes": int(lcc_n),
        "lcc_frac_nodes": float(lcc_frac),
        "clustering_w": float(clust_w),
        "modularity_q": float(mod_q),
        "char_path_length": float(L),
        "small_worldness": float(sigma),
        "mean_betweenness": float(btw),
        "n_nulls_used": int(n_nulls),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = bg.Config(fd_threshold=args.fd, kappa=args.kappa,
                    cache_dir=args.cache_dir, results_dir=args.results_dir)

    manifest_path = args.results_dir / f"manifest_nobump_FD{args.fd}_kappa{args.kappa}.csv"
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}\nRun: python3 supplement_fixA_build_nobump.py --fd {args.fd} --kappa {args.kappa}")

    manifest = pd.read_csv(manifest_path)
    included = manifest[manifest["included"]].copy()

    out_path = _layer4_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    logger.info("FD=%s κ=%s : %d included subjects (no-bump)", cfg.fd_threshold, cfg.kappa, len(included))
    for i, (_, row) in enumerate(included.iterrows(), 1):
        sid = str(row["subject_id"])
        graph_path = Path(row["layer3_cache"])
        if not graph_path.exists():
            logger.warning("missing graphml for %s: %s", sid, graph_path)
            continue
        t0 = time.time()
        metrics = compute_metrics_for_subject_lcc(graph_path, cfg)
        dt = time.time() - t0
        logger.info("[%d/%d] %s  LCC=%d  conn=%s  Cw=%.3f  Q=%.3f  L=%.3f  σ=%.3f  (%.1fs)",
                    i, len(included), sid, metrics["lcc_n_nodes"], metrics["connected_at_requested_kappa"],
                    metrics["clustering_w"], metrics["modularity_q"], metrics["char_path_length"],
                    metrics["small_worldness"], dt)
        rows.append({
            "subject_id": sid,
            "fd": cfg.fd_threshold,
            "kappa": cfg.kappa,
            **metrics,
        })

    df = pd.DataFrame(rows, columns=METRIC_COLS)
    df.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

