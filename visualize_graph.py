from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import build_graphs as bg


def load_graph(path: Path) -> nx.Graph:
    return nx.read_graphml(str(path), node_type=int)


def load_zmatrix_for_subject(subject_id: str, cfg: bg.Config) -> np.ndarray:
    p = (cfg.cache_dir / "layer2"
         / f"{subject_id}_FD{cfg.fd_threshold}_conf{cfg.confound_hash()}_edge{cfg.edge_measure_hash()}.npy")
    return np.load(p)


def schaefer_mni_centroids(cfg: bg.Config) -> np.ndarray:
    """(n_rois, 3) MNI coords for Schaefer-100 ROIs via nilearn."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.plotting import find_parcellation_cut_coords
    atlas = fetch_atlas_schaefer_2018(n_rois=cfg.n_rois, yeo_networks=7)
    return np.asarray(find_parcellation_cut_coords(labels_img=atlas.maps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id")
    ap.add_argument("--fd", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = bg.Config(fd_threshold=args.fd, kappa=args.kappa, cache_dir=args.cache_dir)
    null_h = cfg.null_model_hash()
    g_path = (cfg.cache_dir / "layer3"
              / f"{args.subject_id}_FD{args.fd}_kappa{args.kappa}_null{null_h}.graphml")
    if not g_path.exists():
        sys.exit(f"No layer3 cache for {args.subject_id} at {g_path}")

    G = load_graph(g_path)
    Z = load_zmatrix_for_subject(args.subject_id, cfg)
    n = G.number_of_nodes()
    print(f"{args.subject_id}: {n} nodes, {G.number_of_edges()} edges, "
          f"density={nx.density(G):.3f}, mean_w={np.mean([d['weight'] for _,_,d in G.edges(data=True)]):.3f}")

    Zg = np.zeros_like(Z)
    for u, v, d in G.edges(data=True):
        Zg[int(u), int(v)] = d["weight"]
        Zg[int(v), int(u)] = d["weight"]

    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(Zg, cmap="viridis", aspect="equal")
    ax1.set_title(f"Thresholded Z (κ={args.kappa})")
    ax1.set_xlabel("ROI"); ax1.set_ylabel("ROI")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Fisher z")

    ax2 = fig.add_subplot(1, 3, 2)
    pos = nx.spring_layout(G, seed=cfg.seed, weight="weight", k=0.4, iterations=80)
    deg = np.array([G.degree(n) for n in G.nodes()])
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.25,
                           width=0.2 + 1.5 * (weights - weights.min()) / max(float(np.ptp(weights)), 1e-9))
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=10 + 2 * deg, node_color=deg,
                           cmap="plasma", linewidths=0)
    ax2.set_title("Spring layout (size/color = degree)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    coords = schaefer_mni_centroids(cfg)
    pos_anat = {i: (coords[i, 0], coords[i, 1]) for i in range(coords.shape[0])}
    nx.draw_networkx_edges(G, pos_anat, ax=ax3, alpha=0.2,
                           width=0.2 + 1.5 * (weights - weights.min()) / max(float(np.ptp(weights)), 1e-9))
    nx.draw_networkx_nodes(G, pos_anat, ax=ax3, node_size=12 + 2 * deg,
                           node_color=deg, cmap="plasma", linewidths=0)
    ax3.set_title("Anatomical layout (axial, MNI x/y)")
    ax3.set_xlabel("MNI x (mm)"); ax3.set_ylabel("MNI y (mm)")
    ax3.set_aspect("equal")
    ax3.grid(alpha=0.2)

    fig.suptitle(f"{args.subject_id}  ·  FD={args.fd}  ·  κ={args.kappa}  ·  "
                 f"|E|={G.number_of_edges()}  ·  density={nx.density(G):.3f}")
    fig.tight_layout()

    out = args.out or Path("results") / f"graph_{args.subject_id}_FD{args.fd}_kappa{args.kappa}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
