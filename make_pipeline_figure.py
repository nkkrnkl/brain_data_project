"""Phase-0 deliverable: methods-section pipeline block diagram + methods table CSV.

Outputs:
    results/figures/fig0_pipeline.png
    results/methods_table.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import build_graphs as bg


def make_pipeline_figure(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Two rows of 6 boxes each (last row has 5 boxes + a spacer)
    boxes_top = [
        ("Raw 4D BOLD\n(MNI152NLin2009cAsym)\n168 vols, TR=2s", "#dde7f0"),
        ("Schaefer-100\nparcellation\n(MNI atlas)", "#dde7f0"),
        ("ROI timeseries\n(T × 100)", "#cfe2cb"),
        ("FD-scrub +\nFriston-24 + WM + CSF\n+ DCT (OLS regress)", "#cfe2cb"),
        ("Clean ROI TS\n[Layer 1 cache]", "#a8d3a0"),
        ("Pearson r → Fisher z\nzero negatives + diag", "#f7d8a8"),
    ]
    boxes_bot = [
        ("Z-matrix\n[Layer 2 cache]", "#f4be7c"),
        ("Top-κ density\n+ per-subject\nκ-bump if disconnected", "#f4cccc"),
        ("Connected graph\n[Layer 3 cache]\n(graphml)", "#e89a9a"),
        ("Maslov–Sneppen\nbinary null × 100\n(degree-preserving)", "#cdb4db"),
        ("Graph metrics\nC · Q · L · σ\n· betweenness", "#cdb4db"),
        ("Per-subject metrics\n[Layer 4 cache]\n(CSV per FD,κ)", "#a89cd0"),
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box_w, box_h = 0.88, 1.0

    def draw_row(boxes, y, centers):
        for i, (label, color) in enumerate(boxes):
            x = i + (1 - box_w) / 2
            cx = x + box_w / 2
            centers.append((cx, y + box_h / 2))
            rect = mpatches.FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=1.2, edgecolor="#333", facecolor=color,
            )
            ax.add_patch(rect)
            ax.text(cx, y + box_h / 2, label, ha="center", va="center", fontsize=9)

    centers_top, centers_bot = [], []
    draw_row(boxes_top, 2.5, centers_top)
    draw_row(boxes_bot, 0.9, centers_bot)

    def arrow(p0, p1):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # Horizontal arrows top row
    for i in range(len(centers_top) - 1):
        cx0, cy0 = centers_top[i]
        cx1, cy1 = centers_top[i + 1]
        arrow((cx0 + box_w / 2, cy0), (cx1 - box_w / 2, cy1))
    # Wrap arrow: top-right → bottom-left
    cx_last_top, cy_last_top = centers_top[-1]
    cx_first_bot, cy_first_bot = centers_bot[0]
    arrow((cx_last_top, cy_last_top - box_h / 2), (cx_first_bot, cy_first_bot + box_h / 2))
    # Horizontal arrows bottom row
    for i in range(len(centers_bot) - 1):
        cx0, cy0 = centers_bot[i]
        cx1, cy1 = centers_bot[i + 1]
        arrow((cx0 + box_w / 2, cy0), (cx1 - box_w / 2, cy1))

    ax.text(0.0, 0.18,
            "Locked: 100 ROIs · Friston-24 + WM + CSF + DCT · Fisher-z neg→0 · "
            "100 Maslov–Sneppen binary nulls · seed=20260507\n"
            "Swept: FD ∈ {0.3, 0.5, 0.9} mm    κ ∈ {0.05, 0.10, 0.15, 0.20, 0.25}    "
            "Primary: FD=0.5, κ=0.10",
            ha="left", va="bottom", fontsize=9, style="italic", color="#444",
            transform=ax.transAxes)

    fig.suptitle("Pipeline — Richardson 2018 developmental fMRI → undirected weighted FC graphs",
                 fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    fig_path = make_pipeline_figure(Path("results/figures/fig0_pipeline.png"))
    table_path = bg.dump_methods_table(Path("results/methods_table.csv"))
    print(f"  pipeline diagram → {fig_path}")
    print(f"  methods table    → {table_path}")


if __name__ == "__main__":
    main()
