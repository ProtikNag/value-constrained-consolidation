#!/usr/bin/env python3
"""Setting C retention matrices (paper Figure 7), mean +- sd over seeds.

Reads ``reports/setting_c_reversed/summary.json``. Each cell is raw / joint
ceiling (random floor not subtracted, the paper's Setting C normalization),
averaged over seeds, with the population sd across seeds beneath it. A method
with one seed shows +-0%: CLEAR is seed0 only until seeds 1 and 2 land.

Colour scale and pivot are the ones in ``make_figures.py``, clamped at the
ceiling, so this is a drop-in for the single-seed figure.

Usage::

    python reports/final/atari_reversed/make_retention_multiseed.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from make_figures import CEILING_TICKS, CEILING_TICKTEXT, HERE, REPO, RETENTION_SCALE

SUMMARY = REPO / "reports" / "setting_c_reversed" / "summary.json"
PAPER_PDF = REPO / "paper" / "figures" / "atari" / "retention.pdf"
STEM = "retention_multiseed"
PANELS = [("VCC", "VCC (ours)"), ("CLEAR", "CLEAR"),
          ("CKA-RL", "CKA-RL"), ("CompoNet", "CompoNet")]
LABELS = ["SpaceInv", "Boxing", "Breakout", "Pong", "Q*bert"]
INK, INK_SD = "#212529", "#495057"
LIGHTEN = 0.40   # blend of the retention ramp toward white

CMAP = LinearSegmentedColormap.from_list("retention", [
    (p, tuple(c + (1 - c) * LIGHTEN for c in mpl.colors.to_rgb(h))) for p, h in RETENTION_SCALE])

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "savefig.dpi": 300, "savefig.bbox": "tight", "svg.fonttype": "none",
})


def retention(summary: dict, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Seed mean and population sd of raw / joint, lower triangle, NaN above."""
    joint = np.array([summary["joint_ceilings"][g] for g in summary["order"]])
    stack = []
    for rows in summary["methods"][method]["raw_matrices"].values():
        grid = np.full((5, 5), np.nan)
        for i, row in enumerate(rows):
            grid[i, : len(row)] = np.array(row, dtype=float) / joint[: len(row)]
        stack.append(grid)
    stack = np.array(stack)
    return stack.mean(0), stack.std(0)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    fig.subplots_adjust(wspace=0.05, hspace=0.32, right=0.84)

    for ax, (key, title) in zip(axes.flat, PANELS):
        mean, sd = retention(summary, key)
        for i in range(5):
            for j in range(i + 1):
                shade = min(max(mean[i, j], 0.0), 1.0)
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94,
                                           color=CMAP(shade), lw=0))
                # One ink for every cell: the ramp is lightened enough that dark
                # text clears 5:1 contrast even on its saturated ends.
                ax.text(j + 0.5, i + 0.42, f"{mean[i, j] * 100:.0f}", ha="center",
                        va="center", fontsize=13, fontweight="bold", color=INK)
                ax.text(j + 0.5, i + 0.75, f"±{sd[i, j] * 100:.0f}", ha="center",
                        va="center", fontsize=8.5, color=INK_SD)
        ax.set_xlim(0, 5)
        ax.set_ylim(5, 0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=17, fontweight="bold", color="#212529")
        ax.set_yticks(np.arange(5) + 0.5, LABELS, fontsize=12)
        ax.set_xticks(np.arange(5) + 0.5, LABELS, fontsize=12, rotation=30, ha="right")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        if ax in axes[0]:
            ax.set_xticklabels([])

    cax = fig.add_axes([0.86, 0.30, 0.018, 0.40])
    # Colourbar as vector slices; fig.colorbar would embed a raster gradient.
    edges = np.linspace(0, 1, 201)
    shades = [CMAP((lo + hi) / 2) for lo, hi in zip(edges, edges[1:])]
    for (lo, hi), shade in zip(zip(edges, edges[1:]), shades):
        cax.add_patch(plt.Rectangle((0, lo), 1, hi - lo, facecolor=shade,
                                    edgecolor=shade, lw=0.3))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks(CEILING_TICKS, CEILING_TICKTEXT, fontsize=12)
    for side in cax.spines.values():
        side.set_visible(False)
    cax.set_ylabel("Retention (% of joint ceiling)", fontsize=13)
    fig.supxlabel("Evaluated on  (learning order $\\rightarrow$)", fontsize=16, fontweight="bold",
                  x=0.47, y=0.0)

    for ext in ("pdf", "png", "svg"):
        folder = HERE / ("svg" if ext == "svg" else "png")
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / f"{STEM}.{ext}")
    shutil.copyfile(HERE / "png" / f"{STEM}.pdf", PAPER_PDF)
    print(f"wrote {STEM}.{{pdf,png,svg}} and {PAPER_PDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
