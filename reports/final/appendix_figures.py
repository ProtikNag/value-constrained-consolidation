#!/usr/bin/env python3
"""Vector redraws of the appendix result figures (paper Figures 11 to 14).

    .venv/bin/python reports/final/appendix_figures.py

* ``depth_matrices``      retention matrices across the rollout-depth sweep
* ``depth_perf_cost``     PERF against consolidation wall clock, one point per depth
* ``retention_matrix``    Setting B retention matrices, VCC beside CLEAR
* ``performance_by_band`` Setting B final score per distance band

Data and metrics come from the existing loaders in ``ablation/make_figures.py``
and ``gridworld_v2/make_figures.py``, so seed handling is unchanged. Everything is
drawn as vector patches (no embedded rasters), in Helvetica, at printed size.
Outputs go to ``reports/final/appendix/`` and the PDFs are copied into the paper.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "report"))
from paperfig import AC, TEXTWIDTH, save, style  # noqa: E402

OUT = HERE / "appendix"
PAPER = REPO / "paper" / "figures"


def load_module(path: Path, name: str):
    """Import a figure script by path, so its loaders can be reused."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


# The Setting B score scale of Figure 10, sampled for cells and colourbar alike.
CMIN, CMAX = -0.15, 1.0
CMAP = LinearSegmentedColormap.from_list("score", [
    (0.00, "#DC2626"), (0.09, "#F7CECE"), (0.16, "#F8F9FA"), (1.00, "#2563EB")])
NORM = Normalize(CMIN, CMAX)


def draw_matrix(ax, matrix: list[list[float]], outline_diagonal: bool = False) -> None:
    """Lower-triangular matrix as vector cells, row 1 at the top."""
    n = len(matrix)
    cells, colours = [], []
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            cells.append(Rectangle((c, r), 1, 1))
            colours.append(CMAP(NORM(min(max(value, CMIN), CMAX))))
    # Each cell is stroked in its own colour, which closes the hairline seams PDF
    # viewers draw between abutting vector rectangles.
    ax.add_collection(PatchCollection(cells, facecolors=colours, edgecolors=colours,
                                      linewidths=0.35))
    if outline_diagonal:
        # The diagonal is the score on the task being learned at that moment.
        xs = [v for k in range(n) for v in (k, k + 1)]
        ys = [v for k in range(n) for v in (k, k)]
        ax.step(np.r_[xs, n], np.r_[ys, n], where="post", color=AC["text"], lw=0.8)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def matrix_ticks(ax, n: int, ticks: list[int], label_y: bool) -> None:
    ax.set_xticks([t - 0.5 for t in ticks], [str(t) for t in ticks])
    ax.set_yticks([t - 0.5 for t in ticks], [str(t) for t in ticks] if label_y else [])


def colourbar(fig, cax) -> None:
    """Colourbar as 200 vector slices; fig.colorbar would embed a raster gradient."""
    edges = np.linspace(CMIN, CMAX, 201)
    cax.add_collection(PatchCollection(
        [Rectangle((0, lo), 1, hi - lo) for lo, hi in zip(edges, edges[1:])],
        facecolors=[CMAP(NORM((lo + hi) / 2)) for lo, hi in zip(edges, edges[1:])],
        edgecolors=[CMAP(NORM((lo + hi) / 2)) for lo, hi in zip(edges, edges[1:])],
        linewidths=0.35))
    cax.set_xlim(0, 1)
    cax.set_ylim(CMIN, CMAX)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks([-0.15, 0, 0.25, 0.5, 0.75, 1.0], ["−0.15", "0", "0.25", "0.50", "0.75", "1"])
    cax.tick_params(axis="y", length=2, width=0.6)
    for side in cax.spines.values():
        side.set_visible(False)
    cax.set_ylabel("Normalized score", labelpad=4)


# ── Figure 11: depth matrices ──────────────────────────────────────────────
def depth_matrices(abl) -> None:
    sweep = abl.load_data()["sweep"]
    fig = plt.figure(figsize=(TEXTWIDTH, 1.75))
    grid = fig.add_gridspec(1, 6, width_ratios=[1] * 5 + [0.06], wspace=0.18,
                            left=0.07, right=0.93, bottom=0.17, top=0.80)
    for i, run in enumerate(sweep):
        ax = fig.add_subplot(grid[0, i])
        draw_matrix(ax, run["matrix"], outline_diagonal=True)
        matrix_ticks(ax, 10, [1, 5, 10], label_y=(i == 0))
        ax.set_title(f"$L={run['depth']}$", pad=12)
        ax.text(0.5, 1.03, f"PERF {max(run['perf'], 0.0):.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color=AC["muted"])
        if i == 0:
            ax.set_ylabel("After training task")
        if i == 2:
            ax.set_xlabel("Task evaluated")
    colourbar(fig, fig.add_subplot(grid[0, 5]))
    save(fig, OUT / "depth_matrices", PAPER / "ablation" / "depth_matrices.pdf")


# ── Figure 12: score against cost ──────────────────────────────────────────
def depth_perf_cost(abl) -> None:
    sweep = abl.load_data()["sweep"]
    walls = [r["wall_min"] for r in sweep]
    perfs = [max(r["perf"], 0.0) for r in sweep]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.grid(axis="y")
    ax.plot(walls, perfs, color=AC["blue"], lw=1.6, alpha=0.45, zorder=2)
    ax.scatter(walls, perfs, s=34, color=AC["blue"], edgecolor="white", lw=1.0, zorder=3)
    for r, x, y in zip(sweep, walls, perfs):
        ax.annotate(f"$L={r['depth']}$", (x, y), xytext=(6, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=AC["text"])
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.07, 0.55)  # room under L=1, which sits at zero
    ax.set_xlabel("Consolidation wall clock (minutes)")
    ax.set_ylabel("PERF after ten tasks")
    save(fig, OUT / "depth_perf_cost", PAPER / "ablation" / "depth_perf_cost.pdf")


# ── Figure 13: Setting B retention matrices ────────────────────────────────
def retention_matrix(d: dict) -> None:
    pick = [m for m in d["methods"] if m["label"] in ("DUEL (ours)", "CLEAR")]
    fig = plt.figure(figsize=(TEXTWIDTH, 3.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.10,
                            left=0.07, right=0.92, bottom=0.11, top=0.88)
    for i, m in enumerate(pick):
        ax = fig.add_subplot(grid[0, i])
        draw_matrix(ax, m["matrix"])
        matrix_ticks(ax, 50, [1, 10, 20, 30, 40, 50], label_y=(i == 0))
        name = "VCC (ours)" if m["label"].startswith("DUEL") else m["label"]
        ax.set_title(name, pad=13, color=m["color"])
        mean, sd = m["headline"]["perf"], m["sd_perf"]
        ax.text(0.5, 1.025, f"PERF {mean:.2f} ± {sd:.2f}, {m['n_complete']} seeds",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                color=AC["muted"])
        ax.set_xlabel("Task evaluated")
        if i == 0:
            ax.set_ylabel("After training task")
    colourbar(fig, fig.add_subplot(grid[0, 2]))
    save(fig, OUT / "retention_matrix", PAPER / "gridworld" / "retention_matrix.pdf")


# ── Figure 14: final score per distance band ──────────────────────────────────
def performance_by_band(d: dict, gw) -> list[tuple[str, list[float]]]:
    bands = gw.BANDS
    tasks = d["tasks"]

    def band_mean(values: list[float]) -> list[float]:
        return [statistics.mean(v for v, t in zip(values, tasks) if tuple(t["band"]) == b)
                for b in bands]

    rows = []
    for m in d["methods"]:
        full = [s for s in m["per_seed"] if s["depth"] == gw.N_TASKS]
        per_seed = np.array([band_mean(s["final"]) for s in full])
        sd = per_seed.std(axis=0, ddof=1) if len(full) > 1 else np.zeros(len(bands))
        name = "VCC (ours)" if m["label"].startswith("DUEL") else m["label"]
        rows.append((name, m["color"], per_seed.mean(axis=0), sd))
    joint = band_mean(d["joint_scores"])

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.55))
    ax.grid(axis="y", zorder=0)
    width = 0.8 / len(rows)
    for j, (name, colour, mean, sd) in enumerate(rows):
        xs = np.arange(len(bands)) - 0.4 + width * (j + 0.5)
        ax.bar(xs, mean, width * 0.92, color=colour, zorder=2, label=name, lw=0)
        ax.errorbar(xs, mean, yerr=sd, fmt="none", ecolor=AC["axis"], elinewidth=0.6,
                    capsize=1.2, capthick=0.6, zorder=3)
    for b, value in enumerate(joint):
        ax.plot([b - 0.43, b + 0.43], [value, value], color=AC["text"], lw=0.9,
                ls=(0, (3, 2)), zorder=4)
    ax.axhline(0, color=AC["axis"], lw=0.7, zorder=3)
    ax.set_xticks(range(len(bands)), [f"{lo}–{hi}" for lo, hi in bands])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Shortest-path distance from start to goal (steps)")
    ax.set_ylabel("Final score")
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlim(-0.5, len(bands) - 0.5)
    handles = [Patch(color=c, label=n) for n, c, _, _ in rows]
    handles.append(Line2D([], [], color=AC["text"], lw=0.9, ls=(0, (3, 2)),
                          label="Joint training"))
    ax.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, handleheight=0.8, columnspacing=1.1, borderaxespad=0.3)
    save(fig, OUT / "performance_by_band", PAPER / "gridworld" / "performance_by_band.pdf")
    return [(n, list(np.round(m, 3))) for n, _, m, _ in rows] + [("joint", list(np.round(joint, 3)))]


def main() -> None:
    style()
    abl = load_module(HERE / "ablation" / "make_figures.py", "abl_figs")
    gw = load_module(HERE / "gridworld_v2" / "make_figures.py", "gw_figs")
    depth_matrices(abl)
    depth_perf_cost(abl)
    d = gw.load_data()
    for m in d["methods"]:
        print(f"  {m['label']:<14} seeds {m['n_complete']}  PERF {m['headline']['perf']:.3f}"
              f" ± {m['sd_perf']:.3f}")
    retention_matrix(d)
    for name, values in performance_by_band(d, gw):
        print(f"  band means {name:<14} {values}")


if __name__ == "__main__":
    main()
