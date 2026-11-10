"""Figure 6 of the paper: per-task constraint shortfall before and after consolidation.

Reads the dual trace of VCC seed 4 (Setting B) and compares the shortfall at the
start and end of each task's global phase: (a) the two sorted shortfall curves,
(b) a before/after tally of tasks inside and outside the band. Writes PDF, PNG
and SVG to this folder's pdf/, png/ and svg/.

The constraint is [V_k(pi(w_k)) - V_k(pi(w_[1,k]))]_+^2 <= eps with eps = 0.09, so a task satisfies it
when its shortfall is at or below sqrt(eps) = 0.30.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 4
TRACE = REPO / "reports" / "gridworld_mu0" / "run_logs" / f"seed{SEED}" / "progress.jsonl"
EPS_GAP = 0.09 ** 0.5
CLOSER = 0.1   # a task outside the band counts as moved closer past this drop

AC = dict(bg="#FFFFFF", blue="#2563EB", green="#059669", axis="#495057", grid="#E9ECEF",
          text="#212529", muted="#6C757D", faint="#ADB5BD")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AC["axis"],
    "axes.labelcolor": AC["text"],
    "axes.linewidth": 0.8,
    "xtick.color": AC["muted"],
    "ytick.color": AC["muted"],
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic", "mathtext.bf": "Helvetica:bold", "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
})


def task_shortfalls(path: Path) -> list[tuple[float, float]]:
    """(first, last) current-task shortfall of each task's global phase."""
    by_task: dict[int, list[tuple[int, float]]] = {}
    for line in path.open():
        if '"dual"' in line:
            row = json.loads(line)
            by_task.setdefault(row["task_idx"], []).append((row["iter"], row["shortfall_current"]))
    return [(min(s)[1], max(s)[1]) for s in by_task.values()]


def draw_curves(ax: plt.Axes, pairs: list[tuple[float, float]]) -> None:
    """Panel (a): before and after shortfalls, each curve sorted on its own.

    A quantile view of the whole set of tasks: the gap shows how far the set
    moves down, not how one task changes.
    """
    n = len(pairs)
    xs = range(1, n + 1)
    before = sorted((a for a, _ in pairs), reverse=True)
    after = sorted((b for _, b in pairs), reverse=True)

    for y in (0.5, 0.75, 1.0):
        ax.axhline(y, color=AC["grid"], lw=0.6, zorder=0)
    ax.axhspan(-0.02, EPS_GAP, color=AC["green"], alpha=0.10, lw=0, zorder=0)
    ax.axhline(EPS_GAP, color=AC["green"], lw=1.0, ls=(0, (4, 2.5)), zorder=1)
    ax.annotate(r"$\sqrt{\epsilon} = 0.30$", xy=(1.0, EPS_GAP), xycoords=("axes fraction", "data"),
                xytext=(3, 0), textcoords="offset points", color=AC["green"],
                fontsize=7.5, ha="left", va="center", annotation_clip=False)
    ax.text(3, 0.10, r"$\epsilon$-neighborhood", color=AC["green"], fontsize=7.5,
            fontweight="bold", ha="left", va="center")

    ax.fill_between(xs, after, before, color=AC["blue"], alpha=0.10, lw=0, zorder=1)
    ax.plot(xs, before, color=AC["muted"], lw=1.6, ls=(0, (2.5, 1.5)), zorder=2,
            label="before global phase")
    ax.plot(xs, after, color=AC["blue"], lw=2.0, zorder=3, label="after global phase")

    ax.set_xlim(1, n)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1, 10, 20, 30, 40, n])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.tick_params(length=2.5, labelsize=7.5, pad=2)
    ax.set_xlabel("Tasks, each curve sorted by shortfall", fontsize=8, labelpad=2)
    ax.set_ylabel("Shortfall\n(specialist value − global value)", fontsize=8,
                  labelpad=3, linespacing=1.15)
    ax.legend(loc="upper right", frameon=False, fontsize=7.3, handletextpad=0.4,
              handlelength=1.8, borderaxespad=0.2, labelspacing=0.3, labelcolor=AC["text"])


def draw_bars(ax: plt.Axes, pairs: list[tuple[float, float]]) -> None:
    """Panel (b): two stacked bars, before and after, one segment per outcome.

    Segments are laid out in the same order in both bars, so the tasks that enter
    the band sit directly under the grey they started in.
    """
    stayed_in = sum(a <= EPS_GAP for a, _ in pairs)
    moved_in = sum(a > EPS_GAP >= b for a, b in pairs)
    still_out = [(a, b) for a, b in pairs if a > EPS_GAP and b > EPS_GAP]
    closer = sum(a - b > CLOSER for a, b in still_out)
    unmoved = len(still_out) - closer
    n = len(pairs)

    C_IN, C_CLOSER, C_OUT = AC["green"], "#93B4F5", "#CED4DA"
    before = [(stayed_in, C_IN, f"{stayed_in}", AC["bg"]), (n - stayed_in, C_OUT, f"{n - stayed_in} outside", AC["text"])]
    after = [(stayed_in, C_IN, f"{stayed_in}", AC["bg"]), (moved_in, C_IN, f"{moved_in} moved inside", AC["bg"]),
             (closer, C_CLOSER, f"{closer}", AC["text"]), (unmoved, C_OUT, f"{unmoved}", AC["text"])]
    for y, segments in ((1, before), (0, after)):
        left = 0
        for width, color, label, text_color in segments:
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor=AC["bg"], lw=1.0)
            ax.text(left + width / 2, y, label, color=text_color, fontsize=7.5,
                    ha="center", va="center")
            left += width

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Before\nglobal phase", "After\nglobal phase"], fontsize=8,
                       color=AC["text"], linespacing=1.1)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.42, 1.42)
    ax.set_xticks([0, 10, 20, 30, 40, n])
    ax.tick_params(axis="x", length=2.5, labelsize=7.5, pad=2)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Tasks", fontsize=8, labelpad=1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_IN, C_CLOSER, C_OUT)]
    labels = [r"inside $\epsilon$", "outside, gap reduced", "outside"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.33, 1.04), ncol=3,
              frameon=False, fontsize=7, handlelength=1.0, handleheight=0.8,
              handletextpad=0.35, columnspacing=1.0, borderaxespad=0.2,
              labelcolor=AC["text"])


def draw(pairs: list[tuple[float, float]]) -> plt.Figure:
    """Panel (a) the sorted shortfall curves, panel (b) the before/after tally, side by side."""
    # Sized for the full \textwidth (6.5 in), so fonts print at their nominal size.
    fig = plt.figure(figsize=(6.5, 2.3))
    ax_a = fig.add_axes([0.085, 0.175, 0.40, 0.78])
    ax_b = fig.add_axes([0.665, 0.26, 0.325, 0.50])
    draw_curves(ax_a, pairs)
    draw_bars(ax_b, pairs)
    for x, tag in ((0.005, "(a)"), (0.565, "(b)")):
        fig.text(x, 0.985, tag, fontsize=8.5, fontweight="bold", ha="left", va="top",
                 color=AC["text"])
    return fig


def main() -> None:
    fig = draw(task_shortfalls(TRACE))
    for ext in ("pdf", "png", "svg"):
        (HERE / ext).mkdir(exist_ok=True)
        fig.savefig(HERE / ext / f"constraint_satisfaction_seed{SEED}.{ext}")
    print("wrote", HERE / "pdf" / f"constraint_satisfaction_seed{SEED}.pdf")


if __name__ == "__main__":
    main()
