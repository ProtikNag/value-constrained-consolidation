#!/usr/bin/env python3
"""Ablation figures: the constraint on/off contrast and the rollout-depth sweep.

Two experiments on a 10-task slice of GridWorld-50, both single seed and both at
a reduced budget. They are directional, not full-scale results, and the caveat
is printed on every figure.

THE TWO EXPERIMENTS ARE NOT COMPARABLE TO EACH OTHER. They used different local
budgets, so an absolute PERF from the sweep must never be read against an
absolute PERF from the constraint contrast. Nothing here puts a value from one
experiment on the same axis as a value from the other.

Experiment 1, the constraint (``constraint_ablation/``). Two runs identical
except ``duals.max_value``: 5.0 leaves the multiplier free, and 0.0 pins it to
zero and so removes the current-task constraint entirely. With the constraint
off, the consolidation objective is the past-task sum alone and carries no term
for the task in front of the policy.

Experiment 2, the rollout depth (``gns_sweep/``). Five runs identical except
``global_n_steps``, the PPO rollout depth inside the consolidation phase. The
value estimator stays Monte-Carlo and ``window_H`` is untouched, so this sweep
is about rollout depth and not about the windowed value estimate.

Every metric is recomputed here from each run's ``eval_matrix.json`` rather than
read from ``summary.json``, and the two agree to about 0.001.

Usage::

    python reports/final/ablation/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, height_for, hex_to_rgba, install_template,
)

RUNS = REPO / "reports" / "gridworld_ablations"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 10

FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5

# Palette roles, held fixed across the set so a colour means one thing here and
# in the Setting B figures: blue is the configuration we advocate, amber is the
# one it is measured against, green is a reference level, red is failure.
C_ON = AC["blue"]          # constraint on, and the deepest rollout
C_OFF = AC["amber"]        # constraint off
C_REF = AC["text_faint"]   # score on arrival, a neutral reference
C_RULE = AC["green"]       # the benchmark's success bar

THRESHOLD = 0.6            # the benchmark's own per-task success bar
DEPTHS = [1, 10, 20, 40, 80]
GLOBAL_ITERS = 1800        # consolidation iterations, identical across the sweep

CAVEAT = ("Single seed, ten tasks, reduced budget. Directional rather than "
          "full-scale, and read within this experiment only.")


# ── Data ────────────────────────────────────────────────────────────────────
def metrics(matrix: list[list[float]]) -> dict:
    """PERF, BWT and forgetting from one lower-triangular evaluation matrix.

    PERF is the mean of the final row. BWT compares each earlier task's final
    score against its score on arrival. Forgetting is the mean drop from each
    task's best recorded score to its final one.
    """
    last = len(matrix) - 1
    final = list(matrix[last])
    diagonal = [matrix[k][k] for k in range(last + 1)]
    return dict(
        matrix=matrix, final=final, diagonal=diagonal,
        perf=sum(final) / len(final),
        bwt=sum(final[i] - diagonal[i] for i in range(last)) / last,
        forgetting=sum(max(matrix[k][i] for k in range(i, last + 1)) - final[i]
                       for i in range(last)) / last)


def load_run(subdir: str, name: str, label: str, color: str) -> dict:
    """One run's matrix, metrics, label and consolidation-phase wall clock."""
    base = RUNS / subdir / name
    run = metrics(json.loads((base / "eval_matrix.json").read_text()))
    usage = json.loads((base / "resource_usage.json").read_text())
    wall = sum(phase.get("wall_s") or 0.0
               for task in usage.values() if isinstance(task, dict)
               for key, phase in task.items()
               if key == "global" and isinstance(phase, dict))
    return run | dict(label=label, color=color, wall_min=wall / 60.0)


def load_data() -> dict:
    summary = json.loads((RUNS / "summary.json").read_text())
    constraint = [
        load_run("constraint_ablation", "con", "Constraint on", C_ON),
        load_run("constraint_ablation", "mu0", "Constraint off (μ = 0)", C_OFF),
    ]
    # One hue ramp over the sweep, pale at depth 1 and full blue at depth 80, so
    # the panels read as one ordered family rather than five unrelated series.
    ramp = ["#BFD3F7", "#8FB2F1", "#5F91EB", "#3F79E4", C_ON]
    sweep = [load_run("gns_sweep", f"gns{d}", f"depth {d}", c)
             for d, c in zip(DEPTHS, ramp)]
    for run, depth in zip(sweep, DEPTHS):
        run["depth"] = depth
        run["steps_per_iter"] = summary["gns_sweep"][str(depth)]["collect_steps_per_iter"]
        # All five runs spend the same 1800 consolidation iterations, so total
        # collected steps is the per-iteration figure times that count.
        run["total_steps"] = run["steps_per_iter"] * GLOBAL_ITERS
        # The pushed wall clock is authoritative: the runs were executed back to
        # back on one node, which the per-phase records cannot attest to.
        run["wall_min"] = summary["gns_sweep"][str(depth)]["global_phase_wall_min"]
    return dict(constraint=constraint, sweep=sweep, summary=summary)


# ── Shared helpers ──────────────────────────────────────────────────────────
PAPER_MODE = bool(os.environ.get("FIGS_FOR_PAPER"))
AXIS_TITLE_PX = 56


def add_footnote(fig: go.Figure, text: str, left: int, extra: int = 0) -> None:
    """Footnote under the axis, or bare bottom margin when built for the paper.

    In the paper the LaTeX caption carries this text, so printing it inside the
    artwork would duplicate it and then be cropped at typesetting.
    """
    if PAPER_MODE:
        fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + 10))
        return
    lines = text.count("<br>") + 1
    fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + 15 * lines + 12))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="top",
        yshift=-(AXIS_TITLE_PX + extra), showarrow=False, align="left",
        text=text, font=dict(family=FONT_UI, size=9.5, color=AC["text_muted"]))


# ONE scale, sampled for the cells and for the colourbar, so the legend cannot
# disagree with the matrix. It is the Setting B scale unchanged.
CMIN, CMAX = -0.15, 1.0
MATRIX_SCALE = [
    (0.00, (220, 38, 38)),     # AC red     -- clearly below the floor
    (0.09, (247, 206, 206)),   # pale red   -- at the random floor (score 0)
    (0.16, (248, 249, 250)),   # AC surface
    (1.00, (37, 99, 235)),     # AC blue    -- solved
]


def sample_scale(value: float) -> str:
    """Colour for a score, interpolated in sRGB along MATRIX_SCALE."""
    t = (min(max(value, CMIN), CMAX) - CMIN) / (CMAX - CMIN)
    for (p0, c0), (p1, c1) in zip(MATRIX_SCALE, MATRIX_SCALE[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            rgb = [c0[i] + (c1[i] - c0[i]) * f for i in range(3)]
            return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
    return f"rgb{MATRIX_SCALE[-1][1]}"


def matrix_shapes(runs: list[dict], outline_diagonal: bool = True) -> list[dict]:
    """Cell rectangles for every panel, as one list.

    add_shape is quadratic in the number of shapes already on the figure, so
    every rectangle in a figure is assigned in a single update_layout call.
    """
    shapes: list[dict] = []
    for col, run in enumerate(runs, start=1):
        axes = dict(xref="x" if col == 1 else f"x{col}",
                    yref="y" if col == 1 else f"y{col}")
        for row_idx, row in enumerate(run["matrix"]):
            for col_idx, value in enumerate(row):
                shapes.append(dict(
                    type="rect", x0=col_idx, x1=col_idx + 1,
                    y0=row_idx, y1=row_idx + 1, fillcolor=sample_scale(value),
                    line=dict(width=0.35, color=AC["bg"]), layer="below", **axes))
        if outline_diagonal:
            # The diagonal is the read that matters here: it is the score on the
            # task the policy is currently learning.
            for k in range(len(run["matrix"])):
                shapes.append(dict(
                    type="rect", x0=k, x1=k + 1, y0=k, y1=k + 1,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(width=1.6, color=AC["text_primary"]), **axes))
    return shapes


def style_matrix_axes(fig: go.Figure, n_cols: int, y_title: str) -> None:
    ticks = list(range(N_TASKS))
    for col in range(1, n_cols + 1):
        fig.update_xaxes(
            title=dict(text="Task evaluated", font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[0, N_TASKS], tickmode="array",
            tickvals=[t + 0.5 for t in ticks], ticktext=[str(t + 1) for t in ticks],
            constrain="domain", tickfont=dict(family=FONT_MONO, size=FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)
        fig.update_yaxes(
            title=dict(text=y_title if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[N_TASKS, 0], tickmode="array",
            tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) if col == 1 else "" for t in ticks],
            constrain="domain", scaleanchor="x" if col == 1 else f"x{col}",
            scaleratio=1, tickfont=dict(family=FONT_MONO, size=FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)


def add_colorbar(fig: go.Figure) -> None:
    """Colourbar on an invisible marker trace, so the SVG carries a real gradient."""
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(color=[CMIN], colorscale=[[p, f"rgb{c}"] for p, c in MATRIX_SCALE],
                    cmin=CMIN, cmax=CMAX, showscale=True, opacity=0,
                    colorbar=dict(title=dict(text="Score", side="right",
                                             font=dict(family=FONT_UI, size=FS_AXIS)),
                                  tickvals=[-0.15, 0.0, 0.25, 0.5, 0.75, 1.0],
                                  tickfont=dict(family=FONT_MONO, size=FS_TICK),
                                  thickness=12, len=0.78, outlinewidth=0, x=1.015))),
        row=1, col=1)


def lollipop(fig: go.Figure, x: list, y: list, color: str, name: str,
             width: float = 3.0, size: float = 11.0,
             row: int | None = None, col: int | None = None) -> None:
    """A thin stem from zero with a white-ringed dot at the value.

    Below about six categories a thick bar is mostly ink, and the quiet stem
    leaves room for a dashed reference rule to read as a threshold. row and col
    belong to add_trace rather than to the trace itself, so they are named
    arguments here instead of travelling in a kwargs bag.
    """
    where = {k: v for k, v in (("row", row), ("col", col)) if v is not None}
    for xi, yi in zip(x, y):
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[0, yi], mode="lines", hoverinfo="skip",
            line=dict(color=hex_to_rgba(color, 0.40), width=width),
            showlegend=False), **where)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=name,
        marker=dict(color=color, size=size, line=dict(color=AC["bg"], width=1.8)),
        hoverinfo="skip", showlegend=False), **where)


# ── Figure 1: the two retention matrices ────────────────────────────────────
def fig_constraint_matrices(d: dict) -> go.Figure:
    """Constraint on beside constraint off, on one shared scale.

    The outlined diagonal is the whole argument. With the constraint on, the
    policy reaches a usable score on the task in front of it. With the
    constraint off, only the first task is ever learned on arrival, because the
    objective then holds no term for the current task at all.
    """
    runs = d["constraint"]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09,
        subplot_titles=[f"{r['label']} — PERF {r['perf']:.3f}" for r in runs])
    style_matrix_axes(fig, 2, "After training task")
    add_colorbar(fig)
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=66, r=100, t=48, b=66),
                      shapes=matrix_shapes(runs))
    add_footnote(fig, (
        "Row <i>k</i> is the score on every task seen so far, measured after task "
        "<i>k</i> finished training. Lower-triangular, because a task cannot be<br>"
        "&nbsp;&nbsp;&nbsp;evaluated before it arrives. Red is at or below the random "
        "floor of 0.000 and blue is solved. Both panels share one scale.<br>"
        "&nbsp;&nbsp;&nbsp;<b>The outlined diagonal</b> is the score on the task the "
        "policy is learning at that moment. With the constraint off it is near zero<br>"
        "&nbsp;&nbsp;&nbsp;for every task after the first, because the consolidation "
        "objective carries no term for the current task.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=66, extra=4)
    return fig


def fig_depth_perf_and_cost(d: dict) -> go.Figure:
    """What depth buys and what it costs, on one panel with two y axes.

    Minutes and a normalized score do not share a range, so the cost series
    carries its own axis on the right. Each axis title takes the colour of the
    series it belongs to, which is what lets one panel hold both without a
    legend to decode.
    """
    sweep = d["sweep"]
    depths = [r["depth"] for r in sweep]
    perfs = [r["perf"] for r in sweep]
    walls = [r["wall_min"] for r in sweep]
    C_COST = AC["sienna"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for values, color, fmt, secondary, shift in (
            (walls, C_COST, ".0f", True, -20), (perfs, C_ON, ".2f", False, 20)):
        fig.add_trace(go.Scatter(
            x=depths, y=values, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=hex_to_rgba(color, 0.50), width=2.2,
                      dash="dot" if secondary else "solid")),
            secondary_y=secondary)
        fig.add_trace(go.Scatter(
            x=depths, y=values, mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(color=color, size=11, symbol="diamond" if secondary else "circle",
                        line=dict(color=AC["bg"], width=1.7))),
            secondary_y=secondary)
        for depth, value in zip(depths, values):
            # The lowest cost point would otherwise drop its label below the
            # axis line and onto the first tick, so it goes above instead.
            below = shift < 0 and value > min(values)
            fig.add_annotation(
                x=depth, y=value, text=f"<b>{value:{fmt}}</b>", showarrow=False,
                yshift=shift if below else abs(shift),
                xshift=0 if below else 16,
                yref="y2" if secondary else "y",
                font=dict(family=FONT_MONO, size=10.5, color=color))
    # Direct labels at the right end, in place of a legend.
    fig.add_annotation(
        x=depths[-1], y=perfs[-1], text="<b>PERF</b>", showarrow=False,
        xanchor="left", xshift=10, yshift=-10,
        font=dict(family=FONT_UI, size=12, color=C_ON))
    fig.add_annotation(
        x=depths[-1], y=walls[-1], text="<b>wall clock</b>", showarrow=False,
        xanchor="left", xshift=10, yref="y2",
        font=dict(family=FONT_UI, size=12, color=C_COST))
    fig.update_xaxes(
        title=dict(text="Consolidation rollout depth (PPO steps per collect)",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        tickmode="array", tickvals=depths, ticktext=[str(x) for x in depths],
        range=[-4, 96], tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(
        title=dict(text="PERF after ten tasks",
                   font=dict(family=FONT_UI, size=FS_AXIS, color=C_ON)),
        range=[-0.03, 0.60], dtick=0.1, tickformat=".2f",
        zeroline=True, zerolinewidth=1.1, zerolinecolor=AC["axis"],
        tickfont=dict(family=FONT_MONO, size=FS_TICK, color=C_ON),
        secondary_y=False)
    fig.update_yaxes(
        title=dict(text="Consolidation wall clock (minutes)",
                   font=dict(family=FONT_UI, size=FS_AXIS, color=C_COST)),
        range=[0, 110], dtick=20, showgrid=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK, color=C_COST),
        secondary_y=True)
    fig.update_layout(title=None, margin=dict(l=66, r=112, t=28, b=66))
    add_footnote(fig, (
        "Two y axes, because minutes and a normalized score do not share a "
        "range. Each axis title carries the colour of its series, and the<br>"
        "&nbsp;&nbsp;&nbsp;right-hand range is set wider than the data so the two "
        "curves do not lie on top of one another. <b>Their near-parallel shape is "
        "the result:</b><br>&nbsp;&nbsp;&nbsp;over this range the score rises "
        "about in step with the time it costs, so the vertical gap between them "
        "carries no meaning.<br>"
        "&nbsp;&nbsp;&nbsp;The runs were executed one after another on a single "
        "node, so the wall clock is comparable here. Depth 80 reaches 0.48 for "
        "ten times<br>&nbsp;&nbsp;&nbsp;the wall clock of depth 1, which reaches "
        "nothing at all. No ratio against depth 1 is quoted, because its PERF is "
        "zero.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=66)
    return fig


# ── Figure 8: the five matrices ─────────────────────────────────────────────
def fig_depth_matrices(d: dict) -> go.Figure:
    """One retention matrix per depth, on a shared scale.

    This locates what depth buys. At depth 1 the diagonal never lifts off the
    floor, so the policy acquires nothing. The diagonal fills in as depth grows.
    """
    sweep = d["sweep"]
    fig = make_subplots(
        rows=1, cols=len(sweep), horizontal_spacing=0.022,
        subplot_titles=[f"depth {r['depth']}<br>PERF {r['perf']:.3f}" for r in sweep])
    ticks = [0, 4, 9]
    for col, run in enumerate(sweep, start=1):
        fig.update_xaxes(
            title=dict(text="Task evaluated" if col == 3 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[0, N_TASKS], tickmode="array",
            tickvals=[t + 0.5 for t in ticks], ticktext=[str(t + 1) for t in ticks],
            constrain="domain", tickfont=dict(family=FONT_MONO, size=FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)
        fig.update_yaxes(
            title=dict(text="After training task" if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[N_TASKS, 0], tickmode="array",
            tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) if col == 1 else "" for t in ticks],
            constrain="domain", scaleanchor="x" if col == 1 else f"x{col}",
            scaleratio=1, tickfont=dict(family=FONT_MONO, size=FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)
    add_colorbar(fig)
    for ann in fig.layout.annotations[:len(sweep)]:
        ann.font = dict(family=FONT_UI, size=10.5, color=AC["text_primary"])
        ann.y = (ann.y or 1.0) + 0.045
    fig.update_layout(title=None, margin=dict(l=58, r=86, t=62, b=56),
                      shapes=matrix_shapes(sweep))
    add_footnote(fig, (
        "Row <i>k</i> is the score on every task seen so far, measured after task "
        "<i>k</i> finished training. One shared scale across all five panels.<br>"
        "&nbsp;&nbsp;&nbsp;<b>The outlined diagonal</b> is the score on the task "
        "being learned. At depth 1 it never leaves the floor, and it fills in as "
        "depth grows,<br>&nbsp;&nbsp;&nbsp;so what depth buys here is acquisition "
        "rather than retention.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=58, extra=-20)
    return fig


# ── Figure: retention curve, constraint on against off ──────────────────────
def fig_constraint_retention_curve(d: dict) -> go.Figure:
    """Mean score over the tasks seen so far, as the sequence advances.

    The same read as Figure~3 of the paper, with no uncertainty band, because
    each line is one run. The two curves separate at task 2 and never meet
    again, which is the sequence-level form of the matrix diagonals.
    """
    runs = d["constraint"]
    fig = go.Figure()
    for run in runs:
        xs = list(range(1, N_TASKS + 1))
        # Row k of the matrix, averaged over the tasks it covers.
        ys = [sum(run["matrix"][k]) / len(run["matrix"][k]) for k in range(N_TASKS)]
        run["curve"] = ys
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", hoverinfo="skip", showlegend=False,
            line=dict(color=run["color"], width=2.4),
            marker=dict(color=run["color"], size=7,
                        line=dict(color=AC["bg"], width=1.4))))
        fig.add_annotation(
            x=N_TASKS, y=ys[-1], xanchor="left", xshift=10, yanchor="middle",
            text=f"<b>{run['label']}</b><br>PERF {run['perf']:.3f}",
            showarrow=False, align="left",
            font=dict(family=FONT_UI, size=12, color=run["color"]))
    fig.update_xaxes(
        title=dict(text="Tasks trained", font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.6, N_TASKS + 0.4], dtick=1,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(
        title=dict(text="Mean score over tasks seen",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.0, 1.05], dtick=0.25, tickformat=".2f", zeroline=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_layout(title=None, margin=dict(l=68, r=172, t=20, b=66))
    add_footnote(fig, (
        "After training task <i>t</i>, each line is the mean score of the "
        "deployed policy over tasks 1 to <i>t</i>, which is row <i>t</i> of the "
        "retention matrix<br>&nbsp;&nbsp;&nbsp;averaged along its length. No band "
        "is drawn, because each line is a single run.<br>"
        "&nbsp;&nbsp;&nbsp;The unconstrained run starts higher, because task 1 is "
        "the one task it does learn. It falls below at task 2 and never "
        "recovers.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=68)
    return fig


FIGURES = [
    ("constraint_matrices", fig_constraint_matrices, W_FULL, 420),
    ("constraint_retention_curve", fig_constraint_retention_curve, W_FULL, None),
    ("depth_perf_and_cost", fig_depth_perf_and_cost, W_FULL, None),
    ("depth_matrices", fig_depth_matrices, W_FULL, 340),
]

# Dropped at review. Deleted on every run so a stale PNG cannot linger in the
# folder and be mistaken for current output.
RETIRED = ["constraint_acquire_vs_retain", "constraint_headline",
           "constraint_task_profile", "depth_efficiency", "depth_metrics",
           "depth_perf"]


def main() -> None:
    install_template()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for name in RETIRED:
        for path in (PNG_DIR / f"{name}.png", SVG_DIR / f"{name}.svg"):
            if path.exists():
                path.unlink()
                print(f"  retired {path.name}")
    data = load_data()

    print("\nConstraint on/off, ten tasks, one seed")
    for r in data["constraint"]:
        print(f"  {r['label']:<28} PERF {r['perf']:>6.3f}  BWT {r['bwt']:>+6.3f}"
              f"  forget {r['forgetting']:>5.3f}"
              f"  acquired {sum(1 for v in r['diagonal'] if v >= THRESHOLD)}/{N_TASKS}"
              f"  solved {sum(1 for v in r['final'] if v >= THRESHOLD)}/{N_TASKS}")
    print("\nConsolidation rollout depth sweep")
    for r in data["sweep"]:
        print(f"  depth {r['depth']:>3}  PERF {r['perf']:>6.3f}  BWT {r['bwt']:>+6.3f}"
              f"  forget {r['forgetting']:>5.3f}  wall {r['wall_min']:>5.1f} min"
              f"  {r['steps_per_iter']:>5} steps/iter")
    print()

    for fig_id, builder, width, height in FIGURES:
        fig = builder(data)
        h = height or height_for(width)
        # PNG and SVG come from the SAME figure object, never two code paths.
        fig.write_image(PNG_DIR / f"{fig_id}.png", width=width, height=h, scale=3.125)
        fig.write_image(SVG_DIR / f"{fig_id}.svg", width=width, height=h)
        png, svg = PNG_DIR / f"{fig_id}.png", SVG_DIR / f"{fig_id}.svg"
        for path in (png, svg):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"export failed or empty: {path}")
        raster = svg.read_text().count("<image")
        flag = f"  !! {raster} rasterised <image> tags" if raster else ""
        print(f"  {fig_id:<30} {png.stat().st_size//1024:>5} KB png  "
              f"{svg.stat().st_size//1024:>5} KB svg{flag}")

    print(f"\nwrote {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
