#!/usr/bin/env python3
"""Env-access ablation: replay methods given VCC's live past-environment access.

The question this answers is whether VCC's advantage is bought by re-entering
earlier environments. CLEAR and RECALL are rerun with the same access and no
replay buffer, and the gap does not close.

TWO COMPARISONS ARE IN THE RECORD AND ONLY THE SECOND IS CLEAN. The first ran
every method at one reduced budget, and VCC undertrained the specialists for
tasks 8 and 10, so its own diagonal is zero there and the read is confounded.
The second resumes those two tasks for all three methods, giving the replay
baselines the budget of VCC's deployed policy. That concedes VCC's specialist
phase to them, which makes it the conservative comparison, and it is the one
these figures report. The first is drawn once, beside the second, so the
correction is visible rather than silent.

Nothing here is a new experiment. Every number is recomputed from each run's
eval_matrix.json and agrees with summary.json.

Usage::

    python reports/final/env_access/make_figures.py
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

RUNS = REPO / "reports" / "gridworld_env_access"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 10
FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5

# Palette roles held fixed with the Setting B figures, so a colour means one
# thing across the paper.
C_OURS, C_CLEAR, C_RECALL = AC["blue"], AC["amber"], AC["violet"]
C_RULE = AC["green"]
THRESHOLD = 0.6

CAVEAT = ("Single seed, ten tasks, reduced budget, one task order. Directional "
          "rather than full-scale.")

# (summary key, display label, run directory, colour)
ARMS = [
    ("VCC2", "VCC (ours)", "vcc2_rebudget", C_OURS),
    ("CLEAR_env_v2", "CLEAR, env access", "clear_env_v2", C_CLEAR),
    ("RECALL_env_v2", "RECALL, env access", "recall_env_v2", C_RECALL),
]
FIRST_PASS = [
    ("VCC", "VCC", C_OURS),
    ("CLEAR_env", "CLEAR, env access", C_CLEAR),
    ("RECALL_env", "RECALL, env access", C_RECALL),
]


# ── Data ────────────────────────────────────────────────────────────────────
def metrics(matrix: list[list[float]]) -> dict:
    """PERF, BWT and forgetting from one lower-triangular evaluation matrix."""
    last = len(matrix) - 1
    final = list(matrix[last])
    diagonal = [matrix[k][k] for k in range(last + 1)]
    return dict(
        matrix=matrix, final=final, diagonal=diagonal,
        perf=sum(final) / len(final),
        bwt=sum(final[i] - diagonal[i] for i in range(last)) / last,
        forgetting=sum(max(matrix[k][i] for k in range(i, last + 1)) - final[i]
                       for i in range(last)) / last)


def load_data() -> dict:
    summary = json.loads((RUNS / "summary.json").read_text())
    clean = summary["comparison_2_deployed_budget_matched"]
    first = summary["comparison_1_reduced_equal_budget"]

    arms = []
    for key, label, run, color in ARMS:
        matrix = json.loads((RUNS / "runs" / run / "eval_matrix.json").read_text())
        m = metrics(matrix)
        # The raw matrix and the published summary must agree, or one of them
        # is stale and the figure would be built on the wrong one.
        for name, ours, theirs in (("PERF", m["perf"], clean[key]["PERF"]),
                                   ("BWT", m["bwt"], clean[key]["BWT"]),
                                   ("F", m["forgetting"], clean[key]["forgetting"])):
            assert abs(ours - theirs) < 2e-3, f"{label} {name}: {ours} vs {theirs}"
        arms.append(m | dict(label=label, color=color,
                             budget=clean[key]["budget"]))
    return dict(arms=arms, first=first, clean=clean, summary=summary)


# ── Shared helpers ──────────────────────────────────────────────────────────
PAPER_MODE = bool(os.environ.get("FIGS_FOR_PAPER"))
AXIS_TITLE_PX = 56


def add_footnote(fig: go.Figure, text: str, left: int, extra: int = 0) -> None:
    """Footnote under the axis, or bare bottom margin when built for the paper."""
    if PAPER_MODE:
        fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + 10))
        return
    lines = text.count("<br>") + 1
    fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + 15 * lines + 12))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="top",
        yshift=-(AXIS_TITLE_PX + extra), showarrow=False, align="left",
        text=text, font=dict(family=FONT_UI, size=9.5, color=AC["text_muted"]))


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


def add_colorbar(fig: go.Figure) -> None:
    """Colourbar on an invisible marker trace, so the SVG carries a gradient."""
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


# ── Figure 1: the three retention matrices ──────────────────────────────────
def fig_matrices(d: dict) -> go.Figure:
    """VCC beside both replay baselines once they re-enter the environments.

    add_shape is quadratic in the shapes already present, so all three hundred
    cells are assigned in a single update_layout call.
    """
    arms = d["arms"]
    shapes: list[dict] = []
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.055,
        subplot_titles=[f"{a['label']}<br>PERF {a['perf']:.3f}" for a in arms])
    ticks = list(range(N_TASKS))
    for col, arm in enumerate(arms, start=1):
        axes = dict(xref="x" if col == 1 else f"x{col}",
                    yref="y" if col == 1 else f"y{col}")
        for r, row in enumerate(arm["matrix"]):
            for c, value in enumerate(row):
                shapes.append(dict(type="rect", x0=c, x1=c + 1, y0=r, y1=r + 1,
                                   fillcolor=sample_scale(value),
                                   line=dict(width=0.35, color=AC["bg"]),
                                   layer="below", **axes))
        for k in range(N_TASKS):
            shapes.append(dict(type="rect", x0=k, x1=k + 1, y0=k, y1=k + 1,
                               fillcolor="rgba(0,0,0,0)",
                               line=dict(width=1.5, color=AC["text_primary"]), **axes))
        fig.update_xaxes(
            title=dict(text="Task evaluated" if col == 2 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[0, N_TASKS], tickmode="array",
            tickvals=[t + 0.5 for t in ticks], ticktext=[str(t + 1) for t in ticks],
            constrain="domain", tickfont=dict(family=FONT_MONO, size=9),
            showgrid=False, zeroline=False, row=1, col=col)
        fig.update_yaxes(
            title=dict(text="After training task" if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[N_TASKS, 0], tickmode="array",
            tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) if col == 1 else "" for t in ticks],
            constrain="domain", scaleanchor="x" if col == 1 else f"x{col}",
            scaleratio=1, tickfont=dict(family=FONT_MONO, size=9),
            showgrid=False, zeroline=False, row=1, col=col)
    add_colorbar(fig)
    for ann in fig.layout.annotations[:3]:
        ann.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        ann.y = (ann.y or 1.0) + 0.05
    fig.update_layout(title=None, margin=dict(l=58, r=92, t=64, b=50),
                      shapes=shapes)
    add_footnote(fig, (
        "Row <i>k</i> is the score on every task seen so far, measured after task "
        "<i>k</i> finished training. One shared scale across all three panels.<br>"
        "&nbsp;&nbsp;&nbsp;<b>The outlined diagonal</b> is the score on the task "
        "being learned. Both replay baselines hold what they acquire, and the "
        "columns below<br>&nbsp;&nbsp;&nbsp;their diagonal barely change. VCC's "
        "columns turn blue further down, which is a task recovered after it was "
        "left.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=58, extra=10)
    return fig


# ── Figure 2: mean score over the tasks seen ────────────────────────────────
def fig_retention_curve(d: dict) -> go.Figure:
    """Mean score over the tasks seen so far, as the sequence advances."""
    fig = go.Figure()
    for arm in d["arms"]:
        xs = list(range(1, N_TASKS + 1))
        ys = [sum(arm["matrix"][k]) / len(arm["matrix"][k]) for k in range(N_TASKS)]
        ours = arm["label"].startswith("VCC")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", hoverinfo="skip", showlegend=True,
            name=f"<b>{arm['label']}</b>",
            line=dict(color=arm["color"], width=3.2 if ours else 2.2),
            marker=dict(color=arm["color"], size=9 if ours else 7,
                        line=dict(color=AC["bg"], width=1.4))))
        fig.add_annotation(
            x=N_TASKS, y=ys[-1], xanchor="left", xshift=12, yanchor="middle",
            showarrow=False, text=f"<b>{ys[-1]:.2f}</b>",
            font=dict(family=FONT_MONO, size=12, color=arm["color"]))
    fig.update_xaxes(
        title=dict(text="Tasks trained", font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.6, N_TASKS + 0.6], dtick=1,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(
        title=dict(text="Mean score over tasks seen",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.0, 1.03], dtick=0.25, tickformat=".2f", zeroline=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_layout(
        title=None, margin=dict(l=66, r=96, t=20, b=96), showlegend=True,
        legend=dict(orientation="h", x=0.5, y=-0.23, xanchor="center",
                    yanchor="top", font=dict(family=FONT_UI, size=11.5),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    add_footnote(fig, (
        "After training task <i>t</i>, each line is the mean score of the "
        "deployed policy over tasks 1 to <i>t</i>. No band is drawn, because<br>"
        "&nbsp;&nbsp;&nbsp;each line is a single run. All three re-enter earlier "
        "environments and none keeps a replay buffer.<br>"
        "&nbsp;&nbsp;&nbsp;The replay baselines start near the ceiling and end "
        "below VCC, which starts lowest and climbs.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=66, extra=22)
    return fig


# ── Figure 3: arrival against final, per task ───────────────────────────────
def fig_acquire_vs_final(d: dict) -> go.Figure:
    """Where each task started and where it ended, for all three arms.

    Length and direction are the change, which is why the bars do not run from
    zero. The replay baselines are nearly all stubs: they keep what they get.
    """
    arms = d["arms"]
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.05, shared_yaxes=True,
        subplot_titles=[f"{a['label']} — BWT {a['bwt']:+.3f}" for a in arms])
    tasks = list(range(1, N_TASKS + 1))
    for col, arm in enumerate(arms, start=1):
        for task, start, end in zip(tasks, arm["diagonal"], arm["final"]):
            gained = end >= start
            fig.add_trace(go.Scatter(
                x=[start, end], y=[task, task], mode="lines", hoverinfo="skip",
                showlegend=False,
                line=dict(color=hex_to_rgba(C_OURS if gained else AC["red"], 0.45),
                          width=4.0)), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=arm["diagonal"], y=tasks, mode="markers", hoverinfo="skip",
            showlegend=False,
            marker=dict(symbol="line-ns", size=10,
                        line=dict(color=AC["text_faint"], width=2.0))),
            row=1, col=col)
        fig.add_trace(go.Scatter(
            x=arm["final"], y=tasks, mode="markers", hoverinfo="skip",
            showlegend=False,
            marker=dict(color=arm["color"], size=9,
                        line=dict(color=AC["bg"], width=1.6))), row=1, col=col)
        fig.add_vline(x=THRESHOLD, line=dict(color=C_RULE, width=1.2, dash="dash"),
                      row=1, col=col)
        fig.update_xaxes(
            title=dict(text="Normalized score" if col == 2 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[-0.12, 1.1], dtick=0.5, tickformat=".1f",
            tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=col)
    fig.update_yaxes(
        title=dict(text="Task", font=dict(family=FONT_UI, size=FS_AXIS)),
        autorange="reversed", dtick=1, showgrid=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=1)
    for ann in fig.layout.annotations[:3]:
        ann.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=54, r=22, t=46, b=56))
    add_footnote(fig, (
        "The tick is the task's score on arrival and the dot is its score after "
        "all ten tasks, so each bar's length and direction are<br>"
        "&nbsp;&nbsp;&nbsp;the change. Blue gained, red lost, and those two "
        f"colours mean nothing else here. The dashed rule is the success bar<br>"
        f"&nbsp;&nbsp;&nbsp;at {THRESHOLD:.1f}. Zero is not a baseline, because "
        "a bar encodes a difference. The replay baselines are almost all<br>"
        "&nbsp;&nbsp;&nbsp;stubs: they keep what they acquire and recover "
        "nothing, while VCC's bars move.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=54, extra=-10)
    return fig


# ── Figure 4: the correction, first pass beside the clean one ───────────────
def fig_budget_correction(d: dict) -> go.Figure:
    """PERF under the first reading and under the corrected one.

    The first pass ran every arm at one reduced budget and VCC undertrained its
    specialists for two tasks, which put it below CLEAR. The corrected pass
    resumes those tasks for all three and reverses the order. Showing both is
    what makes the correction checkable rather than asserted.
    """
    first, clean = d["first"], d["clean"]
    pairs = [(label, first[k0]["PERF"], clean[k1]["PERF"], color)
             for (k0, label, color), (k1, _, _, _) in zip(FIRST_PASS, ARMS)]
    fig = go.Figure()
    for i, (label, a, b, color) in enumerate(pairs):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[a, b], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=hex_to_rgba(color, 0.55), width=3.0)))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[a, b], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(color=color, size=12, line=dict(color=AC["bg"], width=1.8))))
        fig.add_annotation(x=0, y=a, text=f"{a:.3f}", showarrow=False,
                           xanchor="right", xshift=-14, yanchor="middle",
                           font=dict(family=FONT_MONO, size=11, color=color))
        # Value travels with the name on the right; two separate labels at
        # 0.616 and 0.580 overlapped each other.
        fig.add_annotation(x=1, y=b, text=f"<b>{label}</b>  {b:.3f}",
                           showarrow=False, xanchor="left", xshift=16,
                           yanchor="middle",
                           font=dict(family=FONT_UI, size=12, color=color))
    for x, head, sub in ((0, "First reading",
                          "one reduced budget for all three"),
                         (1, "Corrected",
                          "tasks 8 and 10 resumed for all three")):
        fig.add_annotation(x=x, y=1.0, yref="paper", yshift=34, showarrow=False,
                           xanchor="center", text=f"<b>{head}</b>",
                           font=dict(family=FONT_UI, size=12.5,
                                     color=AC["text_primary"]))
        fig.add_annotation(x=x, y=1.0, yref="paper", yshift=16, showarrow=False,
                           xanchor="center", text=sub,
                           font=dict(family=FONT_UI, size=10,
                                     color=AC["text_muted"]))
    fig.update_xaxes(range=[-0.42, 1.62], showgrid=False, zeroline=False,
                     showticklabels=False, showline=False, ticks="")
    fig.update_yaxes(
        title=dict(text="PERF after ten tasks",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.52, 0.84], dtick=0.1, tickformat=".2f", zeroline=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_layout(title=None, margin=dict(l=76, r=24, t=58, b=56))
    add_footnote(fig, (
        "The first reading ran every arm at one reduced budget. Under it VCC "
        "left the specialists for tasks 8 and 10 untrained,<br>"
        "&nbsp;&nbsp;&nbsp;so its own diagonal is zero there and CLEAR reads "
        "higher. The corrected pass resumes those two tasks for all<br>"
        "&nbsp;&nbsp;&nbsp;three arms, giving the replay baselines the budget of "
        "VCC's deployed policy rather than its full budget. That<br>"
        "&nbsp;&nbsp;&nbsp;concedes VCC's specialist phase, so the corrected "
        "comparison is the conservative one.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}"
    ), left=76, extra=-44)
    return fig


FIGURES = [
    ("env_access_retention_curve", fig_retention_curve, W_FULL, None),
]

# Dropped at review. Deleted on every run so a stale PNG cannot linger in the
# folder and be mistaken for current output. Their builders are kept, because
# the fifty-task rerun of this experiment may want them back.
RETIRED = ["env_access_matrices", "env_access_acquire_vs_final",
           "env_access_budget_correction"]


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

    print("\nEnv-access ablation, ten tasks, one seed, deployed-budget matched")
    for a in data["arms"]:
        print(f"  {a['label']:<22} PERF {a['perf']:>6.3f}  BWT {a['bwt']:>+6.3f}"
              f"  forget {a['forgetting']:>5.3f}"
              f"  acquired {sum(1 for v in a['diagonal'] if v >= THRESHOLD)}/{N_TASKS}"
              f"  solved {sum(1 for v in a['final'] if v >= THRESHOLD)}/{N_TASKS}")
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
        print(f"  {fig_id:<32} {png.stat().st_size//1024:>5} KB png  "
              f"{svg.stat().st_size//1024:>5} KB svg{flag}")
    print(f"\nwrote {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
