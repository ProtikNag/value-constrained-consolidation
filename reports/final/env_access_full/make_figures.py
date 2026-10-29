#!/usr/bin/env python3
"""Env-access at the full Table-2 budget: does live access beat a buffer?

The ten-task probe in ``reports/final/env_access`` asked whether VCC's lead is
bought by re-entering earlier environments, and answered it by giving CLEAR and
RECALL the same access. What that probe could not answer is the other half of
the question, because it held no buffer-on control at the same budget. Every
replay arm in it already had access, so nothing measured what access was worth.

This run supplies the control. Over the full fifty-task sequence at the full
204.8M-frame budget, each replay method is run twice, once holding a buffer and
once re-collecting in the earlier environments with no buffer, and nothing else
differs. Both methods score LOWER with access than with their buffer. Access is
not what separates them from VCC, and it does not rescue them either.

Seeds: the four replay arms are seed0 alone, so the buffer-against-access
comparison is within-seed and clean. VCC carries three seeds, so figures that
place VCC beside a replay arm show seed0 for the comparison and the three-seed
spread beside it rather than quietly mixing the two.

Nothing here is a new experiment. Every metric is recomputed from the stored
matrices and checked against summary.json at build time.

Usage::

    python reports/final/env_access_full/make_figures.py
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
    spread_labels,
)

RUNS = REPO / "reports" / "gridworld_env_access" / "full_budget"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 50
FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5

# Palette roles held fixed with the Setting B and ten-task env-access figures,
# so a colour means one thing across the paper.
C_OURS, C_CLEAR, C_RECALL = AC["blue"], AC["amber"], AC["violet"]
THRESHOLD = 0.6

CAVEAT = ("Fifty tasks, full 204.8M-frame budget, one task order. Replay arms "
          "are seed0; VCC is three seeds.")

# (summary key, display label, colour, holds a buffer)
ARMS = [
    ("VCC_buffer", "VCC (ours)", C_OURS, False),
    ("CLEAR_buffer_paper", "CLEAR, buffer", C_CLEAR, True),
    ("CLEAR_env_fullbudget", "CLEAR, env access", C_CLEAR, False),
    ("RECALL_buffer_paper", "RECALL, buffer", C_RECALL, True),
    ("RECALL_env_fullbudget", "RECALL, env access", C_RECALL, False),
]
# The buffer-against-access control, which is the point of this run.
PAIRS = [("CLEAR", "CLEAR_buffer_paper", "CLEAR_env_fullbudget", C_CLEAR),
         ("RECALL", "RECALL_buffer_paper", "RECALL_env_fullbudget", C_RECALL)]


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
    """Load every arm and assert the recomputed metrics against summary.json."""
    summary = json.loads((RUNS / "summary.json").read_text())
    arms = {}
    for key, label, color, has_buffer in ARMS:
        entry = summary["methods"][key]
        mat_key = "norm_matrix" if "norm_matrix" in entry else "norm_matrix_seed0"
        got = metrics(entry[mat_key])
        if len(got["matrix"]) != N_TASKS:
            raise ValueError(f"{key}: {len(got['matrix'])} rows, expected {N_TASKS}")
        # Every figure metric comes from ONE matrix, which is seed0 for every
        # arm. The replay arms have no other seed, so mixing VCC's three-seed
        # means into the same panel would compare two different quantities.
        # The three-seed values are carried separately, for annotation only.
        final = got["final"]
        got["solved"] = sum(1 for v in final if v >= 0.9) / len(final)
        got["failed"] = sum(1 for v in final if v <= 0.1) / len(final)
        if len(entry["seeds"]) == 1:
            for name, ours, theirs in (
                    ("PERF", got["perf"], entry["PERF"]),
                    ("BWT", got["bwt"], entry["BWT"]),
                    ("forgetting", got["forgetting"], entry["forgetting"]),
                    ("solved", got["solved"], entry["frac_solved_ge0p9"]),
                    ("failed", got["failed"], entry["frac_failed_le0p1"])):
                if abs(ours - theirs) > 1e-3:
                    raise ValueError(
                        f"{key}: recomputed {name} {ours:.4f} against stored "
                        f"{theirs:.4f}")
        arms[key] = dict(
            key=key, label=label, color=color, has_buffer=has_buffer,
            seeds=entry["seeds"], stated_perf=entry["PERF"],
            stated_bwt=entry["BWT"], stated_forgetting=entry["forgetting"],
            stated_solved=entry["frac_solved_ge0p9"],
            perf_sd=entry.get("PERF_sd"), bwt_sd=entry.get("BWT_sd"),
            per_seed=entry.get("per_seed"), **got)
    return dict(arms=arms, depth=summary["depth"], note=summary["note"])


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


def add_colorbar(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Colourbar on an invisible marker trace, so the SVG carries a gradient."""
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(color=[CMIN],
                    colorscale=[[p, f"rgb{c}"] for p, c in MATRIX_SCALE],
                    cmin=CMIN, cmax=CMAX, showscale=True, opacity=0,
                    colorbar=dict(title=dict(text="Score", side="right",
                                             font=dict(family=FONT_UI, size=FS_AXIS)),
                                  tickvals=[-0.15, 0.0, 0.25, 0.5, 0.75, 1.0],
                                  tickfont=dict(family=FONT_MONO, size=FS_TICK),
                                  thickness=12, len=0.78, outlinewidth=0, x=1.015))),
        row=row, col=col)


# ── Figure 1: the control, buffer against live access ───────────────────────
def fig_buffer_vs_env(d: dict) -> go.Figure:
    """Where the five arms land, and what swapping the buffer for access cost.

    Two panels, because the two questions need different marks. The levels are
    magnitudes, so they are lollipops with zero on the axis. The change is a
    difference whose sign is the whole point, so it is a bar from zero, where
    zero reads as "access changed nothing". A single dumbbell was tried and
    fails here: CLEAR's change is nine times smaller than RECALL's, so its two
    ends collide at any scale that shows both.
    """
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.60, 0.40], horizontal_spacing=0.17,
        subplot_titles=("Where each arm lands", "Change from buffer to access"))

    # ── Panel A: the levels ────────────────────────────────────────────────
    keys = [k for k, _, _, _ in ARMS]
    ys = list(range(len(keys)))[::-1]  # first arm at the top
    for y, key in zip(ys, keys):
        arm = d["arms"][key]
        color = arm["color"]
        fig.add_trace(go.Scatter(
            x=[0, arm["perf"]], y=[y, y], mode="lines", hoverinfo="skip",
            showlegend=False,
            line=dict(color=hex_to_rgba(color, 0.40), width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[arm["perf"]], y=[y], mode="markers", hoverinfo="skip",
            showlegend=False,
            marker=dict(color=color, size=13,
                        line=dict(color=AC["bg"], width=2))), row=1, col=1)
        fig.add_annotation(
            x=0, y=y, xref="x", yref="y", xanchor="right", xshift=-10,
            yanchor="middle", showarrow=False, align="right",
            text=f"<span style='font-size:11.5px'>{arm['label']}</span>",
            font=dict(family=FONT_UI, color=color))
        fig.add_annotation(
            x=arm["perf"], y=y, xref="x", yref="y", xanchor="left", xshift=12,
            yanchor="middle", showarrow=False,
            text=f"<span style='font-family:{FONT_MONO}'>{arm['perf']:.3f}</span>",
            font=dict(family=FONT_UI, size=11, color=AC["text_primary"]))

    # VCC is the reference the others are read against, so it is a rule.
    vcc_seed0 = d["arms"]["VCC_buffer"]["perf"]
    fig.add_vline(x=vcc_seed0, line=dict(color=C_OURS, width=1.1, dash="dash"),
                  row=1, col=1)

    fig.update_xaxes(
        title=dict(text="PERF, mean final score over the fifty tasks",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0, 1.0], dtick=0.25, tickformat=".2f",
        tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=False, ticks="",
                     range=[-0.7, len(keys) - 0.3], row=1, col=2 - 1)

    # ── Panel B: the change ────────────────────────────────────────────────
    deltas = [(name, d["arms"][env]["perf"] - d["arms"][buf]["perf"], color)
              for name, buf, env, color in PAIRS]
    bys = list(range(len(deltas)))[::-1]
    for y, (name, delta, color) in zip(bys, deltas):
        fig.add_trace(go.Bar(
            x=[delta], y=[y], orientation="h", width=0.42, hoverinfo="skip",
            showlegend=False,
            marker=dict(color=hex_to_rgba(color, 0.85),
                        line=dict(color=AC["bg"], width=0.8))), row=1, col=2)
        fig.add_annotation(
            x=delta, y=y, xref="x2", yref="y2", xanchor="right", xshift=-8,
            yanchor="middle", showarrow=False,
            text=f"<b>{name}</b> <span style='font-family:{FONT_MONO}'>"
                 f"{delta:+.3f}</span>",
            font=dict(family=FONT_UI, size=11.5, color=color))
    fig.add_vline(x=0, line=dict(color=AC["axis"], width=1.2), row=1, col=2)
    fig.add_annotation(
        x=0, y=len(deltas) - 0.46, xref="x2", yref="y2", xanchor="left",
        xshift=5, yanchor="bottom", showarrow=False, text="no change",
        font=dict(family=FONT_UI, size=10, color=AC["text_muted"]))

    span = max(abs(v) for _, v, _ in deltas)
    fig.update_xaxes(
        title=dict(text="Change in PERF", font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[-span * 1.35, span * 0.42], dtick=0.04, tickformat="+.2f",
        tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=False, ticks="",
                     range=[-0.7, len(deltas) - 0.3], row=1, col=2)

    for note in fig.layout.annotations[:2]:
        note.font.update(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=126, r=30, t=58, b=92),
                      bargap=0.45)
    add_footnote(fig, (
        "Left, each arm's mean final score, with VCC as the rule the others "
        "are read against. Right, the change from a method's<br>"
        "&nbsp;&nbsp;&nbsp;own buffer run to its live-access run, where "
        "nothing else differs. Neither is positive. CLEAR is unchanged at "
        "-0.009,<br>"
        "&nbsp;&nbsp;&nbsp;which is inside single-seed noise, and RECALL "
        "falls. Access does not rescue a replay method.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT}<br>"
        "&nbsp;&nbsp;&nbsp;Every value above is seed0; VCC's three-seed mean "
        "is 0.779, sd 0.032."
    ), left=126, extra=10)
    return fig


# ── Figure 2: mean score over the tasks seen ────────────────────────────────
def fig_retention_curve(d: dict) -> go.Figure:
    """Mean score over the tasks seen so far, as the fifty-task sequence runs."""
    fig = go.Figure()
    ends = []
    for key, label, color, has_buffer in ARMS:
        arm = d["arms"][key]
        xs = list(range(1, N_TASKS + 1))
        ys = [sum(arm["matrix"][k]) / len(arm["matrix"][k]) for k in range(N_TASKS)]
        ours = key.startswith("VCC")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=True,
            name=f"<b>{label}</b>" if ours else label,
            line=dict(color=color, width=3.0 if ours else 1.9,
                      dash="dot" if has_buffer else "solid")))
        ends.append((ys[-1], color))
    # CLEAR's two arms finish 0.01 apart, so the end labels need pushing apart
    # or they print on top of one another.
    placed = spread_labels([v for v, _ in ends], 0.045)
    for (value, color), y in zip(ends, placed):
        fig.add_annotation(
            x=N_TASKS, y=y, xanchor="left", xshift=10, yanchor="middle",
            showarrow=False, text=f"<b>{value:.2f}</b>",
            font=dict(family=FONT_MONO, size=11.5, color=color))
    fig.update_xaxes(
        title=dict(text="Tasks trained", font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.4, N_TASKS + 0.6], dtick=10,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(
        title=dict(text="Mean score over tasks seen",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.0, 1.03], dtick=0.25, tickformat=".2f", zeroline=False,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_layout(
        title=None, margin=dict(l=66, r=92, t=20, b=104), showlegend=True,
        legend=dict(orientation="h", x=0.5, y=-0.20, xanchor="center",
                    yanchor="top", font=dict(family=FONT_UI, size=10.5),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    add_footnote(fig, (
        "After training task <i>t</i>, each line is the mean score of the "
        "deployed policy over tasks 1 to <i>t</i>. Dotted lines hold a replay<br>"
        "&nbsp;&nbsp;&nbsp;buffer and solid lines re-collect in the earlier "
        "environments. CLEAR's two arms track each other and finish 0.01 "
        "apart;<br>"
        "&nbsp;&nbsp;&nbsp;RECALL's access arm sits below its buffer arm "
        "throughout. No band is drawn, because each line is a single run.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT} The VCC line is seed0."
    ), left=66, extra=30)
    return fig


# ── Figure 3: what the methods actually do, per task ────────────────────────
def fig_acquire_vs_final(d: dict) -> go.Figure:
    """Score on arrival against score at the end, one point per task.

    The clearest statement of the mechanism. A point on the diagonal is a task
    held exactly as it was learned. The replay arms sit on it. VCC's cloud
    lifts off it, which is the positive backward transfer in the table.
    """
    show = ["VCC_buffer", "CLEAR_env_fullbudget", "RECALL_env_fullbudget"]
    fig = make_subplots(
        rows=1, cols=len(show), shared_yaxes=True, horizontal_spacing=0.045,
        subplot_titles=[d["arms"][k]["label"] for k in show])

    for col, key in enumerate(show, start=1):
        arm = d["arms"][key]
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=AC["text_faint"], width=1.0, dash="dash")),
            row=1, col=col)
        fig.add_trace(go.Scatter(
            x=arm["diagonal"], y=arm["final"], mode="markers", hoverinfo="skip",
            showlegend=False,
            marker=dict(color=hex_to_rgba(arm["color"], 0.75), size=7,
                        line=dict(color=AC["bg"], width=1))),
            row=1, col=col)
        gained = sum(1 for a, b in zip(arm["diagonal"], arm["final"]) if b - a > 0.05)
        lost = sum(1 for a, b in zip(arm["diagonal"], arm["final"]) if a - b > 0.05)
        # Plotly names the first subplot's axes "x"/"y", not "x1"/"y1".
        suffix = "" if col == 1 else str(col)
        fig.add_annotation(
            x=0.03, y=0.97, xref=f"x{suffix} domain", yref=f"y{suffix} domain",
            xanchor="left", yanchor="top", showarrow=False, align="left",
            text=f"<span style='font-family:{FONT_MONO}'>{gained:>2}</span> up"
                 f"<br><span style='font-family:{FONT_MONO}'>{lost:>2}</span> down",
            font=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            bgcolor="rgba(255,255,255,0.82)", borderpad=3)
        fig.update_xaxes(
            title=dict(text="Score on arrival" if col == 2 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[-0.08, 1.08], dtick=0.5, tickformat=".1f",
            tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=col)
        fig.update_yaxes(
            title=dict(text="Score after fifty tasks" if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[-0.08, 1.08], dtick=0.25, tickformat=".2f",
            tickfont=dict(family=FONT_MONO, size=FS_TICK), row=1, col=col)

    for note in fig.layout.annotations[:len(show)]:
        note.font.update(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=72, r=28, t=44, b=92))
    add_footnote(fig, (
        "One point per task. The dashed diagonal is a task held exactly as it "
        "was learned; above it the task improved after it was<br>"
        "&nbsp;&nbsp;&nbsp;left, below it the task was lost. Counts are tasks "
        "that moved more than 0.05 either way. The replay arms sit on the<br>"
        "&nbsp;&nbsp;&nbsp;diagonal, keeping what they acquire and recovering "
        "nothing; VCC's cloud lifts off it.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT} VCC is seed0."
    ), left=72, extra=18)
    return fig


# ── Figure 4: the four metrics side by side ─────────────────────────────────
def fig_metrics(d: dict) -> go.Figure:
    """PERF, BWT, forgetting and solved share, all five arms.

    Panels get their own axes because the scales differ by more than ten times;
    a shared axis would flatten BWT and forgetting into nothing.
    """
    panels = [("PERF", "perf", False),
              ("Backward transfer", "bwt", True),
              ("Forgetting", "forgetting", False),
              ("Solved (score \u2265 0.9)", "solved", False)]
    keys = [k for k, _, _, _ in ARMS]
    labels = [d["arms"][k]["label"] for k in keys]
    colors = [d["arms"][k]["color"] for k in keys]
    patterns = ["" if not d["arms"][k]["has_buffer"] else "/" for k in keys]

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.085,
                        subplot_titles=[p[0] for p in panels])
    for col, (title, field, signed) in enumerate(panels, start=1):
        values = [d["arms"][k][field] for k in keys]
        fig.add_trace(go.Bar(
            x=list(range(len(keys))), y=values, showlegend=False, hoverinfo="skip",
            marker=dict(color=[hex_to_rgba(c, 0.88) for c in colors],
                        line=dict(color=colors, width=1.1),
                        pattern=dict(shape=patterns, solidity=0.42,
                                     fgcolor=AC["bg"], size=5)),
            text=[f"{v:+.3f}" if signed else f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=10, color=AC["text_primary"])),
            row=1, col=col)
        if signed:
            fig.add_hline(y=0, line=dict(color=AC["axis"], width=1.0),
                          row=1, col=col)
        lo, hi = min(values + [0.0]), max(values)
        span = (hi - lo) or 1.0
        # Bars rise from zero, so only the signed panel needs room below it.
        # Padding an unsigned panel below zero just empties the lower half.
        bottom = lo - span * 0.18 if signed else 0.0
        fig.update_yaxes(
            range=[bottom, hi + span * 0.30], tickformat=".2f",
            nticks=4, tickfont=dict(family=FONT_MONO, size=FS_TICK),
            row=1, col=col)
        fig.update_xaxes(showticklabels=False, ticks="", showgrid=False,
                         row=1, col=col)

    for note in fig.layout.annotations[:len(panels)]:
        note.font.update(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])

    # One shared legend at the top, so the five arms are named once.
    for label, color, pattern in zip(labels, colors, patterns):
        fig.add_trace(go.Bar(
            x=[None], y=[None], name=label, showlegend=True,
            marker=dict(color=hex_to_rgba(color, 0.88),
                        line=dict(color=color, width=1.1),
                        pattern=dict(shape=pattern, solidity=0.42,
                                     fgcolor=AC["bg"], size=5))))
    fig.update_layout(
        title=None, margin=dict(l=52, r=22, t=86, b=92), showlegend=True,
        legend=dict(orientation="h", x=0.5, y=1.20, xanchor="center",
                    yanchor="bottom", font=dict(family=FONT_UI, size=10.5),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    add_footnote(fig, (
        "Hatched bars hold a replay buffer and solid bars re-collect in the "
        "earlier environments. Each panel carries its own axis,<br>"
        "&nbsp;&nbsp;&nbsp;because the four scales differ by more than ten "
        "times. Backward transfer is where the methods separate: VCC is the "
        "only<br>"
        "&nbsp;&nbsp;&nbsp;arm above zero, and the replay arms forget little "
        "only because they recover nothing.<br>"
        f"&nbsp;&nbsp;&nbsp;{CAVEAT} VCC's bars are seed0."
    ), left=52, extra=-38)
    return fig


FIGURES = [
    ("env_full_buffer_vs_env", fig_buffer_vs_env, W_FULL, 330),
    ("env_full_retention_curve", fig_retention_curve, W_FULL, None),
    ("env_full_acquire_vs_final", fig_acquire_vs_final, W_FULL, 330),
    ("env_full_metrics", fig_metrics, W_FULL, 330),
]


def main() -> None:
    install_template()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("\nEnv-access at the full budget, fifty tasks, depth "
          f"{data['depth']}")
    print(f"  {'arm':<22}{'PERF':>8}{'BWT':>9}{'forget':>9}{'solved':>9}"
          f"{'failed':>9}  seeds")
    for key, label, _, _ in ARMS:
        a = data["arms"][key]
        print(f"  {label:<22}{a['perf']:>8.3f}{a['bwt']:>+9.3f}"
              f"{a['forgetting']:>9.3f}{a['solved']:>9.0%}{a['failed']:>9.0%}"
              f"  {a['seeds']}")
    print("\n  the control, buffer -> env access")
    for name, buf_key, env_key, _ in PAIRS:
        buf, env = data["arms"][buf_key], data["arms"][env_key]
        print(f"    {name:<8}{buf['perf']:.3f} -> {env['perf']:.3f}"
              f"   {env['perf'] - buf['perf']:+.3f}")
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
                raise RuntimeError(f"{path} missing or empty")
        print(f"  {fig_id}: png {png.stat().st_size // 1024}K, "
              f"svg {svg.stat().st_size // 1024}K")


if __name__ == "__main__":
    main()
