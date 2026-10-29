#!/usr/bin/env python3
"""Order-sensitivity figures: V5 (min-max) vs CLEAR against the Joint ceiling.

The report is focused on the **reversed** task order. The canonical order appears
only in Figure 1, which is the order-contrast itself.

Why the Joint ceiling is the denominator everywhere. Joint is one budget-matched
model trained on all five games at once, so it is *order-independent* and gives
the same reference in both orders. The Local specialist is order-dependent (a
task-1 game has no local phase, and later locals start from the evolving global),
so retention against Local mixes forgetting with reference drift. Local is kept
as a raw-score reference in Figure 3 and nowhere else.

Why "previously learned tasks" is the headline statistic. The last task in a
sequence has had nothing trained after it, so its final score measures capacity,
not retention. Including it lets a method that simply overfits the final task
post a high mean. In the reversed order CLEAR does exactly that: Q*bert runs to
15350.8, which is 3.6x the Joint ceiling, and that single cell lifts CLEAR's
five-game mean above V5's while it is forgetting everything else.

Renders with Plotly + Kaleido through report/acviz.py, so the academic template,
palette, and PNG/SVG export contract are applied. Single seed, so no bands or
error bars are drawn anywhere.

Usage::

    python reports/order_sensitivity/make_figures.py [--no-dashboard]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, DIVERGING, FONT_MONO, FONT_UI, W_FULL, W_ONE_HALF,
    add_callout, add_end_labels, append_manifest_entry, export_figure,
    height_for, hex_to_rgba, install_template,
)

# ── Series identity, fixed across every figure ──────────────────────────────
# Academic palette in series order: ours is primary, the baseline is secondary,
# the ceiling is tertiary, the specialist is neutral because it is a reference.
COLOR = {
    "V5": AC["blue"],
    "CLEAR": AC["amber"],
    "Joint": AC["green"],
    "Local": AC["text_muted"],
}
NAME = {
    "V5": "V5 (min-max)",
    "CLEAR": "CLEAR",
    "Joint": "Joint ceiling",
    "Local": "Local specialist",
}

PNG_DIR = HERE / "png"
SVG_DIR = HERE / "svg"
DASH_FIGURES = REPO / "report" / "figures"
DASH_MANIFEST = REPO / "report" / "manifest.json"

GROUP = "reversed_order"
GROUP_TITLE = "Reversed task order: V5 vs CLEAR vs Joint"
PROVENANCE = (
    "Seed 0, greedy-100 evaluation, plain `impala_ac_multihead` net, five Atari "
    "games. **Single seed, so no error bars are drawn and none are invented.** "
    "CLEAR is apples-to-apples with V5: identical net, task set, thresholds, PPO "
    "hyperparameters, frame-matched budget, and equal replay buffer. The only "
    "difference is CLEAR's replay plus policy/value cloning against V5's min-max "
    "dual constraint."
)


# ── Data ────────────────────────────────────────────────────────────────────
def load_data() -> dict:
    """Read the cached numbers transcribed from each run's eval matrix."""
    return json.loads((HERE / "data.json").read_text(encoding="utf-8"))


def reference_in_order(data: dict, key: str, order: list[str]) -> np.ndarray:
    """Re-index a reference vector from the fixed game order into a task order."""
    index = data["references"]["_index"]
    values = data["references"][key]
    return np.array([values[index.index(game)] for game in order], dtype=float)


def matrix(data: dict, key: str) -> np.ndarray:
    """Load one forgetting matrix, unobserved upper triangle as NaN."""
    rows = data["matrices"][key]
    return np.array([[np.nan if v is None else v for v in row] for row in rows],
                    dtype=float)


def retention_matrix(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise score / reference, reference broadcast across training phases."""
    return scores / reference[None, :]


def prior_task_mean(scores: np.ndarray, reference: np.ndarray) -> float:
    """Mean final retention over every task except the last one learned.

    The final task has had nothing trained after it, so its score is not a
    retention measurement and is excluded from the headline statistic.
    """
    final = scores[-1, :-1] / reference[:-1]
    return float(np.mean(final))


# ── Figure 1: the order contrast ────────────────────────────────────────────
def figure_order_contrast(data: dict) -> dict:
    """Slope chart: mean retention on previously learned tasks, by task order."""
    series = {}
    for method, keys in (("V5", ("v5_canonical", "v5_reversed")),
                         ("CLEAR", ("clear_canonical", "clear_reversed"))):
        values = []
        for key, order_name in zip(keys, ("canonical", "reversed")):
            order = data["orders"][order_name]
            joint = reference_in_order(data, "joint", order)
            values.append(prior_task_mean(matrix(data, key), joint))
        series[method] = values

    fig = go.Figure()
    x = [0, 1]
    for method in ("V5", "CLEAR"):
        color = COLOR[method]
        fig.add_trace(go.Scatter(
            x=x, y=series[method], mode="lines+markers+text",
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=9, line=dict(color=AC["bg"], width=1.5)),
            text=[f"{v:.0%}" for v in series[method]],
            # Right-hand values go above their marker so the end labels have room.
            textposition=["middle left", "top center"],
            textfont=dict(family=FONT_MONO, size=12, color=color),
            cliponaxis=False, hoverinfo="skip",
        ))

    add_end_labels(fig, [
        (NAME[m], 1, series[m][1], COLOR[m]) for m in ("V5", "CLEAR")
    ], xshift=34, min_gap=0.09)

    # The crossing is the whole point of the report. Tail placed in data
    # coordinates (axref/ayref) rather than pixel offsets, which is the only way
    # to guarantee the text lands inside the plot area at any export scale.
    fig.add_annotation(
        x=0.5, y=0.653, ax=0.12, ay=0.945, axref="x", ayref="y",
        text="the order flips which<br>method retains more",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        xanchor="center", yanchor="bottom", align="left",
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        bgcolor="rgba(255,255,255,0.86)", borderpad=3,
    )

    fig.update_layout(
        title=dict(text="Retention depends on the task order"),
        xaxis=dict(
            title=dict(text="Task order"),
            tickmode="array", tickvals=x,
            ticktext=["Canonical<br>Q*bert → … → SpaceInv",
                      "Reversed<br>SpaceInv → … → Q*bert"],
            range=[-0.28, 1.28], showgrid=False, ticklen=0,
        ),
        yaxis=dict(
            title=dict(text="Mean retention vs Joint ceiling"),
            range=[0.28, 1.02], tickformat=".0%", nticks=5,
        ),
        margin=dict(l=76, r=150, t=52, b=76),
    )

    stem = "fig1_order_contrast"
    width = W_ONE_HALF
    variant = export_pair(fig, stem, width, height_for(width, 1.28), "Slope")
    return dict(
        id="order_contrast",
        title="Task order flips which method retains more",
        caption=(
            f"Mean retention on previously learned tasks against the Joint "
            f"ceiling: V5 {series['V5'][0]:.0%} → {series['V5'][1]:.0%}, "
            f"CLEAR {series['CLEAR'][0]:.0%} → {series['CLEAR'][1]:.0%}."
        ),
        details=(
            "The one figure here that shows both task orders; every other figure "
            "in this group is the reversed order alone.\n\n"
            "Each point is the mean of final-score / Joint over the four tasks "
            "learned *before* the last one. The last task is excluded because "
            "nothing was trained after it, so its score measures capacity rather "
            "than retention.\n\n"
            "- **Canonical** Q\\*bert → Pong → Breakout → Boxing → SpaceInvaders\n"
            "- **Reversed** SpaceInvaders → Boxing → Breakout → Pong → Q\\*bert\n\n"
            "The lines cross. Neither method is order-robust, and a single-order "
            "result would have supported the opposite conclusion.\n\n" + PROVENANCE
        ),
        variants=[variant],
        metrics=[
            {"name": "V5 canonical", "value": series["V5"][0], "unit": ""},
            {"name": "V5 reversed", "value": series["V5"][1], "unit": ""},
            {"name": "CLEAR canonical", "value": series["CLEAR"][0], "unit": ""},
            {"name": "CLEAR reversed", "value": series["CLEAR"][1], "unit": ""},
        ],
    )


# ── Figure 2: reversed-order retention per game ─────────────────────────────
def figure_reversed_retention(data: dict) -> dict:
    """Horizontal bars: per-game retention vs Joint, final task split out."""
    order = data["orders"]["reversed"]
    short = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    v5 = matrix(data, "v5_reversed")[-1] / joint
    clear = matrix(data, "clear_reversed")[-1] / joint

    prior = list(range(len(order) - 1))
    # y reversed so the first task learned sits at the top of the panel.
    labels = [short[order[i]] for i in prior][::-1]
    v5_prior = [v5[i] for i in prior][::-1]
    clear_prior = [clear[i] for i in prior][::-1]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.74, 0.26], horizontal_spacing=0.10,
        subplot_titles=("Previously learned tasks", "Final task"),
    )

    for column, (names, v5_values, clear_values) in enumerate(
        [(labels, v5_prior, clear_prior),
         ([short[order[-1]]], [v5[-1]], [clear[-1]])], start=1
    ):
        for method, values in (("CLEAR", clear_values), ("V5", v5_values)):
            fig.add_trace(go.Bar(
                y=names, x=values, orientation="h", name=NAME[method],
                marker=dict(color=hex_to_rgba(COLOR[method], 0.88),
                            line=dict(color=COLOR[method], width=1.0)),
                text=[f"{v:.0%}" for v in values],
                textposition="outside", cliponaxis=False,
                textfont=dict(family=FONT_MONO, size=11, color=COLOR[method]),
                hovertemplate="%{y}: %{x:.1%}<extra>" + NAME[method] + "</extra>",
                showlegend=False,
            ), row=1, col=column)
        fig.add_vline(x=1.0, line=dict(color=AC["green"], width=1.2, dash="dash"),
                      row=1, col=column)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT_UI, size=12, color=AC["text_primary"])

    # Every annotation below avoids a numeric y on these categorical axes and
    # avoids leader lines: Plotly grows the autorange to fit an arrow, which
    # collapses the bars into a corner. Categories are addressed by name and
    # nudged in pixels, and the y range is pinned explicitly.
    fig.add_annotation(
        x=1.0, xref="x", y=1.0, yref="paper", text="Joint ceiling",
        showarrow=False, xanchor="center", yanchor="top", yshift=-6,
        font=dict(family=FONT_UI, size=10, color=AC["green"]),
    )
    fig.add_annotation(
        x=1.07, xref="x", y=labels[1], yref="y",
        text="CLEAR keeps 8%<br>of Breakout", showarrow=False,
        xanchor="left", yanchor="middle", align="left",
        font=dict(family=FONT_UI, size=10, color=AC["amber"]),
    )

    # Direct series labels, inside the bars of the top group.
    for method, shift in (("V5", 11), ("CLEAR", -11)):
        fig.add_annotation(
            x=0.02, xref="x", y=labels[-1], yref="y", yshift=shift,
            text=f"<b>{NAME[method]}</b>", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=11, color=AC["bg"]),
        )

    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=AC["grid"],
                     zeroline=False, nticks=5)
    fig.update_xaxes(range=[0, 1.46], title=dict(text="Retention vs Joint ceiling"),
                     row=1, col=1)
    fig.update_xaxes(range=[0, 4.3], nticks=4,
                     title=dict(text="× Joint ceiling (own scale)"), row=1, col=2)
    fig.update_yaxes(showgrid=False, ticklen=0)
    fig.update_yaxes(range=[-0.62, len(labels) - 0.38], row=1, col=1)
    fig.update_yaxes(range=[-0.62, 0.62], row=1, col=2)
    fig.update_layout(
        title=dict(text="Reversed order: what survives to the end of the sequence"),
        barmode="group", bargap=0.34, bargroupgap=0.16, showlegend=False,
        margin=dict(l=92, r=28, t=68, b=68),
    )

    stem = "fig2_reversed_retention"
    width = W_FULL
    variant = export_pair(fig, stem, width, height_for(width, 2.0), "Bars")
    prior_v5 = float(np.mean(v5[:-1]))
    prior_clear = float(np.mean(clear[:-1]))
    return dict(
        id="reversed_retention",
        title="Reversed order: retention against the Joint ceiling",
        caption=(
            f"On the four previously learned tasks V5 keeps {prior_v5:.0%} of the "
            f"Joint ceiling and CLEAR keeps {prior_clear:.0%}."
        ),
        details=(
            "Final score after the whole reversed sequence, divided by the "
            "order-independent Joint ceiling for that game. Tasks run top to "
            "bottom in the order they were learned.\n\n"
            "The final task sits in its own panel on its own scale. Q\\*bert was "
            "learned last, so nothing has been trained after it and its score is "
            "not a retention measurement. CLEAR reaches "
            f"{clear[-1]:.0%} of the ceiling there, which is the cell that lifts "
            "its five-game mean above V5 while it is forgetting the rest.\n\n"
            "- **What to look for.** Breakout. V5 holds "
            f"{v5[2]:.0%}, CLEAR holds {clear[2]:.0%}.\n"
            "- CLEAR is below the ceiling on every previously learned task.\n\n"
            + PROVENANCE
        ),
        variants=[variant],
        metrics=[
            {"name": "V5 prior-task mean", "value": prior_v5, "unit": ""},
            {"name": "CLEAR prior-task mean", "value": prior_clear, "unit": ""},
            {"name": "CLEAR final task", "value": float(clear[-1]), "unit": "x joint"},
        ],
    )


# ── Figure 3: reversed-order raw scores ─────────────────────────────────────
def figure_reversed_raw(data: dict) -> dict:
    """Small multiples of raw final scores, one panel per game, own y-axis."""
    order = data["orders"]["reversed"]
    labels = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    local = reference_in_order(data, "local_reversed", order)
    v5 = matrix(data, "v5_reversed")[-1]
    clear = matrix(data, "clear_reversed")[-1]

    methods = ["Local", "V5", "CLEAR", "Joint"]
    per_game = {"Local": local, "V5": v5, "CLEAR": clear, "Joint": joint}

    # Each panel carries its own y-axis, so panels need real space between them
    # or one panel's tick labels land on the previous panel's bars.
    short_names = {"Local": "Local", "V5": "V5", "CLEAR": "CLEAR", "Joint": "Joint"}
    fig = make_subplots(
        rows=1, cols=len(order), horizontal_spacing=0.075,
        subplot_titles=[f"{i + 1}. {labels[g]}" for i, g in enumerate(order)],
    )
    for column, _game in enumerate(order, start=1):
        values = [per_game[m][column - 1] for m in methods]
        fig.add_trace(go.Bar(
            x=[short_names[m] for m in methods], y=values,
            marker=dict(color=[hex_to_rgba(COLOR[m], 0.88) for m in methods],
                        line=dict(color=[COLOR[m] for m in methods], width=1.0)),
            text=[f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" for v in values],
            textposition="outside", cliponaxis=False,
            textfont=dict(family=FONT_MONO, size=9, color=AC["text_muted"]),
            hovertemplate="%{x}: %{y:,.1f}<extra></extra>", showlegend=False,
        ), row=1, col=column)
        # Raw scales differ by ~700x, so every panel gets headroom of its own.
        top = max(values) * 1.30
        fig.update_yaxes(range=[0, top], row=1, col=column)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT_UI, size=11, color=AC["text_primary"])

    # Parked in the empty upper-left of the Q*bert panel. A leader line would
    # have to leave the panel to find clear space.
    fig.add_annotation(
        x="Local", y=float(clear[-1]) * 0.74, xref="x5", yref="y5",
        text="3.6×<br>ceiling", showarrow=False,
        xanchor="left", yanchor="middle", align="left", xshift=-12,
        font=dict(family=FONT_UI, size=10, color=AC["amber"]),
    )

    fig.update_xaxes(tickfont=dict(size=9), showgrid=False, ticklen=0)
    fig.update_yaxes(showgrid=True, gridcolor=AC["grid"], nticks=5, zeroline=False)
    fig.update_yaxes(title=dict(text="Greedy-100 score"), row=1, col=1)
    fig.update_layout(
        title=dict(text="Reversed order: raw final scores, panels in learning order"),
        showlegend=False, margin=dict(l=76, r=24, t=72, b=56),
    )

    stem = "fig3_reversed_raw_scores"
    width = W_FULL
    variant = export_pair(fig, stem, width, height_for(width, 1.85), "Small multiples")
    return dict(
        id="reversed_raw_scores",
        title="Reversed order: raw final scores",
        caption=(
            "Greedy-100 score on each game after the full reversed sequence, with "
            "the Local specialist and Joint ceiling alongside. Own y-axis per game."
        ),
        details=(
            "The absolute numbers behind the retention ratios. Panels run left to "
            "right in learning order.\n\n"
            "- **Own y-axis per panel, always including zero.** Raw scales differ "
            "by roughly 700x (Pong around 20, Q\\*bert around 4000), so a shared "
            "axis would flatten four of the five panels. No log scale, no "
            "clipping, no normalisation.\n"
            "- **Local** is the single-task specialist and is itself "
            "order-dependent, which is why retention elsewhere in this group is "
            "measured against Joint instead.\n"
            "- **What to look for.** Q\\*bert, where CLEAR reaches 15351 against a "
            "Joint ceiling of 4262.\n\n" + PROVENANCE
        ),
        variants=[variant],
        metrics=[
            {"name": "CLEAR Q*bert", "value": float(clear[-1]), "unit": "score"},
            {"name": "Joint Q*bert", "value": float(joint[-1]), "unit": "score"},
            {"name": "V5 Breakout", "value": float(v5[2]), "unit": "score"},
            {"name": "CLEAR Breakout", "value": float(clear[2]), "unit": "score"},
        ],
    )


# ── Figure 4: reversed-order forgetting trajectories ────────────────────────
def figure_reversed_trajectories(data: dict) -> dict:
    """Small multiples: each game's retention across the training phases after it."""
    order = data["orders"]["reversed"]
    labels = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    tracks = {
        "V5": retention_matrix(matrix(data, "v5_reversed"), joint),
        "CLEAR": retention_matrix(matrix(data, "clear_reversed"), joint),
    }

    # Only the tasks with a phase after them have a trajectory to draw.
    columns = list(range(len(order) - 1))
    phase_labels = [labels[g] for g in order]

    fig = make_subplots(
        rows=1, cols=len(columns), shared_yaxes=True, horizontal_spacing=0.028,
        subplot_titles=[f"{i + 1}. {labels[order[i]]}" for i in columns],
    )
    for column_position, task in enumerate(columns, start=1):
        phases = list(range(task, len(order)))
        for method in ("CLEAR", "V5"):
            values = [tracks[method][p, task] for p in phases]
            fig.add_trace(go.Scatter(
                x=phases, y=values, mode="lines+markers",
                line=dict(color=COLOR[method], width=2.0 if method == "V5" else 1.4),
                marker=dict(color=COLOR[method], size=6,
                            line=dict(color=AC["bg"], width=1.2)),
                hovertemplate=(f"{NAME[method]}<br>after %{{text}}: %{{y:.1%}}"
                               "<extra></extra>"),
                text=[phase_labels[p] for p in phases], showlegend=False,
            ), row=1, col=column_position)
        fig.add_hline(y=1.0, line=dict(color=AC["green"], width=1.0, dash="dash"),
                      row=1, col=column_position)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT_UI, size=11, color=AC["text_primary"])

    # Series labelled directly in the first panel, parked in its empty upper band
    # so neither label sits on a line.
    for method, y_position in (("V5", 1.40), ("CLEAR", 1.24)):
        fig.add_annotation(
            x=0.5, y=y_position, xref="x", yref="y",
            text=f"<b>{NAME[method]}</b>", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=11, color=COLOR[method]),
        )
    fig.add_annotation(
        x=0.15, y=1.0, xref="x", yref="y", text="Joint ceiling", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(family=FONT_UI, size=9.5, color=AC["green"]),
    )
    # xref/yref must name the third panel's axes or the callout lands on panel 1.
    # No leader line: every path from clear space to the 8% point would cross a
    # series line, and the panel it sits in is unambiguous.
    fig.add_annotation(
        x=-0.2, y=0.46, xref="x3", yref="y3",
        text="Breakout collapses<br>once Pong arrives", showarrow=False,
        xanchor="left", yanchor="middle", align="left",
        font=dict(family=FONT_UI, size=10, color=AC["amber"]),
    )

    fig.update_xaxes(
        tickmode="array", tickvals=list(range(len(order))),
        ticktext=[str(p + 1) for p in range(len(order))],
        range=[-0.35, len(order) - 0.65], showgrid=False, zeroline=False,
        title=dict(text="training phase", font=dict(size=10)),
    )
    fig.update_yaxes(range=[0, 1.55], tickformat=".0%", nticks=5,
                     showgrid=True, gridcolor=AC["grid"], zeroline=False)
    fig.update_yaxes(title=dict(text="Retention vs Joint ceiling"), row=1, col=1)
    fig.update_layout(
        title=dict(text="Reversed order: how each task decays after it is learned"),
        showlegend=False, margin=dict(l=80, r=24, t=72, b=64),
    )

    stem = "fig4_reversed_trajectories"
    width = W_FULL
    variant = export_pair(fig, stem, width, height_for(width, 2.15), "Small multiples")
    return dict(
        id="reversed_trajectories",
        title="Reversed order: decay of each task after it is learned",
        caption=(
            "Retention of each game against the Joint ceiling at every later "
            "training phase. V5 dips and recovers on Breakout; CLEAR does not."
        ),
        details=(
            "One panel per task, in learning order. The x-axis is the absolute "
            "training phase, shared across panels, so each line starts on its own "
            "diagonal and shortens as the sequence advances.\n\n"
            "This is the same information as the forgetting matrix, read as a "
            "trajectory instead of a grid, which is what makes the *timing* "
            "visible. Training phase 1 is SpaceInvaders, 2 Boxing, 3 Breakout, "
            "4 Pong, 5 Q\\*bert, matching the panel numbering.\n\n"
            "- **Q\\*bert is absent.** It is the last task in this order, so there "
            "is no later phase to plot.\n"
            "- **What to look for.** Breakout. Both methods leave it near the "
            "ceiling, then V5 falls to 55% and recovers to 70% while CLEAR goes "
            "19% and then 8%.\n"
            "- Boxing shows the opposite shape: CLEAR holds it longer, then drops "
            "below V5 at the end.\n\n" + PROVENANCE
        ),
        variants=[variant],
        metrics=[
            {"name": "V5 Breakout final", "value": float(tracks["V5"][4, 2]), "unit": ""},
            {"name": "CLEAR Breakout final", "value": float(tracks["CLEAR"][4, 2]), "unit": ""},
        ],
    )


# ── Figure 5: forgetting matrices, both orders ──────────────────────────────
# Shared across both variants so a reader switching between them is not misled
# by a shifting colour meaning. Symmetric about the ceiling; the reversed
# Q*bert cell (3.6x) saturates but keeps its printed number.
MATRIX_ZMIN, MATRIX_ZMAX = -0.5, 2.5


def _matrix_panel(data: dict, order_name: str) -> go.Figure:
    """Build the V5-vs-CLEAR retention matrix pair for one task order."""
    order = data["orders"][order_name]
    labels = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    panels = [
        ("V5 (min-max)", retention_matrix(
            matrix(data, f"v5_{order_name}"), joint)),
        ("CLEAR", retention_matrix(
            matrix(data, f"clear_{order_name}"), joint)),
    ]
    columns = [labels[g] for g in order]
    rows = [labels[g] for g in order]

    # Red is *below* the ceiling. acviz.DIVERGING runs blue -> light -> red,
    # which would paint total forgetting in the primary series colour.
    scale = [[stop, color] for stop, color in zip(
        [s for s, _ in DIVERGING], [c for _, c in reversed(DIVERGING)])]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.17,
                        subplot_titles=[name for name, _ in panels])
    for column, (_name, values) in enumerate(panels, start=1):
        fig.add_trace(go.Heatmap(
            z=values, x=columns, y=rows, colorscale=scale,
            zmid=1.0, zmin=MATRIX_ZMIN, zmax=MATRIX_ZMAX, xgap=2, ygap=2,
            hoverongaps=False, showscale=(column == 2),
            hovertemplate="%{y}<br>%{x}: %{z:.1%} of ceiling<extra></extra>",
            colorbar=dict(
                title=dict(text="vs ceiling", font=dict(family=FONT_UI, size=11)),
                tickformat=".0%", thickness=10, len=0.74, outlinewidth=0,
                tickfont=dict(family=FONT_MONO, size=10, color=AC["text_muted"]),
            ),
        ), row=1, col=column)

        for row_index in range(values.shape[0]):
            for column_index in range(row_index + 1):
                value = values[row_index, column_index]
                on_diagonal = row_index == column_index
                fig.add_annotation(
                    x=columns[column_index], y=rows[row_index],
                    xref=f"x{column if column > 1 else ''}",
                    yref=f"y{column if column > 1 else ''}",
                    text=(f"<b>{value:.0%}</b>" if on_diagonal else f"{value:.0%}"),
                    showarrow=False,
                    font=dict(family=FONT_MONO, size=10,
                              color=AC["bg"] if (value < 0.10 or value > 1.95)
                              else AC["text_primary"]),
                )

    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=12, color=AC["text_primary"])

    fig.update_xaxes(title=dict(text="evaluated on (learning order →)"),
                     showgrid=False, ticklen=0, side="bottom")
    fig.update_yaxes(autorange="reversed", showgrid=False, ticklen=0)
    fig.update_yaxes(title=dict(text="training phase (task just consolidated)"),
                     row=1, col=1)
    fig.update_layout(
        title=dict(text=f"{order_name.capitalize()} order: full forgetting matrices"),
        margin=dict(l=124, r=96, t=68, b=72),
    )
    return fig


def figure_matrices(data: dict) -> dict:
    """Forgetting matrices for both orders, reversed first as the paper figure."""
    order = data["orders"]["reversed"]
    joint = reference_in_order(data, "joint", order)
    reversed_values = retention_matrix(matrix(data, "clear_reversed"), joint)

    width = W_FULL
    height = height_for(width, 1.72)
    variants = [
        export_pair(_matrix_panel(data, "reversed"), "fig5_matrices_reversed",
                    width, height, "Reversed order"),
        export_pair(_matrix_panel(data, "canonical"), "fig5_matrices_canonical",
                    width, height, "Canonical order"),
    ]

    return dict(
        id="forgetting_matrices",
        title="Forgetting matrices, both task orders",
        caption=(
            "Every evaluation as a fraction of the Joint ceiling. Bold diagonal is "
            "the task just consolidated; blank upper triangle is never evaluated."
        ),
        details=(
            "Rows are training phases, columns are tasks in learning order. The "
            "**bold diagonal** is each game measured right after its own "
            "consolidation; everything below it is retention. The blank upper "
            "triangle is never evaluated and never imputed.\n\n"
            "- **Reversed is the paper figure**; switch variants for the canonical "
            "order. Both share one colour scale "
            f"({MATRIX_ZMIN:.0%} to {MATRIX_ZMAX:.0%} of the ceiling, centred on "
            "the ceiling) so switching between them is not misleading.\n"
            "- **Red is below the ceiling, blue above.** CLEAR's reversed Q\\*bert "
            "cell is 360% and saturates at the top of the scale; it keeps its "
            "printed number.\n"
            "- **What to look for.** CLEAR's Breakout column in the reversed order "
            "walking 106% → 19% → 8% down the rows, against V5's 103% → 55% → 70%. "
            "In the canonical order the roles swap and it is V5 that loses Boxing "
            "outright, to -38%.\n\n" + PROVENANCE
        ),
        variants=variants,
        metrics=[
            {"name": "CLEAR Breakout (rev, final)",
             "value": float(reversed_values[4, 2]), "unit": ""},
            {"name": "colour range",
             "value": f"{MATRIX_ZMIN:.0%} to {MATRIX_ZMAX:.0%}", "unit": ""},
        ],
    )


# ── Export plumbing ─────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int,
                label: str) -> dict:
    """Export one figure into the report folder and the dashboard figure folder.

    Returns the manifest variant dict, whose paths are relative to ``report/``.
    """
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    DASH_FIGURES.mkdir(parents=True, exist_ok=True)

    # Standalone report keeps the repository's png/ and svg/ split.
    export_figure(fig, PNG_DIR / stem, width, height, label=label)
    svg_source = (PNG_DIR / stem).with_suffix(".svg")
    svg_source.replace(SVG_DIR / f"{stem}.svg")

    # Dashboard copy, exported from the same figure object.
    return export_figure(fig, DASH_FIGURES / stem, width, height, label=label)


def main(argv: list[str] | None = None) -> int:
    """Render every order-sensitivity figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-dashboard", action="store_true",
                        help="skip updating report/manifest.json")
    args = parser.parse_args(argv)

    install_template()
    data = load_data()

    print("order sensitivity | seed 0, greedy-100, reversed-order focus")
    builders = [
        figure_order_contrast,
        figure_reversed_retention,
        figure_reversed_raw,
        figure_reversed_trajectories,
        figure_matrices,
    ]
    for builder in builders:
        entry = builder(data)
        print(f"  {entry['id']}")
        if not args.no_dashboard:
            append_manifest_entry(
                DASH_MANIFEST, group=GROUP, group_title=GROUP_TITLE,
                seed=0, dataset="Atari 5-game sequence (greedy-100)",
                model="impala_ac_multihead", **entry,
            )

    print(f"png: {PNG_DIR}")
    print(f"svg: {SVG_DIR}")
    if not args.no_dashboard:
        print(f"manifest: {DASH_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
