#!/usr/bin/env python3
"""The opening figure: three mechanisms, two stressors.

An introduction figure has to carry one idea, so this one groups methods by the
mechanism they use rather than naming them. A reader counts three things, not
eleven, and every name is waiting in Section 5.

Each band spans the members measured in that column and the line is the
strongest of them, so the baselines are read at their most favourable. Where a
band narrows to a line, one method is left in that group, which is the same
convention the Setting B curves use.

Methods that grow the network are deliberately absent. They are excluded from
the fifty-task setting by construction, so a line for them would break in the
middle column, and the question they raise belongs in Section 2 rather than in
the first figure.

Nothing here is a new experiment. Column A is the Average column of Table 1,
column B is the PERF column of Table 3, and column C is the prior-four PERF
column of Table 4.

Usage::

    python reports/final/motivation/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, hex_to_rgba, install_template,
)

PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"

C_OURS, C_FAIL, C_REPLAY = AC["blue"], AC["red"], AC["amber"]
XS = [0.0, 1.0, 2.0]

# (label, colour, width, members per column). Every measured member is drawn as
# a small dot and the line runs through the strongest of them, so each group is
# read at its most favourable. A column with one member shows one dot, which is
# why no band is drawn: a band that closes to a line reads as an artifact.
# Fine-tuning is left out of the regularization group so the label is exact; it
# changes no value here, since it is never the strongest member.
GROUPS = [
    ("VCC (ours)", C_OURS, 3.6,
     [[0.742], [0.819], [0.670]]),
    ("Replay", C_REPLAY, 2.2,
     [[], [0.686, 0.590], [0.430]]),
    ("Regularization and plasticity", C_FAIL, 2.4,
     [[0.750, 0.683, 0.681], [0.012, -0.004, -0.001], [-0.250]]),
]
# Replay methods are absent from the published table for the left-hand
# benchmark. The series is carried in from the level every measured method
# falls in, as a dotted segment with a hollow marker, so the reader can see
# that it is not a measurement.
REPLAY_ENTRY = 0.715

COLUMNS = [
    ("Established benchmark", "eight to ten modes of one game", "the usual test"),
    ("Longer sequence",       "fifty tasks of one family",      "length"),
    ("Dissimilar tasks",      "five distinct Atari games",      "heterogeneity"),
]


def fig_motivation() -> go.Figure:
    fig = go.Figure()
    X0, X1, YLO, YHI = -0.52, 2.46, -0.34, 1.00

    for i, x in enumerate(XS):
        if i % 2 == 0:
            fig.add_shape(type="rect", x0=x - 0.44, x1=x + 0.44, y0=YLO, y1=YHI,
                          layer="below", fillcolor=AC["surface"], line=dict(width=0))
    # At or below random, drawn as a zone: a score anywhere in it means the
    # same thing.
    fig.add_shape(type="rect", x0=X0, x1=X1, y0=YLO, y1=0.05, layer="below",
                  fillcolor=hex_to_rgba(C_FAIL, 0.07), line=dict(width=0))
    fig.add_annotation(x=X0 + 0.04, y=-0.16, xanchor="left", yanchor="middle",
                       showarrow=False, text="at or below random",
                       font=dict(family=FONT_UI, size=10, color=C_FAIL))

    for label, color, width, cols in GROUPS:
        ours = label.startswith("VCC")
        pts = [(x, ms) for x, ms in zip(XS, cols) if ms]
        xs = [x for x, _ in pts]
        best = [max(ms) for _, ms in pts]

        if label == "Replay":
            # Carried in from the cluster, dotted, so it reads as absent rather
            # than as a measured value.
            fig.add_trace(go.Scatter(
                x=[-0.075, xs[0]], y=[REPLAY_ENTRY, best[0]], mode="lines",
                hoverinfo="skip", showlegend=False,
                line=dict(color=hex_to_rgba(color, 0.45), width=width, dash="dot")))
            fig.add_trace(go.Scatter(
                x=[-0.075], y=[REPLAY_ENTRY], mode="markers", hoverinfo="skip",
                showlegend=False,
                marker=dict(color=AC["bg"], size=11,
                            line=dict(color=hex_to_rgba(color, 0.65), width=2.0))))

        fig.add_trace(go.Scatter(
            x=xs, y=best, mode="lines", hoverinfo="skip", showlegend=True,
            name=f"<b>{label}</b>", line=dict(color=color, width=width)))
        # Every measured member, so the reader sees how many back each point.
        # They are fanned sideways because several of them sit within 0.01 of
        # one another and would otherwise print as a single smudge.
        for x, ms in pts:
            others = [v for v in ms if v < max(ms)]
            for i, v in enumerate(others):
                fig.add_trace(go.Scatter(
                    x=[x + 0.062 * (i + 1)], y=[v], mode="markers",
                    hoverinfo="skip", showlegend=False,
                    marker=dict(color=hex_to_rgba(color, 0.50), size=8,
                                line=dict(color=AC["bg"], width=1.3))))
        fig.add_trace(go.Scatter(
            x=xs, y=best, mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(color=color, size=15 if ours else 11,
                        line=dict(color=AC["bg"], width=2.0))))
        for x, y in zip(xs, best):
            if x < 0.5:
                continue
            above = ours or label.startswith("Regul")
            fig.add_annotation(x=x, y=y, text=f"{y:.2f}", showarrow=False,
                               yshift=22 if above else -24,
                               xshift=0 if above else -26,
                               font=dict(family=FONT_MONO,
                                         size=12 if ours else 11, color=color))

    for x, (head, sub, tag) in zip(XS, COLUMNS):
        for shift, txt, size, col in (
                (46, f"<b>{head}</b>", 13, AC["text_primary"]),
                (29, sub, 10.5, AC["text_muted"]),
                (10, f"<i>{tag}</i>", 10, AC["text_faint"])):
            fig.add_annotation(x=x, y=YHI, yshift=shift, showarrow=False,
                               xanchor="center", text=txt,
                               font=dict(family=FONT_UI, size=size, color=col))

    fig.update_xaxes(range=[X0, X1], showgrid=False, zeroline=False,
                     showticklabels=False, showline=False, ticks="")
    fig.update_yaxes(
        title=dict(text="Normalized performance after the sequence",
                   font=dict(family=FONT_UI, size=12)),
        range=[YLO, YHI + 0.02], dtick=0.25, tickformat=".2f",
        tickfont=dict(family=FONT_MONO, size=10.5), zeroline=False)
    fig.update_layout(
        title=None, margin=dict(l=62, r=30, t=78, b=52), plot_bgcolor=AC["bg"],
        showlegend=True,
        legend=dict(orientation="h", x=0.5, y=-0.04, xanchor="center",
                    yanchor="top", font=dict(family=FONT_UI, size=12),
                    itemsizing="constant", bgcolor="rgba(0,0,0,0)",
                    borderwidth=0))
    return fig


def main() -> None:
    install_template()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig, w, h = fig_motivation(), W_FULL, 440
    # PNG and SVG come from the SAME figure object, never two code paths.
    fig.write_image(PNG_DIR / "motivation.png", width=w, height=h, scale=3.125)
    fig.write_image(SVG_DIR / "motivation.svg", width=w, height=h)
    for p in (PNG_DIR / "motivation.png", SVG_DIR / "motivation.svg"):
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"export failed or empty: {p}")
        print(f"  {p.relative_to(HERE)}  {p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
