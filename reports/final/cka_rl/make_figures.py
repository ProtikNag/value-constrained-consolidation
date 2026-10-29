#!/usr/bin/env python3
"""Finalised paper figures, line 2: the CKA-RL (NeurIPS 2025) comparison.

Figure 1 reproduces the structure of the paper's Table 1 (average performance
and forward transfer across Meta-World, SpaceInvaders and Freeway) with our
DUEL row added.

**Meta-World is left blank for our row**, because that run is still going at a
reduced budget.

Published rows keep the paper's own three-environment Average untouched. Our
Average is over the two environments we have, so it is **not on the same basis**
and is marked with a dagger. Meta-World is where every method scores lowest
(0.00 to 0.46), so dropping it raises the mean: ours' 0.867 is not a like-for-like
0.867 against the published column and must not be read as leading it.

Ranking therefore uses two different quantities on purpose. Published rows are
ordered by their own published Average. Ours is **inserted at the rank it earns
on SpaceInvaders and Freeway**, the only two environments both sides have, which
is third. That is a real comparison on shared ground; what would not be real is
ordering ours' two-environment 0.867 against the published three-environment
column, where it would sort first for no reason but the missing environment.

When Meta-World lands, ours gets a genuine three-environment Average, the dagger
goes away, and the whole table ranks on one quantity.

Usage::

    python reports/final/cka_rl/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, export_figure, install_template,
)

DATA = HERE / "data.json"
PNG_DIR = HERE / "png"
SVG_DIR = HERE / "svg"

ENV_ORDER = ["meta_world", "space_invaders", "freeway", "average"]

# Two banded rows: ours, and the paper's own best method as the thing to beat.
# Tints are the palette's selected-region and warning grounds; acviz.AC carries
# the series and neutral tokens but not these two.
BAND = {
    "DUEL (ours)": ("#EFF6FF", AC["blue"]),
    "CKA-RL": ("#FEF3C7", AC["amber"]),
}


AVERAGED_OVER = ["space_invaders", "freeway"]


def load_data() -> dict:
    """Load the transcribed paper table plus our row."""
    return json.loads(DATA.read_text(encoding="utf-8"))


def fill_average(rows: list[dict]) -> None:
    """Populate each row's ``average``, in place.

    Published rows keep the paper's three-environment Average verbatim. Our row
    has no Meta-World result yet, so its Average is taken over the two
    environments we do have and flagged ``partial``. The two are **not the same
    quantity**: Meta-World is the low-scoring environment for every method, so
    omitting it inflates the mean. Everything downstream that displays or ranks
    has to respect that flag.

    No standard deviation is derived for the partial mean. Combining two
    per-environment standard deviations needs their seed-level covariance, which
    the paper does not report, so any number there would be invented.
    """
    for row in rows:
        published = row.get("average_paper_3env")
        if published is not None:
            row["average"] = published
            row["average_partial"] = False
            continue
        blocks = [row.get(env) for env in AVERAGED_OVER]
        if any(block is None for block in blocks):
            row["average"] = None
            row["average_partial"] = False
            continue
        row["average"] = [
            [sum(block[metric][0] for block in blocks) / len(blocks), None]
            for metric in (0, 1)
        ]
        row["average_partial"] = True


def figure_table1(data: dict) -> None:
    """Table 1 of the CKA-RL paper with our DUEL row added."""
    labels = dict(data["environment_labels"])
    rows = list(data["rows"])
    fill_average(rows)

    # Published rows are ordered by their own published Average.
    ranked = sorted((r for r in rows if not r["average_partial"]),
                    key=lambda r: r["average"][0][0], reverse=True)

    # A partial row is slotted in at the rank it earns on the environments both
    # sides share, not by its own partial mean. Ranking a two-environment mean
    # against the published three-environment column would sort it to the top
    # purely because the missing environment is the low-scoring one.
    ordered = list(ranked)
    for row in (r for r in rows if r["average_partial"]):
        shared = sorted(
            (r for r in rows if all(r.get(env) for env in AVERAGED_OVER)),
            key=lambda r: sum(r[env][0][0] for env in AVERAGED_OVER),
            reverse=True,
        )
        ordered.insert(shared.index(row), row)

    # Column best, computed over every row that actually has the value, so ours
    # competes for bold in the two environments it has.
    def cell(row: dict, env: str, metric: int):
        block = row.get(env)
        return None if block is None else block[metric]

    best: dict[tuple[str, int], float] = {}
    for env in ENV_ORDER:
        for metric in (0, 1):
            # Baseline is excluded from the forward-transfer best. FWT is defined
            # against Baseline's own learning curve (paper eq. 9), so Baseline
            # scores exactly 0 by construction and is the reference, not a
            # competitor. The paper bolds it nowhere in FWT either.
            contenders = [
                r for r in ordered
                if not (metric == 1 and r["method"] == "Baseline")
                # A two-environment mean cannot win the three-environment column.
                and not (env == "average" and r["average_partial"])
            ]
            values = [cell(r, env, metric) for r in contenders]
            values = [v[0] for v in values if v is not None]
            if values:
                best[(env, metric)] = max(values)

    # ── Geometry. Arbitrary units on hidden axes. ──────────────────────────
    label_x = 0.0
    col_x: dict[tuple[str, int], float] = {}
    group_span: dict[str, tuple[float, float]] = {}
    cursor = 21.0
    for env in ENV_ORDER:
        left = cursor
        for metric in (0, 1):
            col_x[(env, metric)] = cursor + 8.6
            cursor += 9.4
        group_span[env] = (left, cursor - 0.8)
        cursor += 2.4
    right_edge = cursor - 2.4

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             hoverinfo="skip", showlegend=False))

    def rule(y: float, x0: float, x1: float, width: float, color: str) -> None:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      line=dict(color=color, width=width), layer="above")

    group_y, metric_y = 2.25, 1.15
    rule(3.0, label_x - 1, right_edge, 1.3, AC["axis"])           # top rule
    for env in ENV_ORDER:
        left, right = group_span[env]
        fig.add_annotation(
            x=(left + right) / 2, y=group_y, text=labels[env], showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(family=FONT_UI, size=10.5, color=AC["text_primary"]),
        )
        rule(group_y - 0.55, left, right, 0.8, AC["border"])       # group underline
    fig.add_annotation(
        x=label_x, y=group_y, text="<b>Method</b>", showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(family=FONT_UI, size=10.5, color=AC["text_primary"]),
    )
    for (env, metric), x in col_x.items():
        fig.add_annotation(
            x=x, y=metric_y, text="PERF." if metric == 0 else "FWT.",
            showarrow=False, xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
        )
    rule(0.6, label_x - 1, right_edge, 1.0, AC["axis"])            # under header

    for index, row in enumerate(ordered):
        y = -index - 0.5
        is_ours = bool(row.get("ours"))
        band = BAND.get(row["method"])
        if band is not None:
            # Tinted band so the two rows that matter are findable at a glance
            # without breaking the ranking they both legitimately sit inside.
            fig.add_shape(
                type="rect", layer="below",
                x0=label_x - 1, x1=right_edge, y0=y - 0.46, y1=y + 0.46,
                fillcolor=band[0], line=dict(width=0),
            )
        accent = band[1] if band is not None else AC["text_primary"]
        fig.add_annotation(
            x=label_x, y=y,
            text=f"<b>{row['method']}</b>" if band is not None else row["method"],
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=9.5, color=accent),
        )
        for env in ENV_ORDER:
            for metric in (0, 1):
                value = cell(row, env, metric)
                x = col_x[(env, metric)]
                if value is None:
                    fig.add_annotation(
                        x=x, y=y, text="—", showarrow=False,
                        xanchor="right", yanchor="middle",
                        font=dict(family=FONT_MONO, size=9,
                                  color=AC["text_faint"]),
                    )
                    continue
                mean, std = value
                is_best = abs(mean - best[(env, metric)]) < 1e-9
                # Ours is quoted to the three decimals the run actually reports;
                # padding it to four would imply precision we do not have.
                body = f"{mean:.3f}" if is_ours else f"{mean:.4f}"
                if is_best:
                    body = f"<b>{body}</b>"
                if env == "average" and row["average_partial"]:
                    body += (f"<span style=\"font-size:8px;"
                             f"color:{AC['text_muted']}\">†</span>")
                if std is not None:
                    body += (f"<span style=\"font-size:6.5px;"
                             f"color:{AC['text_faint']}\">±{std:.2f}</span>")
                fig.add_annotation(
                    x=x, y=y, text=body, showarrow=False,
                    xanchor="right", yanchor="middle",
                    font=dict(family=FONT_MONO, size=8.5, color=accent),
                )

    bottom = -len(ordered)
    rule(bottom - 0.05, label_x - 1, right_edge, 1.3, AC["axis"])  # bottom rule

    footnote = (
        "Published rows transcribed from Hu et al., CKA-RL, NeurIPS 2025, Table 1: "
        "mean over 10 seeds, ± standard deviation. Higher is better throughout; "
        "<b>bold</b> is the best in each<br>"
        "column. Published rows are sorted by their own Average performance. "
        "Baseline is excluded from the forward-transfer best, because forward "
        "transfer is defined<br>"
        "against Baseline's own curve, so its 0.0000 is the reference rather "
        "than a result.<br>"
        "† <b>Ours' Average is over SpaceInvaders and Freeway only</b>, not the "
        "three environments the column above it spans, so the two are not "
        "comparable: Meta-World is<br>"
        "&nbsp;&nbsp;&nbsp;the low-scoring environment for every method, and "
        "omitting it raises the mean. Ours is therefore placed at the rank it "
        "earns on SpaceInvaders and Freeway, the<br>"
        "&nbsp;&nbsp;&nbsp;two environments both sides have, rather than by the "
        "daggered number itself.<br>"
        "— Ours is <b>seed 0 only</b>, so it has no standard deviation, and its "
        "Meta-World run is still going. Ours also has live past-task environment "
        "access, which no baseline<br>"
        "&nbsp;&nbsp;&nbsp;has, and uses over 2× the frames per task. Table 1 "
        "measures plasticity; ours' retention advantage shows in Table 3."
    )
    fig.add_annotation(
        x=label_x - 1, y=bottom - 0.65, text=footnote,
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=7.5, color=AC["text_muted"]),
    )

    fig.update_xaxes(visible=False, range=[label_x - 2, right_edge + 1])
    fig.update_yaxes(visible=False, range=[bottom - 5.6, 3.6])
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=14, r=14, t=12, b=10))

    export_pair(fig, "table1_perf_fwt", 800, 440)


# ── Figures 2+: per-mode final-policy detail ────────────────────────────────
RAW = HERE / "raw"

PASS_COLOR, FAIL_COLOR = AC["blue"], AC["red"]


def tint(hex_color: str, weight: float) -> str:
    """Blend ``hex_color`` toward white by ``weight`` in [0, 1]."""
    channels = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#{:02X}{:02X}{:02X}".format(
        *[round(c + (255 - c) * weight) for c in channels])

ENV_FIGURES = [
    ("space_invaders", "SpaceInvaders", "final_policy_space_invaders"),
    ("freeway", "Freeway", "final_policy_freeway"),
]


def load_per_mode(env_dir: str) -> tuple[list[dict], dict]:
    """Per-mode final-policy records for one environment, in learning order."""
    payload = json.loads(
        (RAW / env_dir / "table3_final_policy.json").read_text(encoding="utf-8"))
    records = [dict(payload["per_mode"][str(mode)], mode=mode)
               for mode in payload["modes"]]
    records.sort(key=lambda r: r["task_idx"])
    return records, payload


def figure_final_policy(env_dir: str, env_label: str, stem: str) -> None:
    """One mode per row: final return against the benchmark's fixed threshold.

    The paper reports only the mean success rate for this table. Everything
    behind that single number is drawn here: each mode's mean greedy return, the
    episode-to-episode spread, the threshold it had to clear, and the resulting
    pass or fail.

    Whiskers are ±1 standard deviation across the 100 evaluation episodes, which
    is the spread of play, not the uncertainty on the mean. The uncertainty on
    the mean is a tenth of that at n=100, so it is far too small to change any
    pass or fail here; the footnote says so rather than drawing a second
    interval nobody can read.
    """
    records, payload = load_per_mode(env_dir)
    count = len(records)
    passed = sum(1 for r in records if r["success"])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34],
                        horizontal_spacing=0.03, shared_yaxes=True)

    # Drawn as a dumbbell rather than as bars from zero. Every threshold here is
    # a different number, so what the reader has to judge is the *margin* over
    # each one; a bar from zero spends most of its ink below the threshold, where
    # nothing is being compared. Dropping the zero baseline also frees the axis
    # to cover only the range the data occupies.
    span_low = min(min(r["threshold"], r["mean_return"] - r["std_return"])
                   for r in records)
    span_high = max(r["mean_return"] + r["std_return"] for r in records)
    pad = (span_high - span_low) * 0.06
    axis_low, axis_high = span_low - pad, span_high + pad

    for index, record in enumerate(records):
        color = PASS_COLOR if record["success"] else FAIL_COLOR
        mean, std = record["mean_return"], record["std_return"]
        threshold = record["threshold"]

        # Faint rail, so the eye can run from the row to its numbers on the right.
        fig.add_trace(go.Scatter(
            x=[axis_low, axis_high], y=[index, index], mode="lines",
            line=dict(color=AC["grid"], width=0.8),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

        # Episode spread, under the margin bar so only the part that reaches
        # past the mean shows. That overhang is the informative half.
        fig.add_trace(go.Scatter(
            x=[mean - std, mean + std], y=[index, index], mode="lines",
            line=dict(color=tint(color, 0.62), width=1.6),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

        # The margin itself: threshold to achieved. Direction carries pass/fail
        # as much as colour does.
        fig.add_trace(go.Scatter(
            x=[threshold, mean], y=[index, index], mode="lines",
            line=dict(color=color, width=4.5),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

        # The bar to clear, as a gate rather than a data point.
        fig.add_shape(
            type="line", x0=threshold, x1=threshold,
            y0=index - 0.26, y1=index + 0.26,
            line=dict(color=AC["text_primary"], width=1.8),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[mean], y=[index], mode="markers",
            marker=dict(color=color, size=9,
                        line=dict(color=AC["bg"], width=1.6)),
            hovertemplate="%{x:.2f}<extra></extra>", showlegend=False,
        ), row=1, col=1)

    # Right panel: the numbers themselves, since the paper prints none of them.
    text_x = [0.30, 0.62, 0.90]
    headers = ["return", "threshold", "×thr"]
    for x, header in zip(text_x, headers):
        fig.add_annotation(
            x=x, y=-0.92, xref="x2", yref="y2", text=header, showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
        )
    for index, record in enumerate(records):
        ratio = record["mean_return"] / record["threshold"]
        color = PASS_COLOR if record["success"] else FAIL_COLOR
        cells = [
            f"{record['mean_return']:,.1f}",
            f"{record['threshold']:,.1f}",
            f"<b>{ratio:.2f}</b>",
        ]
        for x, cell, tone in zip(text_x, cells,
                                 [AC["text_primary"], AC["text_muted"], color]):
            fig.add_annotation(
                x=x, y=index, xref="x2", yref="y2", text=cell, showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(family=FONT_MONO, size=9, color=tone),
            )

    # Legend built from empty traces: the real marks are per-row, so they cannot
    # carry a shared legend entry themselves. Each glyph mirrors its mark.
    for name, color in [("clears threshold", PASS_COLOR),
                        ("below threshold", FAIL_COLOR)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines+markers", name=name,
            line=dict(color=color, width=4.5),
            marker=dict(color=color, size=9,
                        line=dict(color=AC["bg"], width=1.6)),
            showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="threshold",
        marker=dict(symbol="line-ns", size=10,
                    line=dict(color=AC["text_primary"], width=1.8)),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="±1 s.d.",
        line=dict(color=AC["text_faint"], width=1.6),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    fig.update_xaxes(
        title=dict(text="greedy-100 episode return",
                   font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"])),
        range=[axis_low, axis_high],
        showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=False, showline=True, linecolor=AC["border"], ticklen=0,
        tickfont=dict(family=FONT_MONO, size=9, color=AC["text_muted"]),
        row=1, col=1,
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=2)
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(count)),
        ticktext=[f"Mode {r['mode']}" for r in records],
        range=[count - 0.5, -1.25], showgrid=False, zeroline=False,
        showline=False, ticklen=0,
        tickfont=dict(family=FONT_UI, size=10, color=AC["text_primary"]),
        row=1, col=1,
    )
    fig.update_yaxes(visible=False, range=[count - 0.5, -1.25], row=1, col=2)

    rate = payload["success_rate"]
    fig.add_annotation(
        x=0, y=-0.30, xref="paper", yref="paper", xshift=-62,
        text=(f"<b>{passed} of {count} modes clear the benchmark's fixed "
              f"threshold</b> (success rate {rate:.2f}). Each bar runs from a "
              "mode's threshold to the mean of<br>"
              f"{payload['eval_episodes']} greedy episodes of the final policy, "
              "so its length and direction are the margin. The pale line is ±1 "
              "s.d. across those<br>"
              "episodes, which is the spread of play, not the uncertainty on "
              "the mean: at n=100 that uncertainty is a tenth of it, so every "
              "pass and fail<br>"
              "here is decisive. The axis does not start at zero. Modes are in "
              "learning order. Seed 0."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )

    fig.update_layout(
        title=None, bargap=0.34, plot_bgcolor=AC["bg"],
        margin=dict(l=62, r=14, t=40, b=136),
        legend=dict(
            orientation="h", x=0.0, xanchor="left", y=1.02, yanchor="bottom",
            font=dict(family=FONT_UI, size=9.5, color=AC["text_primary"]),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant",
        ),
        showlegend=True,
    )

    export_pair(fig, stem, 560, 196 + 26 * count)


# ── Export ──────────────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    """Write ``png/<stem>.png`` and ``svg/<stem>.svg``, both verified non-empty."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


def main() -> int:
    install_template()
    data = load_data()
    print("reports/final/cka_rl")
    figure_table1(data)
    for env_dir, env_label, stem in ENV_FIGURES:
        figure_final_policy(env_dir, env_label, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
