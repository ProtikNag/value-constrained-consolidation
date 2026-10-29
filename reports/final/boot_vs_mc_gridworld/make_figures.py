#!/usr/bin/env python3
"""Finalised paper figures: bootstrapped vs Monte-Carlo constraint values.

The constraint needs a value for every past task at every check. MC rolls each
task to episode end; the windowed bootstrap walks H steps and then leans on the
frozen local critic. This sweep runs H in {1,3,5,10,20,40,80} against full MC on
the 50-task shared-head GridWorld, three seeds each.

WHAT V_MC IS, because the first draft of these figures got it wrong. Both
estimators are computed at *every* check (crl/ppo/trainer.py: "MC estimate
(always computed)"), and `value_mode` only selects which one drives the
constraint. V_MC is a full-episode rollout and never touches H. So V_MC being
~0.714 under every setting is a CONTROL, not a result: the same instrument reads
the same number because the policies come out equally good. Never plot it as a
function of H; it is a single vertical reference line.

Marks follow reports/final/FIGURE_STYLE.md:
  estimator_bias          dumbbell  -- every row has its own reference (V_MC)
  speed_quality_tradeoff  lollipop  -- magnitudes against one shared rule (zero)
  all_values              booktabs  -- exact numbers, house table style

Usage::

    python reports/final/boot_vs_mc_gridworld/make_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, export_figure, hex_to_rgba, install_template,
)

# TWO data files, deliberately. This script needs PERF, seeds_PERF and
# mc_ref_t_per_check_s, which only data.json carries, so the FIGURES here are
# on the RAW scale. The PAPER quotes the NORMALIZED scale from
# data_normalized.json (PERF 0.5536 MC, 0.5669 at H=1), which is the same
# scale as every other table in the paper. Bias, speedup and break-even are
# metric-independent and identical in both files.
DATA = HERE / "data.json"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
_BUILT: dict = {}

FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5
FS_VALUE, FS_NOTE = 11.0, 9.5

HIGHLIGHT_OURS = hex_to_rgba(AC["blue"], 0.07)    # the operating point
HIGHLIGHT_REF = hex_to_rgba(AC["amber"], 0.09)    # the MC reference row

# From configs/biggrid_50task_*.yaml, used to sanity-check the cost model.
MAX_STEPS = 160
N_ENVS = 16


# --------------------------------------------------------------- data

def load() -> dict:
    return json.loads(DATA.read_text())


def boot_settings(d: dict) -> list[dict]:
    rows = []
    for name, v in d["per_setting"].items():
        if name == "mc":
            continue
        rows.append({
            "H": v["window_H"],
            "PERF": v["PERF"][0], "PERF_sd": v["PERF"][1],
            "BWT": v["BWT"][0], "BWT_sd": v["BWT"][1],
            "forgetting": v["forgetting"][0], "forgetting_sd": v["forgetting"][1],
            "V_mc": v["V_mc"][0], "V_mc_sd": v["V_mc"][1],
            "V_boot": v["V_boot"][0], "V_boot_sd": v["V_boot"][1],
            "bias": v["bias_boot_minus_mc"][0], "bias_sd": v["bias_boot_minus_mc"][1],
            "t": v["t_boot_s"], "speedup": v["speedup_vs_mc"],
            "steps": v["boot_env_steps"], "seeds": v["seeds_PERF"],
        })
    return sorted(rows, key=lambda r: r["H"])


def paired_delta(boot_seeds: list[float], mc_seeds: list[float]) -> tuple[float, float]:
    """Mean and sample sd of the per-seed difference from MC.

    Paired, because every setting ran on the same three seeds. Differencing
    within a seed removes the between-seed term, which is the larger one here.
    """
    diff = np.asarray(boot_seeds, float) - np.asarray(mc_seeds, float)
    return float(diff.mean()), float(diff.std(ddof=1))


def fit_line(rows: list[dict], t_mc: float) -> tuple[float, float, float]:
    H = np.array([r["H"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    slope, intercept = np.polyfit(H, t, 1)
    return float(intercept), float(slope), float((t_mc - intercept) / slope)


# --------------------------------------------------------------- chrome

def visual_len(s: str) -> int:
    s = re.sub(r"<[^>]+>", "", s)
    return len(re.sub(r"&#\d+;|&\w+;", "·", s))


def wrap_note(text: str, max_chars: int) -> str:
    """Hard-wrap to a rendered-width budget. Plotly annotations do not wrap,
    they run off the canvas, and that is invisible rather than an error."""
    lines: list[str] = []
    for para in text.split("<br>"):
        cur = ""
        for word in para.split():
            trial = word if not cur else f"{cur} {word}"
            if visual_len(trial) > max_chars and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    return "<br>".join(lines)


def note_budget(width: int, left: int, right: int) -> int:
    return int((width - left - right) / (FS_NOTE * 0.50))


def footnote(fig: go.Figure, text: str, max_chars: int, y: float = -0.30) -> None:
    fig.add_annotation(x=0, y=y, xref="paper", yref="paper",
                       text=wrap_note(text, max_chars), showarrow=False,
                       xanchor="left", yanchor="top", align="left",
                       font=dict(size=FS_NOTE, color=AC["text_muted"], family=FONT_UI))


def style_axes(fig: go.Figure, **kw) -> None:
    fig.update_xaxes(showgrid=False, showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticks="outside", tickcolor=AC["axis"], ticklen=4,
                     tickfont=dict(size=FS_TICK, color=AC["text_muted"], family=FONT_MONO), **kw)
    fig.update_yaxes(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6, zeroline=False,
                     showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticks="outside", tickcolor=AC["axis"], ticklen=4,
                     tickfont=dict(size=FS_TICK, color=AC["text_muted"], family=FONT_MONO))


def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    _BUILT[stem] = fig
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


# --------------------------------------------------------------- figure 1

def figure_estimator_bias(d: dict) -> None:
    """Dumbbell: the bar runs from each row's reference (V_MC) to what the
    bootstrap actually returned, so its LENGTH is the bias. FIGURE_STYLE.md,
    "How far past a per-row target?".
    """
    rows = boot_settings(d)[::-1]          # H=1 at the top
    v_mc_all = [r["V_mc"] for r in rows]
    v_mc = float(np.mean(v_mc_all))

    fig = go.Figure()
    ys = list(range(len(rows)))

    # one shared reference rule: V_MC does not depend on H
    fig.add_shape(type="line", x0=v_mc, x1=v_mc, y0=-0.75, y1=len(rows) - 0.25,
                  line=dict(color=AC["text_muted"], width=1.4, dash="dash"), layer="below")

    for y, r in zip(ys, rows):
        # Spread first, as a lightened tint running past the dot, so only the
        # overhang shows once the margin bar is drawn over it. Error bars here
        # would straddle the dot and read as a notch cut into the bar.
        fig.add_shape(type="line", x0=r["V_mc"], x1=r["V_boot"] + r["V_boot_sd"],
                      y0=y, y1=y, layer="below",
                      line=dict(color=hex_to_rgba(AC["red"], 0.16), width=11))
        # margin bar: reference -> achieved. Its length is the bias.
        fig.add_shape(type="line", x0=r["V_mc"], x1=r["V_boot"], y0=y, y1=y, layer="below",
                      line=dict(color=hex_to_rgba(AC["red"], 0.50), width=11))
        # gate tick at the reference
        fig.add_shape(type="line", x0=r["V_mc"], x1=r["V_mc"], y0=y - 0.24, y1=y + 0.24,
                      line=dict(color=AC["axis"], width=1.8))

    fig.add_trace(go.Scatter(
        x=[r["V_boot"] for r in rows], y=ys, mode="markers", showlegend=False,
        marker=dict(size=12, color=AC["red"], line=dict(color="#FFFFFF", width=1.8)),
        hoverinfo="skip"))

    # numeric column: the bar length, as a number
    x_num = 0.905
    for y, r in zip(ys, rows):
        fig.add_annotation(
            x=x_num, y=y, xanchor="right", yanchor="middle", showarrow=False,
            text=(f"<b>+{r['bias']:.3f}</b>"
                  f"<span style='font-size:{FS_NOTE}px;color:{AC['text_faint']}'>"
                  f" ±{r['bias_sd']:.3f}</span>"),
            font=dict(family=FONT_MONO, size=FS_VALUE, color=AC["text_primary"]))
    fig.add_annotation(x=x_num, y=len(rows) - 0.35, xanchor="right", yanchor="bottom",
                       showarrow=False, text="<b>over-estimate</b>",
                       font=dict(family=FONT_UI, size=FS_TICK, color=AC["text_muted"]))
    fig.add_annotation(x=v_mc, y=len(rows) - 0.35, xanchor="center", yanchor="bottom",
                       showarrow=False,
                       text=f"<b>V<sub>MC</sub> = {v_mc:.3f}</b>, identical for every row",
                       font=dict(family=FONT_UI, size=FS_TICK, color=AC["text_muted"]))

    fig.update_layout(width=W_FULL, height=int(W_FULL * 0.62), showlegend=False,
                      plot_bgcolor=AC["bg"], margin=dict(l=96, r=20, t=34, b=150))
    style_axes(fig)
    fig.update_xaxes(range=[0.695, 0.925],
                     title=dict(text="constraint value estimate for a past task",
                                font=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_UI)))
    fig.update_yaxes(showgrid=False, tickmode="array", tickvals=ys,
                     ticktext=[f"H = {r['H']}" for r in rows],
                     tickfont=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_MONO),
                     range=[-0.75, len(rows) + 0.35], showline=False, ticks="")

    footnote(fig,
             "Each bar runs from that run's own V<sub>MC</sub> gate to what the bootstrap returned, so bar length "
             "is the bias; the pale overhang past the dot is 1 s.d. across 3 seeds. At H=1 the estimate leans "
             "almost entirely on the frozen local critic and overshoots by +0.122; by H=80 real reward has "
             "displaced most of it and the overshoot is +0.072. "
             "<b>V<sub>MC</sub> is not a function of H.</b> Both estimators run at every check and value_mode only "
             "picks which drives the constraint, so V<sub>MC</sub> is a full-episode rollout that never sees H. "
             "Its flatness across rows is the control, not a finding.",
             note_budget(W_FULL, 96, 20), y=-0.22)
    export_pair(fig, "estimator_bias", W_FULL, int(W_FULL * 0.62))


# --------------------------------------------------------------- figure 2

def figure_speed_quality(d: dict) -> None:
    """(a) lollipop of the quality change against MC, zero-based so stem length
    is the real magnitude; (b) the affine cost model behind the speedup."""
    rows = boot_settings(d)
    mc = d["per_setting"]["mc"]
    t_mc = d["mc_ref_t_per_check_s"]
    mc_perf_sd = mc["PERF"][1]
    dperf = [paired_delta(r["seeds"], mc["seeds_PERF"]) for r in rows]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.115,
                        subplot_titles=("(a) Every window leaves quality where MC left it",
                                        "(b) Why the speedup saturates"))

    # ---- (a) lollipop of paired delta PERF, zero = MC
    xs = list(range(len(rows)))
    fig.add_shape(type="rect", x0=-0.6, x1=len(rows) - 0.4, y0=-mc_perf_sd, y1=mc_perf_sd,
                  fillcolor=hex_to_rgba(AC["text_faint"], 0.20), line=dict(width=0),
                  layer="below", row=1, col=1)
    fig.add_hline(y=0, line=dict(color=AC["axis"], width=1.3, dash="dash"), row=1, col=1)
    for x, (m, _sd) in zip(xs, dperf):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=m,
                      line=dict(color=hex_to_rgba(AC["blue"], 0.40), width=3),
                      layer="below", row=1, col=1)
    fig.add_trace(go.Scatter(
        x=xs, y=[m for m, _ in dperf], mode="markers", showlegend=False,
        marker=dict(size=11, color=AC["blue"], line=dict(color="#FFFFFF", width=1.6)),
        error_y=dict(type="data", array=[sd for _, sd in dperf],
                     color=hex_to_rgba(AC["blue"], 0.45), thickness=1.1, width=4),
        hoverinfo="skip"), row=1, col=1)
    fig.add_annotation(x=len(rows) - 0.45, y=mc_perf_sd, xref="x", yref="y",
                       xanchor="right", yanchor="bottom", showarrow=False, yshift=2,
                       text=f"MC's own seed spread, ±{mc_perf_sd:.4f}",
                       font=dict(size=FS_NOTE, color=AC["text_muted"], family=FONT_UI))

    # ---- (b) cost model
    a, b, h_equiv = fit_line(rows, t_mc)
    grid = np.linspace(0, max(h_equiv * 1.08, 95), 200)
    fig.add_trace(go.Scatter(x=grid, y=1000 * (a + b * grid), mode="lines", showlegend=False,
                             line=dict(color=AC["blue"], width=1.5, dash="dot"),
                             hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[r["H"] for r in rows], y=[1000 * r["t"] for r in rows],
                             mode="markers", showlegend=False,
                             marker=dict(size=10, color=AC["blue"],
                                         line=dict(color="#FFFFFF", width=1.6)),
                             hoverinfo="skip"), row=1, col=2)
    fig.add_hline(y=1000 * t_mc, line=dict(color=AC["amber"], width=1.6, dash="dash"), row=1, col=2)
    fig.add_annotation(x=3, y=1000 * t_mc, xanchor="left", yanchor="top", yshift=-5,
                       showarrow=False, text=f"<b>full-episode MC, {t_mc*1000:.0f} ms</b>",
                       font=dict(size=FS_NOTE, color=AC["amber"], family=FONT_UI), row=1, col=2)
    fig.add_annotation(x=h_equiv, y=1000 * t_mc, ax=-4, ay=46, axref="pixel", ayref="pixel",
                       showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.2,
                       arrowcolor=AC["red"], xanchor="right", align="right",
                       text=f"<b>fit reaches MC at H≈{h_equiv:.0f}</b><br>"
                            f"<span style='font-size:9px'>episode cap is {MAX_STEPS}</span>",
                       font=dict(size=FS_NOTE, color=AC["red"], family=FONT_UI),
                       bgcolor="rgba(255,255,255,0.88)", borderpad=2, row=1, col=2)
    fig.add_annotation(x=26, y=a * 1000, xanchor="left", yanchor="top", yshift=-6,
                       showarrow=False, text=f"{a*1000:.1f} ms fixed cost at H=0",
                       font=dict(size=FS_NOTE, color=AC["text_muted"], family=FONT_UI),
                       row=1, col=2)

    fig.update_layout(width=W_FULL, height=int(W_FULL * 0.58), showlegend=False,
                      margin=dict(l=76, r=24, t=64, b=140))
    for ann, x0 in zip(fig.layout.annotations[:2], (0.0, 0.5575)):
        ann.font = dict(size=FS_PANEL, color=AC["text_primary"], family=FONT_UI)
        ann.x, ann.xanchor = x0, "left"

    style_axes(fig)
    fig.update_xaxes(tickmode="array", tickvals=xs,
                     ticktext=[f"{r['H']}<br><span style='font-size:9px'>"
                               f"{r['speedup']:.0f}×</span>" for r in rows],
                     range=[-0.6, len(rows) - 0.4],
                     title=dict(text="bootstrap window H, with its per-check speedup",
                                font=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_UI)),
                     row=1, col=1)
    fig.update_yaxes(tickformat="+.3f",
                     title=dict(text="change in PERF vs MC, paired by seed",
                                font=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_UI)),
                     row=1, col=1)
    fig.update_xaxes(range=[0, max(h_equiv * 1.08, 95)],
                     title=dict(text="bootstrap window H",
                                font=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_UI)),
                     row=1, col=2)
    fig.update_yaxes(title=dict(text="ms per constraint check",
                                font=dict(size=FS_AXIS, color=AC["text_primary"], family=FONT_UI)),
                     row=1, col=2)

    best = max(rows, key=lambda r: r["speedup"])
    bi = rows.index(best)
    footnote(fig,
             f"(a) Stems are the per-seed change from MC, so zero is MC itself and stem length is the whole "
             f"effect. Every window lands inside MC's own seed spread once its bars are read. Across an "
             f"{max(r['speedup'] for r in rows)/min(r['speedup'] for r in rows):.0f}-fold range of cost the "
             f"constraint is indifferent to how its value is estimated. "
             f"(b) Cost is affine in H, {a*1000:.1f} ms fixed plus {b*1000:.2f} ms per step, which is why speedup "
             f"saturates instead of scaling as 1/H. The fit reaches MC's {t_mc*1000:.0f} ms at H≈{h_equiv:.0f} "
             f"against the {MAX_STEPS}-step episode cap, the consistency check on the model.",
             note_budget(W_FULL, 76, 24), y=-0.44)
    export_pair(fig, "speed_quality_tradeoff", W_FULL, int(W_FULL * 0.58))


# --------------------------------------------------------------- table

def figure_all_values(d: dict) -> None:
    """Every number, in the house booktabs style of gridworld/headline_metrics.

    No H column: the row label already says H. No env-step column: it is exactly
    16*H and carries nothing the label does not. No V_MC column: it is the same
    0.714 on every row by construction, so it belongs in the note, not in eight
    identical cells.
    """
    rows = boot_settings(d)
    mc = d["per_setting"]["mc"]
    t_mc = d["mc_ref_t_per_check_s"]
    dperf = [paired_delta(r["seeds"], mc["seeds_PERF"]) for r in rows]
    v_mc_all = [mc["V_mc"][0]] + [r["V_mc"] for r in rows]

    # (key, header line 1, header line 2, sense, decimals, signed, better)
    cols = [("PERF", "Average", "performance", "higher is better", 4, False, +1),
            ("dPERF", "Change vs MC", "paired by seed", "0 is MC", 4, True, None),
            ("BWT", "Backward", "transfer", "higher is better", 4, True, +1),
            ("forgetting", "Forgetting", "", "lower is better", 4, False, -1),
            ("V_boot", "V<sub>boot</sub>", "", "", 4, False, None),
            ("bias", "Bias vs", "V<sub>MC</sub>", "smaller is better", 4, True, -1),
            ("ms", "ms per", "check", "lower is better", 1, False, -1),
            ("speedup", "Speedup", "", "higher is better", 2, False, +1)]
    # 16 units per numeric column: a cell is "0.8714 ±0.0061", ~12 units wide
    # at this canvas, and the header senses ("smaller is better") need the rest.
    label_x = 0.0
    col_x = [45.0, 61.0, 77.0, 93.0, 109.0, 125.0, 137.0, 151.0]
    right_edge = col_x[-1] + 2

    table: list[dict] = [{
        "label": "MC (full episode)", "colour": AC["amber"], "band": HIGHLIGHT_REF,
        "PERF": mc["PERF"], "dPERF": None, "BWT": mc["BWT"],
        "forgetting": mc["forgetting"], "V_boot": None, "bias": None,
        "ms": (t_mc * 1000, None), "speedup": (1.0, None)}]
    for r, dp in zip(rows, dperf):
        best = r["speedup"] == max(x["speedup"] for x in rows)
        table.append({
            "label": f"Bootstrap H = {r['H']}",
            "colour": AC["blue"] if best else AC["text_primary"],
            "band": HIGHLIGHT_OURS if best else None,
            "PERF": (r["PERF"], r["PERF_sd"]), "dPERF": dp,
            "BWT": (r["BWT"], r["BWT_sd"]),
            "forgetting": (r["forgetting"], r["forgetting_sd"]),
            "V_boot": (r["V_boot"], r["V_boot_sd"]), "bias": (r["bias"], r["bias_sd"]),
            "ms": (r["t"] * 1000, None), "speedup": (r["speedup"], None)})

    best_of = {}
    for key, _h1, _h2, _s, _dp, _sg, better in cols:
        if better is None:
            continue
        vals = [row[key][0] for row in table if row.get(key) is not None]
        best_of[key] = max(vals) if better > 0 else min(vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             hoverinfo="skip", showlegend=False))

    def rule(y: float, width: float) -> None:
        fig.add_shape(type="line", x0=label_x - 1, x1=right_edge, y0=y, y1=y,
                      line=dict(color=AC["axis"], width=width), layer="above")

    header_y = 0.0
    rule(header_y + 1.05, 1.4)
    rule(header_y - 0.62, 1.1)
    fig.add_annotation(x=label_x, y=header_y + 0.12, text="<b>constraint value estimator</b>",
                       showarrow=False, xanchor="left", yanchor="middle",
                       font=dict(family=FONT_UI, size=FS_TICK, color=AC["text_muted"]))
    for x, (_k, head1, head2, sense, _dp, _sg, _b) in zip(col_x, cols):
        fig.add_annotation(x=x, y=header_y + (0.46 if head2 else 0.28), text=f"<b>{head1}</b>",
                           showarrow=False, xanchor="right", yanchor="middle",
                           font=dict(family=FONT_UI, size=FS_AXIS, color=AC["text_primary"]))
        if head2:
            fig.add_annotation(x=x, y=header_y + 0.12, text=f"<b>{head2}</b>", showarrow=False,
                               xanchor="right", yanchor="middle",
                               font=dict(family=FONT_UI, size=FS_AXIS, color=AC["text_primary"]))
        if sense:
            fig.add_annotation(x=x, y=header_y - 0.28, text=sense, showarrow=False,
                               xanchor="right", yanchor="middle",
                               font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_faint"]))

    for index, row in enumerate(table):
        y = -index - 0.95
        if row["band"]:
            fig.add_shape(type="rect", layer="below", x0=label_x - 1, x1=right_edge,
                          y0=y - 0.45, y1=y + 0.45, fillcolor=row["band"], line=dict(width=0))
        fig.add_annotation(x=label_x, y=y, text=f"<b>{row['label']}</b>", showarrow=False,
                           xanchor="left", yanchor="middle",
                           font=dict(family=FONT_UI, size=FS_AXIS, color=row["colour"]))
        for x, (key, _h1, _h2, _s, dps, signed, _b) in zip(col_x, cols):
            cell = row.get(key)
            if cell is None:
                fig.add_annotation(x=x, y=y, text="—", showarrow=False, xanchor="right",
                                   yanchor="middle",
                                   font=dict(family=FONT_UI, size=FS_VALUE, color=AC["text_faint"]))
                continue
            mean, sd = cell
            body = f"{mean:+.{dps}f}" if signed else f"{mean:.{dps}f}"
            if key == "speedup":
                body += "×"
            if key in best_of and abs(mean - best_of[key]) < 1e-12:
                body = f"<b>{body}</b>"
            if sd is not None and sd > 1e-12:
                body += (f"<span style=\"font-size:{FS_NOTE}px;color:{AC['text_faint']}\">"
                         f" ±{sd:.{dps}f}</span>")
            fig.add_annotation(x=x, y=y, text=body, showarrow=False, xanchor="right",
                               yanchor="middle",
                               font=dict(family=FONT_MONO, size=FS_VALUE, color=AC["text_primary"]))

    bottom = -len(table) - 0.5
    rule(bottom, 1.4)
    note_lines = [
        "50-task shared-head GridWorld, 3 seeds per setting, ±1 s.d. across seeds. "
        "<b>Bold</b> is the best value in each column.",
        "<b>Change vs MC is paired</b>: the per-seed difference, which removes the between-seed term and is the "
        "column to read for whether a window changed anything.",
        f"<b>V<sub>MC</sub> is omitted because it is constant</b> at {np.mean(v_mc_all):.4f} "
        f"({min(v_mc_all):.4f}–{max(v_mc_all):.4f}). Both estimators run at every check and value_mode only "
        "selects which drives the constraint,",
        "so V<sub>MC</sub> is a full-episode rollout that never sees H. Its flatness is the control, not a "
        "finding. Bias is V<sub>boot</sub> − V<sub>MC</sub> on the same runs.",
        "Shaded rows are the MC reference and H = 1, the operating point. ms per check was measured on a shared "
        "cluster, so read it as indicative.",
        "Forward transfer is absent throughout: it needs a from-scratch single-task baseline that this sweep does "
        "not have.",
    ]
    fig.add_annotation(x=label_x - 1, y=bottom - 0.30, text="<br>".join(note_lines),
                       showarrow=False, xanchor="left", yanchor="top", align="left",
                       font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_muted"]))

    top_y = header_y + 1.5
    bottom_y = bottom - 0.55 - 0.40 * len(note_lines)
    fig.update_xaxes(visible=False, range=[label_x - 2, right_edge + 1])
    fig.update_yaxes(visible=False, range=[bottom_y, top_y])
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=20, r=20, t=14, b=12))
    width = int(W_FULL * 1.70)
    export_pair(fig, "all_values", width, round(30 * (top_y - bottom_y)) + 26)


def write_table(d: dict) -> None:
    """The same columns as figure_all_values, as LaTeX for the paper."""
    rows = boot_settings(d)
    mc = d["per_setting"]["mc"]
    t_mc = d["mc_ref_t_per_check_s"]
    dperf = [paired_delta(r["seeds"], mc["seeds_PERF"]) for r in rows]

    def pm(m: float, sd: float, dp: int = 4, signed: bool = False) -> str:
        head = f"{m:+.{dp}f}" if signed else f"{m:.{dp}f}"
        return f"${head}\\pm{sd:.{dp}f}$"

    out = [
        "% GridWorld 50-task: MC vs windowed-bootstrap constraint values.",
        "% Generated by reports/final/boot_vs_mc_gridworld/make_figures.py -- do not hand-edit.",
        "% V_MC is omitted: constant across settings by construction. See the note.",
        r"\begin{tabular}{lcccccrr}",
        r"\toprule",
        r"Estimator & PERF & $\Delta$PERF & BWT & Forgetting & $V_{\mathrm{boot}}$"
        r" & Bias & Speedup \\",
        r"\midrule",
        f"MC (full episode) & {pm(*mc['PERF'])} & --- & {pm(*mc['BWT'])} & "
        f"{pm(*mc['forgetting'])} & --- & --- & $1.00\\times$ \\\\",
    ]
    for r, dp in zip(rows, dperf):
        out.append(
            f"Bootstrap $H={r['H']}$ & {pm(r['PERF'], r['PERF_sd'])} & "
            f"{pm(dp[0], dp[1], signed=True)} & {pm(r['BWT'], r['BWT_sd'])} & "
            f"{pm(r['forgetting'], r['forgetting_sd'])} & {pm(r['V_boot'], r['V_boot_sd'])} & "
            f"{pm(r['bias'], r['bias_sd'], signed=True)} & ${r['speedup']:.2f}\\times$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_boot_vs_mc.tex").write_text("\n".join(out) + "\n")
    print("  table_boot_vs_mc.tex")


# --------------------------------------------------------------- verify

def verify() -> list[str]:
    from PIL import Image
    problems = []
    d = load()
    rows = boot_settings(d)
    _, _, h_equiv = fit_line(rows, d["mc_ref_t_per_check_s"])

    if not 0 < h_equiv <= MAX_STEPS:
        problems.append(f"cost model implies H={h_equiv:.0f}, outside (0,{MAX_STEPS}]")
    for r in rows:
        if r["steps"] != N_ENVS * r["H"]:
            problems.append(f"H={r['H']}: env steps {r['steps']} != {N_ENVS}*H")

    # V_MC must stay flat: it is the control. If it ever moves with H, the
    # "same instrument, same reading" claim in every note here is wrong.
    v = [r["V_mc"] for r in rows]
    if max(v) - min(v) > 0.01:
        problems.append(f"V_MC varies by {max(v)-min(v):.4f} across H; it is no longer a control")

    for stem, fig in _BUILT.items():
        xa = fig.full_figure_for_development(warn=False).layout.xaxis
        if xa.type == "log" and xa.range is not None and xa.range[1] - xa.range[0] > 4:
            problems.append(f"{stem}: log x-axis spans too many decades; check annotation units")
    for stem in ("estimator_bias", "speed_quality_tradeoff", "all_values"):
        png, svg = PNG_DIR / f"{stem}.png", SVG_DIR / f"{stem}.svg"
        for p in (png, svg):
            if not p.exists():
                problems.append(f"missing {p.name}")
        if svg.exists() and b"image/png" in svg.read_bytes():
            problems.append(f"{svg.name} embeds raster data")
        if png.exists() and Image.open(png).size[0] < 1800:
            problems.append(f"{png.name} is under 300 dpi at print width")
    return problems


def main() -> int:
    install_template()
    d = load()
    print("boot-vs-MC figures")
    figure_estimator_bias(d)
    figure_speed_quality(d)
    figure_all_values(d)
    write_table(d)
    problems = verify()
    if problems:
        print("\nFAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nverify: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
