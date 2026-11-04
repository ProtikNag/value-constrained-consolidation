#!/usr/bin/env python3
"""The constraint ablation at full scale: GridWorld-50, three seeds per setting.

Three settings share one figure: mu pinned at 0, the uniform-weight average
(``global_current_uniform``, current task at 1/k), and VCC (seeds 2, 3 and 4).
Matrices are the seed mean; the curve carries a +-1 sample sd
band. Uniform matrices come from ``reports/gridworld_mu0/summary.json``.

Usage::

    python reports/final/ablation/make_mu0_full50.py
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import make_figures as mf
from acviz import FONT_UI, W_FULL, height_for, hex_to_rgba, install_template

VCC_RUNS = [mf.REPO / "reports" / "biggrid50v2" / name for name in
            ("duel_v2_fix_fresh_seed2", "duel_v2_fix_fresh_seed3", "duel_v2_fix_fresh_seed4")]
MU0_RUNS = [mf.REPO / "reports" / "gridworld_mu0" / f"duel_v2_mu0_seed{s}" for s in range(3)]
SUMMARY = mf.REPO / "reports" / "gridworld_mu0" / "summary.json"
STEM = "mu0_full50"


def load_setting(matrices: list, label: str, color: str) -> dict:
    """Seed-mean matrix, per-seed metrics (sample sd) and the curve of one setting."""
    per_seed = [mf.metrics(m) for m in matrices]
    n = len(per_seed[0]["matrix"])
    mean = [[float(np.mean([s["matrix"][k][i] for s in per_seed])) for i in range(k + 1)]
            for k in range(n)]
    curves = np.array([[np.mean(s["matrix"][k]) for k in range(n)] for s in per_seed])
    stat = {key: (float(np.mean([s[key] for s in per_seed])),
                  float(np.std([s[key] for s in per_seed], ddof=1)))
            for key in ("perf", "forgetting")}
    share = lambda test: 100 * float(np.mean([np.mean([test(v) for v in s["final"]])
                                              for s in per_seed]))
    return dict(matrix=mean, label=label, color=color, stat=stat, n=n,
                curve_mean=curves.mean(0), curve_sd=curves.std(0, ddof=1),
                ge09=share(lambda v: v >= 0.9), le01=share(lambda v: v <= 0.1))


def main() -> None:
    install_template()
    mf.PAPER_MODE = True
    read = lambda paths: [json.loads((p / "eval_matrix.json").read_text()) for p in paths]
    uniform = json.loads(SUMMARY.read_text())["methods"]["uniform"]["raw_matrices"]
    # Order follows the appendix text: mu=0, then the uniform average, then VCC.
    runs = [load_setting(read(MU0_RUNS), "μ = 0", mf.AC["amber"]),
            load_setting(uniform, "Uniform weights", mf.AC["green"]),
            load_setting(read(VCC_RUNS), "VCC", mf.C_ON)]
    mf.N_TASKS = runs[0]["n"]
    for r in runs:
        (p, ps), (f, fs) = r["stat"]["perf"], r["stat"]["forgetting"]
        print(f"  {r['label']:<16} PERF {p:.3f}±{ps:.3f}  F {f:.3f}±{fs:.3f}"
              f"  >=0.9 {r['ge09']:.0f}%  <=0.1 {r['le01']:.0f}%")
    export(merged(runs, [0, 9, 19, 29, 39, 49]), "constraint_combined", 600)


def merged(runs: list[dict], ticks: list[int]) -> go.Figure:
    """Matrices on the top row, the retention curves across the bottom row."""
    n, cols = runs[0]["n"], len(runs)
    fig = make_subplots(
        rows=2, cols=cols, specs=[[{}] * cols, [{"colspan": cols}] + [None] * (cols - 1)],
        row_heights=[0.40, 0.60], horizontal_spacing=0.05, vertical_spacing=0.14,
        subplot_titles=[f"<b>{r['label']}</b><br>PERF {r['stat']['perf'][0]:.2f} ± "
                        f"{r['stat']['perf'][1]:.2f}" for r in runs] + [""])
    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_UI, size=mf.FS_PANEL, color=mf.AC["text_primary"])

    # Top row: seed-mean matrices, diagonal outlined, shared colourbar beside them.
    shapes = mf.matrix_shapes(runs)
    for shape in shapes:
        transparent = shape["fillcolor"] == "rgba(0,0,0,0)"
        shape["line"] = dict(width=0.8, color=mf.AC["text_primary"]) if transparent \
            else dict(width=0)
    fig.update_layout(shapes=shapes)
    for col in range(1, cols + 1):
        fig.update_xaxes(
            title=dict(text="Task evaluated", font=dict(family=FONT_UI, size=mf.FS_AXIS),
                       standoff=4),
            range=[0, n], tickmode="array", tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) for t in ticks], constrain="domain",
            tickfont=dict(family=mf.FONT_MONO, size=mf.FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)
        fig.update_yaxes(
            title=dict(text="After training task" if col == 1 else None,
                       font=dict(family=FONT_UI, size=mf.FS_AXIS)),
            range=[n, 0], tickmode="array", tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) if col == 1 else "" for t in ticks],
            constrain="domain", scaleanchor="x" if col == 1 else f"x{col}", scaleratio=1,
            tickfont=dict(family=mf.FONT_MONO, size=mf.FS_TICK),
            showgrid=False, zeroline=False, row=1, col=col)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(color=[mf.CMIN], colorscale=[[p, f"rgb{c}"] for p, c in mf.MATRIX_SCALE],
                    cmin=mf.CMIN, cmax=mf.CMAX, showscale=True, opacity=0,
                    colorbar=dict(title=dict(text="Score", side="right",
                                             font=dict(family=FONT_UI, size=mf.FS_AXIS)),
                                  tickvals=[-0.15, 0.0, 0.25, 0.5, 0.75, 1.0],
                                  tickfont=dict(family=mf.FONT_MONO, size=mf.FS_TICK),
                                  thickness=11, len=0.34, y=0.80, yanchor="middle",
                                  x=1.01, outlinewidth=0))), row=1, col=1)

    # Bottom row: seed-mean curves with a +-1 sd band. The three end values sit
    # within 0.1 of each other, so the labels stack in the empty lower-left corner.
    xs = list(range(1, n + 1))
    curve_axes = f"x{cols + 1}", f"y{cols + 1}"
    for r, label_y in zip(runs, (0.44, 0.49, 0.54)):
        lo, hi = r["curve_mean"] - r["curve_sd"], r["curve_mean"] + r["curve_sd"]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(hi) + list(lo[::-1]), fill="toself",
                                 fillcolor=hex_to_rgba(r["color"], 0.16), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(r["curve_mean"]), mode="lines+markers",
                                 hoverinfo="skip", showlegend=False,
                                 line=dict(color=r["color"], width=2.0),
                                 marker=dict(color=r["color"], size=3.5,
                                             line=dict(color=mf.AC["bg"], width=0.8))),
                      row=2, col=1)
        fig.add_annotation(x=4, y=label_y, xref=curve_axes[0], yref=curve_axes[1],
                           xanchor="left", showarrow=False,
                           text=f"<b>━ {r['label']}</b>",
                           font=dict(family=FONT_UI, size=11.5, color=r["color"]))
    fig.update_xaxes(
        title=dict(text="Tasks trained", font=dict(family=FONT_UI, size=mf.FS_AXIS),
                   standoff=4),
        range=[0.5, n + 0.5], tickmode="array", tickvals=[1, 10, 20, 30, 40, 50],
        tickfont=dict(family=mf.FONT_MONO, size=mf.FS_TICK), row=2, col=1)
    fig.update_yaxes(
        title=dict(text="Mean score over tasks seen", font=dict(family=FONT_UI, size=mf.FS_AXIS)),
        range=[0.40, 1.02], dtick=0.1, tickformat=".2f", zeroline=False,
        tickfont=dict(family=mf.FONT_MONO, size=mf.FS_TICK), row=2, col=1)

    # Panel letters, top-left of each row.
    for text, y in (("(a)", 1.03), ("(b)", 0.54)):
        fig.add_annotation(xref="paper", yref="paper", x=-0.09, y=y, text=f"<b>{text}</b>",
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(family=FONT_UI, size=13, color=mf.AC["text_primary"]))
    fig.update_layout(title=None, margin=dict(l=62, r=72, t=52, b=46))
    return fig


def export(fig: go.Figure, name: str, height: int | None) -> None:
    """PNG and SVG from the same figure object, as in make_figures.py."""
    h = height or height_for(W_FULL)
    png, svg = mf.PNG_DIR / f"{STEM}_{name}.png", mf.SVG_DIR / f"{STEM}_{name}.svg"
    fig.write_image(png, width=W_FULL, height=h, scale=3.125)
    fig.write_image(svg, width=W_FULL, height=h)
    fig.write_image(png.with_suffix(".pdf"), width=W_FULL, height=h)
    for path in (png, svg):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"export failed or empty: {path}")
    print(f"  {png.name}  {svg.name}")


if __name__ == "__main__":
    main()
