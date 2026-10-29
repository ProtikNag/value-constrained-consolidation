#!/usr/bin/env python3
"""Finalised paper figures: BigGrid v2, the goal-conditioned 50-task tier.

This tier replaces the v1 GridWorld set. v1 was defective in ways that made its
numbers unreadable: the random floor sat at 0.742 so a method was graded mostly
on free reward, eight tasks were unsolvable, and slip correlated +0.552 with
score, so the benchmark partly measured its own confound. v2 fixes all three and
adds the thing v1 never had, a difficulty axis (spawn distance to the goal) that
is verified decorrelated from the other construction axes.

THREE SEEDS PER METHOD, AT UNEQUAL DEPTHS. Every seed of a method is pooled
under one label, including the arms whose configuration differs (DUEL's fresh
specialist init, CLEAR's and RECALL's replay budget). Those differences are
temporary and will be unified when the outstanding runs land, so nothing here
separates them.

NOTHING IS HELD FORWARD. Where seeds end at different depths, each plotted value
is a mean over the seeds that actually reached that depth, and the uncertainty
band is the min-to-max range over those same seeds. The band therefore narrows
to nothing where only one seed remains, which is the visual signal that the
sample has thinned. A held-forward placeholder would have been worse than
nothing here: CReLU's two short seeds stop at task 5 around 0.18, while its full
seed ends at 0.00, so carrying the short value to task 50 would have tripled a
score that is really zero.

Usage::

    python reports/final/gridworld_v2/make_figures.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, add_band, height_for, hex_to_rgba,
    install_template,
)

RUNS = REPO / "reports" / "biggrid50v2"
DUEL_TRACE = RUNS / "duel_v2_fix_seed0"      # the only run with a dual trace
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 50

FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5

# Palette roles, fixed across the set so a colour means one thing.
C_OURS = AC["blue"]
C_CEIL = AC["green"]       # joint / oracle ceiling
C_FAIL = AC["red"]
C_REF = AC["text_faint"]   # neutral reference

THRESHOLD = 0.6            # the benchmark's own per-task success bar
BANDS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 80)]

# (run-directory prefixes, display label, colour). This is the display order
# everywhere. Every directory matching "<prefix>_seed<digits>" EXACTLY is loaded
# and keyed by its seed number, so a method may draw its seeds from more than
# one prefix. DUEL does: seed 0 warm-starts each specialist and seeds 1 and 2
# re-initialise it. Those are pooled per the current instruction to treat the
# unmatched arms as matched until the replacement runs land.
METHODS = [
    (["duel_v2_fix", "duel_v2_fix_fresh"], "DUEL (ours)", C_OURS),
    (["biggrid50v2_clear"], "CLEAR", AC["amber"]),
    (["biggrid50v2_recall"], "RECALL", AC["violet"]),
    (["biggrid50v2_cka_rl"], "CKA-RL", AC["teal"]),
    (["biggrid50v2_cbp"], "CbpNet", AC["rose"]),
    (["biggrid50v2_crelu"], "CReLU", AC["sienna"]),
    (["biggrid50v2_finetune"], "Fine-tuning", AC["red"]),
    (["biggrid50v2_scratch"], "From-scratch", C_REF),
]

# learned_vs_retained shows only the methods that retain anything. The five that
# collapse produce four identical panels of the same point cloud on the floor.
QUADRANT_METHODS = ["DUEL (ours)", "CLEAR", "RECALL", "CKA-RL"]


# ── Data ────────────────────────────────────────────────────────────────────
def load_seeds(prefixes: list[str]) -> list[tuple[int, str, list[list[float]]]]:
    """(seed number, directory name, eval matrix) for one method, seed-ordered.

    The suffix is matched exactly against "_seed<digits>". A plain glob would
    pull "duel_v2_fix_fresh_seed1" in under the prefix "duel_v2_fix" as well as
    its own, and the seed would be counted twice.
    """
    out = []
    for prefix in prefixes:
        pattern = re.compile(re.escape(prefix) + r"_seed(\d+)$")
        for d in RUNS.glob(f"{prefix}_seed*"):
            match = pattern.fullmatch(d.name)
            path = d / "eval_matrix.json"
            if match and path.exists():
                out.append((int(match.group(1)), d.name,
                            json.loads(path.read_text())))
    return sorted(out)


def seed_metrics(matrix: list[list[float]], depth: int | None = None) -> dict:
    """PERF, BWT and forgetting for one seed, read at that seed's own depth.

    Forgetting is the drop from each task's best recorded score to its final
    one, averaged over the tasks that have been left behind. It separates a
    task that was never learned from one that was learned and then lost, which
    BWT alone cannot do.

    Truncating to ``depth`` is legitimate: the first j phases of a continual run
    are the same whatever happens later, so row j-1 IS the depth-j result.
    """
    k = len(matrix) if depth is None else depth
    final = matrix[k - 1]
    if k > 1:
        bwt = statistics.mean(final[i] - matrix[i][i] for i in range(k - 1))
        forgetting = statistics.mean(
            max(matrix[t][i] for t in range(i, k)) - final[i]
            for i in range(k - 1))
    else:
        bwt = forgetting = float("nan")
    return dict(depth=k, perf=statistics.mean(final), bwt=bwt,
                forgetting=forgetting, final=final,
                diag=[matrix[i][i] for i in range(k)])


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and population-free s.d.; s.d. is 0.0 for a single value."""
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan"), float("nan")
    return (statistics.mean(clean),
            statistics.stdev(clean) if len(clean) > 1 else 0.0)


def pooled_matrix(seeds: list) -> tuple[list[list[float]], list[int]]:
    """Cell-wise mean over seeds, plus the seed count behind each ROW.

    A partial seed contributes only the rows it reached, so the deeper rows of
    the averaged matrix rest on fewer seeds than the shallow ones. The per-row
    count is returned so the caption can say where that changes.
    """
    depth = max(len(m) for _, _, m in seeds)
    matrix, counts = [], []
    for r in range(depth):
        rows = [m[r] for _, _, m in seeds if len(m) > r]
        counts.append(len(rows))
        matrix.append([statistics.mean(row[c] for row in rows)
                       for c in range(r + 1)])
    return matrix, counts


def progress_rows(kind: str) -> list[dict]:
    marker = f'"{kind}"'
    return [json.loads(line) for line in (DUEL_TRACE / "progress.jsonl").open()
            if marker in line]


def load_data() -> dict:
    manifest = json.loads((REPO / "configs" / "biggrid50v2.manifest.json").read_text())
    summary = json.loads((RUNS / "setting_b_v2_summary.json").read_text())
    joint = json.loads((RUNS / "joint_sep_matched.json").read_text())[0]
    floor_ceiling = json.loads((RUNS / "floor_ceiling.json").read_text())
    specialist = {r["task"]: r["score"] for r in
                  json.loads((RUNS / "specialist_learnability.json").read_text())}

    summary_key = {"DUEL (ours)": "DUEL", "CLEAR": "CLEAR", "RECALL": "RECALL",
                   "CKA-RL": "CKA_RL", "CbpNet": "CbpNet", "CReLU": "CReLU",
                   "Fine-tuning": "finetune", "From-scratch": "from_scratch"}

    methods = []
    for prefixes, label, color in METHODS:
        seeds = load_seeds(prefixes)
        if not seeds:
            continue
        per_seed = [seed_metrics(m) for _, _, m in seeds]
        matrix, row_counts = pooled_matrix(seeds)
        entry = summary["methods"].get(summary_key[label], {})
        # Forward transfer is an AUC over within-phase evaluations, so it comes
        # from the run summary and cannot be recomputed here. Two defects in
        # that file have to be filtered.
        #
        # First, an FWT of exactly 0.0 on a method that is not from-scratch is
        # the summary's not-computed default rather than a measurement:
        # from-scratch is 0 by construction, and every method really measured
        # came back with a non-round value.
        #
        # Second, the summary's top-level FWT_raw_gap is seed 0's value, not a
        # mean over seeds. Average the seeds that carry one instead.
        measured = [s.get("FWT_raw_gap") for s in entry.get("seeds", [])]
        measured = [v for v in measured
                    if v is not None and (label == "From-scratch" or v != 0.0)]
        fwt = statistics.mean(measured) if measured else None
        # Cross-method comparison happens at ONE depth or it says nothing: a
        # retaining method's PERF climbs as the sequence shortens, so pooling
        # seeds of different lengths can and does reverse a ranking. Every
        # headline number is therefore the mean over the seeds that reached the
        # FULL 50 tasks, read there, with the s.d. across those same seeds. A
        # seed that stopped early contributes to the curves and the matrix,
        # where its own depth is respected, but not to a 50-task figure.
        full = [m for _, _, m in seeds if len(m) == N_TASKS]
        if not full:                      # nothing finished yet: fall back
            full = [max(seeds, key=lambda s: len(s[2]))[2]]
        at_full = [seed_metrics(m) for m in full]
        headline = {k: mean_sd([s[k] for s in at_full])[0]
                    for k in ("perf", "bwt", "forgetting")}
        headline["depth"] = len(full[0])
        # Per-task columns come from the seeds pooled, so a method with three
        # complete seeds is scored over 150 tasks rather than 50.
        headline["final"] = [v for s in at_full for v in s["final"]]
        methods.append(dict(
            label=label, color=color, seeds=seeds, per_seed=per_seed,
            headline=headline, n_complete=len(at_full),
            sd_perf=mean_sd([s["perf"] for s in at_full])[1],
            sd_bwt=mean_sd([s["bwt"] for s in at_full])[1],
            sd_forgetting=mean_sd([s["forgetting"] for s in at_full])[1],
            runs=[name for _, name, _ in seeds],
            n_seeds=len(seeds), depths=[s["depth"] for s in per_seed],
            max_depth=max(s["depth"] for s in per_seed),
            min_depth=min(s["depth"] for s in per_seed),
            complete=all(s["depth"] == N_TASKS for s in per_seed),
            matrix=matrix, row_counts=row_counts,
            perf=mean_sd([s["perf"] for s in per_seed]),
            bwt=mean_sd([s["bwt"] for s in per_seed]),
            forgetting=mean_sd([s["forgetting"] for s in per_seed]),
            # Every seed's points together, for the scatter and box panels.
            all_diag=[v for s in per_seed for v in s["diag"]],
            all_final=[v for s in per_seed for v in s["final"]],
            fwt=fwt))

    # Third defect, and the one that cannot be filtered per method: if two
    # different methods report byte-identical forward transfer, one run's value
    # has been copied onto the other. An AUC gap over 50 phases does not
    # coincide to 17 significant figures. Both are dropped rather than one
    # chosen, because the file does not say which is the real one.
    seen: dict[float, list[dict]] = {}
    for m in methods:
        if m["fwt"] is not None and m["label"] != "From-scratch":
            seen.setdefault(m["fwt"], []).append(m)
    for value, clash in seen.items():
        if len(clash) > 1:
            names = ", ".join(m["label"] for m in clash)
            print(f"  !! forward transfer identical across {names} "
                  f"({value:.6f}); dropped as a copy, not a measurement")
            for m in clash:
                m["fwt"] = None

    dual = progress_rows("dual")
    return dict(
        tasks=manifest["tasks"], methods=methods,
        duel=next(m for m in methods if m["label"].startswith("DUEL")),
        joint_scores=joint["scores"], joint_mean=joint["mean"],
        floor=[r["random"]["mean"] for r in floor_ceiling],
        oracle=[r["oracle"]["mean"] for r in floor_ceiling],
        specialist=[specialist[i + 1] for i in range(N_TASKS)],
        dual=dual)


# ── Shared helpers ──────────────────────────────────────────────────────────
FOOTNOTE_SIZE = 8.5
AXIS_TITLE_PX = 60
FOOTNOTE_LEAD = 13


# Set FIGS_FOR_PAPER=1 to drop the in-figure footnotes. In the report folder a
# figure has to stand on its own, so it carries its own caption. In the paper
# the LaTeX caption says the same thing, and the embedded copy is then both
# redundant and too small to read at column width, while its reserved margin is
# dead space on the page.
PAPER_MODE = bool(os.environ.get("FIGS_FOR_PAPER"))


def add_footnote(fig: go.Figure, text: str, left: int, extra: int = 0) -> None:
    """Place a hand-wrapped footnote under the x-axis title, reserving space."""
    if PAPER_MODE:
        fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + 10))
        return
    lines = text.count("<br>") + 1
    fig.update_layout(margin=dict(b=AXIS_TITLE_PX + extra + lines * FOOTNOTE_LEAD + 16))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="top",
        xshift=-left, yshift=-(AXIS_TITLE_PX + extra), showarrow=False,
        align="left", text=text,
        font=dict(family=FONT_UI, size=FOOTNOTE_SIZE, color=AC["text_muted"]))


def seed_note(methods: list[dict], per_line: int = 99) -> str:
    """Each method's seed depths, wrapped at ``per_line`` entries.

    The line grows as runs deepen (every depth gains a digit at task 10 and
    again at 100), so a figure that fitted it last week can overflow its canvas
    this week. Wrapping is cheaper than re-tuning each footnote.
    """
    parts = [f"{m['label'].replace(' (ours)', '')} "
             f"{'/'.join(str(d) for d in sorted(m['depths'], reverse=True))}"
             for m in methods]
    lines = [", ".join(parts[i:i + per_line])
             for i in range(0, len(parts), per_line)]
    return ",<br>&nbsp;&nbsp;&nbsp;".join(lines)


def curve_stats(m: dict) -> tuple[list[int], list[float], list[float], list[float], int]:
    """Mean, min and max of the mean-over-seen-tasks curve, per task depth.

    At each depth only the seeds that actually reached it are counted, so no
    value is invented. The range collapses to a single line past the point
    where one seed is left, which is what marks the thinning sample.
    """
    curves = [[statistics.mean(row) for row in matrix] for _, _, matrix in m["seeds"]]
    xs, mean, lo, hi = [], [], [], []
    for i in range(m["max_depth"]):
        here = [c[i] for c in curves if len(c) > i]
        xs.append(i + 1)
        mean.append(statistics.mean(here))
        lo.append(min(here))
        hi.append(max(here))
    return xs, mean, lo, hi, m["min_depth"]


# ONE scale, sampled for both the cells and the colourbar, so the legend cannot
# disagree with the matrix. Scores run from about -0.05 to 1.005.
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


def row_count_spans(counts: list[int]) -> str:
    """'rows 1-20 over 3 seeds, 21-36 over 2' from a per-row seed count."""
    spans, start = [], 0
    for i in range(1, len(counts) + 1):
        if i == len(counts) or counts[i] != counts[start]:
            spans.append(f"{start + 1}–{i} over {counts[start]}")
            start = i
    return ", ".join(spans)


# ── Figure 1: retention matrices, ours beside the runner-up ─────────────────
def fig_retention_matrix(d: dict) -> go.Figure:
    """Two seed-averaged evaluation matrices side by side, on one scale.

    Each cell is the mean over every seed that reached that row, so the deeper
    rows rest on fewer seeds than the shallow ones. The panel subtitle says
    where that changes.

    Drawn as explicit rectangles rather than go.Heatmap, which rasterises its
    cells into an embedded bitmap and would ship a blurry matrix into a paper.
    The ~2,550 rectangles are assigned in ONE call: add_shape is quadratic in
    the number of shapes already present, which measured 98.8s at 1,200 shapes
    and put this figure alone at about seven minutes.
    """
    shapes: list[dict] = []
    rival = next(m for m in d["methods"] if m["label"] == "CLEAR")
    panels = [d["duel"], rival]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=[
            f"{p['label']} — PERF {p['headline']['perf']:.3f} at 50 tasks"
            + (f", {p['n_complete']} seeds" if p["n_complete"] > 1 else "")
            for p in panels])
    for col, panel in enumerate(panels, start=1):
        axes = dict(xref="x" if col == 1 else f"x{col}",
                    yref="y" if col == 1 else f"y{col}")
        for row_idx, row in enumerate(panel["matrix"]):
            for col_idx, value in enumerate(row):
                shapes.append(dict(
                    type="rect", x0=col_idx, x1=col_idx + 1,
                    y0=row_idx, y1=row_idx + 1, fillcolor=sample_scale(value),
                    line=dict(width=0.35, color=AC["bg"]), layer="below", **axes))
        if panel["max_depth"] < N_TASKS:
            shapes.append(dict(type="rect", x0=0, x1=N_TASKS,
                               y0=panel["max_depth"], y1=N_TASKS,
                               fillcolor=AC["surface"], line=dict(width=0),
                               layer="below", **axes))
        ticks = [0, 9, 19, 29, 39, 49]
        # constrain="domain" plus matching ranges keeps the cells flush against
        # both axis lines: no gap between the last row and the x-axis.
        fig.update_xaxes(
            title=dict(text="Task evaluated", font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[0, N_TASKS], tickmode="array", tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) for t in ticks], constrain="domain",
            tickfont=dict(family=FONT_MONO, size=FS_TICK), showgrid=False,
            zeroline=False, row=1, col=col)
        fig.update_yaxes(
            title=dict(text="After training task" if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS_AXIS)),
            range=[N_TASKS, 0], tickmode="array", tickvals=[t + 0.5 for t in ticks],
            ticktext=[str(t + 1) for t in ticks], constrain="domain",
            scaleanchor="x" if col == 1 else "x2", scaleratio=1,
            tickfont=dict(family=FONT_MONO, size=FS_TICK), showgrid=False,
            zeroline=False, row=1, col=col)
    # Colourbar on an invisible marker trace -> a real SVG gradient, sampled
    # from the SAME scale as the cells, so red appears in the legend too.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(color=[CMIN], colorscale=[[p, f"rgb{c}"] for p, c in MATRIX_SCALE],
                    cmin=CMIN, cmax=CMAX, showscale=True, opacity=0,
                    colorbar=dict(title=dict(text="Score", side="right",
                                             font=dict(family=FONT_UI, size=FS_AXIS)),
                                  tickvals=[-0.15, 0.0, 0.25, 0.5, 0.75, 1.0],
                                  tickfont=dict(family=FONT_MONO, size=FS_TICK),
                                  thickness=12, len=0.74, outlinewidth=0, x=1.015))),
        row=1, col=1)
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=70, r=100, t=48, b=66),
                      shapes=shapes)
    add_footnote(fig, (
        "Row <i>k</i> is the score on every task seen so far, measured after task "
        "<i>k</i> finished training; lower-triangular because a task cannot<br>"
        "&nbsp;&nbsp;&nbsp;be evaluated before it arrives. Red is at or below the random "
        "floor (0.000), blue is solved. The scale is shared by both panels<br>"
        "&nbsp;&nbsp;&nbsp;and spans the full range the cells use, red included.<br>"
        "&nbsp;&nbsp;&nbsp;Every cell is a mean over the seeds that reached its row, so "
        "the lower rows average fewer seeds than the upper ones:<br>"
        f"&nbsp;&nbsp;&nbsp;rows {row_count_spans(panels[0]['row_counts'])} seeds for "
        f"{panels[0]['label'].replace(' (ours)', '')}, "
        f"rows {row_count_spans(panels[1]['row_counts'])} for {panels[1]['label']}.<br>"
        "&nbsp;&nbsp;&nbsp;The PERF in each title is read at 50 tasks, from the one run "
        "of that method that has finished, so the two are comparable."
    ), left=70, extra=4)
    return fig


# ── Figure 2: retention trajectories, all methods ───────────────────────────
def fig_retention_curves(d: dict) -> go.Figure:
    """Mean score over tasks seen, as the sequence advances, per method.

    The band is the min-to-max range across seeds at each depth, over the seeds
    that reached it. Where a method is down to one seed the range has no width,
    so a line without a band around it is a single run.
    """
    fig = go.Figure()
    for m in d["methods"]:
        xs, mean, lo, hi, overlap = curve_stats(m)
        if m["n_seeds"] > 1:
            add_band(fig, xs, lo, hi, m["color"], 0.18)
        fig.add_trace(go.Scatter(
            x=xs, y=mean, mode="lines",
            name=f"{m['label']}  ({m['n_seeds']} seeds)",
            line=dict(width=2.4 if m["label"].startswith("DUEL") else 1.6,
                      color=m["color"]),
            hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(
        x=[1, N_TASKS], y=[d["joint_mean"]] * 2, mode="lines", name="Joint ceiling",
        line=dict(width=1.3, color=C_CEIL, dash="dash"),
        hoverinfo="skip", showlegend=True))
    fig.update_xaxes(title=dict(text="Tasks trained",
                                font=dict(family=FONT_UI, size=FS_AXIS)),
                     range=[0.5, N_TASKS + 0.5], dtick=10,
                     tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(title=dict(text="Mean score over tasks seen",
                                font=dict(family=FONT_UI, size=FS_AXIS)),
                     range=[-0.12, 1.08], dtick=0.25, tickformat=".2f",
                     tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_layout(title=None, margin=dict(l=68, r=180, t=16, b=64),
                      showlegend=True,
                      legend=dict(x=1.015, y=1.0, yanchor="top", xanchor="left",
                                  font=dict(family=FONT_UI, size=10.5),
                                  bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                  itemsizing="constant"))
    add_footnote(fig, (
        "Each line is one method's mean score over the tasks it has seen so far, "
        "averaged across the seeds that reached that depth.<br>"
        "&nbsp;&nbsp;&nbsp;Shaded bands are the min-to-max range over those same seeds. "
        "No value is held forward: where a band narrows to nothing, only one<br>"
        "&nbsp;&nbsp;&nbsp;seed is still running, and the line past that point is that "
        "single run. The joint ceiling trains one network on all 50 at once.<br>"
        f"&nbsp;&nbsp;&nbsp;Seed depths — {seed_note(d['methods'], 5)}."
    ), left=68)
    return fig


# ── Figure 3: per-task score distribution, by method ────────────────────────
def fig_score_distribution(d: dict) -> go.Figure:
    """Where every task's score sits, when learned and at the end.

    One mark: the density. No inner box, no whisker caps and no point cloud on
    top of it, because three overlaid encodings of the same 150 numbers fight
    each other and the shape loses.

    The mean is a diamond rather than Plotly's mean LINE. These scores are
    strongly bimodal, piling up at the floor and at solved, so the mean falls in
    the thin waist between the two lobes, and a line there is only as wide as
    the density it sits in, which is almost nothing. That is why the mean could
    not be seen. A marker does not depend on the density under it.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=("Score when the task was learned",
                                        "Score at the end of the sequence"))
    methods = d["methods"]
    for col, key in ((1, "all_diag"), (2, "all_final")):
        for m in methods:
            fig.add_trace(go.Violin(
                x=m[key], name=m["label"], orientation="h",
                # spanmode="hard" stops the kernel spilling past the data, which
                # on a bounded score would draw density above 1.0.
                spanmode="hard", bandwidth=0.035,
                points=False, box_visible=False, meanline_visible=False,
                fillcolor=hex_to_rgba(m["color"], 0.45),
                line=dict(color=m["color"], width=1.0),
                width=0.9, hoverinfo="skip", showlegend=False), row=1, col=col)
            mean = statistics.mean(m[key])
            fig.add_trace(go.Scatter(
                x=[mean], y=[m["label"]], mode="markers",
                marker=dict(symbol="diamond", size=9,
                            color=AC["text_primary"],
                            line=dict(width=1.6, color=AC["bg"])),
                hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_annotation(
                xref="x domain" if col == 1 else "x2 domain", x=0.995,
                xanchor="right", y=m["label"], yanchor="middle", showarrow=False,
                text=f"<b>{mean:.2f}</b>",
                font=dict(family=FONT_MONO, size=11, color=AC["text_primary"]),
                bgcolor="rgba(255,255,255,0.88)", borderpad=1, row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(family=FONT_UI, size=11),
                     tickmode="array", tickvals=[m["label"] for m in methods],
                     ticktext=[f"{m['label']}  ({m['n_seeds']} seeds)"
                               for m in methods], row=1, col=1)
    for col in (1, 2):
        fig.update_xaxes(range=[-0.20, 1.18], dtick=0.25, tickformat=".2f",
                         tickfont=dict(family=FONT_MONO, size=FS_TICK),
                         title=dict(text="Score (0 = random floor, ~1 = solved)",
                                    font=dict(family=FONT_UI, size=FS_AXIS)),
                         row=1, col=col)
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"])
    fig.update_layout(title=None, margin=dict(l=140, r=28, t=88, b=86),
                      violingap=0.22, violingroupgap=0)
    # One key for the diamond, outside the panels.
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.125, xanchor="left", yanchor="bottom",
        showarrow=False,
        text=(f"<span style='color:{AC['text_primary']}'>◆</span>  mean, "
              "printed at each panel's right edge"),
        font=dict(family=FONT_UI, size=10.5, color=AC["text_primary"]))
    add_footnote(fig, (
        "Every task of every seed goes into the density, so a method with three "
        "complete seeds is 150 scores. The width of each shape is<br>"
        "&nbsp;&nbsp;&nbsp;how many of them sit at that score. The two lobes are the "
        "result: on this benchmark a task is almost always either solved or<br>"
        "&nbsp;&nbsp;&nbsp;on the floor, and the left panel shows that every method "
        "reaches the upper lobe while only three stay there.<br>"
        "&nbsp;&nbsp;&nbsp;v2 needs no normalisation: its random floor is −0.003 and its "
        "oracle ceiling 0.982, so raw scores already read on a 0-to-1 scale."
    ), left=140)
    return fig


# ── Figure 4: learned versus retained, the methods that retain ──────────────
def fig_learned_vs_retained(d: dict) -> go.Figure:
    """Score when a task was learned against its score at the end.

    The quadrants are mirror images: a task learned poorly that ends strong was
    RESCUED by later training; one learned strongly that ends poorly was LOST.

    Square panels on a narrow canvas. The axes carry the same range, so a panel
    wider than it is tall puts the no-change diagonal at the wrong slope and the
    eye stops reading distance from it correctly. Nothing is written inside a
    panel: the shaded corners are the message and a text box sitting on them
    competes with it.
    """
    lo, hi = 0.25, 0.60
    methods = [m for m in d["methods"] if m["label"] in QUADRANT_METHODS]
    cols = 2
    rows = -(-len(methods) // cols)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.085, vertical_spacing=0.24)
    for i, m in enumerate(methods):
        r, c = i // cols + 1, i % cols + 1
        pairs = list(zip(m["all_diag"], m["all_final"]))
        total = len(pairs)
        rescued = sum(1 for a, b in pairs if a < lo and b > hi)
        lost = sum(1 for a, b in pairs if a > hi and b < lo)
        better = sum(1 for a, b in pairs if b > a)
        fig.add_shape(type="rect", x0=-0.18, x1=lo, y0=hi, y1=1.14,
                      fillcolor=hex_to_rgba(C_OURS, 0.11), line=dict(width=0),
                      layer="below", row=r, col=c)
        fig.add_shape(type="rect", x0=hi, x1=1.14, y0=-0.18, y1=lo,
                      fillcolor=hex_to_rgba(C_FAIL, 0.11), line=dict(width=0),
                      layer="below", row=r, col=c)
        fig.add_shape(type="line", x0=-0.18, y0=-0.18, x1=1.14, y1=1.14,
                      line=dict(color=AC["border"], width=1.1), layer="below",
                      row=r, col=c)
        fig.add_trace(go.Scatter(
            x=m["all_diag"], y=m["all_final"], mode="markers",
            marker=dict(size=7.6, color=hex_to_rgba(m["color"], 0.50),
                        line=dict(width=0.8, color=AC["bg"])),
            hoverinfo="skip", showlegend=False), row=r, col=c)
        # Name and counts ABOVE the panel, in the gap the row spacing leaves.
        for y, text, size, color in (
                (1.30, f"<b>{m['label']}</b>", 12.5, AC["text_primary"]),
                (1.155, f"<b>{better / total:.0%}</b> ended better", 11, m["color"]),
                (1.025, f"rescued <b>{rescued}</b> · lost <b>{lost}</b> "
                        f"of {total}", 11, m["color"])):
            fig.add_annotation(
                xref="x domain", yref="y domain", x=0, y=y, xanchor="left",
                yanchor="bottom", showarrow=False, text=text,
                font=dict(family=FONT_UI, size=size, color=color), row=r, col=c)
        fig.update_xaxes(range=[-0.18, 1.14], dtick=0.5, tickformat=".1f",
                         tickfont=dict(family=FONT_MONO, size=11), row=r, col=c)
        fig.update_yaxes(range=[-0.18, 1.14], dtick=0.5, tickformat=".1f",
                         tickfont=dict(family=FONT_MONO, size=11), row=r, col=c)
    # The quadrant key sits above everything, stacked, outside every panel.
    for i, (color, text) in enumerate((
            (C_OURS, "RESCUED: learned poorly (&lt;0.25), ended strong (&gt;0.60)"),
            (C_FAIL, "LOST: learned strong (&gt;0.60), ended poorly (&lt;0.25)"))):
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=1.200 - i * 0.030, xanchor="left",
            yanchor="bottom", showarrow=False,
            text=f"<span style='color:{color}'>■</span>  {text}",
            font=dict(family=FONT_UI, size=10.5, color=AC["text_primary"]))
    fig.update_layout(title=None, margin=dict(l=64, r=16, t=150, b=92))
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0, yshift=-48,
                       xanchor="center", yanchor="top", showarrow=False,
                       text="Score when the task was learned",
                       font=dict(family=FONT_UI, size=13, color=AC["text_primary"]))
    fig.add_annotation(xref="paper", yref="paper", x=0, xshift=-50, y=0.5,
                       yanchor="middle", xanchor="center", showarrow=False,
                       textangle=-90, text="Score at the end of the sequence",
                       font=dict(family=FONT_UI, size=13, color=AC["text_primary"]))
    add_footnote(fig, (
        "One point per task per seed, every seed pooled. The diagonal is<br>"
        "&nbsp;&nbsp;&nbsp;no change: above it a task improved after being left,<br>"
        "&nbsp;&nbsp;&nbsp;below it the task was forgotten. The shaded quadrants<br>"
        "&nbsp;&nbsp;&nbsp;are mirror images — 0.25 is barely off the random floor,<br>"
        "&nbsp;&nbsp;&nbsp;0.60 is the success bar. Only the four methods that<br>"
        "&nbsp;&nbsp;&nbsp;retain anything are shown; the other four collapse onto<br>"
        "&nbsp;&nbsp;&nbsp;the floor and produce identical panels.<br>"
        "&nbsp;&nbsp;&nbsp;Seed depths — DUEL 50/32/8, CLEAR 50/36/20,<br>"
        "&nbsp;&nbsp;&nbsp;RECALL 50/20/20, CKA-RL 50/8/8."
    ), left=64, extra=26)
    return fig


# ── Figure 5: score against the difficulty axis ─────────────────────────────
def fig_performance_by_band(d: dict) -> go.Figure:
    """Every method against the designed difficulty axis, with the joint ceiling.

    Bars from zero with a spread across seeds, because the quantity is a
    magnitude and the reader is comparing heights within each band. The ceiling
    is a rule rather than a ninth bar: drawn as a bar it reads as a competitor.
    """
    xs = list(range(len(BANDS)))

    # A seed that reached 8 tasks touches one or two tasks of each band, and a
    # mean over two tasks is not a band score: unchecked it put CKA-RL at 0.64
    # on the hardest band off a single lucky task. Half the band is the floor.
    MIN_TASKS_PER_BAND = 5

    def band_mean(values: list[float], depth: int) -> list[float]:
        """Mean score per band, for bands this run covered at least half of."""
        out = []
        for band in BANDS:
            vals = [v for i, (t, v) in enumerate(zip(d["tasks"], values))
                    if tuple(t["band"]) == band and i < depth]
            out.append(statistics.mean(vals)
                       if len(vals) >= MIN_TASKS_PER_BAND else float("nan"))
        return out

    fig = go.Figure()
    for m in d["methods"]:
        per_seed = [band_mean(s["final"], s["depth"]) for s in m["per_seed"]]
        means, errs = [], []
        for b in range(len(BANDS)):
            here = [c[b] for c in per_seed if c[b] == c[b]]
            mu, sd = mean_sd(here)
            means.append(mu)
            errs.append(sd)
        fig.add_trace(go.Bar(
            x=xs, y=means, name=m["label"],
            marker=dict(color=m["color"],
                        line=dict(width=0.5, color=AC["bg"])),
            error_y=dict(type="data", array=errs, visible=True,
                         color=AC["axis"], thickness=1.0, width=2.2),
            hoverinfo="skip", showlegend=True))
    joint = band_mean(d["joint_scores"], N_TASKS)
    # One dashed segment per band group, so the ceiling reads as a reference
    # against that group rather than as a trend across the bands.
    for b, value in enumerate(joint):
        # Half the group width, so the rule sits over its own bars and not into
        # the gap on either side. Group width is 1 - bargap = 0.88.
        fig.add_shape(type="line", x0=b - 0.37, x1=b + 0.37, y0=value, y1=value,
                      line=dict(color=C_CEIL, width=2.0, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="Joint ceiling",
        line=dict(color=C_CEIL, width=1.8, dash="dot"), showlegend=True))
    fig.update_xaxes(title=dict(text="Shortest-path distance from spawn to goal (steps)",
                                font=dict(family=FONT_UI, size=FS_AXIS)),
                     tickmode="array", tickvals=xs,
                     ticktext=[f"{lo}–{hi}" for lo, hi in BANDS],
                     tickfont=dict(family=FONT_MONO, size=FS_TICK),
                     range=[-0.55, len(BANDS) - 0.45], showgrid=False)
    fig.update_yaxes(title=dict(text="Mean score", font=dict(family=FONT_UI, size=FS_AXIS)),
                     range=[-0.08, 1.12], dtick=0.25, tickformat=".2f",
                     tickfont=dict(family=FONT_MONO, size=FS_TICK),
                     # The five collapsed methods sit on zero, so the baseline
                     # has to be drawn or their bars have nothing to stand on.
                     zeroline=True, zerolinecolor=AC["axis"], zerolinewidth=1.0)
    fig.update_layout(title=None, margin=dict(l=68, r=172, t=16, b=78),
                      showlegend=True, barmode="group", bargap=0.12,
                      bargroupgap=0.0,
                      legend=dict(x=1.015, y=1.0, xanchor="left", yanchor="top",
                                  font=dict(family=FONT_UI, size=10.5),
                                  bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                  itemsizing="constant"))
    add_footnote(fig, (
        "Ten tasks per band. Each bar is the mean over seeds of that seed's mean "
        "score on the band's tasks, and the whisker is ±1 s.d. across<br>"
        "&nbsp;&nbsp;&nbsp;seeds. A seed scores a band only if it reached at least "
        "five of that band's ten tasks, so a short seed contributes to the near<br>"
        "&nbsp;&nbsp;&nbsp;bands and not the far ones, and a bar with no whisker rests "
        "on one seed.<br>"
        "&nbsp;&nbsp;&nbsp;The dashed rule in each group is the joint ceiling for that "
        "band, one network trained on all 50 at the same 204.8M-frame budget.<br>"
        f"&nbsp;&nbsp;&nbsp;Seed depths — {seed_note(d['methods'], 5)}."
    ), left=68)
    return fig


# ── Figure 6: does the constraint actually get satisfied? ───────────────────
def fig_constraint_satisfaction(d: dict) -> go.Figure:
    """Did the constraint bind, and did the dual then satisfy it?

    This is a DISTRIBUTION plot, not a per-task one. Each curve is the 49
    shortfalls of its own kind, sorted from largest to smallest, and the x
    position is a rank in that sorted list rather than a task index. The two
    curves are sorted SEPARATELY, so rank r on one is generally a different
    task from rank r on the other and the vertical distance between them is
    not any one task's improvement.

    What it does show is the thing a reviewer asks first: the whole
    distribution of shortfalls moved down, and most of it moved below the
    threshold. Training order was tried instead and is worse for that purpose,
    because both curves then zig-zag and the shift is impossible to see. The
    per-task facts the ranking cannot carry are stated as counts in the header.

    The constraint is F = [V_L - V_G]_+^2 <= eps with eps = 0.09, so it is met
    when the shortfall itself is at or below sqrt(0.09) = 0.30.
    """
    EPS_GAP = 0.09 ** 0.5
    by_task: dict[int, list[tuple[int, float]]] = {}
    for r in d["dual"]:
        by_task.setdefault(r["task_idx"], []).append((r["iter"], r["shortfall_current"]))
    pairs = []
    for k in sorted(by_task):
        series = sorted(by_task[k])
        pairs.append((series[0][1], series[-1][1]))
    n = len(pairs)
    starts = sorted((a for a, _ in pairs), reverse=True)
    ends = sorted((b for _, b in pairs), reverse=True)
    xs = list(range(1, n + 1))

    met = sum(1 for _, b in pairs if b <= EPS_GAP)
    worse = sum(1 for a, b in pairs if b > a + 1e-6)
    bound = sum(1 for a, _ in pairs if a > EPS_GAP)

    fig = go.Figure()
    fig.add_hrect(y0=-0.02, y1=EPS_GAP, fillcolor=hex_to_rgba(C_CEIL, 0.08),
                  line=dict(width=0), layer="below")
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1], y=starts + ends[::-1], fill="toself",
        fillcolor=hex_to_rgba(C_OURS, 0.13), line=dict(width=0), mode="lines",
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=starts, mode="lines", name="start of the global phase",
        line=dict(width=1.9, color=AC["text_muted"], dash="dot"),
        hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(
        x=xs, y=ends, mode="lines", name="end of the global phase",
        line=dict(width=2.3, color=C_OURS),
        hoverinfo="skip", showlegend=True))
    fig.add_hline(y=EPS_GAP, line=dict(color=C_CEIL, width=1.4, dash="dash"))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name=f"constraint satisfied below (√ε = {EPS_GAP:.2f})",
        line=dict(color=C_CEIL, width=1.4, dash="dash"), showlegend=True))
    # Where each curve crosses the threshold: the count of tasks still outside
    # it, read straight off the x axis.
    for values, color, label, dy in ((starts, AC["text_muted"], "start", 26),
                                     (ends, C_OURS, "end", -26)):
        crossing = sum(1 for v in values if v > EPS_GAP)
        if 0 < crossing < n:
            fig.add_annotation(
                x=crossing, y=EPS_GAP, ax=0, ay=dy, showarrow=True, arrowhead=0,
                arrowwidth=1.0, arrowcolor=color,
                text=f"{crossing} still outside at the {label}",
                font=dict(family=FONT_UI, size=9.5, color=color),
                bgcolor="rgba(255,255,255,0.86)", borderpad=2,
                xanchor="left" if label == "start" else "right")
    fig.update_xaxes(
        title=dict(text="Rank, 1 = largest shortfall (each curve sorted on its own)",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[0.5, n + 0.5], dtick=10,
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    fig.update_yaxes(
        title=dict(text="Current-task shortfall  [V<sub>L</sub> − V<sub>G</sub>]<sub>+</sub>",
                   font=dict(family=FONT_UI, size=FS_AXIS)),
        range=[-0.02, 1.02], dtick=0.25, tickformat=".2f",
        tickfont=dict(family=FONT_MONO, size=FS_TICK))
    # Legend and counts both sit ABOVE the plotting area, on their own lines.
    fig.update_layout(
        title=None, margin=dict(l=78, r=28, t=80, b=66), showlegend=True,
        legend=dict(orientation="h", x=0, y=1.055, xanchor="left",
                    yanchor="bottom", font=dict(family=FONT_UI, size=10.5),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.175, xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        text=(f"<b>{bound}</b> of {n} tasks start outside the constraint "
              f"&nbsp;·&nbsp; <b>{met}</b> end satisfied "
              f"&nbsp;·&nbsp; <b>{worse}</b> end worse than they started"),
        font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"]))
    add_footnote(fig, (
        "The shortfall is how far the global model's value sits below its frozen "
        "specialist's on the task it is learning. Each task gives two of<br>"
        "&nbsp;&nbsp;&nbsp;them: one at the first dual refresh of its global phase and "
        "one at the last.<br>"
        "<b>Reading the x axis.</b> This is a distribution, not a task sequence. Each "
        "curve holds the same 49 numbers it always had, sorted from<br>"
        "&nbsp;&nbsp;&nbsp;largest to smallest, and x is the position in THAT curve's "
        "own sorted list. The two curves are sorted independently, so rank 10 on<br>"
        "&nbsp;&nbsp;&nbsp;one is usually a different task from rank 10 on the other. "
        "The shaded gap is therefore how far the whole distribution moved down,<br>"
        "&nbsp;&nbsp;&nbsp;not any single task's improvement. Read per-task facts from "
        "the counts above, which are computed task by task.<br>"
        "&nbsp;&nbsp;&nbsp;Task 1 has no global phase and is absent. DUEL seed 0 only, "
        "the one run that logs a dual trace."
    ), left=78)
    return fig


# ── Figure 7: headline table ────────────────────────────────────────────────
def fig_headline_metrics(d: dict) -> go.Figure:
    """The numbers, as a booktabs table drawn with shapes on hidden axes.

    Every value is the mean over the seeds that completed all 50 tasks, read at
    50, with the s.d. across those same seeds beside it. Seeds that stopped
    early are excluded here and appear only in the curves and the matrix, where
    their own depth is respected. Pooling seeds of different lengths into one
    mean was tried and discarded: it put CLEAR above DUEL on PERF purely
    because CLEAR's unfinished seeds were shorter, and a short sequence has
    less to retain.
    """
    def share(values: list[float], test) -> float:
        return sum(1 for v in values if test(v)) / len(values)

    HEADS = [("", ""), ("PERF", "↑"), ("Forgetting", "↓"),
             ("Backward<br>transfer", "↑"), ("Forward<br>transfer", "↑"),
             ("≥0.9", "↑"), ("≤0.1", "↓")]
    rows: list[tuple] = [("REFERENCE POINTS", None, None)]
    for label, values in (("Random policy", d["floor"]),
                          ("BFS shortest-path oracle", d["oracle"]),
                          ("Specialist, 4.096M frames each", d["specialist"]),
                          ("Joint, 204.8M frames", d["joint_scores"])):
        rows.append((label, [
            (statistics.mean(values), 0.0), None, None, None,
            (share(values, lambda v: v >= 0.9), 0.0),
            (share(values, lambda v: v <= 0.1), 0.0)], None))
    rows.append(("CONTINUAL, 204.8M GLOBAL FRAMES", None, None))
    for m in d["methods"]:
        h = m["headline"]
        rows.append((m["label"], [
            (h["perf"], m["sd_perf"]),
            (h["forgetting"], m["sd_forgetting"]),
            (h["bwt"], m["sd_bwt"]),
            None if m["fwt"] is None else (m["fwt"], 0.0),
            (share(h["final"], lambda v: v >= 0.9), 0.0),
            (share(h["final"], lambda v: v <= 0.1), 0.0)], m))

    # Best value per column, over the continual rows only. Reference rows are
    # not competitors, and a column where the oracle wins says nothing.
    continual = [r for r in rows if r[2] is not None]
    best = []
    for col in range(6):
        vals = [r[1][col][0] for r in continual
                if r[1][col] is not None and r[1][col][0] == r[1][col][0]]
        lower_is_better = col in (1, 5)
        best.append((min(vals) if lower_is_better else max(vals)) if vals else None)

    fig = go.Figure()
    y = 0.0
    header_h, row_h, sect_h = 0.086, 0.050, 0.034
    xs = [0.013, 0.418, 0.568, 0.718, 0.845, 0.930, 1.0]
    aligns = ["left"] + ["right"] * 6

    fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                  line=dict(color=AC["axis"], width=1.3))
    y -= header_h
    for x, align, (text, arrow) in zip(xs, aligns, HEADS):
        fig.add_annotation(x=x, y=y + header_h * 0.64, text=f"<b>{text}</b>",
                           xanchor=align, yanchor="middle", showarrow=False,
                           font=dict(family=FONT_UI, size=9.5, color=AC["text_primary"]))
        if arrow:
            fig.add_annotation(
                x=x, y=y + header_h * 0.17, xanchor=align, yanchor="middle",
                showarrow=False,
                text=f"{arrow} better",
                font=dict(family=FONT_UI, size=7.4, color=AC["text_muted"]))
    fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                  line=dict(color=AC["axis"], width=1.0))

    for label, cells, method in rows:
        if cells is None:
            y -= sect_h
            fig.add_annotation(x=xs[0], y=y + sect_h * 0.3, text=label.upper(),
                               xanchor="left", yanchor="middle", showarrow=False,
                               font=dict(family=FONT_UI, size=8.5,
                                         color=AC["text_muted"]))
            continue
        y -= row_h
        mid = y + row_h * 0.34
        band = ("#EFF6FF" if label.startswith("DUEL")
                else "#FEF3C7" if label.startswith("CLEAR") else None)
        if band:
            fig.add_shape(type="rect", x0=0, x1=1, y0=y, y1=y + row_h,
                          fillcolor=band, line=dict(width=0), layer="below")
        fig.add_annotation(
            x=xs[0], y=mid, text=(f"<b>{label}</b>" if method else label),
            xanchor="left", yanchor="middle", showarrow=False,
            font=dict(family=FONT_UI, size=10,
                      color=method["color"] if method else AC["text_primary"]))
        if method is not None:
            spread = (f"{method['n_complete']} of {method['n_seeds']} seeds at 50"
                      if method["n_seeds"] > 1
                      else f"{method['n_complete']} seed")
            fig.add_annotation(
                x=xs[0] + 0.168, y=mid, text=spread, xanchor="left",
                yanchor="middle", showarrow=False,
                font=dict(family=FONT_UI, size=8, color=AC["text_muted"]))
        for col, (x, align) in enumerate(zip(xs[1:], aligns[1:])):
            cell = cells[col]
            if cell is None:
                fig.add_annotation(x=x, y=mid, text="—", xanchor=align,
                                   yanchor="middle", showarrow=False,
                                   font=dict(family=FONT_MONO, size=9.5,
                                             color=AC["text_muted"]))
                continue
            value, sd = cell
            as_pct = col in (4, 5)
            shown = f"{value:.0%}" if as_pct else (
                f"{value:+.3f}" if col == 2 and method else f"{value:.3f}")
            is_best = (method is not None and best[col] is not None
                       and abs(value - best[col]) < 1e-12)
            text = f"<b>{shown}</b>" if is_best else shown
            if method is not None and sd > 0 and not as_pct:
                text += (f" <span style='font-size:7.4px;color:{AC['text_muted']}'>"
                         f"±{sd:.3f}</span>")
            fig.add_annotation(
                x=x, y=mid, text=text, xanchor=align, yanchor="middle",
                showarrow=False,
                font=dict(family=FONT_MONO, size=9.5, color=AC["text_primary"]))
    fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                  line=dict(color=AC["axis"], width=1.3))
    fig.add_annotation(
        x=0, y=y - 0.026, xanchor="left", yanchor="top", showarrow=False, align="left",
        text=("Every value is the mean over that method's seeds that completed all "
              "50 tasks, read at 50, with ±1 s.d. across those same<br>"
              "&nbsp;&nbsp;&nbsp;seeds beside it. The count of complete seeds is given "
              "per row, and a row with one of them carries no ±. Seeds still "
              "running<br>&nbsp;&nbsp;&nbsp;are excluded here and appear only in the "
              "curves and the matrix, where their own depth is respected. "
              "<b>Bold</b> is the best<br>&nbsp;&nbsp;&nbsp;value in each column, over "
              "the continual rows only. Seed depths —<br>"
              f"&nbsp;&nbsp;&nbsp;{seed_note(d['methods'], 5)}.<br>"
              "Forgetting is the drop from each task's best recorded score to its last "
              "one. Forward transfer is the raw within-phase gap against<br>"
              "&nbsp;&nbsp;&nbsp;from-scratch, an area under a learning curve that "
              "cannot be recomputed from an evaluation matrix; it reads — where the<br>"
              "&nbsp;&nbsp;&nbsp;run summary has not yet measured it. Reference rows are "
              "not continual methods and do not compete for the best value."),
        font=dict(family=FONT_UI, size=8, color=AC["text_muted"]))
    fig.update_xaxes(visible=False, range=[-0.015, 1.03])
    fig.update_yaxes(visible=False, range=[y - 0.215, 0.02])
    fig.update_layout(title=None, margin=dict(l=14, r=14, t=12, b=12),
                      plot_bgcolor=AC["bg"])
    return fig


# ── Driver ──────────────────────────────────────────────────────────────────
FIGURES = [
    ("retention_matrix", fig_retention_matrix, W_FULL, 430),
    ("retention_curves", fig_retention_curves, W_FULL, None),
    ("score_distribution", fig_score_distribution, W_FULL, 530),
    # Narrow canvas on purpose: the quadrant panels have to be square.
    ("learned_vs_retained", fig_learned_vs_retained, 445, 700),
    ("performance_by_band", fig_performance_by_band, W_FULL, None),
    ("constraint_satisfaction", fig_constraint_satisfaction, W_FULL, None),
    ("headline_metrics", fig_headline_metrics, W_FULL, 620),
]

# Dropped after review. Deleted on every run so a stale PNG cannot linger in the
# folder and be mistaken for current output.
RETIRED = ["acquisition_vs_retention", "axis_confound", "specialist_margin",
           "method_comparison", "acquisition_curves", "dual_dynamics",
           "compute_cost", "budget_sensitivity"]


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

    print(f"\nBigGrid v2 — joint ceiling {data['joint_mean']:.3f}, "
          f"oracle {statistics.mean(data['oracle']):.3f}")
    for m in data["methods"]:
        depths = "/".join(str(x) for x in sorted(m["depths"], reverse=True))
        h = m["headline"]
        print(f"  {m['label']:<14} PERF {h['perf']:>6.3f} ±{m['sd_perf']:.3f}"
              f"  BWT {h['bwt']:>+6.3f} ±{m['sd_bwt']:.3f}"
              f"  forget {h['forgetting']:>5.3f}"
              f"   {m['n_complete']}/{m['n_seeds']} seeds complete; depths {depths}")
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
        print(f"  {fig_id:<22} {png.stat().st_size//1024:>5} KB png  "
              f"{svg.stat().st_size//1024:>5} KB svg{flag}")

    print(f"\nwrote {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
