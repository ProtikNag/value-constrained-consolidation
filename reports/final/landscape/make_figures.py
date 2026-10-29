#!/usr/bin/env python3
"""One task boundary in parameter space, told in order.

A task boundary is two phases. The specialist is cloned from the global policy
carried in from the previous task and trained on the new task alone. The global
policy is then consolidated from that same point under a constraint that only
exists once the specialist does, because the specialist's value IS the floor.
Both phases leave from one place, so both are drawn on one plane taken from the
two recorded paths together.

Three figures, in the order the phases happen.

``specialist_phase_3d``   the specialist, read against the two objectives it
                          trades off. No constraint line: at this point in the
                          sequence there is no constraint to draw.
``consolidation_phase_3d`` the consolidation, with the region the constraint
                          admits raised out of the greyed remainder.
``task_boundary_2d``      the whole boundary flat, in three acts.

Usage::

    python reports/final/landscape/make_figures.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import AC, FONT_MONO, FONT_UI, install_template  # noqa: E402

DATA = HERE / "data"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
TASK = 5
PAPER_MODE = bool(os.environ.get("FIGS_FOR_PAPER"))

C_LOCAL = AC["amber"]        # the specialist branch
C_GLOBAL = AC["blue"]        # the consolidation branch, and "ours" everywhere
C_ORIGIN = AC["text_primary"]
C_LIMIT = AC["green"]   # distinct from both ends of a blue-red scale
PAD = 2.6                    # data units kept around the two paths

# Low must not read as the page. Diverging blue to red through a pale middle,
# which is the palette's diverging scale rather than a rainbow.
DIVERGE = [[0.0, "#132A82"], [0.16, "#2449C2"], [0.38, "#4C54D6"],
           [0.58, "#7C3AED"], [0.78, "#A52C93"], [1.0, "#B91C1C"]]
# Ruled-out ground keeps its shape but loses its colour, so the two regions
# read as areas rather than as a line drawn across one surface.
GREYS = [[0.0, "#CED4DA"], [1.0, "#6C757D"]]


def load() -> dict:
    return json.loads((DATA / f"surface_k{TASK}.json").read_text())


def smooth(grid, k: int = 1):
    """Box blur over a (2k+1) window, used for the constraint curve only."""
    g = np.asarray(grid, dtype=float)
    padded = np.pad(g, k, mode="edge")
    return np.mean([padded[i:i + g.shape[0], j:j + g.shape[1]]
                    for i in range(2 * k + 1) for j in range(2 * k + 1)], axis=0)


def window(d: dict, keys: tuple[str, ...]) -> tuple:
    """Crop the measured grid to the branches named, so the view is close in."""
    xs, ys = np.array(d["xs"]), np.array(d["ys"])
    pts = [q for key in keys for q in d["paths"][key]]
    x0 = max(xs[0], min(q["x"] for q in pts) - PAD)
    x1 = min(xs[-1], max(q["x"] for q in pts) + PAD)
    y0 = max(ys[0], min(q["y"] for q in pts) - PAD)
    y1 = min(ys[-1], max(q["y"] for q in pts) + PAD)
    ix = np.where((xs >= x0) & (xs <= x1))[0]
    iy = np.where((ys >= y0) & (ys <= y1))[0]
    return xs[ix], ys[iy], ix, iy, (x0, x1, y0, y1)


def crop(grid, ix, iy) -> np.ndarray:
    return np.asarray(grid, dtype=float)[np.ix_(iy, ix)]


def lookup(xs, ys, grid, pts) -> np.ndarray:
    """Bilinear height of `grid` at the given (x, y) points."""
    g = np.asarray(grid, dtype=float)
    xi = np.clip(np.interp(pts[:, 0], xs, np.arange(len(xs))), 0, len(xs) - 1.001)
    yi = np.clip(np.interp(pts[:, 1], ys, np.arange(len(ys))), 0, len(ys) - 1.001)
    x0, y0 = xi.astype(int), yi.astype(int)
    fx, fy = xi - x0, yi - y0
    return ((1 - fx) * (1 - fy) * g[y0, x0] + fx * (1 - fy) * g[y0, x0 + 1]
            + (1 - fx) * fy * g[y0 + 1, x0] + fx * fy * g[y0 + 1, x0 + 1])


def contour_xy(xs, ys, field, level):
    """Vertices of the `level` contour of `field`, via matplotlib, not drawn.

    Plotly's Surface can contour its own coordinates but not a second field, so
    the constraint curve is extracted here and laid on the surface by looking
    its height up in the objective grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tmp, ax = plt.subplots()
    cs = ax.contour(xs, ys, field, levels=[level])
    segs = [np.asarray(seg) for per_level in cs.allsegs for seg in per_level]
    plt.close(tmp)
    return [seg for seg in segs if len(seg) > 2]


# ── 3D ──────────────────────────────────────────────────────────────────────
LIFT = 0.075
# Strip every word from the 3D figures, for dropping into slides and
# annotating there: no axis titles, no tick labels, no A/B/C letters, no
# heading, no colourbar, no key. The surfaces, the branches, the markers and
# the grid, nothing else. The grid darkens a step so it still reads once the
# axis labels that framed it are gone.
BARE = False
GRID_BARE = "#CED4DA"
# Bare mode drops the heading, the colourbar and the axis labels, so the art
# no longer fills the canvas the framed version was sized for. Pulling the
# camera in takes the slack out, which matters in slides: the white border
# otherwise travels with the object and makes it awkward to align.
BARE_ZOOM = 0.86


def zoom(cam: dict) -> dict:
    """Same angle, closer in, when the figure is bare."""
    if not BARE:
        return cam
    eye = {k: v * BARE_ZOOM for k, v in cam["eye"].items()}
    return dict(cam, eye=eye)


def scene_axes(zlabel: str) -> dict:
    grid = GRID_BARE if BARE else AC["grid"]
    common = dict(showbackground=True, backgroundcolor=AC["bg"],
                  gridcolor=grid, zerolinecolor=grid,
                  showticklabels=not BARE,
                  tickfont=dict(family=FONT_MONO, size=9,
                                color=AC["text_muted"]))
    font = dict(family=FONT_UI, size=11, color=AC["text_primary"])

    def title(text):
        return dict(text="" if BARE else text, font=font)

    return dict(
        xaxis=dict(title=title("PC 1"), **common),
        yaxis=dict(title=title("PC 2"), **common),
        zaxis=dict(title=title(zlabel), range=[0, 1.18], **common))


def add_branch(fig, scene, xs, ys, field, pts, color, lift=LIFT):
    """The branch on the surface, with a cone at the end for direction."""
    px = np.array([q["x"] for q in pts])
    py = np.array([q["y"] for q in pts])
    pz = lookup(xs, ys, np.nan_to_num(field, nan=0.0),
                np.stack([px, py], 1)) + lift
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=color, width=8), scene=scene))
    # Direction of travel, from the last stretch of the path.
    ux, uy, uz = px[-1] - px[-6], py[-1] - py[-6], pz[-1] - pz[-6]
    fig.add_trace(go.Cone(
        x=[px[-1]], y=[py[-1]], z=[pz[-1]], u=[ux], v=[uy], w=[uz],
        sizemode="absolute", sizeref=0.6, anchor="tip", showscale=False,
        colorscale=[[0, color], [1, color]], hoverinfo="skip", scene=scene))
    return px, py, pz


def add_marker(fig, scene, x, y, z, letter, color, symbol):
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers" if BARE else "markers+text", text=[f"{letter}"],
        textposition="top center",
        textfont=dict(family=FONT_UI, size=14, color=color),
        marker=dict(color=color, size=9, symbol=symbol,
                    line=dict(color=AC["bg"], width=2)),
        hoverinfo="skip", showlegend=False, scene=scene))


def fig_specialist(d: dict) -> tuple[go.Figure, int, int]:
    """The specialist phase against the two objectives it trades off."""
    xs, ys, ix, iy, _ = window(d, ("local",))
    past = crop(d["Z_past_score"], ix, iy)
    new = crop(d["Z_new_score"], ix, iy)
    pts = d["paths"]["local"]
    # Leaning toward PC 2 rather than looking straight down it, so the task-5
    # cliff is seen obliquely instead of edge on.
    cam = dict(eye=dict(x=1.06, y=-1.01, z=0.74), up=dict(x=0, y=0, z=1))

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.03,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=None if BARE else
                       (f"height = mean score on tasks 1-{TASK - 1}",
                        f"height = score on task {TASK}"))

    for col, field, zlab in ((1, past, f"tasks 1-{TASK - 1}"),
                             (2, new, f"task {TASK}")):
        scene = f"scene{col if col > 1 else ''}"
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=field, cmin=float(np.nanmin(field)), cmax=1,
            colorscale=DIVERGE, showscale=(col == 2) and not BARE,
            opacity=0.98, hoverinfo="skip", scene=scene,
            contours=dict(z=dict(show=True, usecolormap=False,
                                 color="rgba(73,80,87,0.22)", width=1)),
            colorbar=dict(title=dict(text="normalized<br>score", side="top",
                                     font=dict(family=FONT_UI, size=10)),
                          tickvals=[0, 0.5, 1.0],
                          tickfont=dict(family=FONT_MONO, size=9.5),
                          thickness=11, len=0.55, outlinewidth=0,
                          x=1.005, y=0.5)))
        px, py, pz = add_branch(fig, scene, xs, ys, field, pts, C_LOCAL)
        add_marker(fig, scene, px[0], py[0], pz[0], "A", C_ORIGIN, "circle")
        add_marker(fig, scene, px[-1], py[-1], pz[-1], "B", C_LOCAL, "square")
        dom = (dict(x=[0.0, 0.5] if col == 1 else [0.5, 1.0], y=[0.0, 1.0])
               if BARE else
               dict(x=[0.02, 0.47] if col == 1 else [0.53, 0.98], y=[0.0, 1.0]))
        fig.update_layout(**{scene: dict(**scene_axes(zlab), camera=cam,
                                         domain=dom,
                                         aspectratio=dict(x=1.0, y=1.0, z=0.62))})

    for note in fig.layout.annotations[:2]:
        note.font.update(family=FONT_UI, size=12, color=AC["text_primary"])
    if BARE:
        fig.update_layout(title=None, showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        return fig, 900, 440
    fig.add_annotation(
        xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
        yanchor="bottom", yshift=26, showarrow=False, align="left",
        text=f"<b>Specialist phase.</b>&nbsp;&nbsp;"
             f"<span style='color:{C_ORIGIN}'>&#9679;</span> <b>A</b>, the "
             f"global policy carried in from task {TASK - 1}, where specialist "
             f"training starts.&nbsp;&nbsp;"
             f"<span style='color:{C_LOCAL}'>&#9632;</span> <b>B</b>, the "
             f"trained specialist. Same path on both surfaces.",
        font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"]))
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=4, r=28, t=78, b=22))
    return fig, 900, 440


def fig_task5(d: dict) -> tuple[go.Figure, int, int]:
    """The task-5 cliff on its own, at a size worth projecting.

    The same window, camera and height field as the right half of
    `specialist_phase_3d`, which is panel (e) of the overview. This is the
    surface the constraint is cut from: the specialist climbs it, and the
    near-vertical face is why the admissible region has the shape it does.
    """
    xs, ys, ix, iy, _ = window(d, ("local",))
    field = crop(d["Z_new_score"], ix, iy)
    pts = d["paths"]["local"]
    scene = "scene"

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=field, cmin=float(np.nanmin(field)), cmax=1,
        colorscale=DIVERGE, showscale=not BARE, opacity=0.98,
        hoverinfo="skip", scene=scene,
        contours=dict(z=dict(show=True, usecolormap=False,
                             color="rgba(73,80,87,0.22)", width=1)),
        colorbar=dict(title=dict(text="normalized<br>score", side="top",
                                 font=dict(family=FONT_UI, size=10)),
                      tickvals=[0, 0.5, 1.0],
                      tickfont=dict(family=FONT_MONO, size=9.5),
                      thickness=11, len=0.55, outlinewidth=0, x=1.0, y=0.5)))
    px, py, pz = add_branch(fig, scene, xs, ys, field, pts, C_LOCAL)
    add_marker(fig, scene, px[0], py[0], pz[0], "A", C_ORIGIN, "circle")
    add_marker(fig, scene, px[-1], py[-1], pz[-1], "B", C_LOCAL, "square")

    fig.update_layout(**{scene: dict(
        **scene_axes(f"task {TASK}"),
        camera=dict(eye=dict(x=1.14, y=-1.09, z=0.80), up=dict(x=0, y=0, z=1)),
        domain=dict(x=[0.0, 1.0 if BARE else 0.93], y=[0.0, 1.0]),
        aspectratio=dict(x=1.0, y=1.0, z=0.62))})
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=0, r=0 if BARE else 18, t=0, b=0))
    return fig, 720, 560


def fig_consolidation(d: dict) -> tuple[go.Figure, int, int]:
    """Consolidation, with the ruled-out ground drained of colour.

    Height here is the mean over ALL FIVE tasks seen, not tasks 1 to 4. That is
    the quantity on which A and C are comparable: A has not learned task 5 yet,
    so judging it on the earlier four alone flatters it and makes consolidation
    look like a loss. On all five, A is 0.56 and C is 0.53.
    """
    xs, ys, ix, iy, _ = window(d, ("global",))
    n_past = d["n_past"]
    past = crop(d["Z_past_score"], ix, iy)
    new_f = crop(d["Z_new_score"], ix, iy)
    seen = (n_past * past + new_f) / (n_past + 1)
    short = smooth(crop(d["Z_shortfall"], ix, iy))
    budget = d["eps"] ** 0.5
    ok = short <= budget
    pts = d["paths"]["global"]
    # From the other side, so the branch runs across the view rather than away.
    cam = zoom(dict(eye=dict(x=-0.92, y=-1.32, z=0.74),
                    up=dict(x=0, y=0, z=1)))
    scene = "scene"
    lo_z, hi_z = float(np.nanmin(seen)), float(np.nanmax(seen))

    fig = go.Figure()
    # Ruled out: same terrain, no colour. An area, not a line.
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=np.where(ok, np.nan, seen), cmin=lo_z, cmax=hi_z,
        colorscale=GREYS, showscale=False, opacity=0.85, hoverinfo="skip",
        scene=scene,
        contours=dict(z=dict(show=True, usecolormap=False,
                             color="rgba(73,80,87,0.22)", width=1))))
    # Admissible: the measurement, in full colour.
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=np.where(ok, seen, np.nan), cmin=lo_z, cmax=hi_z,
        colorscale=DIVERGE, showscale=not BARE, opacity=1.0, hoverinfo="skip",
        scene=scene,
        contours=dict(z=dict(show=True, usecolormap=False,
                             color="rgba(73,80,87,0.22)", width=1)),
        colorbar=dict(title=dict(text="mean score,<br>all 5 tasks", side="top",
                                 font=dict(family=FONT_UI, size=10)),
                      tickvals=[round(lo_z, 1), round((lo_z + hi_z) / 2, 1),
                                round(hi_z, 1)],
                      tickfont=dict(family=FONT_MONO, size=9.5),
                      thickness=11, len=0.55, outlinewidth=0, x=1.0, y=0.5)))

    for seg in contour_xy(xs, ys, short, budget):
        z = lookup(xs, ys, seen, seg) + LIFT * 1.5
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=z, mode="lines", hoverinfo="skip",
            showlegend=False, line=dict(color=C_LIMIT, width=5), scene=scene))

    px, py, pz = add_branch(fig, scene, xs, ys, seen, pts, C_GLOBAL)
    add_marker(fig, scene, px[0], py[0], pz[0], "A", C_ORIGIN, "circle")
    add_marker(fig, scene, px[-1], py[-1], pz[-1], "C", C_GLOBAL, "diamond")

    # Off-scene swatches, so the two regions are named in a key rather than in
    # the caption.
    for label, colour in () if BARE else (
            ("admissible under the constraint", "#7C3AED"),
            ("ruled out by the constraint", "#9AA2AA")):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", name=label,
            marker=dict(size=11, symbol="square", color=colour),
            showlegend=True, hoverinfo="skip", scene=scene))

    axes = scene_axes(f"mean score, tasks 1-{TASK}")
    axes["zaxis"]["range"] = [max(0.0, lo_z - 0.05), hi_z + 0.12]
    fig.update_layout(**{scene: dict(**axes, camera=cam,
                                     domain=dict(x=[0.0, 1.0 if BARE else 0.97],
                                                 y=[0.0, 1.0]),
                                     aspectratio=dict(x=1.0, y=1.0, z=0.62))})
    if BARE:
        fig.update_layout(title=None, showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        return fig, 700, 420
    fig.add_annotation(
        xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
        yanchor="bottom", yshift=20, showarrow=False, align="left",
        text=f"<b>Consolidation phase.</b>&nbsp;&nbsp;"
             f"<span style='color:{C_ORIGIN}'>&#9679;</span> <b>A</b>, the "
             f"same start.&nbsp;&nbsp;"
             f"<span style='color:{C_GLOBAL}'>&#9670;</span> <b>C</b>, the "
             f"global policy carried on to task {TASK + 1}.&nbsp;&nbsp;"
             f"<span style='color:{C_LIMIT}'>&#9472;&#9472;</span> the "
             f"constraint edge.<br>Height is the mean score over all "
             f"{TASK} tasks seen, which is what A and C are comparable on.",
        font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"]))
    fig.update_layout(
        title=None, showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.0, xanchor="left",
                    yanchor="bottom", font=dict(family=FONT_UI, size=10.5),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    itemsizing="constant"),
        margin=dict(l=4, r=24, t=84, b=14))
    return fig, 700, 440


# ── 2D: the boundary in three acts ──────────────────────────────────────────
def arrow_2d(fig, pts, color, xref, yref, at: float = 0.62,
             span: float = 0.10):
    """A short arrow partway along a branch, showing direction of travel.

    Placed mid-path rather than at the end, where the head would sit under the
    endpoint marker and never be seen.
    """
    n = len(pts)
    tip, tail = pts[int(n * (at + span))], pts[int(n * at)]
    fig.add_annotation(
        x=tip["x"], y=tip["y"], ax=tail["x"], ay=tail["y"],
        xref=xref, yref=yref, axref=xref, ayref=yref,
        showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2.6,
        arrowcolor=color, text="")


def fig_flat(d: dict) -> tuple[go.Figure, int, int]:
    """Three panels in the order the phases happen."""
    xs, ys, ix, iy, (x0, x1, y0, y1) = window(d, ("local", "global"))
    n_past = d["n_past"]
    past = crop(d["Z_past_score"], ix, iy)
    new = crop(d["Z_new_score"], ix, iy)
    seen = (n_past * past + new) / (n_past + 1)
    short = smooth(crop(d["Z_shortfall"], ix, iy))
    budget = d["eps"] ** 0.5
    gl, lo = d["paths"]["global"], d["paths"]["local"]

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.035,
        subplot_titles=(
            f"1.  specialist training, on tasks 1-{TASK - 1}",
            f"2.  the specialist sets the floor on task {TASK}",
            f"3.  consolidation, on all {TASK} tasks seen"))

    panels = [(1, past, False, "local"), (2, new, True, None),
              (3, seen, True, "global")]
    for col, field, show_limit, branch in panels:
        suffix = "" if col == 1 else str(col)
        xref, yref = f"x{suffix}", f"y{suffix}"
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=field, hoverinfo="skip", showlegend=False,
            contours=dict(start=0.0, end=1.0, size=0.0625, coloring="fill"),
            colorscale=DIVERGE, showscale=(col == 3),
            line=dict(width=0.4, color="rgba(73,80,87,0.18)"),
            colorbar=dict(
                title=dict(text="normalized<br>score", side="top",
                           font=dict(family=FONT_UI, size=10)),
                tickvals=[0, 0.5, 1.0],
                tickfont=dict(family=FONT_MONO, size=9.5),
                thickness=10, len=0.62, outlinewidth=0, x=1.015, y=0.5)),
            row=1, col=col)

        # The constraint does not exist during specialist training: it is
        # defined by the specialist's value, which only exists once B does.
        if show_limit:
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=short, hoverinfo="skip", showlegend=False,
                contours=dict(start=budget, end=budget, size=1,
                              coloring="none"),
                line=dict(color=C_LIMIT, width=2.2), showscale=False),
                row=1, col=col)

        if branch is not None:
            pts = d["paths"][branch]
            color = C_LOCAL if branch == "local" else C_GLOBAL
            fig.add_trace(go.Scatter(
                x=[q["x"] for q in pts], y=[q["y"] for q in pts],
                mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color=color, width=2.6)), row=1, col=col)
            arrow_2d(fig, pts, color, xref, yref)

        marks = {1: [("A", gl[0], C_ORIGIN, "circle"),
                     ("B", lo[-1], C_LOCAL, "square")],
                 2: [("A", gl[0], C_ORIGIN, "circle"),
                     ("B", lo[-1], C_LOCAL, "square")],
                 3: [("A", gl[0], C_ORIGIN, "circle"),
                     ("C", gl[-1], C_GLOBAL, "diamond")]}[col]
        for letter, q, color, symbol in marks:
            fig.add_trace(go.Scatter(
                x=[q["x"]], y=[q["y"]], mode="markers", showlegend=False,
                hoverinfo="skip",
                marker=dict(color=color, size=13, symbol=symbol,
                            line=dict(color=AC["bg"], width=1.8))),
                row=1, col=col)
            fig.add_annotation(
                x=q["x"], y=q["y"], xref=xref, yref=yref, xanchor="center",
                yanchor="middle", yshift=-16 if letter == "C" else 16,
                showarrow=False, text=f"<b>{letter}</b>",
                font=dict(family=FONT_UI, size=12, color=color),
                bgcolor="rgba(255,255,255,0.88)", borderpad=1)

        fig.update_xaxes(
            title=dict(text="PC 1", font=dict(family=FONT_UI, size=11)),
            range=[x0, x1], showgrid=False, zeroline=False,
            tickfont=dict(family=FONT_MONO, size=9), row=1, col=col)
        fig.update_yaxes(
            title=dict(text="PC 2" if col == 1 else None,
                       font=dict(family=FONT_UI, size=11)),
            range=[y0, y1], showgrid=False, zeroline=False,
            scaleanchor="x" if col == 1 else f"x{col}", scaleratio=1,
            tickfont=dict(family=FONT_MONO, size=9), row=1, col=col)

    for note in fig.layout.annotations[:3]:
        note.font.update(family=FONT_UI, size=11.5, color=AC["text_primary"])

    panel_w = (WIDTH_2D - 54 - 86) * (1 - 2 * 0.035) / 3
    height = int(round(panel_w / ((x1 - x0) / (y1 - y0)) + 54 + 52 + KEY_2D))

    glyph = {"A": "&#9679;", "B": "&#9632;", "C": "&#9670;"}
    rows = [("A", gl[0], C_ORIGIN, "global in", f"carried from task {TASK - 1}"),
            ("B", lo[-1], C_LOCAL, "specialist out", "sets the floor"),
            ("C", gl[-1], C_GLOBAL, "global out", f"carried to task {TASK + 1}")]
    for i, (letter, q, color, name, sub) in enumerate(rows):
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=0.0, xanchor="left",
            yanchor="top", yshift=-52 - i * 18, showarrow=False, align="left",
            text=f"<span style='color:{color}'>{glyph[letter]}</span>&nbsp; "
                 f"<b>{letter}</b>&nbsp;&nbsp;{name} <span style='color:"
                 f"{AC['text_muted']}'>({sub})</span>&nbsp;&nbsp;&nbsp;"
                 f"<span style='font-family:{FONT_MONO}'>tasks 1-{TASK - 1} "
                 f"{q['past_score']:+.2f}&nbsp;&nbsp;task {TASK} "
                 f"{q['new_score']:+.2f}&nbsp;&nbsp;all {TASK} "
                 f"{(n_past * q['past_score'] + q['new_score']) / (n_past + 1):+.2f}"
                 f"</span>",
            font=dict(family=FONT_UI, size=10, color=AC["text_primary"]))

    if not PAPER_MODE:
        newf = np.array(d["Z_new_score"])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=0.0, xanchor="left",
            yanchor="top", yshift=-52 - len(rows) * 18 - 10, showarrow=False,
            align="left",
            text=f"Read left to right. Panel 1 carries no constraint line "
                 f"because at that point there is none: the floor is the "
                 f"specialist's own value,<br>"
                 f"which exists only once B does. Panel 2 is where it comes "
                 f"from. Panel 3 is read on all {TASK} tasks seen, which is "
                 f"what A and C<br>"
                 f"are comparable on: A has not learned task {TASK} yet, so "
                 f"the earlier four alone flatter it. On all {TASK}, A is "
                 f"{(n_past * gl[0]['past_score'] + gl[0]['new_score']) / (n_past + 1):.2f} "
                 f"and C is "
                 f"{(n_past * gl[-1]['past_score'] + gl[-1]['new_score']) / (n_past + 1):.2f}.<br>"
                 f"Task {TASK} is near-binary over this plane, "
                 f"{(newf >= 0.9).mean():.0%} above 0.9 and "
                 f"{(newf <= 0.1).mean():.0%} below 0.1, so the green line "
                 f"follows a cliff and &#949; barely moves it.<br>"
                 f"Plane: the two leading principal components of the "
                 f"{len(gl)} consolidation and {len(lo)} specialist iterates, "
                 f"carrying {d['explained_variance']:.0%} of their variance. "
                 f"{d['n_episodes']} greedy<br>"
                 f"episodes at every point of a "
                 f"{d['grid_n']}&#215;{d['grid_n']} grid, unsmoothed; only the "
                 f"green curve is blurred.",
            font=dict(family=FONT_UI, size=9.5, color=AC["text_muted"]))

    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=54, r=86, t=54, b=52 + KEY_2D))
    return fig, WIDTH_2D, height


WIDTH_2D = 900
KEY_2D = 76 if PAPER_MODE else 162


# ── The whole boundary in one figure: flat on top, surfaces below ───────────
PANEL_COLS = 3
# Plotly renders a 3D scene with WebGL and can only embed it in an SVG as a
# bitmap; `scale` is the only lever on its sharpness. It oversamples 2x on
# top of this, so 2 gives roughly 4x the panel size, which is ample for
# print without the file turning into megabytes of base64.
SVG_SCALE = 2
# Each 3D panel keeps the box its standalone figure gives it, 390 by 340, so it
# is rendered at the same size and the same camera rather than squeezed into a
# smaller cell. The flat row is the one that gives way.
SOLID_W, SOLID_H = 390, 340
FS = dict(title=14, letter=16, legend=12.5, key=11.5, axis=13, tick=10.5,
          scene_title=12, scene_tick=10)
# The flat panels sit narrower than their column, centred over the surface
# below. With the 3D boxes fixed at their standalone size, this is what keeps
# the whole figure wider than it is tall.
FLAT_FRAC = 0.70


def fig_combined(d: dict) -> tuple[go.Figure, int, int]:
    """Six panels: the three acts flat (a-c) and as surfaces (d-f).

    Columns are the three acts and rows the two renderings, so each column is
    one question answered twice. One crop serves all six, which is what lets
    PC 1 and PC 2 be labelled once instead of six times.
    """
    xs, ys, ix, iy, (x0, x1, y0, y1) = window(d, ("local", "global"))
    n_past = d["n_past"]
    past = crop(d["Z_past_score"], ix, iy)
    new = crop(d["Z_new_score"], ix, iy)
    seen = (n_past * past + new) / (n_past + 1)
    short = smooth(crop(d["Z_shortfall"], ix, iy))
    budget = d["eps"] ** 0.5
    ok = short <= budget
    gl, lo = d["paths"]["global"], d["paths"]["local"]

    m_l, m_r, m_t, hgap_px, vgap_px = 56, 104, 150, 34, 40
    col_w = SOLID_W
    width = m_l + m_r + 3 * col_w + 2 * hgap_px
    # Equal aspect on the flat row, measured on the narrowed panel.
    flat_h = (col_w * FLAT_FRAC) / ((x1 - x0) / (y1 - y0))
    m_b = 56
    total = flat_h + SOLID_H
    hgap = hgap_px / (width - m_l - m_r)
    height = int(round(total + vgap_px + m_t + m_b))
    vgap = vgap_px / total
    solid_frac = SOLID_H / total * (1 - vgap)

    fig = make_subplots(
        rows=2, cols=3, horizontal_spacing=hgap, vertical_spacing=vgap,
        row_heights=[flat_h / total, SOLID_H / total],
        shared_xaxes=True, shared_yaxes=True,
        specs=[[{"type": "xy"}] * 3, [{"type": "surface"}] * 3],
        subplot_titles=(f"specialist phase &#183; tasks 1-{TASK - 1}",
                        f"specialist phase &#183; task {TASK}",
                        f"consolidation &#183; all {TASK} tasks seen",
                        "", "", ""))

    # ── Row 1, flat ────────────────────────────────────────────────────────
    for col, field, show_limit, branch in [(1, past, False, "local"),
                                           (2, new, True, None),
                                           (3, seen, True, "global")]:
        suffix = "" if col == 1 else str(col)
        xref, yref = f"x{suffix}", f"y{suffix}"
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=field, hoverinfo="skip", showlegend=False,
            contours=dict(start=0.0, end=1.0, size=0.0625, coloring="fill"),
            colorscale=DIVERGE, showscale=(col == 3),
            line=dict(width=0.4, color="rgba(73,80,87,0.18)"),
            colorbar=dict(
                title=dict(text="normalized<br>score", side="top",
                           font=dict(family=FONT_UI, size=FS["axis"])),
                tickvals=[0, 0.5, 1.0],
                tickfont=dict(family=FONT_MONO, size=FS["tick"]),
                thickness=13, len=0.42, outlinewidth=0, x=1.015, y=0.76)),
            row=1, col=col)
        if show_limit:
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=short, hoverinfo="skip", showlegend=False,
                contours=dict(start=budget, end=budget, size=1,
                              coloring="none"),
                line=dict(color=C_LIMIT, width=2.4), showscale=False),
                row=1, col=col)
        if branch is not None:
            pts = d["paths"][branch]
            colour = C_LOCAL if branch == "local" else C_GLOBAL
            fig.add_trace(go.Scatter(
                x=[q["x"] for q in pts], y=[q["y"] for q in pts], mode="lines",
                hoverinfo="skip", showlegend=False,
                line=dict(color=colour, width=3.0)), row=1, col=col)
            arrow_2d(fig, pts, colour, xref, yref)
        marks = {1: [("A", gl[0], C_ORIGIN, "circle"),
                     ("B", lo[-1], C_LOCAL, "square")],
                 2: [("A", gl[0], C_ORIGIN, "circle"),
                     ("B", lo[-1], C_LOCAL, "square")],
                 3: [("A", gl[0], C_ORIGIN, "circle"),
                     ("C", gl[-1], C_GLOBAL, "diamond")]}[col]
        for letter, q, colour, symbol in marks:
            fig.add_trace(go.Scatter(
                x=[q["x"]], y=[q["y"]], mode="markers", showlegend=False,
                hoverinfo="skip",
                marker=dict(color=colour, size=14, symbol=symbol,
                            line=dict(color=AC["bg"], width=1.8))),
                row=1, col=col)
            fig.add_annotation(
                x=q["x"], y=q["y"], xref=xref, yref=yref, xanchor="center",
                yanchor="middle", yshift=-17 if letter == "C" else 17,
                showarrow=False, text=f"<b>{letter}</b>",
                font=dict(family=FONT_UI, size=FS["title"], color=colour),
                bgcolor="rgba(255,255,255,0.88)", borderpad=2)
        fig.update_xaxes(
            range=[x0, x1], showgrid=False, zeroline=False,
            title=dict(text="PC 1" if col == 2 else None,
                       font=dict(family=FONT_UI, size=FS["axis"])),
            tickfont=dict(family=FONT_MONO, size=FS["tick"]), row=1, col=col)
        fig.update_yaxes(
            range=[y0, y1], showgrid=False, zeroline=False,
            title=dict(text="PC 2" if col == 1 else None,
                       font=dict(family=FONT_UI, size=FS["axis"])),
            scaleanchor=f"x{suffix}", scaleratio=1,
            tickfont=dict(family=FONT_MONO, size=FS["tick"]), row=1, col=col)
        cw_f = (1 - 2 * hgap) / PANEL_COLS
        left = (col - 1) * (cw_f + hgap) + cw_f * (1 - FLAT_FRAC) / 2
        fig.update_xaxes(domain=[left, left + cw_f * FLAT_FRAC],
                         row=1, col=col)

    # ── Row 2, surfaces: the standalone settings, unchanged ────────────────
    # Closer in than the standalone figures, because the surface row here has
    # no tick labels or axis titles around it to fill the cell.
    tighten = 0.90
    cam_spec = dict(eye=dict(x=1.06 * tighten, y=-1.01 * tighten,
                             z=0.74 * tighten), up=dict(x=0, y=0, z=1))
    cam_cons = dict(eye=dict(x=-0.87 * tighten, y=-1.25 * tighten,
                             z=0.70 * tighten), up=dict(x=0, y=0, z=1))
    cw = (1 - 2 * hgap) / PANEL_COLS
    # The surface row carries no text of its own. Its panels are read off the
    # flat row above them, which shares the crop, the colourbar and the key,
    # so axis titles, tick labels and A/B/C letters here only repeat what is
    # already said once and crowd three small scenes while doing it.
    global BARE
    was_bare, BARE = BARE, True
    for idx, which, branch, zlab, cam, masked, ztop in [
            (4, "Z_past_score", "local", f"tasks 1-{TASK - 1}", cam_spec,
             False, 1.18),
            (5, "Z_new_score", "local", f"task {TASK}", cam_spec, False, 1.18),
            (6, "seen", "global", f"all {TASK} tasks", cam_cons, True, None)]:
        scene = f"scene{idx - 3 if idx > 4 else ''}"
        # Its own window, so the panel is framed exactly as the standalone is.
        sxs, sys, six, siy, _ = window(d, (branch,))
        if which == "seen":
            field = ((n_past * crop(d["Z_past_score"], six, siy)
                      + crop(d["Z_new_score"], six, siy)) / (n_past + 1))
        else:
            field = crop(d[which], six, siy)
        sshort = smooth(crop(d["Z_shortfall"], six, siy))
        mask = (sshort <= budget) if masked else None
        xs_s, ys_s = sxs, sys
        pts = d["paths"][branch]
        colour = C_LOCAL if branch == "local" else C_GLOBAL
        lo_z = float(np.nanmin(field))
        full = field                      # the curve and branch need it unmasked
        if mask is not None:
            fig.add_trace(go.Surface(
                x=xs_s, y=ys_s, z=np.where(mask, np.nan, field), cmin=0.0,
                cmax=1, colorscale=GREYS, showscale=False, opacity=0.85,
                hoverinfo="skip", scene=scene,
                contours=dict(z=dict(show=True, usecolormap=False,
                                     color="rgba(73,80,87,0.20)", width=1))))
            field = np.where(mask, field, np.nan)
        fig.add_trace(go.Surface(
            x=xs_s, y=ys_s, z=field, cmin=0.0, cmax=1, colorscale=DIVERGE,
            showscale=False, opacity=0.98, hoverinfo="skip", scene=scene,
            contours=dict(z=dict(show=True, usecolormap=False,
                                 color="rgba(73,80,87,0.20)", width=1))))
        if mask is not None:
            for seg in contour_xy(xs_s, ys_s, sshort, budget):
                z = lookup(xs_s, ys_s, full, seg) + LIFT
                fig.add_trace(go.Scatter3d(
                    x=seg[:, 0], y=seg[:, 1], z=z, mode="lines",
                    hoverinfo="skip", showlegend=False,
                    line=dict(color=C_LIMIT, width=5), scene=scene))
        px, py, pz = add_branch(fig, scene, xs_s, ys_s, full, pts, colour)
        add_marker(fig, scene, px[0], py[0], pz[0], "A", C_ORIGIN, "circle")
        end = "C" if branch == "global" else "B"
        add_marker(fig, scene, px[-1], py[-1], pz[-1], end, colour,
                   "diamond" if end == "C" else "square")
        axes = scene_axes(zlab)
        hi_z = float(np.nanmax(full))
        axes["zaxis"]["range"] = [max(0.0, lo_z - 0.05),
                                  ztop if ztop else hi_z + 0.12]
        for key in ("xaxis", "yaxis", "zaxis"):
            axes[key]["tickfont"] = dict(family=FONT_MONO,
                                         size=FS["scene_tick"],
                                         color=AC["text_muted"])
            axes[key]["title"]["font"] = dict(family=FONT_UI,
                                              size=FS["scene_title"],
                                              color=AC["text_primary"])
        if not BARE:
            axes["zaxis"]["title"]["text"] = zlab
        col = idx - 3
        x_start = (col - 1) * (cw + hgap)
        fig.update_layout(**{scene: dict(
            **axes, camera=cam, aspectratio=dict(x=1.0, y=1.0, z=0.62),
            domain=dict(x=[x_start + (0.048 if idx == 6 else 0.014),
                           x_start + cw], y=[0.0, solid_frac]))})

    BARE = was_bare

    # ── One legend for everything that is not a value ──────────────────────
    for name, colour, kind in [("specialist phase", C_LOCAL, "line"),
                               ("consolidation phase", C_GLOBAL, "line"),
                               ("constraint edge", C_LIMIT, "line"),
                               ("admissible", "#7C3AED", "square"),
                               ("ruled out", "#9AA2AA", "square")]:
        common = dict(x=[None], y=[None], name=name, showlegend=True,
                      hoverinfo="skip")
        if kind == "line":
            fig.add_trace(go.Scatter(mode="lines",
                                     line=dict(color=colour, width=4),
                                     **common))
        else:
            fig.add_trace(go.Scatter(
                mode="markers",
                marker=dict(size=14, symbol="square", color=colour), **common))

    for note in fig.layout.annotations[:3]:
        note.font.update(family=FONT_UI, size=FS["title"],
                         color=AC["text_primary"])

    for i, (row, col) in enumerate([(1, 1), (1, 2), (1, 3),
                                    (2, 1), (2, 2), (2, 3)]):
        fig.add_annotation(
            xref="paper", yref="paper", xanchor="left", yanchor="bottom",
            x=(col - 1) * (cw + hgap),
            y=1.004 if row == 1 else solid_frac - 0.012, showarrow=False,
            text=f"<b>({chr(97 + i)})</b>",
            font=dict(family=FONT_UI, size=FS["letter"],
                      color=AC["text_primary"]))

    glyph = {"A": "&#9679;", "B": "&#9632;", "C": "&#9670;"}
    for i, (letter, q, colour, name) in enumerate(
            [("A", gl[0], C_ORIGIN, "global in, carried from task 4"),
             ("B", lo[-1], C_LOCAL, "specialist out, sets the floor"),
             ("C", gl[-1], C_GLOBAL, "global out, carried to task 6")]):
        allf = (n_past * q["past_score"] + q["new_score"]) / (n_past + 1)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
            yanchor="bottom", yshift=34 + (2 - i) * 21, showarrow=False,
            align="left",
            text=f"<span style='color:{colour}'>{glyph[letter]}</span>&nbsp; "
                 f"<b>{letter}</b>&nbsp;&nbsp;{name}&nbsp;&nbsp;&nbsp;"
                 f"<span style='font-family:{FONT_MONO}'>tasks 1-{TASK - 1} "
                 f"{q['past_score']:+.2f}&nbsp;&nbsp;task {TASK} "
                 f"{q['new_score']:+.2f}&nbsp;&nbsp;all {TASK} "
                 f"{allf:+.2f}</span>",
            font=dict(family=FONT_UI, size=FS["key"],
                      color=AC["text_primary"]))

    fig.update_layout(
        title=None, showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.124, xanchor="left",
                    yanchor="bottom", font=dict(family=FONT_UI,
                                                size=FS["legend"]),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    itemsizing="constant"),
        margin=dict(l=m_l, r=m_r, t=m_t, b=m_b))
    return fig, int(width), height


def clean_svg(path: Path) -> str:
    """Strip elements that paint nothing but still arrive as slide objects.

    Plotly writes a <g> for every trace type it knows, used or not, a clip
    rectangle for each of those empty layers, and a rectangle of solid paper
    behind everything. On a slide each one is a selectable object: the paper
    rectangle in particular sits over whatever is behind the figure and is
    invisible against a white background, so it is found by accident rather
    than on purpose. None of them carry any of the figure.
    """
    t = path.read_text()
    g0, r0 = t.count("<g "), t.count("<rect")
    # The paper rectangle, whichever way Plotly wrote it. A transparent
    # background is not "no background": it is a rect with fill-opacity 0,
    # which is the worst kind, invisible against anything and still there.
    t = re.sub(r'<rect[^>]*fill-opacity:\s*0\s*[;"][^>]*/>', "", t)
    # Only the full-canvas one. The white pads behind the A, B and C labels
    # are also white rects, at 88% opacity, and they must stay: without them
    # the letters sit straight on the map and stop being readable.
    t = re.sub(r'<rect x="0" y="0" width="[\d.]+" height="[\d.]+"\s*'
               r'style="fill:\s*rgb\(255,\s*255,\s*255\);\s*'
               r'fill-opacity:\s*1;?"\s*/>', "", t)
    prev = None
    while prev != t:            # a group of empty groups is itself empty
        prev = t
        t = re.sub(r"<g\b[^>]*/>", "", t)
        t = re.sub(r"<g\b[^>]*>\s*</g>", "", t)
    # clip paths nothing references any more
    for cid in re.findall(r'<clipPath[^>]*id="([^"]+)"', t):
        if t.count(cid) < 2:
            t = re.sub(r'<clipPath[^>]*id="%s".*?</clipPath>' % re.escape(cid),
                       "", t, flags=re.S)
    path.write_text(t)
    return (f"-{g0 - t.count('<g ')} groups, "
            f"-{r0 - t.count('<rect')} rects")


FIGURES = [("task_boundary_combined", fig_combined),
           ("specialist_phase_3d", fig_specialist),
           ("consolidation_phase_3d", fig_consolidation),
           ("task_boundary_2d", fig_flat)]
# Built only by --bare: a single surface, for a slide of its own.
BARE_ONLY = [("specialist_task5_3d", fig_task5)]


def main() -> None:
    install_template()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    d = load()

    gl, lo = d["paths"]["global"], d["paths"]["local"]
    print(f"\ntask {d['task']}, run {d['run']}, grid {d['grid_n']}"
          f"x{d['grid_n']} at {d['n_episodes']} episodes")
    print(f"  plane: {d['explained_variance']:.1%} of both paths' variance")
    print(f"  {'':18}{'tasks 1-4':>11}{'task 5':>9}")
    for label, q in (("A global in", gl[0]), ("B specialist out", lo[-1]),
                     ("C global out", gl[-1])):
        print(f"  {label:<18}{q['past_score']:>11.3f}{q['new_score']:>9.3f}")
    print()

    figures = ([(i, b) for i, b in FIGURES if i.endswith("_3d")] + BARE_ONLY
               if BARE else FIGURES)
    for fig_id, builder in figures:
        if BARE:
            fig_id += "_bare"
        fig, width, height = builder(d)
        # PNG and SVG come from the SAME figure object, never two code paths.
        fig.write_image(PNG_DIR / f"{fig_id}.png", width=width, height=height,
                        scale=3.125)
        fig.write_image(SVG_DIR / f"{fig_id}.svg", width=width,
                        height=height, scale=SVG_SCALE)
        png, svg = PNG_DIR / f"{fig_id}.png", SVG_DIR / f"{fig_id}.svg"
        stripped = clean_svg(svg)
        for p in (png, svg):
            if not p.exists() or p.stat().st_size == 0:
                raise RuntimeError(f"{p} missing or empty")
        print(f"  {fig_id}: {width}x{height}, png "
              f"{png.stat().st_size // 1024}K, svg "
              f"{svg.stat().st_size // 1024}K {stripped}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bare", action="store_true",
                    help="the two 3D figures with no text at all, for slides; "
                         "writes *_3d_bare.{png,svg}")
    BARE = ap.parse_args().bare
    main()
