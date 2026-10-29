#!/usr/bin/env python3
"""Academic figure helpers: the canonical implementation of Track F.

This module is the executable form of ``algorithmic-art-academic/SKILL.md``
Sections F.3 through F.8. Import it from any figure script:

    from acviz import (AC, AC_SERIES, EXPORT_SCALE, W_FULL, add_band,
                       add_callout, add_end_labels, append_manifest_entry,
                       export_figure, install_template, write_gif)

Nothing here draws a figure. It supplies the template, the annotation
conventions, the export contract, and the manifest writer.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import plotly.io as pio

# ── Palette ─────────────────────────────────────────────────────────────────
AC: dict[str, str] = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "violet": "#7C3AED", "teal": "#0891B2", "rose": "#BE185D", "sienna": "#92400E",
    "bg": "#FFFFFF", "surface": "#F8F9FA", "border": "#DEE2E6",
    "axis": "#495057", "grid": "#E9ECEF",
    "text_primary": "#212529", "text_muted": "#6C757D", "text_faint": "#ADB5BD",
}

AC_SERIES: list[str] = [
    AC["blue"], AC["amber"], AC["green"], AC["red"],
    AC["violet"], AC["teal"], AC["rose"], AC["sienna"],
]

FONT_UI = "Inter, Helvetica, Arial, sans-serif"
FONT_TITLE = "Source Serif 4, Georgia, serif"
FONT_MONO = "JetBrains Mono, Menlo, Consolas, monospace"

# Sequential and diverging scales built from the palette. Never rainbow, never jet.
SEQ_BLUE = [[0.0, AC["surface"]], [1.0, AC["blue"]]]
DIVERGING = [[0.0, AC["blue"]], [0.5, AC["surface"]], [1.0, AC["red"]]]

# ── Print geometry ──────────────────────────────────────────────────────────
CSS_DPI, PRINT_DPI = 96, 300
EXPORT_SCALE = PRINT_DPI / CSS_DPI  # 3.125 -> 300 dpi at the intended print width


def width_px(print_inches: float) -> int:
    """Plotly figure width in CSS px for a given intended print width in inches."""
    return int(round(print_inches * CSS_DPI))


W_SINGLE_COL = width_px(3.25)  # 312 px  -> 975 px PNG
W_ONE_HALF = width_px(5.00)    # 480 px  -> 1500 px PNG
W_FULL = width_px(6.75)        # 648 px  -> 2025 px PNG


def height_for(width: int, ratio: float = 1.55) -> int:
    """Height in px for a given width, defaulting to the standard line-plot ratio."""
    return int(round(width / ratio))


# ── Template ────────────────────────────────────────────────────────────────
def install_template(name: str = "academic", set_default: bool = True) -> go.layout.Template:
    """Register the academic Plotly template: white ground, Tufte spine, faint grid.

    Args:
        name: template name to register under.
        set_default: also make it the default template for new figures.

    Returns:
        The registered template object.
    """
    axis_common = dict(
        showline=True, linecolor=AC["axis"], linewidth=1.2, mirror=False,
        ticks="outside", tickcolor=AC["axis"], ticklen=4, tickwidth=1.0,
        tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
        title=dict(font=dict(family=FONT_UI, size=13, color=AC["text_primary"])),
        zeroline=False,
    )
    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=AC["bg"],
            plot_bgcolor=AC["bg"],
            colorway=AC_SERIES,
            font=dict(family=FONT_UI, size=12, color=AC["text_primary"]),
            title=dict(
                font=dict(family=FONT_TITLE, size=16, color=AC["text_primary"]),
                x=0.0, xanchor="left", y=0.97, yanchor="top",
            ),
            # The wide right margin holds the direct end-of-line series labels.
            margin=dict(l=64, r=110, t=52, b=56),
            showlegend=False,
            # Horizontal grid only; no vertical grid outside heatmaps.
            xaxis=dict(showgrid=False, **axis_common),
            yaxis=dict(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6, **axis_common),
            hoverlabel=dict(font=dict(family=FONT_UI, size=12)),
        )
    )
    pio.templates[name] = template
    if set_default:
        pio.templates.default = name
    return template


# ── Color helpers ───────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an ``rgba(...)`` string at the given alpha."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def series_color(index: int) -> str:
    """Palette color for series ``index``, cycling after eight series."""
    return AC_SERIES[index % len(AC_SERIES)]


# ── Annotation conventions ──────────────────────────────────────────────────
def add_band(
    fig: go.Figure,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    color: str,
    alpha: float = 0.20,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add an uncertainty band (mean ± std, or a CI) as a filled polygon.

    Uncertainty is never omitted when repeated runs exist.
    """
    trace = go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor=hex_to_rgba(color, alpha),
        line=dict(width=0),
        # mode is NOT optional. Plotly defaults a Scatter to "lines+markers"
        # below 20 points, so a band over a short series sprouts a marker at
        # every polygon vertex -- and because no marker colour is set, those
        # markers take a colour from the template colorway by trace index, not
        # from this band. A 5-point band then shows blue dots on a brown series.
        mode="lines",
        hoverinfo="skip",
        showlegend=False,
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def spread_labels(values: Sequence[float], min_gap: float) -> list[float]:
    """Nudge label positions apart so none collide, preserving their order.

    Args:
        values: label anchor positions in data units, in any order.
        min_gap: minimum separation to enforce, in the same data units.

    Returns:
        Adjusted positions aligned with the input order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    # Single upward pass: each label is pushed just clear of the one below it.
    for rank in range(1, len(order)):
        below, current = order[rank - 1], order[rank]
        if adjusted[current] - adjusted[below] < min_gap:
            adjusted[current] = adjusted[below] + min_gap
    return adjusted


def add_end_labels(
    fig: go.Figure,
    labels: Iterable[tuple[str, float, float, str]],
    xshift: int = 8,
    size: int = 12,
    min_gap: float | None = None,
) -> None:
    """Direct-label series at the right end of each line instead of a legend box.

    Args:
        fig: the figure to annotate.
        labels: iterable of ``(name, x_last, y_last, color)``.
        xshift: horizontal offset in px from the line end.
        size: label font size in px.
        min_gap: when given, the minimum vertical separation between labels in
            y-axis data units. Labels closer than this are nudged apart.
    """
    items = list(labels)
    positions = [y for _, _, y, _ in items]
    if min_gap is not None:
        positions = spread_labels(positions, min_gap)

    for (name, x_last, _, color), y_label in zip(items, positions):
        fig.add_annotation(
            x=x_last, y=y_label, text=f"<b>{name}</b>", showarrow=False,
            xanchor="left", xshift=xshift, yanchor="middle",
            font=dict(family=FONT_UI, size=size, color=color),
        )


def add_callout(
    fig: go.Figure,
    x: float,
    y: float,
    text: str,
    ax: int = 28,
    ay: int = -30,
    color: str | None = None,
) -> None:
    """Annotate the single most important result with one short callout."""
    fig.add_annotation(
        x=x, y=y, text=text,
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=ax, ay=ay, xanchor="left",
        font=dict(family=FONT_UI, size=11, color=color or AC["text_primary"]),
        bgcolor="rgba(255,255,255,0.82)", borderpad=3,
    )


# ── Export contract ─────────────────────────────────────────────────────────
def export_figure(
    fig: go.Figure, stem: Path, width: int, height: int, label: str = "Figure"
) -> dict[str, Any]:
    """Export a figure to PNG (300 dpi equivalent) and SVG, and verify both.

    Args:
        fig: the figure to export.
        stem: output path without an extension, e.g. ``report/figures/loss``.
        width: figure width in CSS px.
        height: figure height in CSS px.
        label: variant label recorded in the manifest entry.

    Returns:
        A manifest variant dict with ``label``, ``png``, ``svg`` and ``html``,
        with paths made relative to ``report/``.

    Raises:
        RuntimeError: either export is missing or zero bytes.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")

    fig.write_image(png, width=width, height=height, scale=EXPORT_SCALE)
    fig.write_image(svg, width=width, height=height)  # vector: scale is moot

    for path in (png, svg):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"export failed or empty: {path}")

    return {
        "label": label,
        "png": _relative_to_report(png),
        "svg": _relative_to_report(svg),
        "html": None,
    }


EMBED_STYLE = (
    "<style>html,body{margin:0;padding:0;height:100%;background:#FFFFFF;"
    "font-family:Inter,Helvetica,Arial,sans-serif}</style>"
)


def export_interactive(fig: go.Figure, stem: Path, auto_play: bool = False) -> str:
    """Write an interactive Plotly HTML export and return its report-relative path.

    The page is styled for iframe embedding: no body margin (Plotly sizes its
    div to 100% of the body, and the browser default 8px margin would otherwise
    push the x-axis title out of view) and a white ground matching the figure.

    Args:
        fig: the figure to export.
        stem: output path without an extension.
        auto_play: whether an animated figure starts playing on load. Off by
            default so the reader presses Play deliberately.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".html")

    document = fig.to_html(include_plotlyjs="cdn", full_html=True, auto_play=auto_play)
    document = document.replace("</head>", f"{EMBED_STYLE}</head>", 1)
    path.write_text(document, encoding="utf-8")

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"interactive export failed or empty: {path}")
    return _relative_to_report(path)


def write_gif(
    fig: go.Figure,
    frames: Sequence[go.Frame],
    stem: Path,
    width: int,
    height: int,
    fps: int = 8,
    scale: float = 1.5,
) -> str:
    """Render each Plotly frame with Kaleido and stitch them into a GIF.

    Args:
        fig: the base figure supplying the layout.
        frames: the animation frames, in order.
        stem: output path without an extension.
        width: frame width in CSS px.
        height: frame height in CSS px.
        fps: playback rate.
        scale: raster scale for each frame; 1.5 keeps GIFs legible but small.

    Returns:
        The report-relative path of the written GIF.
    """
    import imageio.v3 as iio

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".gif")

    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, frame in enumerate(frames):
            snapshot = go.Figure(data=frame.data, layout=fig.layout)
            if getattr(frame, "layout", None) is not None:
                snapshot.update_layout(frame.layout)
            frame_path = Path(tmp) / f"frame_{index:04d}.png"
            snapshot.write_image(frame_path, width=width, height=height, scale=scale)
            images.append(iio.imread(frame_path))
        iio.imwrite(path, images, duration=int(1000 / fps), loop=0)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"gif export failed or empty: {path}")
    return _relative_to_report(path)


def _relative_to_report(path: Path) -> str:
    """Express a path relative to the ``report/`` directory, as the manifest requires."""
    parts = Path(path).as_posix().split("/")
    if "report" in parts:
        return "/".join(parts[parts.index("report") + 1:])
    return Path(path).as_posix()


# ── Manifest ────────────────────────────────────────────────────────────────
def append_manifest_entry(
    manifest_path: Path,
    *,
    id: str,
    group: str,
    group_title: str,
    title: str,
    caption: str,
    details: str,
    variants: list[dict[str, Any]],
    seed: int | None = None,
    dataset: str | None = None,
    model: str | None = None,
    animation: dict[str, str] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append one figure entry to the manifest, replacing any entry with the same id.

    The manifest is created if absent. Ordering is stable: a replaced entry keeps
    its original position so reruns do not reshuffle the dashboard.

    Raises:
        ValueError: the caption exceeds the 160-character limit.
    """
    if len(caption) > 160:
        raise ValueError(f"caption for {id!r} is {len(caption)} chars, limit is 160")

    entry: dict[str, Any] = {
        "id": id,
        "group": group,
        "group_title": group_title,
        "title": title,
        "caption": caption,
        "details": details,
        "seed": seed,
        "dataset": dataset,
        "model": model,
        "variants": variants,
        "animation": animation,
        "metrics": metrics or [],
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Any = {"figures": []}
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    figures = payload if isinstance(payload, list) else payload.setdefault("figures", [])
    for index, existing in enumerate(figures):
        if existing.get("id") == id:
            figures[index] = entry
            break
    else:
        figures.append(entry)

    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return entry
