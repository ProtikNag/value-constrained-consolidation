"""A band must never draw markers, at any series length.

Plotly resolves a Scatter's default mode to "lines+markers" below 20 points, so
an uncertainty band over a short series grew a marker at every polygon vertex,
coloured from the template colorway rather than from the band. On BigGrid v2
that put blue and violet dots on the brown CReLU and teal CKA-RL bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

from acviz import add_band  # noqa: E402


def test_band_never_draws_markers() -> None:
    """Short bands are the regression: 5 points makes a 10-point polygon."""
    for n in (2, 5, 8, 19, 40):
        fig = go.Figure()
        xs = list(range(n))
        add_band(fig, xs, [0.0] * n, [1.0] * n, "#2563EB", 0.2)
        mode = fig.data[-1].mode
        assert mode == "lines", f"n={n}: mode is {mode!r}, markers would render"


if __name__ == "__main__":
    test_band_never_draws_markers()
    print("ok")
