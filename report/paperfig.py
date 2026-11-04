"""Shared matplotlib style for the paper's appendix figures.

Figures are drawn at their printed size (width in inches = the width they take on
the page), so a 9 pt label here is a 9 pt label in the PDF. Helvetica regular and
bold are registered from ``report/fonts`` (extracted from the macOS system
collection, which matplotlib cannot read directly). PDFs embed TrueType
(fonttype 42) so text stays selectable and vector.

    from paperfig import AC, TEXTWIDTH, save, style
    style()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))
    save(fig, out_dir / "name", paper_pdf=REPO / "paper/figures/.../name.pdf")
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TEXTWIDTH = 6.5          # TMLR text block, inches

AC = dict(
    blue="#2563EB", amber="#D97706", green="#059669", red="#DC2626",
    violet="#7C3AED", teal="#0891B2", rose="#BE185D", sienna="#92400E",
    bg="#FFFFFF", surface="#F8F9FA", border="#DEE2E6", axis="#495057",
    grid="#E9ECEF", text="#212529", muted="#6C757D", faint="#ADB5BD",
)
SERIES = [AC[k] for k in ("blue", "amber", "green", "red", "violet", "teal", "rose", "sienna")]


SYSTEM_HELVETICA = Path("/System/Library/Fonts/Helvetica.ttc")
FACES = ("Helvetica", "Helvetica Bold", "Helvetica Oblique", "Helvetica Bold Oblique")


def extract_fonts() -> None:
    """Write the Helvetica faces out of the macOS collection, which matplotlib cannot read.

    The fonts are Apple-licensed, so they are regenerated locally and never committed.
    Off macOS this is a no-op and the figures fall back to Arial.
    """
    out = HERE / "fonts"
    if not SYSTEM_HELVETICA.exists() or len(list(out.glob("*.ttf"))) == len(FACES):
        return
    from fontTools.ttLib import TTCollection
    out.mkdir(exist_ok=True)
    for face in TTCollection(str(SYSTEM_HELVETICA)).fonts:
        name = face["name"].getDebugName(4)
        if name in FACES:
            face.save(str(out / f"{name.replace(' ', '-')}.ttf"))


def style(base: float = 8.5) -> None:
    """Register Helvetica and set the paper rcParams. ``base`` is the body size in pt."""
    extract_fonts()
    for ttf in (HERE / "fonts").glob("*.ttf"):
        fm.fontManager.addfont(str(ttf))
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.titleweight": "bold",
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 0.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AC["axis"],
        "axes.labelcolor": AC["text"],
        "axes.titlecolor": AC["text"],
        "xtick.color": AC["muted"],
        "ytick.color": AC["muted"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": AC["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def save(fig: plt.Figure, stem: Path, paper_pdf: Path | None = None) -> None:
    """Write ``stem``.pdf/.png/.svg and optionally copy the PDF into the paper."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    if paper_pdf is not None:
        Path(paper_pdf).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stem.with_suffix(".pdf"), paper_pdf)
    plt.close(fig)
