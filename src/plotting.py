"""Shared matplotlib style so every figure in the repo reads as one system.

Colour is assigned by role: a fixed, colour-blind-safe categorical order for
data series, neutral ink for reference lines, and a reserved "good" green for
the chemical-accuracy band. Grid and axes are deliberately recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are only ever written to disk.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical series colours, always used in this order (blue, orange).
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

#: Reserved status colour for "inside the accuracy target".
GOOD = "#0ca30c"

DPI = 200


def apply_style() -> None:
    """Install the repo-wide matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "mathtext.default": "regular",
        }
    )


def set_title(ax: Axes, title: str, subtitle: str | None = None) -> None:
    """Add a left-aligned title with an optional secondary line underneath."""
    ax.set_title(
        title,
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=30 if subtitle else 12,
    )
    if subtitle:
        ax.text(
            0.0,
            1.025,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=INK_SECONDARY,
            va="bottom",
            ha="left",
        )


def save_figure(fig: Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` (creating parent directories) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
