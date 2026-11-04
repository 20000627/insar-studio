"""Matplotlib rendering for maps, network diagrams and quality panels.

All figures use the same paper/ink/teal/amber visual language as the UI.
Rendering happens on the server; the browser displays PNGs and overlays its
own interactive canvas controls.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Arc, ConnectionPatch, FancyArrowPatch
from scipy import ndimage

PAPER = "#F3F0E8"
INK = "#17231F"
MUTED = "#6A756F"
TEAL = "#0E6B5C"
TEAL_LIGHT = "#7FB3A6"
AMBER = "#C47A2C"
RED = "#A6432F"
GRID = "#D8D2C4"

_SEQ = ["#F3F0E8", "#DCE8E1", "#B6D3C7", "#7FB3A6", "#3E8C7B", "#0E6B5C", "#08483E", "#032A24"]
_DIV = ["#8C2F1E", "#C47A2C", "#E6C79C", "#F3F0E8", "#A9CFC4", "#3E8C7B", "#0E6B5C", "#053B33"]


def _register_fonts() -> None:
    preferred = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred if name in available]
    if chosen:
        plt.rcParams["font.family"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_register_fonts()
plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": False,
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)


def seq_cmap() -> ListedColormap:
    return ListedColormap(_SEQ)


def div_cmap() -> ListedColormap:
    return ListedColormap(_DIV)


def _stretch(data: np.ndarray, mask: np.ndarray | None = None, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float64)
    if mask is not None:
        arr = arr[mask & np.isfinite(arr)]
    else:
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _clean_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.7)


def save_amplitude_preview(
    amp: np.ndarray,
    mask: np.ndarray | None,
    path: Path,
    *,
    title: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _stretch(amp, mask, 1, 99)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(amp, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors=[TEAL], linewidths=0.5, alpha=0.5)
    _clean_axes(ax)
    if title:
        ax.set_title(title, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_raster(
    data: np.ndarray,
    path: Path,
    *,
    mask: np.ndarray | None = None,
    title: str = "",
    subtitle: str = "",
    unit: str = "",
    cmap: str = "seq",
    divergent: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    overlay: np.ndarray | None = None,
    overlay_cmap: str = "gray",
    overlay_alpha: float = 0.35,
    colorbar: bool = True,
    figsize: tuple[float, float] = (5.4, 4.2),
    bounds: tuple[float, float, float, float] | None = None,
    contour_levels: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32).copy()
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    if vmin is None or vmax is None:
        vmin2, vmax2 = _stretch(arr, None, 2, 98)
        if symmetric:
            lim = max(abs(vmin2), abs(vmax2))
            vmin, vmax = -lim, lim
        elif divergent:
            vmin, vmax = vmin2, vmax2
        else:
            vmin, vmax = vmin2, vmax2
    cmap_obj = div_cmap() if divergent else seq_cmap()
    fig, ax = plt.subplots(figsize=figsize)
    if overlay is not None:
        ov = np.asarray(overlay, dtype=np.float32)
        ov = np.where(np.isfinite(ov), ov, np.nanmedian(ov))
        ax.imshow(ov, cmap=overlay_cmap, alpha=overlay_alpha, interpolation="bilinear")
    if divergent:
        norm = TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax)) if vmin < 0 < vmax else None
        im = ax.imshow(arr, cmap=cmap_obj, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
    if contour_levels and np.isfinite(arr).any():
        try:
            ax.contour(arr, levels=contour_levels, colors=[INK], linewidths=0.35, alpha=0.45)
        except Exception:
            pass
    _clean_axes(ax)
    if bounds:
        west, south, east, north = bounds
        # Show geographic extent while keeping raster pixel coordinates off
        ax.set_xlabel(f"{west:.3f}°E – {east:.3f}°E", color=MUTED, fontsize=7.5)
        ax.set_ylabel(f"{south:.3f}°N – {north:.3f}°N", color=MUTED, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left", pad=18)
    if subtitle:
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, color=MUTED, fontsize=7.5, va="bottom")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
        cbar.outline.set_edgecolor(GRID)
        cbar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
        if unit:
            cbar.set_label(unit, color=MUTED, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_wrapped_phase(
    phase: np.ndarray,
    coherence: np.ndarray,
    path: Path,
    *,
    mask: np.ndarray | None = None,
    title: str = "Wrapped phase",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    ax = axes[0]
    cmap = plt.get_cmap("twilight")
    ax.imshow(np.where(mask, phase, np.nan) if mask is not None else phase, cmap=cmap, vmin=-math.pi, vmax=math.pi, interpolation="nearest")
    _clean_axes(ax)
    ax.set_title(title, loc="left")
    ax = axes[1]
    ax.imshow(np.where(mask, coherence, np.nan) if mask is not None else coherence, cmap=seq_cmap(), vmin=0, vmax=1, interpolation="nearest")
    _clean_axes(ax)
    ax.set_title("Coherence", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_network(
    pairs: Iterable[dict[str, Any]],
    dates: list[str],
    bperp: list[float],
    path: Path,
    *,
    ref_index: int = 0,
) -> Path:
    """Baseline–time network diagram with every pair drawn as an arc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import date

    d0 = date.fromisoformat(dates[0])
    x = np.asarray([(date.fromisoformat(d) - d0).days for d in dates], dtype=float)
    y = np.asarray(bperp, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for p in pairs:
        i, j = int(p["ref"]), int(p["sec"])
        color = TEAL if (j - i) <= 2 else (TEAL_LIGHT if (j - i) <= 5 else AMBER)
        ax.plot([x[i], x[j]], [y[i], y[j]], color=color, lw=0.8, alpha=0.55, zorder=1)
    ax.scatter(x, y, s=34, facecolor=PAPER, edgecolor=INK, linewidth=1.0, zorder=3)
    ax.scatter([x[ref_index]], [y[ref_index]], s=64, facecolor=AMBER, edgecolor=AMBER, zorder=4)
    ax.axhline(0, color=GRID, lw=0.7, zorder=0)
    ax.set_xlabel("Days since first acquisition")
    ax.set_ylabel("Perpendicular baseline (m)")
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"Small-baseline network · {len(list(pairs))} pairs", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_series(
    dates: list[str],
    series: dict[str, np.ndarray],
    path: Path,
    *,
    title: str = "LOS displacement time series",
    unit: str = "mm",
    uncertainty: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import date

    d0 = date.fromisoformat(dates[0])
    x = np.asarray([(date.fromisoformat(d) - d0).days for d in dates], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [TEAL, AMBER, "#5B6FA8", RED]
    for k, (label, y) in enumerate(series.items()):
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, color=colors[k % len(colors)], lw=1.4, marker="o", ms=2.5, label=label)
        if uncertainty is not None and k == 0:
            ax.fill_between(x, y - uncertainty, y + uncertainty, color=colors[0], alpha=0.13, lw=0)
    ax.axhline(0, color=GRID, lw=0.7)
    ax.set_xlabel("Days since first acquisition")
    ax.set_ylabel(unit)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_histogram(
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 48,
    mask: np.ndarray | None = None,
    color: str = TEAL,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, dtype=float)
    if mask is not None:
        v = v[mask]
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.hist(v, bins=bins, color=color, alpha=0.85, edgecolor=PAPER, linewidth=0.4)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_pipeline_diagram(path: Path, steps: list[tuple[str, str]]) -> Path:
    """A compact horizontal flow diagram for reports / the results page."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 1.4))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, sub) in enumerate(steps):
        x = i + 0.5
        ax.text(x, 0.62, title, ha="center", va="center", fontsize=8, color=INK, weight="bold")
        ax.text(x, 0.28, sub, ha="center", va="center", fontsize=6.5, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.94, 0.62), xytext=(i + 0.62, 0.62), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
        ax.add_patch(plt.Rectangle((i + 0.08, 0.12), 0.78, 0.72, facecolor="#FFFFFF" if i % 2 == 0 else "#F8F5EC", edgecolor=GRID, lw=0.7, zorder=-1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
