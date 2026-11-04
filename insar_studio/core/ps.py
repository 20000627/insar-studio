"""Persistent-scatterer candidate selection.

A true PS workflow needs full-resolution, single-master phase and a per-point
deformation model.  This module implements the two selection diagnostics used
by that workflow -- amplitude dispersion and interferometric coherence -- and
returns a high-quality point mask plus per-point weights.  The time-series
solver is shared with SBAS, but it runs only on the selected points and uses
the point quality as a pixel weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .io import Stack
from .utils import downsample_real


def amplitude_dispersion(stack: Stack, multilook: int = 1, max_images: int = 12) -> np.ndarray:
    n = stack.n_dates
    step = max(1, n // max(1, max_images))
    amps = np.asarray([np.abs(stack.slc[i]) for i in range(0, n, step)], dtype=np.float32)
    mean = np.maximum(amps.mean(axis=0), 1e-6)
    adi = amps.std(axis=0) / mean
    return downsample_real(adi, multilook).astype(np.float32)


def select_ps(
    stack: Stack,
    coherence: np.ndarray,
    valid: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Select PS candidates and return masks / diagnostics.

    ``coherence`` is the pair stack (n_pairs, rows, cols) on the multilooked
    grid.  ``valid`` is the corresponding boolean mask.
    """
    m = max(1, int(params.get("multilook", 1)))
    adi = amplitude_dispersion(stack, m)
    rows, cols = valid.shape[1:]
    adi = adi[:rows, :cols]
    if adi.shape != (rows, cols):
        adi = np.pad(adi, ((0, rows - adi.shape[0]), (0, cols - adi.shape[1])), mode="edge")
    mean_coh = np.nanmean(coherence, axis=0)
    base = np.asarray(valid, dtype=bool).any(axis=0)
    adi_max = float(params.get("ps_adi_max", 0.28))
    coh_min = float(params.get("ps_coherence_min", 0.62))

    candidate = base & (adi <= adi_max) & (mean_coh >= coh_min)
    # If the thresholds are too strict for a small demo, relax them once and
    # keep the best 1.5% of pixels so the technique still produces a result.
    relaxed = False
    if candidate.sum() < max(30, 0.002 * base.sum()):
        relaxed = True
        score = (1.0 / (adi + 0.03)) * np.clip(mean_coh, 0, 1)
        score[~base] = -np.inf
        k = max(30, int(0.015 * base.sum()))
        flat = np.argsort(score.ravel())[-k:]
        candidate = np.zeros_like(base)
        candidate.ravel()[flat] = True
        candidate &= base & np.isfinite(score)
    # remove isolated salt noise in the ADI map (but preserve genuine points)
    opened = ndimage.binary_opening(candidate, structure=np.ones((2, 2)))
    if opened.sum() >= max(20, 0.35 * candidate.sum()):
        candidate = opened
    weight = np.clip((1.0 - np.minimum(adi / max(adi_max, 1e-6), 1.0)) * mean_coh, 0.04, 1.0)
    weight[~candidate] = 0.0
    stats = {
        "n_candidates": int(candidate.sum()),
        "density": float(candidate.sum() / max(base.sum(), 1)),
        "adi_median": float(np.nanmedian(adi[candidate])) if candidate.any() else 0.0,
        "coherence_median": float(np.nanmedian(mean_coh[candidate])) if candidate.any() else 0.0,
        "adi_max": adi_max,
        "coherence_min": coh_min,
        "relaxed": relaxed,
    }
    return {
        "mask": candidate,
        "weight": weight.astype(np.float32),
        "adi": adi.astype(np.float32),
        "mean_coherence": mean_coh.astype(np.float32),
        "stats": stats,
    }


def save_ps_figure(result: dict[str, Any], root: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plotting import PAPER, INK, MUTED, TEAL, AMBER, GRID, seq_cmap

        adi = result["adi"]
        mask = result["mask"]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        ax = axes[0]
        vmin, vmax = np.nanpercentile(adi, [2, 98])
        im = ax.imshow(adi, cmap=seq_cmap(), vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title("Amplitude dispersion index", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
        ax = axes[1]
        ax.imshow(mask, cmap="gray", interpolation="nearest")
        ys, xs = np.nonzero(mask)
        if len(xs):
            ax.scatter(xs, ys, s=0.6, color=AMBER, alpha=0.7, linewidth=0)
        ax.set_title(f"Persistent scatterer candidates · {mask.sum():,} px", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.patch.set_facecolor(PAPER)
        fig.tight_layout()
        out = root / "products" / "figures" / "ps_candidates.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return "products/figures/ps_candidates.png"
    except Exception:
        return None
