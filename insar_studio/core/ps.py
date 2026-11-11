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


def amplitude_dispersion(
    stack: Stack,
    multilook: int = 1,
    max_images: int = 12,
    *,
    block_rows: int | None = None,
) -> np.ndarray:
    """Estimate amplitude dispersion without materialising every image.

    The old implementation built a list of up to 12 full-resolution amplitude
    arrays.  This version accumulates first and second moments one row block at
    a time, so peak memory is proportional to the block height, not the whole
    stack.  The result is numerically the population standard deviation divided
    by the mean amplitude.
    """
    n = stack.n_dates
    m = max(1, int(multilook))
    rows, cols = int(stack.slc.shape[1]), int(stack.slc.shape[2])
    out_rows, out_cols = rows // m, cols // m
    if out_rows <= 0 or out_cols <= 0:
        return np.empty((max(out_rows, 0), max(out_cols, 0)), dtype=np.float32)

    step = max(1, n // max(1, max_images))
    indices = list(range(0, n, step))
    count = max(1, len(indices))

    if block_rows is None:
        # sum, sum of squares, one amplitude block and its square: ~32 B/px.
        per_row = max(1, cols * 32)
        budget_rows = max(m, (128 * 1024 * 1024) // per_row)
        block_rows = max(m, (budget_rows // m) * m)
    else:
        block_rows = max(m, (max(1, int(block_rows)) // m) * m)

    adi = np.empty((out_rows, out_cols), dtype=np.float32)
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        height = r1 - r0
        sum_amp = np.zeros((height, cols), dtype=np.float64)
        sum_sq = np.zeros((height, cols), dtype=np.float64)
        for i in indices:
            amp = np.abs(np.asarray(stack.slc[i, r0:r1, :], dtype=np.complex64)).astype(np.float64)
            amp[~np.isfinite(amp)] = 0.0
            sum_amp += amp
            sum_sq += amp * amp
        mean = sum_amp / count
        variance = np.maximum(sum_sq / count - mean * mean, 0.0)
        block_adi = (np.sqrt(variance) / np.maximum(mean, 1e-6)).astype(np.float32)

        block_ds = downsample_real(block_adi, m)
        out0 = r0 // m
        adi[out0 : out0 + block_ds.shape[0], : block_ds.shape[1]] = block_ds
    return adi


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
    # Accumulate finite pair coherence one pair at a time instead of creating a
    # full NaN mask for the whole cube.
    total = np.zeros((rows, cols), dtype=np.float64)
    count = np.zeros((rows, cols), dtype=np.int32)
    for pair_coh in coherence:
        finite = np.isfinite(pair_coh)
        total += np.where(finite, pair_coh, 0.0)
        count += finite
    mean_coh = np.divide(
        total,
        count,
        out=np.zeros((rows, cols), dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)
    base = np.asarray(valid, dtype=bool).any(axis=0)
    adi_max = float(params.get("ps_adi_max", 0.28))
    coh_min = float(params.get("ps_coherence_min", 0.62))

    candidate = base & (adi <= adi_max) & (mean_coh >= coh_min)
    # If the thresholds are too strict for a small demo, relax them once and
    # keep the best 1.5% of pixels so the technique still produces a result.
    # Never invent candidates when no pair coherence is finite: in that case a
    # PS solution would be meaningless and the caller should report emptiness.
    relaxed = False
    finite_coh = np.isfinite(mean_coh) & (count > 0)
    if (
        candidate.sum() < max(30, 0.002 * base.sum())
        and finite_coh.any()
        and float(np.nanmax(mean_coh[finite_coh])) > 0
    ):
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
