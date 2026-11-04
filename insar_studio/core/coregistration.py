"""Image coregistration.

Coarse-to-fine phase correlation on amplitude patches, robust outlier
rejection, optional affine refinement and cubic resampling of the complex
SLC.  Sub-pixel accuracy is validated by the synthetic demo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, estimate_transform, warp

from .io import Stack, load_stack, safe_join
from .plotting import save_histogram
from .utils import atomic_write_json, read_json


Progress = Callable[[float, str], None]


def _log_amp(amp: np.ndarray) -> np.ndarray:
    a = np.log1p(np.abs(amp).astype(np.float32))
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return a
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _patch_offsets(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    tile: int = 64,
    upsample: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = ref.shape
    centers = []
    shifts = []
    weights = []
    step_r = max(tile // 2, 1)
    step_c = max(tile // 2, 1)
    for r in range(tile // 2, rows - tile // 2, step_r):
        for c in range(tile // 2, cols - tile // 2, step_c):
            r0, r1 = r - tile // 2, r + tile // 2
            c0, c1 = c - tile // 2, c + tile // 2
            pr = ref[r0:r1, c0:c1]
            pm = mov[r0:r1, c0:c1]
            if pr.std() < 0.015 or pm.std() < 0.015:
                continue
            try:
                shift, error, _ = phase_cross_correlation(
                    pr, pm, upsample_factor=upsample, normalization=None
                )
            except Exception:
                continue
            if not np.all(np.isfinite(shift)):
                continue
            if abs(shift[0]) > tile / 2 or abs(shift[1]) > tile / 2:
                continue
            centers.append((r, c))
            shifts.append((shift[0], shift[1]))
            weights.append(float(pr.std() * pm.std()))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(shifts, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _robust_filter(centers: np.ndarray, shifts: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(shifts) < 3:
        return centers, shifts, weights
    med = np.median(shifts, axis=0)
    mad = np.median(np.abs(shifts - med), axis=0) * 1.4826
    mad = np.maximum(mad, 0.08)
    keep = np.all(np.abs(shifts - med) < 3.5 * mad, axis=1)
    if keep.sum() < 3:
        keep = np.ones(len(shifts), dtype=bool)
    return centers[keep], shifts[keep], weights[keep]


def estimate_transform(
    ref_amp: np.ndarray,
    mov_amp: np.ndarray,
    *,
    model: str = "affine",
    tile: int = 64,
    upsample: int = 20,
) -> tuple[AffineTransform, dict[str, Any]]:
    ref = _log_amp(ref_amp)
    mov = _log_amp(mov_amp)
    centers, shifts, weights = _patch_offsets(ref, mov, tile=tile, upsample=upsample)
    if len(centers) == 0:
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None)
        centers = np.asarray([(ref.shape[0] / 2, ref.shape[1] / 2)])
        shifts = np.asarray([(shift[0], shift[1])])
        weights = np.asarray([1.0])
    centers, shifts, weights = _robust_filter(centers, shifts, weights)

    # output point -> input point
    dst = centers.copy()
    src = centers - shifts
    info: dict[str, Any] = {
        "n_tiles": int(len(centers)),
        "median_shift_row": float(np.median(shifts[:, 0])),
        "median_shift_col": float(np.median(shifts[:, 1])),
        "shift_scatter_row": float(np.std(shifts[:, 0])),
        "shift_scatter_col": float(np.std(shifts[:, 1])),
        "model": model,
    }

    if model == "translation" or len(centers) < 4:
        t = np.median(shifts, axis=0)
        transform = AffineTransform(translation=(-t[1], -t[0]))  # x=col, y=row
        # For warp(inverse_map), translation maps output -> input = output - shift
        info["model"] = "translation"
        info["shift_row"] = float(t[0])
        info["shift_col"] = float(t[1])
        return transform, info

    try:
        # skimage uses (x, y) = (col, row)
        src_xy = np.column_stack([src[:, 1], src[:, 0]])
        dst_xy = np.column_stack([dst[:, 1], dst[:, 0]])
        # Estimate output -> input directly; warp() consumes this as inverse_map.
        transform = estimate_transform("affine", dst_xy, src_xy)
        pred = transform(dst_xy)
        residual = pred - src_xy
        rms = float(np.sqrt(np.mean(residual**2)))
        info["affine_rms_px"] = rms
        # A wild affine fit is worse than a pure translation
        if not np.all(np.isfinite(transform.params)) or rms > 0.75:
            t = np.median(shifts, axis=0)
            transform = AffineTransform(translation=(-t[1], -t[0]))
            info["model"] = "translation_fallback"
            info["shift_row"] = float(t[0])
            info["shift_col"] = float(t[1])
        else:
            info["matrix"] = transform.params.tolist()
    except Exception:
        t = np.median(shifts, axis=0)
        transform = AffineTransform(translation=(-t[1], -t[0]))
        info["model"] = "translation_fallback"
        info["shift_row"] = float(t[0])
        info["shift_col"] = float(t[1])
    return transform, info


def _apply_transform(image: np.ndarray, transform: AffineTransform, *, fast: bool = False) -> np.ndarray:
    """Resample a complex image. ``transform`` maps output coords to input."""
    if fast:
        # ``transform`` maps output -> input; ndimage.shift needs the inverse
        # (the displacement of image content).
        t = transform.translation  # (x, y), output -> input
        shift = (-t[1], -t[0])  # (row, col), content displacement
        return ndimage.shift(
            image.real, shift=shift, order=3, mode="nearest"
        ) + 1j * ndimage.shift(image.imag, shift=shift, order=3, mode="nearest")
    real = warp(
        image.real,
        inverse_map=transform,
        order=3,
        mode="edge",
        preserve_range=True,
        clip=False,
    )
    imag = warp(
        image.imag,
        inverse_map=transform,
        order=3,
        mode="edge",
        preserve_range=True,
        clip=False,
    )
    return (real + 1j * imag).astype(np.complex64)


def _coherence(a: np.ndarray, b: np.ndarray, window: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    prod = b * np.conj(a)
    num = uniform_filter(prod.real, window) + 1j * uniform_filter(prod.imag, window)
    p1 = uniform_filter(np.abs(a) ** 2, window)
    p2 = uniform_filter(np.abs(b) ** 2, window)
    return (np.abs(num) / np.sqrt(np.maximum(p1 * p2, 1e-12))).astype(np.float32)


def coregister_stack(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Coregister all acquisitions to the reference and persist the stack."""
    root = Path(project_root)
    params = params or {}
    stack = load_stack(root)
    n = stack.n_dates
    ref_index = int(params.get("ref_index", stack.meta.get("ref_index", 0)))
    model = str(params.get("model", "affine"))
    tile = int(params.get("tile", 64))
    upsample = int(params.get("upsample", 20))
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_amp = stack.amplitude(ref_index)
    out = np.empty_like(stack.slc, dtype=np.complex64)
    out[ref_index] = stack.slc[ref_index]
    transforms: list[dict[str, Any]] = []
    shifts = np.zeros((n, 2), dtype=np.float32)
    gains: list[float] = []
    if progress:
        progress(0.02, f"配准参考影像：{stack.dates[ref_index]}")

    for i in range(n):
        if i == ref_index:
            transforms.append({"index": i, "date": stack.dates[i], "model": "reference", "n_tiles": 0})
            gains.append(1.0)
            continue
        mov = np.asarray(stack.slc[i])
        transform, info = estimate_transform(ref_amp, np.abs(mov), model=model, tile=tile, upsample=upsample)
        fast = info.get("model", "").startswith("translation")
        out[i] = _apply_transform(mov, transform, fast=fast)
        info.update({"index": i, "date": stack.dates[i]})
        transforms.append(info)
        t = transform.translation
        shifts[i] = (t[1], t[0])

        # coherence before / after on a central subset (quality evidence)
        sl = (slice(stack.rows // 5, stack.rows * 4 // 5), slice(stack.cols // 5, stack.cols * 4 // 5))
        c_before = float(np.nanmean(_coherence(np.asarray(stack.slc[ref_index])[sl], mov[sl], 5)))
        c_after = float(np.nanmean(_coherence(np.asarray(stack.slc[ref_index])[sl], out[i][sl], 5)))
        gains.append(c_after / max(c_before, 1e-6))
        if progress:
            progress(0.04 + 0.86 * (i + 1) / n, f"配准 {i + 1}/{n} · {stack.dates[i]} · 偏移 {t[1]:+.2f}/{t[0]:+.2f} px")

    coreg_path = out_dir / "slc_coreg.npy"
    np.save(coreg_path, out, allow_pickle=False)

    coh_before = []
    coh_after = []
    for i in range(n):
        if i == ref_index:
            continue
        sl = (slice(stack.rows // 5, stack.rows * 4 // 5), slice(stack.cols // 5, stack.cols * 4 // 5))
        coh_before.append(float(np.nanmean(_coherence(np.asarray(stack.slc[ref_index])[sl], np.asarray(stack.slc[i])[sl], 5))))
        coh_after.append(float(np.nanmean(_coherence(np.asarray(stack.slc[ref_index])[sl], out[i][sl], 5))))

    result = {
        "ref_index": ref_index,
        "ref_date": stack.dates[ref_index],
        "model": model,
        "n_dates": n,
        "transforms": transforms,
        "shifts": shifts.tolist(),
        "coherence_before": float(np.mean(coh_before)) if coh_before else 1.0,
        "coherence_after": float(np.mean(coh_after)) if coh_after else 1.0,
        "coherence_gain": float(np.mean(coh_after) - np.mean(coh_before)) if coh_before else 0.0,
        "stack_path": "data/processed/slc_coreg.npy",
        "params": params,
    }
    atomic_write_json(out_dir / "coreg.json", result)

    try:
        save_histogram(
            np.asarray(coh_after) - np.asarray(coh_before) if coh_before else np.zeros(1),
            root / "products" / "figures" / "coreg_coherence_gain.png",
            title="Coherence gain after coregistration",
            xlabel="Δ coherence",
            color="#0E6B5C",
        )
        # amplitude difference image
        from .plotting import save_raster

        diff = np.abs(out[ref_index] - np.asarray(stack.slc[ref_index]))
        save_raster(
            diff,
            root / "products" / "figures" / "coreg_residual.png",
            title="Coregistration residual amplitude",
            subtitle="|reference − resampled reference|",
            cmap="seq",
            figsize=(5.2, 4.0),
        )
    except Exception:
        pass

    if progress:
        progress(0.94, f"配准完成 · 相干性 {result['coherence_before']:.2f} → {result['coherence_after']:.2f}")
    return result


def load_coregistered(project_root: Path) -> np.ndarray:
    path = Path(project_root) / "data" / "processed" / "slc_coreg.npy"
    if not path.exists():
        raise FileNotFoundError("尚未完成配准，请先运行配准步骤")
    return np.load(path, mmap_mode="r", allow_pickle=False)
