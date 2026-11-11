"""Small numerical and filesystem helpers shared by the processing engine."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


TWO_PI = 2.0 * math.pi


def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap phase to (-pi, pi]."""
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def phase_std(x: np.ndarray) -> float:
    """Circular standard deviation of phase in radians."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    r = np.abs(np.mean(np.exp(1j * x)))
    r = min(max(r, 1e-12), 1.0)
    return float(math.sqrt(max(0.0, -2.0 * math.log(r))))


def temporal_coherence(residual: np.ndarray) -> np.ndarray:
    """Coherence-like stability metric from a phase residual stack.

    ``residual`` has shape (n_dates, ...).  Values close to 1 mean the
    inverted model explains the observations with a stable phase.
    """
    if residual.shape[0] == 0:
        return np.ones(residual.shape[1:], dtype=np.float32)
    return np.abs(np.mean(np.exp(1j * residual), axis=0)).astype(np.float32)


def robust_sigma(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def nan_gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter that ignores NaNs (normalised convolution)."""
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    num = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[~valid] = np.nan
    return out


def fill_nan_nearest(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.all(np.isfinite(x)):
        return x.copy()
    valid = np.isfinite(x)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return x[tuple(idx)]


def stable_reference(quality: np.ndarray, valid: np.ndarray | None = None, search: int = 12) -> tuple[int, int]:
    """Pick a stable reference pixel: highest quality near the scene centre.

    A small central window avoids edge artefacts while still allowing the
    user to override the choice in expert mode.
    """
    q = np.asarray(quality, dtype=np.float64)
    if valid is not None:
        q = np.where(valid, q, -np.inf)
    rows, cols = q.shape
    cr, cc = rows // 2, cols // 2
    r0, r1 = max(0, cr - search), min(rows, cr + search + 1)
    c0, c1 = max(0, cc - search), min(cols, cc + search + 1)
    window = q[r0:r1, c0:c1]
    if not np.isfinite(window).any():
        window = q
        r0 = c0 = 0
    idx = int(np.nanargmax(window))
    rr, cc2 = np.unravel_index(idx, window.shape)
    return int(r0 + rr), int(c0 + cc2)


def downsample_complex(arr: np.ndarray, factor: int) -> np.ndarray:
    """Uniform complex multilook by an integer factor (edge cropped)."""
    if factor <= 1:
        return arr
    rows = arr.shape[0] // factor * factor
    cols = arr.shape[1] // factor * factor
    cropped = arr[:rows, :cols]
    return cropped.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def downsample_real(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    rows = arr.shape[0] // factor * factor
    cols = arr.shape[1] // factor * factor
    cropped = np.asarray(arr)[:rows, :cols]
    return cropped.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def downsample_complex_pair(
    ref: np.ndarray,
    sec: np.ndarray,
    factor: int,
    *,
    block_budget_bytes: int = 256 * 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multilook a complex pair in row blocks.

    Returns ``(ifg, ref_m, sec_m)``.  The block loop avoids materialising
    full-resolution ``ref``, ``sec`` and ``ifg`` copies at the same time, which
    is the main peak-memory spike when a pair is much larger than RAM.
    """
    factor = max(1, int(factor))
    if factor <= 1:
        return np.asarray(sec) * np.conj(np.asarray(ref)), ref, sec
    from .memory import choose_block_rows

    rows = ref.shape[0] // factor * factor
    cols = ref.shape[1] // factor * factor
    out_shape = (rows // factor, cols // factor)
    ifg = np.empty(out_shape, dtype=np.complex64)
    ref_m = np.empty(out_shape, dtype=np.complex64)
    sec_m = np.empty(out_shape, dtype=np.complex64)
    block_rows = choose_block_rows(
        rows,
        cols,
        budget_bytes=max(16 * 1024 * 1024, int(block_budget_bytes)),
        arrays=3,
        itemsize=8,
        multiple=factor,
    )
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        a = np.asarray(ref[r0:r1, :cols])
        b = np.asarray(sec[r0:r1, :cols])
        out_r0 = r0 // factor
        out_r1 = r1 // factor
        ifg[out_r0:out_r1] = downsample_complex(b * np.conj(a), factor)
        ref_m[out_r0:out_r1] = downsample_complex(a, factor)
        sec_m[out_r0:out_r1] = downsample_complex(b, factor)
    return ifg, ref_m, sec_m


def upsample_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    rr = np.minimum((np.arange(rows) * arr.shape[0] // rows), arr.shape[0] - 1)
    cc = np.minimum((np.arange(cols) * arr.shape[1] // cols), arr.shape[1] - 1)
    return arr[np.ix_(rr, cc)]


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy / path / date objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sanitise at the final write boundary: every stage summary, job record and
    # cache file goes through this helper, so a stage that accidentally copies
    # request parameters cannot persist a GACOS token or download password.
    from ..params import redact_secrets

    payload = json.dumps(redact_secrets(json_safe(data)), ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def format_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    return f"{minutes / 60.0:.1f} 小时"


def parse_date(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def date_range_pairs(dates: Iterable[str]) -> np.ndarray:
    ds = [parse_date(d) for d in dates]
    return np.asarray([(d - ds[0]).days for d in ds], dtype=np.float64)


def resample_to_shape(arr: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    if arr.shape == shape:
        return np.asarray(arr)
    zoom = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    out = ndimage.zoom(arr, zoom, order=order, mode="nearest")
    if out.shape != shape:
        out = out[: shape[0], : shape[1]]
        if out.shape != shape:
            pad_r = shape[0] - out.shape[0]
            pad_c = shape[1] - out.shape[1]
            out = np.pad(out, ((0, max(0, pad_r)), (0, max(0, pad_c))), mode="edge")
    return out
