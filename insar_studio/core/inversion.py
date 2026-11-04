"""SBAS / PS / DS time-series inversion.

The core solver estimates a displacement history and an optional DEM error
from a redundant set of unwrapped pair phases.  It supports weighted least
squares, Tikhonov smoothing, robust re-weighting, loop-closure down-weighting
and iterative spatio-temporal atmospheric separation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from ..config import DEFAULT_PROCESSING
from .geotiff import write_geotiff
from .io import load_stack
from .plotting import save_raster, save_series
from .utils import (
    atomic_write_json,
    downsample_real,
    json_safe,
    nan_gaussian_filter,
    read_json,
    stable_reference,
    temporal_coherence,
    wrap_phase,
)


Progress = Callable[[float, str], None]


def _pair_phase_path(root: Path, pair_id: str, corrected: bool) -> Path:
    base = root / "interferograms" / pair_id
    if corrected:
        candidate = base / "unwrapped_atmcorr.npy"
        if candidate.exists():
            return candidate
    return base / "unwrapped.npy"


def load_pair_cube(
    root: Path,
    pairs: list[dict[str, Any]],
    *,
    corrected: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    phases = []
    cohs = []
    masks = []
    kept = []
    shape = None
    for p in pairs:
        phase_path = _pair_phase_path(root, p["id"], corrected)
        coh_path = root / "interferograms" / p["id"] / "coherence.npy"
        mask_path = root / "interferograms" / p["id"] / "mask.npy"
        if not phase_path.exists() or not coh_path.exists() or not mask_path.exists():
            continue
        phase = np.load(phase_path, mmap_mode="r", allow_pickle=False)
        coh = np.load(coh_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False).astype(bool) & np.isfinite(phase)
        if shape is None:
            shape = phase.shape
        if phase.shape != shape or coh.shape != shape or mask.shape != shape:
            continue
        phases.append(np.asarray(phase, dtype=np.float32))
        cohs.append(np.asarray(coh, dtype=np.float32))
        masks.append(mask)
        kept.append(p)
    if not phases:
        raise RuntimeError("没有可用的干涉对，请先完成干涉处理")
    return (
        np.stack(phases),
        np.stack(cohs),
        np.stack(masks),
        kept,
    )


def _time_design(
    pairs: list[dict[str, Any]],
    n_dates: int,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = len(pairs)
    A = np.zeros((p, n_dates), dtype=np.float64)
    for k, pair in enumerate(pairs):
        A[k, int(pair["ref"])] -= 1.0
        A[k, int(pair["sec"])] += 1.0
    G = A.copy()
    if n_dates >= 3 and regularization > 0:
        D = np.zeros((n_dates - 2, n_dates), dtype=np.float64)
        for i in range(n_dates - 2):
            D[i, i : i + 3] = (1.0, -2.0, 1.0)
        G = np.vstack([A, D * float(regularization)])
    return G, A


def _sampling_days(stack_meta: dict[str, Any]) -> np.ndarray:
    from .utils import date_range_pairs

    days = np.asarray(stack_meta.get("sampling_days") or [], dtype=np.float64)
    if len(days) != len(stack_meta["dates"]):
        days = date_range_pairs(stack_meta["dates"])
    return days


def _reference_pixel(root: Path, mask: np.ndarray, params: dict[str, Any] | None = None) -> tuple[int, int]:
    p = params or {}
    if str(p.get("reference_mode", "auto")).lower() == "manual":
        try:
            rr = int(p.get("reference_row"))
            cc = int(p.get("reference_col"))
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
                return rr, cc
        except (TypeError, ValueError):
            pass
    ref = read_json(root / "interferograms" / "reference.json", {}) or {}
    r, c = int(ref.get("row", mask.shape[0] // 2)), int(ref.get("col", mask.shape[1] // 2))
    r = min(max(r, 0), mask.shape[0] - 1)
    c = min(max(c, 0), mask.shape[1] - 1)
    if not mask[r, c]:
        rr, cc = stable_reference(mask.astype(float), mask)
        return rr, cc
    return r, c


def _estimate_dem_error(
    phase: np.ndarray,
    mask: np.ndarray,
    pairs: list[dict[str, Any]],
    stack_meta: dict[str, Any],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Separate baseline-dependent DEM error using a smooth deformation model.

    A free per-date phase history and a DEM-error column are mathematically
    collinear (b = A @ bperp), so the DEM term is estimated against a
    low-order temporal basis (linear + annual + semi-annual) and then removed.
    The full, unconstrained history is recovered afterwards.
    """
    p, rows, cols = phase.shape
    n_dates = len(stack_meta["dates"])
    days = _sampling_days(stack_meta)
    A = _time_design(pairs, n_dates, 0.0)[1]
    t = (days - days[0]) / 365.25
    t = t - t.mean()
    B = np.column_stack(
        [
            t,
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t),
            np.cos(4 * np.pi * t),
        ]
    )
    AB = A @ B
    wavelength = float(stack_meta["wavelength_m"])
    k = 4.0 * np.pi / wavelength
    R = float(stack_meta.get("slant_range_m", 705000.0))
    sin_inc = np.sin(np.deg2rad(float(stack_meta["incidence_deg"])))
    bperp = stack_meta.get("bperp_m", [0.0] * n_dates)
    b = np.asarray(
        [k * (bperp[int(q["sec"])] - bperp[int(q["ref"])]) / (R * sin_inc) for q in pairs],
        dtype=np.float64,
    )
    if np.std(b) < 1e-10 or n_dates < 6:
        return phase, np.zeros((rows, cols), dtype=np.float32), {"enabled": False, "reason": "基线变化不足或历元过少"}

    G = np.column_stack([AB, b])
    col_scale = np.std(G, axis=0)
    col_scale[col_scale < 1e-9] = 1.0
    Gs = G / col_scale
    # tiny ridge keeps the separation stable when bperp correlates with time
    ridge = np.sqrt(1e-4) * np.eye(Gs.shape[1])
    Gs = np.vstack([Gs, ridge])
    w = np.sqrt(np.clip(weights, 0.02, 1.0))[:, None]
    Gsw = Gs[:p] * w
    Gsw = np.vstack([Gsw, Gs[p:]])
    zero_pad = np.zeros((Gs.shape[0] - p, phase.shape[1] * phase.shape[2]), dtype=np.float32)
    phi = np.where(mask, phase, 0.0).reshape(p, -1).astype(np.float32)
    Y = np.vstack([phi * w, zero_pad])
    coef = (np.linalg.pinv(Gsw) @ Y).astype(np.float32)
    eps = coef[-1] / col_scale[-1]
    eps = eps.reshape(rows, cols)
    coverage = mask.mean(axis=0)
    eps = np.where(coverage >= 0.45, eps, 0.0)
    corrected = phase - b[:, None, None] * eps[None, :, :]
    cond = float(np.linalg.cond(Gsw))
    info = {
        "enabled": True,
        "condition": cond,
        "warning": "基线-时间相关性较强，DEM 误差估计不确定度较大" if cond > 1e4 else "",
        "median_abs_m": float(np.nanmedian(np.abs(eps))) if np.isfinite(eps).any() else 0.0,
    }
    return corrected.astype(np.float32), eps.astype(np.float32), info


def _solve_weighted(
    phase: np.ndarray,
    mask: np.ndarray,
    coh: np.ndarray,
    pairs: list[dict[str, Any]],
    G: np.ndarray,
    A: np.ndarray,
    weights: np.ndarray,
    n_dates: int,
    iterations: int,
    extra_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p, rows, cols = phase.shape
    phase_flat = phase.reshape(p, -1)
    mask_flat = mask.reshape(p, -1)
    coverage = mask_flat.mean(axis=0)
    phi = np.where(mask_flat, phase_flat, 0.0).astype(np.float32)
    w = weights.astype(np.float64).copy()
    pw = np.ones(rows * cols, dtype=np.float64)
    if extra_weight is not None:
        pw = np.clip(np.asarray(extra_weight, dtype=np.float64).reshape(-1), 1e-3, 1.0)
    sqrt_w = np.sqrt(w)[:, None] * np.sqrt(pw)[None, :]
    extra_rows = G.shape[0] - p
    # Regularisation rows are not tied to the pixel quality weight.
    sqrt_w_full = np.vstack([sqrt_w, np.ones((extra_rows, phi.shape[1]), dtype=np.float64)])
    zeros_extra = np.zeros((extra_rows, phi.shape[1]), dtype=np.float32)
    X = np.zeros((n_dates, rows * cols), dtype=np.float32)
    if not (coverage >= 0.25).any():
        return X.reshape(n_dates, rows, cols), coverage.reshape(rows, cols)

    for it in range(max(1, iterations)):
        # The design matrix is identical for every pixel; only the diagonal
        # weights differ.  Batched normal equations are far cheaper than a
        # per-pixel SVD and fit comfortably in memory for a 24-date stack.
        Y = np.vstack([phi * sqrt_w, zeros_extra])
        if it > 0:
            pred = A @ X
            miss = ~mask_flat
            Y[:p] = np.where(miss, pred * sqrt_w, Y[:p])
        sw2 = (sqrt_w_full**2).astype(np.float64)
        block = 4096
        X = np.empty((n_dates, phi.shape[1]), dtype=np.float32)
        for s in range(0, phi.shape[1], block):
            e = min(phi.shape[1], s + block)
            W = sw2[:, s:e]
            GtWG = np.einsum("mn,mp,ml->nlp", G, W, G, optimize=True)
            GtWy = np.einsum("mn,mp,mp->np", G, W, Y[:, s:e], optimize=True)
            GtWG = GtWG + np.eye(G.shape[1], dtype=np.float64)[:, :, None] * 1e-9
            X[:, s:e] = np.linalg.solve(GtWG.transpose(2, 0, 1), GtWy.T).T.astype(np.float32)
        model = (G[:p] @ X).astype(np.float32)
        resid = np.where(mask_flat, phase_flat - model, np.nan)
        med = np.nanmedian(np.abs(resid), axis=1)
        sigma = np.nanmedian(med) if np.isfinite(med).any() else 1.0
        sigma = max(float(sigma), 1e-3)
        ratio = med / (3.0 * sigma)
        new_w = np.clip(1.0 / (1.0 + ratio**2), 0.04, 1.0)
        w = weights * new_w
        sqrt_w = np.sqrt(w)[:, None] * np.sqrt(pw)[None, :]
    return X.reshape(n_dates, rows, cols), coverage.reshape(rows, cols)


def _fit_velocity(disp_mm: np.ndarray, days: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, rows, cols = disp_mm.shape
    t = days.astype(np.float64)
    t = t - t.mean()
    denom = float(np.sum(t**2))
    if denom <= 0:
        return np.zeros((rows, cols), dtype=np.float32), np.zeros((rows, cols), dtype=np.float32)
    y = disp_mm.reshape(n, -1)
    v = (t[:, None] * y).sum(axis=0) / denom * 365.25
    intercept = y.mean(axis=0)
    pred = intercept[None, :] + (v / 365.25)[None, :] * t[:, None]
    rmse = np.sqrt(np.mean((y - pred) ** 2, axis=0))
    return v.reshape(rows, cols).astype(np.float32), rmse.reshape(rows, cols).astype(np.float32)


def _aps_screens(
    phase: np.ndarray,
    mask: np.ndarray,
    pairs: list[dict[str, Any]],
    n_dates: int,
    x_3d: np.ndarray,
    days: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Separate a spatially coherent, temporally high-frequency screen.

    A free per-date phase history can absorb atmospheric delay.  We therefore
    fit a smooth temporal model (linear + annual + semi-annual), treat the
    high-frequency residual as the atmosphere/noise candidate, and retain its
    spatially coherent component.  This is the standard temporal high-pass
    approximation used when no external GACOS product is available.
    """
    rows, cols = phase.shape[1:]
    t = (days - days[0]) / 365.25
    t = t - t.mean()
    B = np.column_stack(
        [
            np.ones_like(t),
            t,
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t),
            np.cos(4 * np.pi * t),
        ]
    )
    Xf = x_3d.reshape(n_dates, -1).astype(np.float64)
    coef = np.linalg.pinv(B) @ Xf
    resid = (Xf - B @ coef).reshape(n_dates, rows, cols)
    screens = np.zeros((n_dates, rows, cols), dtype=np.float32)
    for i in range(n_dates):
        sm = ndimage.gaussian_filter(
            np.nan_to_num(resid[i], nan=0.0), sigma=sigma, mode="nearest"
        )
        sm -= np.nanmedian(sm)
        screens[i] = sm.astype(np.float32)
    return screens


def invert_arrays(
    root: Path,
    phase: np.ndarray,
    coh: np.ndarray,
    mask: np.ndarray,
    pairs: list[dict[str, Any]],
    stack_meta: dict[str, Any],
    params: dict[str, Any],
    *,
    extra_mask: np.ndarray | None = None,
    extra_weight: np.ndarray | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    p = {**DEFAULT_PROCESSING, **params}
    n_dates = len(stack_meta["dates"])
    days = _sampling_days(stack_meta)
    G, A = _time_design(pairs, n_dates, float(p.get("regularization", 0.08)))
    pair_weights = np.clip(np.nanmean(coh, axis=(1, 2)), 0.04, 1.0) ** 2
    closure = read_json(root / "interferograms" / "summary.json", {}) or {}
    flags = closure.get("closure", {}).get("flagged_pairs", {}) if isinstance(closure.get("closure"), dict) else {}
    if flags:
        for k, pair in enumerate(pairs):
            if pair["id"] in flags:
                pair_weights[k] *= 0.35

    phase_corr = phase.copy()
    eps = np.zeros(phase.shape[1:], dtype=np.float32)
    dem_info: dict[str, Any] = {"enabled": False}
    if p.get("estimate_dem_error", True):
        if progress:
            progress(0.05, "估计与剥离 DEM 误差")
        phase_corr, eps, dem_info = _estimate_dem_error(phase_corr, mask, pairs, stack_meta, pair_weights)

    screens = None
    n_aps = int(p.get("aps_iterations", 0)) if p.get("aps_method", "spatiotemporal") != "off" else 0
    X = None
    coverage = np.zeros(phase.shape[1:], dtype=np.float32)
    for aps_it in range(max(0, n_aps) + 1):
        if progress:
            progress(0.10 + 0.76 * aps_it / max(1, n_aps + 1), "时序反演" if aps_it == 0 else f"大气误差分离 · 第 {aps_it} 轮")
        X, coverage = _solve_weighted(
            phase_corr,
            mask,
            coh,
            pairs,
            G,
            A,
            pair_weights,
            n_dates,
            int(p.get("robust_iterations", 3)),
            extra_weight=extra_weight,
        )
        if aps_it < n_aps:
            screens = _aps_screens(
                phase_corr,
                mask,
                pairs,
                n_dates,
                X,
                days,
                float(p.get("aps_sigma_px", 10.0)),
            )
            for k, pair in enumerate(pairs):
                phase_corr[k] = phase_corr[k] - (screens[int(pair["sec"])] - screens[int(pair["ref"])])

    X = X.copy()
    X -= X[0]
    coverage_valid = coverage >= 0.45
    if extra_mask is not None:
        coverage_valid &= extra_mask
    coverage_valid &= np.isfinite(eps)
    rr, cc = _reference_pixel(root, coverage_valid, p)

    # Force a single, documented spatial reference.  DEM/APS corrections and
    # regularisation perturb the raw reference pixel, so the output must be
    # re-referenced explicitly; otherwise every product carries an undocumented
    # gauge that cannot be compared with ground truth or other dates.
    X -= X[:, rr, cc][:, None, None]

    disp = X * float(stack_meta["wavelength_m"]) / (4.0 * np.pi) * 1000.0
    model = (A @ X.reshape(n_dates, -1)).reshape(len(pairs), *phase.shape[1:])
    resid = np.where(mask, phase_corr - model, np.nan)
    tcoh = temporal_coherence(np.nan_to_num(resid, nan=0.0))
    valid = coverage_valid & np.isfinite(disp).all(axis=0)
    disp[:, ~valid] = np.nan
    eps = np.where(valid, eps, np.nan)
    sq = np.nan_to_num(resid, nan=0.0) ** 2
    cnt = np.isfinite(resid).sum(axis=0)
    resid_std = np.sqrt(np.divide(sq.sum(axis=0), cnt, out=np.zeros(sq.shape[1:], dtype=np.float64), where=cnt > 0))
    disp_std = resid_std * float(stack_meta["wavelength_m"]) / (4.0 * np.pi) * 1000.0
    vel, lin_rmse = _fit_velocity(np.nan_to_num(disp, nan=0.0), days, valid)
    vel = np.where(valid, vel, np.nan)

    tcoh = np.where(valid, tcoh, np.nan)
    rmse = np.where(valid, disp_std, np.nan)
    return {
        "disp_mm": disp.astype(np.float32),
        "velocity_mm_yr": vel.astype(np.float32),
        "dem_error_m": eps.astype(np.float32),
        "temporal_coherence": tcoh.astype(np.float32),
        "residual_mm": rmse.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "valid": valid.astype(bool),
        "reference_pixel": [int(rr), int(cc)],
        "screens": screens,
        "linear_rmse_mm": np.where(valid, lin_rmse, np.nan).astype(np.float32),
        "days": days,
        "dem_info": dem_info,
    }


def _save_products(
    root: Path,
    result: dict[str, Any],
    stack_meta: dict[str, Any],
    params: dict[str, Any],
    technique: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    products = root / "products"
    fig = products / "figures"
    geo = products / "geotiff"
    arrays = products / "arrays"
    csv_dir = products / "csv"
    for d in (fig, geo, arrays, csv_dir):
        d.mkdir(parents=True, exist_ok=True)
    m = max(1, int(params.get("multilook", 2)))
    grid_meta = {
        **stack_meta,
        "rows": int(result["valid"].shape[0]),
        "cols": int(result["valid"].shape[1]),
        "lon_step": float(stack_meta.get("lon_step", 0.0004)) * m,
        "lat_step": float(stack_meta.get("lat_step", -0.0004)) * m,
    }
    np.save(arrays / "disp_ts_mm.npy", result["disp_mm"], allow_pickle=False)
    np.save(arrays / "velocity_mm_yr.npy", result["velocity_mm_yr"], allow_pickle=False)
    np.save(arrays / "dem_error_m.npy", result["dem_error_m"], allow_pickle=False)
    np.save(arrays / "temporal_coherence.npy", result["temporal_coherence"], allow_pickle=False)
    np.save(arrays / "residual_mm.npy", result["residual_mm"], allow_pickle=False)
    np.save(arrays / "valid.npy", result["valid"], allow_pickle=False)
    if result.get("screens") is not None:
        np.save(arrays / "aps_screens.npy", result["screens"], allow_pickle=False)

    bounds = (
        float(grid_meta.get("corner_lon", 0.0)),
        float(grid_meta.get("corner_lat", 0.0)) + int(grid_meta["rows"]) * float(grid_meta["lat_step"]),
        float(grid_meta.get("corner_lon", 0.0)) + int(grid_meta["cols"]) * float(grid_meta["lon_step"]),
        float(grid_meta.get("corner_lat", 0.0)),
    )
    if bounds[1] > bounds[3]:
        bounds = (bounds[0], bounds[3], bounds[2], bounds[1])
    write_geotiff(geo / "velocity_mm_yr.tif", result["velocity_mm_yr"], corner_lon=float(grid_meta["corner_lon"]), corner_lat=float(grid_meta["corner_lat"]), lon_step=float(grid_meta["lon_step"]), lat_step=float(grid_meta["lat_step"]))
    write_geotiff(geo / "dem_error_m.tif", result["dem_error_m"], corner_lon=float(grid_meta["corner_lon"]), corner_lat=float(grid_meta["corner_lat"]), lon_step=float(grid_meta["lon_step"]), lat_step=float(grid_meta["lat_step"]))
    write_geotiff(geo / "temporal_coherence.tif", result["temporal_coherence"], corner_lon=float(grid_meta["corner_lon"]), corner_lat=float(grid_meta["corner_lat"]), lon_step=float(grid_meta["lon_step"]), lat_step=float(grid_meta["lat_step"]))
    for idx, d in enumerate(stack_meta["dates"]):
        write_geotiff(geo / f"disp_{d.replace('-', '')}.tif", result["disp_mm"][idx], corner_lon=float(grid_meta["corner_lon"]), corner_lat=float(grid_meta["corner_lat"]), lon_step=float(grid_meta["lon_step"]), lat_step=float(grid_meta["lat_step"]), compress=True)

    vel = result["velocity_mm_yr"]
    valid = result["valid"]
    save_raster(
        vel,
        fig / "velocity.png",
        mask=valid,
        title="LOS velocity",
        subtitle="Positive towards satellite · mm/yr",
        unit="mm/yr",
        divergent=True,
        symmetric=True,
        bounds=bounds,
        overlay=np.asarray(load_stack(root).dem, dtype=np.float32)[:: max(1, int(params.get("multilook", 2))), :: max(1, int(params.get("multilook", 2)))][: vel.shape[0], : vel.shape[1]],
        overlay_alpha=0.18,
    )
    save_raster(
        result["dem_error_m"],
        fig / "dem_error.png",
        mask=valid,
        title="Estimated DEM error",
        subtitle="Jointly estimated from baseline-dependent phase",
        unit="m",
        divergent=True,
        symmetric=True,
        bounds=bounds,
    )
    save_raster(
        result["temporal_coherence"],
        fig / "temporal_coherence.png",
        mask=valid,
        title="Temporal coherence",
        subtitle="Stability of the inverted phase history",
        unit="",
        vmin=0,
        vmax=1,
        bounds=bounds,
    )
    save_raster(
        result["residual_mm"],
        fig / "residual.png",
        mask=valid,
        title="Residual scatter",
        subtitle="Phase-to-displacement residual · mm",
        unit="mm",
        bounds=bounds,
    )

    # representative time series
    vv = np.where(valid, vel, np.nan)
    if np.isfinite(vv).any():
        idx = np.nanargmax(np.abs(vv - np.nanmedian(vv)))
        rr, cc = np.unravel_index(idx, vv.shape)
        series = {
            "Inverted LOS": result["disp_mm"][:, rr, cc],
        }
        save_series(
            stack_meta["dates"],
            series,
            fig / "timeseries_peak.png",
            title=f"Peak deformation · row {rr} col {cc}",
            uncertainty=result["residual_mm"][:, rr, cc] if False else None,
        )
        # CSV series for the peak and reference pixels
        import csv

        rr2, cc2 = result["reference_pixel"]
        with open(csv_dir / "timeseries_points.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "peak_row", "peak_col", "peak_disp_mm", "ref_row", "ref_col", "ref_disp_mm"])
            for i, d in enumerate(stack_meta["dates"]):
                w.writerow([d, rr, cc, float(result["disp_mm"][i, rr, cc]), rr2, cc2, float(result["disp_mm"][i, rr2, cc2])])

    summary = {
        "technique": technique,
        "n_dates": len(stack_meta["dates"]),
        "n_pairs": len(result.get("pairs", [])),
        "velocity_median_mm_yr": float(np.nanmedian(vel)) if np.isfinite(vel).any() else 0.0,
        "velocity_p05_mm_yr": float(np.nanpercentile(vel, 5)) if np.isfinite(vel).any() else 0.0,
        "velocity_p95_mm_yr": float(np.nanpercentile(vel, 95)) if np.isfinite(vel).any() else 0.0,
        "temporal_coherence_median": float(np.nanmedian(result["temporal_coherence"])) if np.isfinite(result["temporal_coherence"]).any() else 0.0,
        "residual_median_mm": float(np.nanmedian(result["residual_mm"])) if np.isfinite(result["residual_mm"]).any() else 0.0,
        "dem_error_median_m": float(np.nanmedian(result["dem_error_m"])) if np.isfinite(result["dem_error_m"]).any() else 0.0,
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.sum() / max(valid.size, 1)),
        "reference_pixel": result["reference_pixel"],
        "grid": {
            "rows": int(valid.shape[0]),
            "cols": int(valid.shape[1]),
            "corner_lon": float(grid_meta["corner_lon"]),
            "corner_lat": float(grid_meta["corner_lat"]),
            "lon_step": float(grid_meta["lon_step"]),
            "lat_step": float(grid_meta["lat_step"]),
        },
        "bounds": bounds,
        "params": params,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        summary.update(extra)
    atomic_write_json(products / "inversion.json", summary)
    return summary


def run_dinsar(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Produce differential-interferometry products from a single pair."""
    from .plotting import save_wrapped_phase

    root = Path(project_root)
    p = {**DEFAULT_PROCESSING, **(params or {})}
    stack = load_stack(root)
    network = read_json(root / "data" / "network.json", {}) or {}
    pairs = network.get("pairs", [])
    if not pairs:
        raise RuntimeError("尚未生成干涉网络")
    pair = pairs[0]
    wanted = p.get("dinsar_pair")
    if wanted:
        if isinstance(wanted, str):
            for candidate in pairs:
                if str(candidate.get("id")) == wanted:
                    pair = candidate
                    break
        elif isinstance(wanted, (list, tuple)) and len(wanted) == 2:
            key = tuple(sorted((int(wanted[0]), int(wanted[1]))))
            for candidate in pairs:
                if (int(candidate.get("ref")), int(candidate.get("sec"))) == key:
                    pair = candidate
                    break
    pid = str(pair["id"])
    pair_dir = root / "interferograms" / pid
    if progress:
        progress(0.08, f"D-InSAR：读取干涉对 {pid}")
    phase_path = pair_dir / "unwrapped.npy"
    corrected = pair_dir / "unwrapped_atmcorr.npy"
    if corrected.exists():
        phase = np.load(corrected, allow_pickle=False)
    elif phase_path.exists():
        phase = np.load(phase_path, allow_pickle=False)
    else:
        raise RuntimeError("干涉对尚未解缠")
    coh = np.load(pair_dir / "coherence.npy", allow_pickle=False)
    mask = np.load(pair_dir / "mask.npy", allow_pickle=False).astype(bool) & np.isfinite(phase)
    if mask.sum() < 20:
        raise RuntimeError("有效像元不足")
    k = 4.0 * np.pi / float(stack.meta["wavelength_m"])
    disp_mm = (phase * stack.meta["wavelength_m"] / (4.0 * np.pi) * 1000.0).astype(np.float32)
    # Re-reference the pair to the same stable pixel used by the rest of the
    # chain.  The raw phase is only referenced implicitly; without an explicit
    # gauge the map carries an arbitrary phase offset that shows up as a large
    # bias during validation and export.
    rr, cc = _reference_pixel(root, mask, p)
    if np.isfinite(disp_mm[rr, cc]):
        disp_mm = (disp_mm - disp_mm[rr, cc]).astype(np.float32)
    valid = mask & np.isfinite(disp_mm)
    products = root / "products"
    arrays = products / "arrays"
    geo = products / "geotiff"
    fig = products / "figures"
    for d in (arrays, geo, fig):
        d.mkdir(parents=True, exist_ok=True)
    np.save(arrays / "disp_pair_mm.npy", disp_mm, allow_pickle=False)
    np.save(arrays / "disp_ts_mm.npy", disp_mm[None, :, :], allow_pickle=False)
    np.save(arrays / "velocity_mm_yr.npy", np.full(disp_mm.shape, np.nan, dtype=np.float32), allow_pickle=False)
    np.save(arrays / "dem_error_m.npy", np.full(disp_mm.shape, np.nan, dtype=np.float32), allow_pickle=False)
    np.save(arrays / "temporal_coherence.npy", np.where(valid, coh, np.nan).astype(np.float32), allow_pickle=False)
    np.save(arrays / "residual_mm.npy", np.where(valid, 0.0, np.nan).astype(np.float32), allow_pickle=False)
    np.save(arrays / "valid.npy", valid, allow_pickle=False)
    m = max(1, int(p.get("multilook", 2)))
    grid = {
        "rows": int(valid.shape[0]),
        "cols": int(valid.shape[1]),
        "corner_lon": float(stack.meta.get("corner_lon", 0.0)),
        "corner_lat": float(stack.meta.get("corner_lat", 0.0)),
        "lon_step": float(stack.meta.get("lon_step", 0.0004)) * m,
        "lat_step": float(stack.meta.get("lat_step", -0.0004)) * m,
    }
    bounds = (
        grid["corner_lon"],
        grid["corner_lat"] + grid["rows"] * grid["lat_step"],
        grid["corner_lon"] + grid["cols"] * grid["lon_step"],
        grid["corner_lat"],
    )
    if bounds[1] > bounds[3]:
        bounds = (bounds[0], bounds[3], bounds[2], bounds[1])
    save_raster(
        disp_mm,
        fig / "dinsar_displacement.png",
        mask=valid,
        title=f"D-InSAR LOS displacement · {pair['ref_date']} → {pair['sec_date']}",
        subtitle=f"Δt {pair['dt_days']} d · B⊥ {pair['bperp_diff']:+.1f} m",
        unit="mm",
        divergent=True,
        symmetric=True,
        bounds=bounds,
    )
    save_raster(
        coh,
        fig / "dinsar_coherence.png",
        mask=valid,
        title="Interferometric coherence",
        unit="",
        vmin=0,
        vmax=1,
        bounds=bounds,
    )
    try:
        wrapped = np.load(pair_dir / "wrapped.npy", allow_pickle=False)
        save_wrapped_phase(wrapped, coh, fig / "dinsar_wrapped.png", mask=valid, title=f"Wrapped phase · {pid}")
    except Exception:
        pass
    summary = {
        "technique": "dinsar",
        "pair": pair,
        "n_dates": len(stack.meta["dates"]),
        "n_pairs": 1,
        "reference_pixel": [int(rr), int(cc)],
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.sum() / max(valid.size, 1)),
        "displacement_median_mm": float(np.nanmedian(disp_mm[valid])),
        "displacement_p05_mm": float(np.nanpercentile(disp_mm[valid], 5)),
        "displacement_p95_mm": float(np.nanpercentile(disp_mm[valid], 95)),
        "coherence_median": float(np.nanmedian(coh[valid])),
        "grid": grid,
        "bounds": bounds,
        "params": p,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(products / "inversion.json", summary)
    if progress:
        progress(0.97, f"D-InSAR 完成 · 中位位移 {summary['displacement_median_mm']:.2f} mm")
    return summary


def run_inversion(
    project_root: Path,
    technique: str = "sbas",
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    p = {**DEFAULT_PROCESSING, **(params or {})}
    stack = load_stack(root)
    network = read_json(root / "data" / "network.json")
    if not network or not network.get("pairs"):
        raise RuntimeError("尚未生成干涉网络")
    pairs = network["pairs"]
    technique = (technique or stack.meta.get("technique", "sbas")).lower()
    if technique == "dinsar":
        return run_dinsar(root, p, progress)
    if progress:
        progress(0.02, "载入干涉相位")
    phase, coh, mask, kept = load_pair_cube(root, pairs, corrected=True)
    if progress:
        progress(0.12, f"载入 {len(kept)} 个干涉对")

    extra_mask = None
    extra_weight = None
    extra: dict[str, Any] = {}
    if technique == "ps":
        from .ps import save_ps_figure, select_ps

        m = max(1, int(p.get("multilook", 1)))
        ps = select_ps(stack, coh, mask, {**p, "multilook": m})
        extra_mask = ps["mask"]
        extra_weight = ps["weight"]
        extra = {"ps": ps["stats"], **ps["stats"]}
        fig = save_ps_figure(ps, root)
        if fig:
            extra["figure"] = fig
    elif technique == "ds":
        linked = read_json(root / "data" / "processed" / "ds_reference.json", {}) or {}
        extra = {
            "ds_patch": int(p.get("ds_patch", 9)),
            "ds_phase_linking": "eigenvalue",
            "ds_reference_pixel": [linked.get("row"), linked.get("col")],
        }

    result = invert_arrays(
        root,
        phase,
        coh,
        mask,
        kept,
        stack.meta,
        p,
        extra_mask=extra_mask,
        extra_weight=extra_weight,
        progress=progress,
    )
    result["pairs"] = kept
    summary = _save_products(root, result, stack.meta, p, technique, extra=extra)
    if progress:
        progress(0.97, f"反演完成 · 中位速率 {summary['velocity_median_mm_yr']:.2f} mm/yr")
    return summary


def extract_series(project_root: Path, row: int, col: int) -> dict[str, Any]:
    root = Path(project_root)
    arrays = root / "products" / "arrays"
    meta = read_json(root / "data" / "stack" / "meta.json")
    inv = read_json(root / "products" / "inversion.json")
    disp = np.load(arrays / "disp_ts_mm.npy", allow_pickle=False)
    valid = np.load(arrays / "valid.npy", allow_pickle=False)
    if not (0 <= row < disp.shape[1] and 0 <= col < disp.shape[2]):
        raise IndexError("像元超出范围")
    return {
        "row": int(row),
        "col": int(col),
        "dates": meta["dates"],
        "disp_mm": disp[:, row, col].tolist(),
        "valid": bool(valid[row, col]),
        "velocity_mm_yr": float(np.load(arrays / "velocity_mm_yr.npy", allow_pickle=False)[row, col]),
        "temporal_coherence": float(np.load(arrays / "temporal_coherence.npy", allow_pickle=False)[row, col]),
        "inversion": inv or {},
    }
