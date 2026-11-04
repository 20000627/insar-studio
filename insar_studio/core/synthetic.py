"""Reproducible synthetic Sentinel-1-like stack.

The demo is not a toy interferogram: it contains a realistic mixture of

* spatially varying linear + seasonal + transient deformation,
* topography-correlated and turbulent atmospheric delay,
* residual DEM error proportional to the perpendicular baseline,
* orbit ramps, decorrelation, water, and sub-pixel coregistration shifts.

The processing chain can therefore be validated against ground truth while
the same code paths are used for real stacks.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import (
    DEFAULT_AZIMUTH_SPACING,
    DEFAULT_INCIDENCE,
    DEFAULT_RANGE_SPACING,
    DEMO_COLS,
    DEMO_INTERVAL_DAYS,
    DEMO_N_DATES,
    DEMO_ROWS,
    DEMO_START,
    S1_WAVELENGTH,
)
from .utils import atomic_write_json, downsample_complex, wrap_phase


def _normalise(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    std = a.std()
    if std < 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / std


def _terrain(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1)
    yn = y / max(rows - 1, 1)
    dem = (
        110.0
        + 95.0 * np.exp(-(((xn - 0.28) ** 2) / 0.035 + ((yn - 0.35) ** 2) / 0.08))
        + 130.0 * np.exp(-(((xn - 0.72) ** 2) / 0.05 + ((yn - 0.62) ** 2) / 0.12))
        + 55.0 * np.exp(-(((xn - 0.58) ** 2) / 0.012 + ((yn - 0.22) ** 2) / 0.03))
        + 22.0 * np.sin(4.0 * np.pi * xn + 0.6) * np.cos(3.0 * np.pi * yn)
        + 8.0 * _normalise(ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 18))
    )
    # a river valley
    river = np.exp(-(((yn - (0.52 + 0.18 * np.sin(3.0 * np.pi * xn))) ** 2) / 0.0012))
    dem -= 38.0 * river
    return dem.astype(np.float32)


def _water_and_veg(rows: int, cols: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1)
    yn = y / max(rows - 1, 1)
    water = ((((xn - 0.79) ** 2) / 0.010 + ((yn - 0.75) ** 2) / 0.005) < 1.0)
    river = np.abs(yn - (0.52 + 0.18 * np.sin(3.0 * np.pi * xn))) < 0.018
    water |= river
    veg = ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 9)
    veg = (veg > np.percentile(veg, 55)).astype(np.float32)
    veg = ndimage.gaussian_filter(veg, 4)
    veg[water] = 0.0
    return water, np.clip(veg, 0, 1).astype(np.float32)


def _deformation(
    rows: int,
    cols: int,
    dates: list[date],
    sampling_days: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1)
    yn = y / max(rows - 1, 1)

    # LOS velocity, mm/yr (positive towards the satellite)
    vel = (
        -7.2
        + 4.8 * (xn - 0.5)
        - 21.0 * np.exp(-(((xn - 0.68) ** 2) / 0.020 + ((yn - 0.32) ** 2) / 0.030))
        + 7.0 * np.exp(-(((xn - 0.32) ** 2) / 0.030 + ((yn - 0.70) ** 2) / 0.045))
        + 2.4 * np.sin(2.0 * np.pi * yn + 0.7)
    ).astype(np.float32)

    seasonal_amp = (2.1 + 1.1 * xn + 0.8 * np.cos(2.0 * np.pi * yn)).astype(np.float32)
    doy = np.asarray([d.timetuple().tm_yday for d in dates], dtype=np.float64)
    seasonal = np.sin(2.0 * np.pi * (doy - 78.0) / 365.25)

    # A slow landslide activation on the western slope
    event_day = 198.0
    event = 1.0 / (1.0 + np.exp(-(sampling_days - event_day) / 15.0))
    landslide_mask = np.exp(
        -(((xn - 0.235) ** 2) / 0.0045 + ((yn - 0.635) ** 2) / 0.017)
    ).astype(np.float32)
    # small injection-like transient near the centre
    transient_map = 5.5 * np.exp(
        -(((xn - 0.52) ** 2) / 0.006 + ((yn - 0.47) ** 2) / 0.010)
    )

    ty = sampling_days / 365.25
    n = len(dates)
    disp = np.empty((n, rows, cols), dtype=np.float32)
    for i in range(n):
        d = (
            vel * ty[i]
            + seasonal_amp * seasonal[i]
            + 21.0 * landslide_mask * event[i]
            + transient_map * np.clip((sampling_days[i] - 238.0) / 70.0, 0.0, 1.0)
        )
        disp[i] = d.astype(np.float32)

    info = {
        "event_day": event_day,
        "landslide_amplitude_mm": 21.0,
        "velocity_range_mm_yr": [float(np.nanmin(vel)), float(np.nanmax(vel))],
        "seasonal_amplitude_mm": float(np.nanmax(seasonal_amp)),
    }
    return disp, vel, info


def _baselines(n: int, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, 17.0, size=n)
    b = np.cumsum(steps)
    b -= b[0]
    # keep the stack within a realistic small-baseline envelope
    scale = 145.0 / max(np.max(np.abs(b)), 1.0)
    b *= scale
    return b.astype(np.float32)


def _atmosphere(
    n: int,
    rows: int,
    cols: int,
    dem: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty((n, rows, cols), dtype=np.float32)
    dem_n = _normalise(dem)
    for i in range(n):
        turb = ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 15)
        turb += 0.45 * ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 4)
        turb = _normalise(turb) * rng.uniform(2.2, 6.0)
        stratified = rng.uniform(-2.6, 2.6) * dem_n
        # long-wavelength delay screen
        y, x = np.mgrid[0:rows, 0:cols]
        wave = 1.2 * np.sin(2 * np.pi * x / (cols * rng.uniform(1.8, 3.2)) + rng.uniform(0, 6.28))
        out[i] = (turb + stratified + wave).astype(np.float32)
    # remove the per-date mean: interferometric atmosphere is differential
    out -= out.mean(axis=(1, 2), keepdims=True)
    return out


def _orbit_ramp(n: int, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1) - 0.5
    yn = y / max(rows - 1, 1) - 0.5
    out = np.empty((n, rows, cols), dtype=np.float32)
    for i in range(n):
        a = rng.normal(0.0, 0.22)
        b = rng.normal(0.0, 0.22)
        c = rng.normal(0.0, 0.10)
        out[i] = (a * xn + b * yn + c * xn * yn).astype(np.float32)
    out -= out.mean(axis=(1, 2), keepdims=True)
    return out


def generate_demo_dataset(
    root: Path,
    *,
    rows: int = DEMO_ROWS,
    cols: int = DEMO_COLS,
    n_dates: int = DEMO_N_DATES,
    interval_days: int = DEMO_INTERVAL_DAYS,
    start: str = DEMO_START,
    seed: int = 20260317,
    wavelength: float = S1_WAVELENGTH,
    incidence: float = DEFAULT_INCIDENCE,
) -> dict[str, Any]:
    """Create a complete synthetic stack on disk and return its metadata."""
    root = Path(root)
    stack_dir = root / "data" / "stack"
    stack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = [date.fromisoformat(start) + timedelta(days=i * interval_days) for i in range(n_dates)]
    sampling = np.asarray([(d - dates[0]).days for d in dates], dtype=np.float64)
    dem_true = _terrain(rows, cols, rng)
    # External DEM has a smooth vertical bias (vegetation / geoid / resampling)
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1)
    yn = y / max(rows - 1, 1)
    dem_bias = 7.5 + 2.8 * np.sin(2 * np.pi * xn + 0.4) + 1.6 * np.cos(2 * np.pi * yn)
    dem_ref = (dem_true + dem_bias).astype(np.float32)

    water, veg = _water_and_veg(rows, cols, rng)
    disp, vel, def_info = _deformation(rows, cols, dates, sampling, rng)
    bperp = _baselines(n_dates, rng)
    atm = _atmosphere(n_dates, rows, cols, dem_true, rng)
    ramp = _orbit_ramp(n_dates, rows, cols, rng)

    slant_range = 705_000.0
    sin_inc = np.sin(np.deg2rad(incidence))
    k = 4.0 * np.pi / wavelength

    # amplitude: terrain shading + texture + water
    gy, gx = np.gradient(dem_true)
    shade = 0.65 + 0.35 * np.tanh(0.9 * (-0.7 * gx - 0.4 * gy) / (np.abs(gx).mean() + 1e-6))
    texture = 0.72 + 0.28 * _normalise(ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 2))
    amp_base = shade * texture
    amp_base[water] *= 0.18
    amp_base += 0.04 * rng.normal(size=(rows, cols))

    # per-date phase noise increases in vegetation and over water
    sigma_phase = 0.10 + 0.42 * veg + 1.15 * water.astype(np.float32)
    sigma_phase = sigma_phase.astype(np.float32)

    shifts = np.zeros((n_dates, 2), dtype=np.float32)
    for i in range(1, n_dates):
        shifts[i, 0] = np.clip(rng.normal(0.0, 0.85), -2.2, 2.2)
        shifts[i, 1] = np.clip(rng.normal(0.0, 0.75), -2.0, 2.0)

    slc = np.empty((n_dates, rows, cols), dtype=np.complex64)
    topo_phase = np.empty((n_dates, rows, cols), dtype=np.float32)
    for i in range(n_dates):
        topo = k * bperp[i] * dem_true / (slant_range * sin_inc)
        phase = (
            topo
            + k * disp[i] / 1000.0
            + k * atm[i] / 1000.0
            + ramp[i]
            + rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32) * sigma_phase
        )
        # occasional strongly decorrelated acquisition (rain / ionosphere)
        if i == 9:
            phase += rng.normal(0.0, 0.55, size=(rows, cols)).astype(np.float32)
        amp = amp_base * (1.0 + 0.10 * rng.normal(size=(rows, cols)))
        img = (amp * np.exp(1j * phase)).astype(np.complex64)
        if i > 0:
            img = ndimage.shift(
                img.real, shift=(shifts[i, 1], shifts[i, 0]), order=3, mode="nearest"
            ) + 1j * ndimage.shift(
                img.imag, shift=(shifts[i, 1], shifts[i, 0]), order=3, mode="nearest"
            )
        slc[i] = img.astype(np.complex64)
        topo_phase[i] = topo.astype(np.float32)

    # valid data mask: water and a two-pixel border are excluded
    valid = ~water
    valid[:2, :] = valid[-2:, :] = False
    valid[:, :2] = valid[:, -2:] = False

    # true mean coherence (for diagnostics / demo GACOS)
    coh_true = np.empty((n_dates, rows, cols), dtype=np.float32)
    tau = np.where(water, 22.0, 95.0)
    base_coh = np.where(water, 0.28, 0.86 - 0.26 * veg)
    for i in range(n_dates):
        coh_true[i] = base_coh * np.exp(-np.abs(sampling[i]) / tau)

    # ---- write stack ----
    np.save(stack_dir / "slc.npy", slc, allow_pickle=False)
    np.save(stack_dir / "dem.npy", dem_ref, allow_pickle=False)
    np.save(stack_dir / "dem_true.npy", dem_true, allow_pickle=False)
    np.save(stack_dir / "mask.npy", valid, allow_pickle=False)
    np.save(stack_dir / "water.npy", water, allow_pickle=False)
    np.savez_compressed(
        stack_dir / "truth.npz",
        disp_los_mm=disp,
        velocity_mm_yr=vel,
        dem_true=dem_true,
        dem_ref=dem_ref,
        atmosphere_mm=atm,
        coherence_true=coh_true,
        shifts=shifts,
        bperp=bperp,
        sampling_days=sampling,
    )

    meta: dict[str, Any] = {
        "schema": "insar-studio/stack/v1",
        "name": "Synthetic Alpine Valley",
        "source": "demo",
        "demo": True,
        "created_at": date.today().isoformat(),
        "platform": "Sentinel-1A",
        "orbit_direction": "DESCENDING",
        "relative_orbit": 42,
        "rows": rows,
        "cols": cols,
        "n_dates": n_dates,
        "dates": [d.isoformat() for d in dates],
        "sampling_days": sampling.tolist(),
        "wavelength_m": wavelength,
        "incidence_deg": incidence,
        "range_spacing_m": DEFAULT_RANGE_SPACING,
        "azimuth_spacing_m": DEFAULT_AZIMUTH_SPACING,
        "slant_range_m": slant_range,
        "bperp_m": bperp.tolist(),
        "slc_path": "data/stack/slc.npy",
        "dem_path": "data/stack/dem.npy",
        "mask_path": "data/stack/mask.npy",
        "water_path": "data/stack/water.npy",
        "truth_path": "data/stack/truth.npz",
        "corner_lon": 106.52,
        "corner_lat": 29.63,
        "lon_step": 0.00042,
        "lat_step": -0.00036,
        "crs": "EPSG:4326",
        "ref_index": 0,
        "units": {"slc": "complex digital number", "dem": "metre", "displacement": "mm"},
        "notes": "Synthetic stack with known deformation, atmosphere and DEM error. Built for validation and training.",
        "demo_truth": def_info,
        "pixel_area_m2": DEFAULT_RANGE_SPACING * DEFAULT_AZIMUTH_SPACING,
    }
    atomic_write_json(stack_dir / "meta.json", meta)

    # A compact preview (amplitude of the reference image) for the data page
    try:
        from .plotting import save_amplitude_preview

        save_amplitude_preview(
            np.abs(slc[0]),
            valid,
            root / "data" / "preview_amplitude.png",
            title="Synthetic amplitude (reference)",
        )
    except Exception:
        pass
    return meta


def load_truth(root: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(root) / meta.get("truth_path", "data/stack/truth.npz")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
