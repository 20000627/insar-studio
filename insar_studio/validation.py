"""Validation against the synthetic truth (demo projects only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.io import load_stack
from .core.utils import atomic_write_json, downsample_real, read_json


def _slope_velocity(disp: np.ndarray, days: np.ndarray) -> np.ndarray:
    t = np.asarray(days, dtype=np.float64)
    t = t - t.mean()
    y = disp.reshape(disp.shape[0], -1)
    v = (t[:, None] * y).sum(axis=0) / max(float((t**2).sum()), 1e-9) * 365.25
    return v.reshape(disp.shape[1:])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if v.size else float("nan")


def validate_demo(project_root: Path, *, force: bool = False) -> dict[str, Any] | None:
    root = Path(project_root)
    truth_path = root / "data" / "stack" / "truth.npz"
    inv = read_json(root / "products" / "inversion.json", {}) or {}
    if not truth_path.exists() or not inv:
        return None
    cache = root / "products" / "validation.json"
    if cache.exists() and not force:
        cached = read_json(cache, {}) or {}
        if cached.get("created_at") and cached.get("technique") == inv.get("technique"):
            return cached
    try:
        stack = load_stack(root)
        m = max(1, int(inv.get("params", {}).get("multilook", 2)))
        with np.load(truth_path) as truth:
            disp_true = truth["disp_los_mm"].astype(np.float32)
            dem_true = (truth["dem_true"] - truth["dem_ref"]).astype(np.float32)
            atm_true = truth["atmosphere_mm"].astype(np.float32)
            bperp = truth["bperp"].astype(np.float32)
        days = np.asarray(stack.meta.get("sampling_days", []), dtype=np.float64)
        if len(days) != len(stack.dates):
            days = np.arange(len(stack.dates), dtype=np.float64) * 12.0
        technique = str(inv.get("technique", "sbas")).lower()
        base = {
            "technique": technique,
            "truth": "synthetic",
            "note": "合成数据真值对比，仅用于演示与算法回归，不代表真实场景精度。",
            "created_at": inv.get("created_at", ""),
        }
        if technique == "dinsar":
            pair = inv.get("pair", {})
            ref_i = int(pair.get("ref", 0))
            sec_i = int(pair.get("sec", 1))
            arr = np.load(root / "products" / "arrays" / "disp_pair_mm.npy", allow_pickle=False)
            d_true = downsample_real(disp_true[sec_i] - disp_true[ref_i], m)[: arr.shape[0], : arr.shape[1]]
            atm = read_json(root / "data" / "atmosphere" / "atmosphere.json", {}) or {}
            if atm.get("method") == "off":
                d_true = d_true + downsample_real(atm_true[sec_i] - atm_true[ref_i], m)[: arr.shape[0], : arr.shape[1]]
            # The pair product is re-referenced to a stable pixel; the truth
            # must be re-referenced identically before differencing.
            ref = inv.get("reference_pixel") or [d_true.shape[0] // 2, d_true.shape[1] // 2]
            rr = min(max(int(ref[0]), 0), d_true.shape[0] - 1)
            cc = min(max(int(ref[1]), 0), d_true.shape[1] - 1)
            d_true = d_true - d_true[rr, cc]
            valid = np.load(root / "products" / "arrays" / "valid.npy", allow_pickle=False).astype(bool)
            diff = arr - d_true
            base.update(
                {
                    "valid_pixels": int(valid.sum()),
                    "reference_pixel": [int(rr), int(cc)],
                    "displacement_rmse_mm": _rmse(diff[valid]),
                    "displacement_bias_mm": float(np.nanmean(diff[valid])) if np.isfinite(diff[valid]).any() else float("nan"),
                    "displacement_p95_mm": float(np.nanpercentile(np.abs(diff[valid]), 95)) if np.isfinite(diff[valid]).any() else float("nan"),
                    "displacement_correlation": _safe_corr(arr[valid], d_true[valid]),
                }
            )
            rmse = base.get("displacement_rmse_mm", 0.0)
            base["grade"] = "优秀" if rmse < 3 else ("良好" if rmse < 6 else ("可用" if rmse < 12 else "需检查"))
            atomic_write_json(cache, base)
            return base

        disp = np.load(root / "products" / "arrays" / "disp_ts_mm.npy", allow_pickle=False)
        valid = np.load(root / "products" / "arrays" / "valid.npy", allow_pickle=False).astype(bool)
        vel = np.load(root / "products" / "arrays" / "velocity_mm_yr.npy", allow_pickle=False)
        n = disp.shape[0]
        td = np.stack([downsample_real(disp_true[i] - disp_true[0], m) for i in range(n)])
        td = td[:, : disp.shape[1], : disp.shape[2]]
        # The product is spatially referenced to a stable pixel: subtract the
        # truth history of the same pixel before computing any metric.
        ref = inv.get("reference_pixel") or [td.shape[1] // 2, td.shape[2] // 2]
        rr = min(max(int(ref[0]), 0), td.shape[1] - 1)
        cc = min(max(int(ref[1]), 0), td.shape[2] - 1)
        td = td - td[:, rr, cc][:, None, None]
        diff = disp - td
        v_true = _slope_velocity(td, days[:n])
        dvec = valid
        vdiff = vel - v_true
        tvals = []
        for i in range(n):
            e = diff[i][dvec]
            e = e[np.isfinite(e)]
            tvals.append(float(np.sqrt(np.mean(e**2))) if e.size else 0.0)
        dvals = diff[:, dvec]
        dvals = dvals[np.isfinite(dvals)]
        base.update(
            {
                "valid_pixels": int(valid.sum()),
                "displacement_rmse_mm": _rmse(dvals),
                "displacement_bias_mm": float(np.mean(dvals)) if dvals.size else float("nan"),
                "displacement_p95_mm": float(np.percentile(np.abs(dvals), 95)) if dvals.size else float("nan"),
                "per_date_rmse_mm": tvals,
                "velocity_rmse_mm_yr": _rmse(vdiff[dvec]),
                "velocity_bias_mm_yr": float(np.nanmean(vdiff[dvec])) if np.isfinite(vdiff[dvec]).any() else float("nan"),
                "velocity_correlation": _safe_corr(vel[dvec], v_true[dvec]),
                "velocity_truth_median_mm_yr": float(np.nanmedian(v_true[dvec])) if np.isfinite(v_true[dvec]).any() else float("nan"),
                "velocity_est_median_mm_yr": float(np.nanmedian(vel[dvec])) if np.isfinite(vel[dvec]).any() else float("nan"),
            }
        )
        dem = np.load(root / "products" / "arrays" / "dem_error_m.npy", allow_pickle=False)
        if np.isfinite(dem).any():
            dt = downsample_real(dem_true, m)[: dem.shape[0], : dem.shape[1]]
            ok = dvec & np.isfinite(dem) & np.isfinite(dt)
            if ok.sum() > 10 and float(np.std(dem[ok])) > 1e-6 and float(np.std(dt[ok])) > 1e-6:
                base["dem_error_median_m"] = float(np.nanmedian(dem[ok]))
                base["dem_truth_median_m"] = float(np.nanmedian(dt[ok]))
                corr = _safe_corr(dem[ok], dt[ok])
                if corr is not None:
                    base["dem_correlation"] = corr
        atm = read_json(root / "data" / "atmosphere" / "atmosphere.json", {}) or {}
        if atm.get("validation"):
            base["atmosphere"] = atm["validation"]
        # A compact, honest headline grade: displacement RMSE against the
        # centimetre-scale signal, not a made-up "accuracy" claim.
        rmse = base.get("displacement_rmse_mm", 0.0)
        base["grade"] = "优秀" if rmse < 3 else ("良好" if rmse < 6 else ("可用" if rmse < 12 else "需检查"))
        atomic_write_json(cache, base)
        return base
    except Exception:
        return None
