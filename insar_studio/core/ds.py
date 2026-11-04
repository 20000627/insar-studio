"""Distributed-scatterer (DS) phase linking.

The implementation estimates a complex coherence matrix in a local window,
extracts the dominant eigenvector (the EMI / phase-linking solution), and uses
the linked phase to rebuild every interferometric pair.  This suppresses
speckle without the resolution loss of a large boxcar and is well suited to
vegetated or bare-soil surfaces.

The processing is deliberately explicit so that every intermediate product can
be inspected: covariance matrix, linked phase, DS coherence and the rebuilt
pair stack.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from .io import load_stack
from .plotting import save_raster
from .unwrapping import unwrap
from .utils import (
    atomic_write_json,
    downsample_complex,
    downsample_real,
    read_json,
    stable_reference,
    wrap_phase,
)

Progress = Callable[[float, str], None]


def _filter_covariance(prod: np.ndarray, size: int) -> np.ndarray:
    real = ndimage.uniform_filter(prod.real, size=size, mode="nearest")
    imag = ndimage.uniform_filter(prod.imag, size=size, mode="nearest")
    return (real + 1j * imag).astype(np.complex64)


def phase_link(
    coreg: np.ndarray,
    stack_meta: dict[str, Any],
    valid_full: np.ndarray,
    *,
    multilook: int = 2,
    patch: int = 9,
    ref_index: int = 0,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Estimate linked DS phase from a coregistered SLC stack."""
    n = int(coreg.shape[0])
    if n < 6:
        raise ValueError("DS 相位链接至少需要 6 个历元")
    m = max(1, int(multilook))
    patch = max(3, int(patch) | 1)
    k = 4.0 * np.pi / float(stack_meta["wavelength_m"])
    bperp = np.asarray(stack_meta.get("bperp_m", [0.0] * n), dtype=np.float64)
    dem = downsample_real(np.asarray(load_stack_shape_dem(stack_meta)), m)
    # The DEM is passed separately by the caller through stack_meta["_dem"];
    # fall back to zeros if it is unavailable.
    dem = np.asarray(stack_meta.get("_dem", dem), dtype=np.float32)
    if dem.ndim == 2:
        dem = downsample_real(dem, m)
    sin_inc = np.sin(np.deg2rad(float(stack_meta["incidence_deg"])))
    slant = float(stack_meta.get("slant_range_m", 705000.0))

    rows = coreg.shape[1] // m * m
    cols = coreg.shape[2] // m * m
    slc = np.empty((n, rows // m, cols // m), dtype=np.complex64)
    if progress:
        progress(0.03, "DS：去除地形相位并多视")
    for i in range(n):
        topo = k * bperp[i] * np.asarray(stack_meta["_dem_full"], dtype=np.float32) / (slant * sin_inc)
        z = np.asarray(coreg[i, :rows, :cols]) * np.exp(-1j * topo[:rows, :cols])
        slc[i] = downsample_complex(z.astype(np.complex64), m)
        if progress and i % max(1, n // 6) == 0:
            progress(0.05 + 0.15 * i / n, f"DS：地形相位校正 {i + 1}/{n}")

    diag = np.empty((n, rows // m, cols // m), dtype=np.float32)
    for i in range(n):
        diag[i] = _filter_covariance(slc[i] * np.conj(slc[i]), patch).real
    amp_ok = np.min(diag, axis=0) > 1e-8
    base = downsample_real(valid_full.astype(np.float32), m) > 0.55
    base = base[: rows // m, : cols // m] & amp_ok

    coh = np.empty((n, n, rows // m, cols // m), dtype=np.complex64)
    if progress:
        progress(0.22, "DS：估计复数相干矩阵")
    for i in range(n):
        coh[i, i] = 1.0
        for j in range(i + 1, n):
            prod = slc[i] * np.conj(slc[j])
            c = _filter_covariance(prod, patch)
            c /= np.maximum(np.sqrt(diag[i] * diag[j]), 1e-9)
            coh[i, j] = c
            coh[j, i] = np.conj(c)
        if progress:
            progress(0.22 + 0.42 * (i + 1) / n, f"DS：相干矩阵 {i + 1}/{n}")
    coh = np.clip(coh, -1, 1, out=coh)

    flat_base = base.ravel()
    idx = np.flatnonzero(flat_base)
    rows2, cols2 = base.shape
    linked = np.zeros((n, rows2, cols2), dtype=np.complex64)
    ds_coh = np.zeros((rows2, cols2), dtype=np.float32)
    if idx.size == 0:
        raise RuntimeError("DS 有效像元不足，请降低掩膜阈值")

    chunk = 2048
    if progress:
        progress(0.68, f"DS：特征分解 · {idx.size:,} 个像元")
    for start in range(0, idx.size, chunk):
        sel = idx[start : start + chunk]
        rr, cc = np.unravel_index(sel, (rows2, cols2))
        mats = coh[:, :, rr, cc].transpose(2, 0, 1).astype(np.complex128)
        vals, vecs = np.linalg.eigh(mats)
        v = vecs[:, :, -1]
        theta = np.angle(v)
        theta = theta - theta[:, ref_index : ref_index + 1]
        linked[:, rr, cc] = np.exp(1j * theta).T.astype(np.complex64)
        eig = vals[:, -1]
        trace = vals.sum(axis=1)
        ds_coh[rr, cc] = np.clip(eig / np.maximum(trace, 1e-9), 0, 1).astype(np.float32)
        if progress:
            progress(0.68 + 0.22 * min(1.0, (start + chunk) / idx.size), f"DS：特征分解 {min(idx.size, start + chunk):,}/{idx.size:,}")

    return {
        "linked": linked,
        "coherence_matrix": coh,
        "ds_coherence": ds_coh,
        "base_mask": base,
        "multilook": m,
        "patch": patch,
        "shape": (rows2, cols2),
    }


def load_stack_shape_dem(stack_meta: dict[str, Any]) -> np.ndarray:
    dem = stack_meta.get("_dem")
    if dem is not None:
        return np.asarray(dem)
    return np.zeros((int(stack_meta.get("rows", 1)), int(stack_meta.get("cols", 1))), dtype=np.float32)


def run_ds_interferometry(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Phase-link the stack and rebuild the network interferograms."""
    root = Path(project_root)
    p = dict(params or {})
    stack = load_stack(root)
    coreg_path = root / "data" / "processed" / "slc_coreg.npy"
    if not coreg_path.exists():
        raise FileNotFoundError("请先完成 SLC 配准")
    coreg = np.load(coreg_path, mmap_mode="r", allow_pickle=False)
    network = read_json(root / "data" / "network.json", {}) or {}
    pairs = network.get("pairs", [])
    if not pairs:
        raise FileNotFoundError("尚未生成干涉网络")
    m = max(1, int(p.get("multilook", 2)))
    ref_index = int(p.get("ref_index", network.get("stats", {}).get("ref_index", 0)))
    stack_meta = dict(stack.meta)
    stack_meta["_dem_full"] = np.asarray(stack.dem, dtype=np.float32)
    if progress:
        progress(0.01, "DS 相位链接：准备数据")
    link = phase_link(
        coreg,
        stack_meta,
        stack.mask,
        multilook=m,
        patch=int(p.get("ds_patch", 9)),
        ref_index=ref_index,
        progress=progress,
    )
    linked = link["linked"]
    coh_mat = link["coherence_matrix"]
    ds_coh = link["ds_coherence"]
    base = link["base_mask"]
    ds_min = float(p.get("ds_coherence_min", 0.18))
    mask_base = base & (ds_coh >= ds_min)
    if mask_base.sum() < 50:
        mask_base = base
    out_dir = root / "data" / "processed"
    np.save(out_dir / "ds_linked.npy", linked, allow_pickle=False)
    np.save(out_dir / "ds_coherence.npy", ds_coh, allow_pickle=False)

    # a common reference point for all pairs
    ref_coh = np.abs(coh_mat[0, -1]) if len(pairs) else ds_coh
    rr, cc = stable_reference(ds_coh, mask_base)
    # a high-quality point near the scene centre is more stable than a random
    # maximum; fall back to the DS coherence maximum only when needed.
    if not mask_base[rr, cc]:
        rr, cc = np.unravel_index(int(np.nanargmax(np.where(mask_base, ds_coh, -1))), ds_coh.shape)
    atomic_write_json(root / "data" / "processed" / "ds_reference.json", {"row": int(rr), "col": int(cc)})

    k = 4.0 * np.pi / float(stack.meta["wavelength_m"])
    metas = []
    t0 = time.time()
    filter_name = str(p.get("filter", "boxcar")).lower()
    for idx, pair in enumerate(pairs):
        if progress:
            progress(0.90 + 0.06 * idx / max(len(pairs), 1), f"DS 重建干涉对 {idx + 1}/{len(pairs)}")
        sec, ref = int(pair["sec"]), int(pair["ref"])
        wrapped = np.angle(linked[sec] * np.conj(linked[ref])).astype(np.float32)
        coh = np.abs(coh_mat[sec, ref]).astype(np.float32)
        # linked phase is already spatially regularised; a light boxcar only
        # removes residual pixel-to-pixel noise.
        if filter_name in ("boxcar", "gaussian", "goldstein", "nlmeans", "nlm"):
            from .filters import filter_complex

            z = np.exp(1j * wrapped)
            z = filter_complex(z, "boxcar", size=int(p.get("ds_filter_size", 3)))
            wrapped = np.angle(z).astype(np.float32)
        valid = mask_base & (coh >= float(p.get("min_coherence", 0.18))) & np.isfinite(wrapped)
        if valid.sum() < 20:
            valid = mask_base & np.isfinite(wrapped)
        unwrapped, unwrap_info = unwrap(
            wrapped,
            coh,
            valid,
            method=p.get("unwrap_method", "quality"),
            coherence_min=float(p.get("unwrap_coherence_min", 0.12)),
            snaphu_exe=str(p.get("snaphu_exe", "snaphu")),
        )
        if np.isfinite(unwrapped[rr, cc]):
            unwrapped = unwrapped - unwrapped[rr, cc]
        else:
            r2, c2 = stable_reference(coh, valid)
            if np.isfinite(unwrapped[r2, c2]):
                unwrapped = unwrapped - unwrapped[r2, c2]
        unwrapped = unwrapped.astype(np.float32)
        if p.get("remove_ramp", False):
            try:
                from .interferometry import _remove_plane

                unwrapped, ramp_info = _remove_plane(unwrapped, coh, valid)
            except Exception:
                ramp_info = {"removed": False}
        else:
            ramp_info = {"removed": False}
        disp = (unwrapped * stack.meta["wavelength_m"] / (4.0 * np.pi) * 1000.0).astype(np.float32)
        valid &= np.isfinite(unwrapped)
        pair_dir = root / "interferograms" / pair["id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        np.save(pair_dir / "wrapped.npy", wrapped, allow_pickle=False)
        np.save(pair_dir / "coherence.npy", coh, allow_pickle=False)
        np.save(pair_dir / "unwrapped.npy", unwrapped, allow_pickle=False)
        np.save(pair_dir / "displacement_mm.npy", disp, allow_pickle=False)
        np.save(pair_dir / "mask.npy", valid, allow_pickle=False)
        meta = {
            **pair,
            "rows": int(wrapped.shape[0]),
            "cols": int(wrapped.shape[1]),
            "multilook": m,
            "wavelength_m": float(stack.meta["wavelength_m"]),
            "incidence_deg": float(stack.meta["incidence_deg"]),
            "ref_pixel": [int(rr), int(cc)],
            "topo_removed": True,
            "ramp": ramp_info,
            "unwrap": unwrap_info,
            "technique": "ds",
            "ds": {"patch": int(p.get("ds_patch", 9)), "phase_linking": "eigenvalue"},
            "stats": {
                "coherence_mean": float(np.nanmean(coh[valid])) if valid.any() else 0.0,
                "coherence_p90": float(np.nanpercentile(coh[valid], 90)) if valid.any() else 0.0,
                "valid_fraction": float(valid.sum() / max(base.sum(), 1)),
                "unwrapped_fraction": float(np.isfinite(unwrapped[valid]).sum() / max(valid.sum(), 1)),
                "displacement_p05_mm": float(np.nanpercentile(disp[valid], 5)) if valid.any() else 0.0,
                "displacement_p95_mm": float(np.nanpercentile(disp[valid], 95)) if valid.any() else 0.0,
            },
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(pair_dir / "meta.json", meta)
        metas.append(meta)

    from .interferometry import _closure_stats

    closure = _closure_stats(root, pairs, metas)
    summary = {
        "n_pairs": len(metas),
        "multilook": m,
        "technique": "ds",
        "reference_pixel": [int(rr), int(cc)],
        "reference_info": {"row": int(rr), "col": int(cc), "ds_coherence": float(ds_coh[rr, cc])},
        "mean_coherence": float(np.mean([x["stats"]["coherence_mean"] for x in metas])) if metas else 0.0,
        "mean_valid_fraction": float(np.mean([x["stats"]["valid_fraction"] for x in metas])) if metas else 0.0,
        "ds_coherence_median": float(np.nanmedian(ds_coh[mask_base])) if mask_base.any() else 0.0,
        "ds_pixels": int(mask_base.sum()),
        "ds_patch": int(p.get("ds_patch", 9)),
        "closure": closure,
        "params": p,
        "elapsed_s": round(time.time() - t0, 2),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(root / "interferograms" / "summary.json", summary)
    try:
        save_raster(
            ds_coh,
            root / "products" / "figures" / "ds_coherence.png",
            mask=mask_base,
            title="DS phase-linking coherence",
            subtitle="Largest eigenvalue / trace of the local coherence matrix",
            unit="",
            vmin=0,
            vmax=1,
            figsize=(5.2, 4.0),
        )
    except Exception:
        pass
    if progress:
        progress(0.98, f"DS 干涉完成 · {len(metas)} 对 · 相位链接像元 {mask_base.sum():,}")
    return summary
