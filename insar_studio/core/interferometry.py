"""Interferogram formation, coherence, filtering and unwrapping."""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from ..config import DEFAULT_PROCESSING
from .filters import filter_complex
from .io import load_stack, safe_join
from .memory import MIB, choose_block_rows
from .memory import ensure_memory_budget, estimate_interferometry_peak
from .plotting import save_raster, save_wrapped_phase
from .unwrapping import unwrap
from .utils import (
    atomic_write_json,
    downsample_complex_pair,
    downsample_real,
    json_safe,
    read_json,
    stable_reference,
    wrap_phase,
)


Progress = Callable[[float, str], None]


def _downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.copy()
    return downsample_real(mask.astype(np.float32), factor) > 0.55


def _coherence(ifg: np.ndarray, ref_amp: np.ndarray, sec_amp: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)
    num = ndimage.uniform_filter(ifg.real, w) + 1j * ndimage.uniform_filter(ifg.imag, w)
    p1 = ndimage.uniform_filter(ref_amp**2, w)
    p2 = ndimage.uniform_filter(sec_amp**2, w)
    coh = np.abs(num) / np.sqrt(np.maximum(p1 * p2, 1e-12))
    return np.clip(coh, 0, 1).astype(np.float32)


def reference_pixel(
    coreg: np.ndarray,
    mask: np.ndarray,
    multilook: int,
    *,
    ref_index: int = 0,
) -> tuple[int, int, dict[str, Any]]:
    """Select a stable, high-quality reference pixel near the scene centre."""
    n = coreg.shape[0]
    step = max(1, n // 8)
    # Accumulate first and second amplitude moments instead of stacking up to
    # eight full-resolution images.  On a 10000 x 10000 scene the old stack
    # alone needed about 3.2 GiB.
    mean_amp = np.zeros(coreg.shape[1:], dtype=np.float32)
    mean_sq = np.zeros(coreg.shape[1:], dtype=np.float32)
    count = 0
    for i in range(0, n, step):
        amp = np.abs(np.asarray(coreg[i])).astype(np.float32)
        mean_amp += amp
        mean_sq += amp * amp
        count += 1
    if count == 0:
        raise RuntimeError("SLC 栈为空，无法选择参考像元")
    mean_amp /= count
    mean_sq /= count
    variance = np.maximum(mean_sq - mean_amp * mean_amp, 0.0)
    adi = (np.sqrt(variance) / np.maximum(mean_amp, 1e-6)).astype(np.float32)
    adi_m = downsample_real(adi, multilook)
    mask_m = _downsample_mask(mask, multilook)
    quality = 1.0 / (adi_m + 0.03)
    quality[~mask_m] = -np.inf
    row, col = stable_reference(quality, mask_m, search=max(6, min(adi_m.shape) // 8))
    info = {
        "row": int(row),
        "col": int(col),
        "adi": float(adi_m[row, col]),
        "multilook": int(multilook),
    }
    return int(row), int(col), info


def _remove_ramp(
    phase: np.ndarray,
    coherence: np.ndarray,
    valid: np.ndarray,
    *,
    order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and remove a low-order polynomial ramp from unwrapped phase.

    ``order=1`` removes ``a*x + b*y + c``; ``order=2`` additionally removes
    ``d*x² + e*y² + f*x*y``.  The fit uses high-coherence pixels only so that
    noisy or decorrelated areas do not bias the ramp.
    """
    rows, cols = phase.shape
    order = 2 if int(order) >= 2 else 1
    good = valid & (coherence >= 0.55) & np.isfinite(phase)
    idx = np.flatnonzero(good.ravel())
    if idx.size < 200:
        return phase, {"removed": False, "order": order, "reason": "有效高相干像元不足"}
    if idx.size > 12000:
        idx = np.random.default_rng(7).choice(idx, 12000, replace=False)
    # Only the sampled pixels need coordinates; a full mgrid for a large scene
    # would allocate two int64 arrays over the whole image.
    jr, jc = np.unravel_index(idx, (rows, cols))
    x = (jc / max(cols - 1, 1)) - 0.5
    y = (jr / max(rows - 1, 1)) - 0.5
    z = phase.ravel()[idx]
    columns = [x, y, np.ones_like(x)]
    names = ["x", "y", "intercept"]
    if order >= 2:
        columns.extend([x * x, y * y, x * y])
        names.extend(["x2", "y2", "xy"])
    A = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    # Subtract the low-order surface in row blocks so the float64 ramp never
    # exists as a full-image array.
    xn = (np.arange(cols, dtype=np.float64) / max(cols - 1, 1)) - 0.5
    xn2 = xn * xn
    out = np.array(phase, dtype=np.float32, copy=True)
    block_rows = choose_block_rows(rows, cols, budget_bytes=64 * MIB, arrays=4, itemsize=8)
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        yn = (np.arange(r0, r1, dtype=np.float64) / max(rows - 1, 1))[:, None] - 0.5
        term = coef[0] * xn[None, :] + coef[1] * yn + coef[2]
        if order >= 2:
            term += coef[3] * xn2[None, :] + coef[4] * yn * yn + coef[5] * xn[None, :] * yn
        out[r0:r1] = (phase[r0:r1] - term).astype(np.float32)
    return out, {
        "removed": True,
        "order": order,
        "terms": names,
        "coefficients": {name: float(value) for name, value in zip(names, coef)},
        "n_fit_pixels": int(idx.size),
    }


def _remove_plane(phase: np.ndarray, coherence: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible plane-only ramp removal."""
    return _remove_ramp(phase, coherence, valid, order=1)


def _save_pair_figures(
    root: Path,
    pair: dict[str, Any],
    wrapped: np.ndarray,
    coh: np.ndarray,
    unwrapped: np.ndarray,
    disp_mm: np.ndarray,
    valid: np.ndarray,
) -> None:
    fig_dir = root / "products" / "figures" / "pairs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = pair["id"]
    try:
        save_wrapped_phase(
            wrapped,
            coh,
            fig_dir / f"{stem}_wrapped.png",
            mask=valid,
            title=f"Wrapped phase · {pair['ref_date']} → {pair['sec_date']}",
        )
        save_raster(
            disp_mm,
            fig_dir / f"{stem}_unwrapped.png",
            mask=valid & np.isfinite(disp_mm),
            title=f"Unwrapped LOS displacement · {pair['ref_date']} → {pair['sec_date']}",
            subtitle=f"Δt {pair['dt_days']} d · B⊥ {pair['bperp_diff']:+.1f} m",
            unit="mm",
            symmetric=True,
            divergent=True,
            figsize=(5.4, 4.2),
        )
    except Exception:
        pass


def process_pair(
    root: Path,
    coreg: np.ndarray,
    stack_meta: dict[str, Any],
    pair: dict[str, Any],
    params: dict[str, Any],
    *,
    dem_full: np.ndarray | None = None,
    mask_full: np.ndarray | None = None,
    ref_pixel: tuple[int, int] | None = None,
    save_figures: bool = True,
) -> dict[str, Any]:
    p = {**DEFAULT_PROCESSING, **params}
    m = max(1, int(p.get("multilook", 2)))
    i, j = int(pair["ref"]), int(pair["sec"])
    ref_full = np.asarray(coreg[i])
    sec_full = np.asarray(coreg[j])
    rows = ref_full.shape[0] // m * m
    cols = ref_full.shape[1] // m * m
    if m > 1:
        ifg, ref_m, sec_m = downsample_complex_pair(
            ref_full[:rows, :cols],
            sec_full[:rows, :cols],
            m,
        )
    else:
        ref = ref_full[:rows, :cols]
        sec = sec_full[:rows, :cols]
        ifg = sec * np.conj(ref)
        ref_m = ref
        sec_m = sec
    ref_amp = np.abs(ref_m).astype(np.float32)
    sec_amp = np.abs(sec_m).astype(np.float32)

    if dem_full is None or mask_full is None:
        stack = load_stack(root)
        dem_full = stack.dem
        mask_full = stack.mask
    dem = downsample_real(np.asarray(dem_full)[:rows, :cols], m).astype(np.float32)
    mask = _downsample_mask(np.asarray(mask_full)[:rows, :cols], m)

    wavelength = float(stack_meta["wavelength_m"])
    incidence = float(np.deg2rad(stack_meta["incidence_deg"]))
    slant_range = float(stack_meta.get("slant_range_m", 705000.0))
    k = 4.0 * np.pi / wavelength
    bdiff = float(pair["bperp_diff"])
    topo = (k * bdiff / (slant_range * np.sin(incidence))) * dem
    ifg = ifg * np.exp(-1j * topo)
    ifg = ifg.astype(np.complex64)

    coh = _coherence(ifg, ref_amp, sec_amp, int(p.get("coherence_window", 5)))
    filtered = filter_complex(
        ifg,
        p.get("filter", "goldstein"),
        alpha=p.get("goldstein_alpha", 0.65),
        patch=p.get("goldstein_patch", 32),
        size=int(p.get("boxcar_size", 5)),
        sigma=p.get("gaussian_sigma", 2.0),
    )
    wrapped = np.angle(filtered).astype(np.float32)
    valid = mask & (coh >= float(p.get("min_coherence", 0.25))) & np.isfinite(wrapped)

    unwrapped, unwrap_info = unwrap(
        wrapped,
        coh,
        valid,
        method=p.get("unwrap_method", "quality"),
        coherence_min=float(p.get("unwrap_coherence_min", 0.18)),
        snaphu_exe=str(p.get("snaphu_exe", "snaphu")),
    )

    # consistent reference pixel
    if ref_pixel is None:
        ref_pixel = stable_reference(coh, valid)
    rr, cc = ref_pixel
    if 0 <= rr < unwrapped.shape[0] and 0 <= cc < unwrapped.shape[1] and np.isfinite(unwrapped[rr, cc]):
        unwrapped = unwrapped - unwrapped[rr, cc]
    else:
        # fall back to the best coherent pixel in this pair
        rr, cc = stable_reference(coh, valid)
        if np.isfinite(unwrapped[rr, cc]):
            unwrapped = unwrapped - unwrapped[rr, cc]
    unwrapped = unwrapped.astype(np.float32)

    ramp_info = {"removed": False}
    if p.get("remove_ramp", True):
        unwrapped, ramp_info = _remove_ramp(
            unwrapped,
            coh,
            valid,
            order=int(p.get("ramp_order", 1)),
        )

    disp_mm = (unwrapped * wavelength / (4.0 * np.pi) * 1000.0).astype(np.float32)
    valid_out = valid & np.isfinite(unwrapped)

    pair_dir = root / "interferograms" / pair["id"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.save(pair_dir / "wrapped.npy", wrapped, allow_pickle=False)
    np.save(pair_dir / "coherence.npy", coh, allow_pickle=False)
    np.save(pair_dir / "unwrapped.npy", unwrapped, allow_pickle=False)
    np.save(pair_dir / "displacement_mm.npy", disp_mm, allow_pickle=False)
    np.save(pair_dir / "mask.npy", valid_out, allow_pickle=False)

    stats = {
        "coherence_mean": float(np.nanmean(coh[mask])) if mask.any() else 0.0,
        "coherence_p90": float(np.nanpercentile(coh[mask], 90)) if mask.any() else 0.0,
        "valid_fraction": float(valid_out.sum() / max(mask.sum(), 1)),
        "unwrapped_fraction": float(np.isfinite(unwrapped[valid]).sum() / max(valid.sum(), 1)),
        "displacement_p05_mm": float(np.nanpercentile(disp_mm[valid_out], 5)) if valid_out.any() else 0.0,
        "displacement_p95_mm": float(np.nanpercentile(disp_mm[valid_out], 95)) if valid_out.any() else 0.0,
    }
    meta = {
        **pair,
        "rows": int(ifg.shape[0]),
        "cols": int(ifg.shape[1]),
        "multilook": m,
        "wavelength_m": wavelength,
        "incidence_deg": float(stack_meta["incidence_deg"]),
        "ref_pixel": [int(rr), int(cc)],
        "topo_removed": True,
        "ramp": ramp_info,
        "unwrap": unwrap_info,
        "stats": stats,
        "params": {
            "filter": p.get("filter"),
            "coherence_window": p.get("coherence_window"),
            "min_coherence": p.get("min_coherence"),
            "remove_ramp": p.get("remove_ramp"),
            "ramp_order": int(p.get("ramp_order", 1)) if p.get("remove_ramp", True) else None,
        },
        "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(pair_dir / "meta.json", meta)
    if save_figures:
        _save_pair_figures(root, pair, wrapped, coh, unwrapped, disp_mm, valid_out)
    return meta


def process_interferograms(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
    *,
    save_figures: bool = False,
) -> dict[str, Any]:
    root = Path(project_root)
    p = {**DEFAULT_PROCESSING, **(params or {})}
    coreg_path = root / "data" / "processed" / "slc_coreg.npy"
    if not coreg_path.exists():
        raise FileNotFoundError("请先完成 SLC 配准，再生成干涉图")
    coreg = np.load(coreg_path, mmap_mode="r", allow_pickle=False)
    stack = load_stack(root)
    network = read_json(root / "data" / "network.json")
    if not network or not network.get("pairs"):
        raise FileNotFoundError("尚未生成干涉网络，请先在「网络」页创建网络")
    pairs = network["pairs"]
    m = max(1, int(p.get("multilook", 2)))
    ensure_memory_budget(estimate_interferometry_peak(stack.rows, stack.cols, m), "干涉")
    ref_row, ref_col, ref_info = reference_pixel(
        coreg, stack.mask, m, ref_index=int(network.get("stats", {}).get("ref_index", 0))
    )
    atomic_write_json(root / "interferograms" / "reference.json", ref_info)

    metas = []
    t0 = time.time()
    for idx, pair in enumerate(pairs):
        if progress:
            progress(
                0.02 + 0.92 * idx / max(len(pairs), 1),
                f"干涉对 {idx + 1}/{len(pairs)} · {pair['ref_date']} → {pair['sec_date']}",
            )
        meta = process_pair(
            root,
            coreg,
            stack.meta,
            pair,
            p,
            dem_full=stack.dem,
            mask_full=stack.mask,
            ref_pixel=(ref_row, ref_col),
            save_figures=save_figures,
        )
        metas.append(meta)

    # loop-closure quality: wrapped phase closure for every triangle
    closure = _closure_stats(root, pairs, metas)
    summary = {
        "n_pairs": len(metas),
        "multilook": m,
        "reference_pixel": [ref_row, ref_col],
        "reference_info": ref_info,
        "mean_coherence": float(np.mean([m["stats"]["coherence_mean"] for m in metas])) if metas else 0.0,
        "mean_valid_fraction": float(np.mean([m["stats"]["valid_fraction"] for m in metas])) if metas else 0.0,
        "closure": closure,
        "params": p,
        "elapsed_s": round(time.time() - t0, 2),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(root / "interferograms" / "summary.json", summary)
    if progress:
        progress(0.97, f"干涉处理完成 · {len(metas)} 对 · 平均相干性 {summary['mean_coherence']:.2f}")
    return summary


def _closure_stats(root: Path, pairs: list[dict[str, Any]], metas: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {(int(p["ref"]), int(p["sec"])): p for p in pairs}
    # A small LRU keeps a triangle's three arrays hot without retaining every
    # wrapped product.  On large networks the old unbounded cache could hold
    # hundreds of full-resolution arrays in RAM.
    wrapped_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
    max_cached = 6

    def get(pair: tuple[int, int]) -> np.ndarray | None:
        if pair not in by_id:
            return None
        if pair in wrapped_cache:
            wrapped_cache.move_to_end(pair)
            return wrapped_cache[pair]
        path = root / "interferograms" / by_id[pair]["id"] / "wrapped.npy"
        if not path.exists():
            return None
        wrapped_cache[pair] = np.load(path, allow_pickle=False)
        while len(wrapped_cache) > max_cached:
            wrapped_cache.popitem(last=False)
        return wrapped_cache[pair]

    closures = []
    pair_flags: dict[str, int] = {}
    keys = sorted(by_id)
    for a, b in keys:
        for c in range(b + 1, max(k[1] for k in keys) + 1):
            if (b, c) not in by_id or (a, c) not in by_id:
                continue
            p1, p2, p3 = get((a, b)), get((b, c)), get((a, c))
            if p1 is None or p2 is None or p3 is None:
                continue
            val = np.abs(wrap_phase(p1 + p2 - p3))
            med = float(np.nanmedian(val))
            closures.append(med)
            if med > 1.2:
                for pid in (by_id[(a, b)]["id"], by_id[(b, c)]["id"], by_id[(a, c)]["id"]):
                    pair_flags[pid] = pair_flags.get(pid, 0) + 1
    return {
        "n_triangles": len(closures),
        "median_closure_rad": float(np.median(closures)) if closures else 0.0,
        "p90_closure_rad": float(np.percentile(closures, 90)) if closures else 0.0,
        "flagged_pairs": pair_flags,
        "warning": "存在闭合差较大的干涉对，建议检查解缠或降低相干性阈值" if pair_flags else "",
    }


def load_pair(root: Path, pair_id: str) -> dict[str, Any]:
    pair_dir = Path(root) / "interferograms" / pair_id
    meta = read_json(pair_dir / "meta.json")
    if not meta:
        raise FileNotFoundError(f"干涉对不存在: {pair_id}")
    arrays = {}
    for name, fname in (
        ("wrapped", "wrapped.npy"),
        ("coherence", "coherence.npy"),
        ("unwrapped", "unwrapped.npy"),
        ("displacement", "displacement_mm.npy"),
        ("mask", "mask.npy"),
    ):
        path = pair_dir / fname
        if path.exists():
            arrays[name] = np.load(path, allow_pickle=False)
    return {"meta": meta, "arrays": arrays}
