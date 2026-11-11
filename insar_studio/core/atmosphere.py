"""Atmospheric phase screen (APS) correction.

Three routes are supported:

``gacos``
    External GACOS zenith delay products.  Four sources are supported: the
    official asynchronous service (imported tar.gz / email links), demo
    screens, a local directory, or a custom synchronous online adapter.  The
    per-date delay is converted to LOS and differentiated for every
    interferometric pair.
``empirical``
    A robust phase-elevation regression per pair, optionally including a
    long-wavelength plane.  This is the classic stratified-troposphere
    correction and needs no external weather model.
``off``
    Keep the raw unwrapped phase.  The temporal high-pass separation in the
    inversion can still be enabled.

All products are written next to the interferograms so that the inversion can
pick them up without any additional configuration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from .gacos import load_screens
from .io import load_stack
from .plotting import save_raster
from .utils import (
    atomic_write_json,
    downsample_real,
    read_json,
    robust_sigma,
)

Progress = Callable[[float, str], None]


def _phase_scale(stack_meta: dict[str, Any]) -> float:
    return 4.0 * np.pi / float(stack_meta["wavelength_m"])


def _pair_dirs(root: Path, pairs: list[dict[str, Any]]) -> list[Path]:
    return [root / "interferograms" / str(p["id"]) for p in pairs]


def _load_pair_array(root: Path, pair_id: str, name: str) -> np.ndarray | None:
    path = root / "interferograms" / pair_id / name
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _grid_bounds(stack_meta: dict[str, Any], shape: tuple[int, int], multilook: int) -> tuple[float, float, float, float]:
    west = float(stack_meta.get("corner_lon", 0.0))
    north = float(stack_meta.get("corner_lat", 0.0))
    lon_step = float(stack_meta.get("lon_step", 0.0004)) * multilook
    lat_step = float(stack_meta.get("lat_step", -0.0004)) * multilook
    return (west, north + shape[0] * lat_step, west + shape[1] * lon_step, north)


def _truth_diagnostics(
    root: Path,
    stack_meta: dict[str, Any],
    pairs: list[dict[str, Any]],
    corrected: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    shape: tuple[int, int],
    multilook: int,
) -> dict[str, Any] | None:
    truth_path = root / "data" / "stack" / "truth.npz"
    if not truth_path.exists():
        return None
    try:
        with np.load(truth_path) as truth:
            if "atmosphere_mm" not in truth.files:
                return None
            atm = truth["atmosphere_mm"].astype(np.float32)
            disp = truth["disp_los_mm"].astype(np.float32)
        m = max(1, multilook)
        k = _phase_scale(stack_meta)
        rows, cols = shape
        ref_info = read_json(root / "interferograms" / "reference.json", {}) or {}
        ref_r = min(max(int(ref_info.get("row", rows // 2)), 0), rows - 1)
        ref_c = min(max(int(ref_info.get("col", cols // 2)), 0), cols - 1)
        before = []
        after = []
        before_atm = []
        for p in pairs:
            sec, ref = int(p["sec"]), int(p["ref"])
            id_ = str(p["id"])
            if id_ not in corrected:
                continue
            mask = masks[id_]
            # deformation-only truth, multilook averaged
            def _ml(arr: np.ndarray) -> np.ndarray:
                return downsample_real(arr, m)[:rows, :cols]

            d_pair = _ml(disp[sec] - disp[ref])
            a_pair = _ml(atm[sec] - atm[ref])
            # pair phases are referenced to the same stable pixel
            d_pair = d_pair - d_pair[ref_r, ref_c]
            a_pair = a_pair - a_pair[ref_r, ref_c]
            phase_d = k * d_pair / 1000.0
            phase_a = k * a_pair / 1000.0
            # compare unwrapped pair phase against truth deformation + atmosphere
            raw = corrected.get(f"raw_{id_}")
            corr = corrected.get(id_)
            if raw is None or corr is None:
                continue
            before.append(float(np.nanmedian(np.abs((raw - phase_d)[mask]))))
            after.append(float(np.nanmedian(np.abs((corr - phase_d)[mask]))))
            before_atm.append(float(np.nanmedian(np.abs((raw - phase_d - phase_a)[mask]))))
        if not after:
            return None
        wav_mm = float(stack_meta["wavelength_m"]) / (4.0 * np.pi) * 1000.0
        return {
            "available": True,
            "pair_residual_before_mm": float(np.median(before) * wav_mm),
            "pair_residual_after_mm": float(np.median(after) * wav_mm),
            "pair_residual_noise_floor_mm": float(np.median(before_atm) * wav_mm),
            "note": "以合成真值为参照的成对相位残差（中位绝对值，含未建模形变/解缠误差），仅用于验证。",
        }
    except Exception:
        return None


def _save_corrected(
    root: Path,
    pairs: list[dict[str, Any]],
    corrected: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    screen_stack: np.ndarray | None,
    dates: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    out_dir = root / "data" / "atmosphere"
    out_dir.mkdir(parents=True, exist_ok=True)
    before_std = []
    after_std = []
    for p in pairs:
        pid = str(p["id"])
        if pid not in corrected:
            continue
        pair_dir = root / "interferograms" / pid
        mask = masks[pid]
        corr = corrected[pid].astype(np.float32)
        np.save(pair_dir / "unwrapped_atmcorr.npy", corr, allow_pickle=False)
        b = raw.get(pid)
        if b is not None:
            before_std.append(float(robust_sigma(b[mask])))
        after_std.append(float(robust_sigma(corr[mask])))
        meta = read_json(pair_dir / "meta.json", {}) or {}
        meta["atmosphere"] = {
            "method": summary["method"],
            "source": summary.get("source", ""),
            "corrected_at": summary.get("created_at", ""),
        }
        atomic_write_json(pair_dir / "meta.json", meta)
    summary["pair_phase_std_before_rad"] = float(np.median(before_std)) if before_std else 0.0
    summary["pair_phase_std_after_rad"] = float(np.median(after_std)) if after_std else 0.0
    if summary["pair_phase_std_before_rad"] > 1e-9:
        summary["phase_std_reduction_pct"] = float(
            100.0 * (1.0 - summary["pair_phase_std_after_rad"] / summary["pair_phase_std_before_rad"])
        )
    else:
        summary["phase_std_reduction_pct"] = 0.0
    if screen_stack is not None:
        np.save(out_dir / "screens.npy", screen_stack.astype(np.float32), allow_pickle=False)
        summary["screens_path"] = "data/atmosphere/screens.npy"
        summary["dates"] = dates
    atomic_write_json(out_dir / "atmosphere.json", summary)
    return summary


def _diagnostic_figure(
    root: Path,
    screens: np.ndarray | None,
    dates: list[str],
    pairs: list[dict[str, Any]],
    raw: dict[str, np.ndarray],
    corrected: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plotting import PAPER, INK, MUTED, TEAL, AMBER, GRID

        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
        if screens is not None and len(dates):
            ax = axes[0]
            im = ax.imshow(screens[len(screens) // 2], cmap="RdBu_r", interpolation="nearest")
            ax.set_title(f"APS screen · {dates[len(dates) // 2]}", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
            cb.set_label("mm", color=MUTED, fontsize=7)

        ax = axes[1]
        b = []
        a = []
        for p in pairs:
            pid = str(p["id"])
            if pid in raw and pid in corrected:
                m = masks[pid]
                b.extend(np.asarray(raw[pid][m]).ravel()[:: max(1, int(m.sum() / 8000))])
                a.extend(np.asarray(corrected[pid][m]).ravel()[:: max(1, int(m.sum() / 8000))])
        bins = np.linspace(-3.2, 3.2, 70)
        if b:
            ax.hist(np.clip(b, -3.2, 3.2), bins=bins, color=AMBER, alpha=0.55, label="校正前", density=True)
        if a:
            ax.hist(np.clip(a, -3.2, 3.2), bins=bins, color=TEAL, alpha=0.55, label="校正后", density=True)
        ax.set_title("Pair phase distribution (rad)", loc="left")
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.6)

        ax = axes[2]
        pid_list = [str(p["id"]) for p in pairs if str(p["id"]) in raw and str(p["id"]) in corrected]
        bvals = [float(robust_sigma(raw[k][masks[k]])) for k in pid_list]
        avals = [float(robust_sigma(corrected[k][masks[k]])) for k in pid_list]
        ax.scatter(bvals, avals, s=10, color=TEAL, alpha=0.65, linewidth=0)
        lim = max([1e-6] + bvals + avals)
        ax.plot([0, lim], [0, lim], color=GRID, lw=0.8, ls="--")
        ax.set_xlabel("Before (rad)", color=MUTED)
        ax.set_ylabel("After (rad)", color=MUTED)
        ax.set_title("Phase scatter per pair", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.patch.set_facecolor(PAPER)
        fig.tight_layout()
        out = root / "products" / "figures" / "atmosphere.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        summary["figure"] = "products/figures/atmosphere.png"
    except Exception:
        pass


def correct_gacos(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    params = params or {}
    stack = load_stack(root)
    network = read_json(root / "data" / "network.json", {}) or {}
    pairs = network.get("pairs", [])
    if not pairs:
        raise RuntimeError("尚未生成干涉网络")
    first = _load_pair_array(root, str(pairs[0]["id"]), "unwrapped.npy")
    if first is None:
        raise RuntimeError("尚未生成干涉图，无法进行大气校正")
    shape = first.shape
    m = max(1, int(params.get("multilook", 2)))
    bounds = _grid_bounds(stack.meta, shape, m)
    if progress:
        progress(0.04, "载入 GACOS 逐日大气延迟")
    screens_by_date, info = load_screens(
        root,
        params,
        target_shape=shape,
        target_bounds=bounds,
    )
    missing = [d for d in stack.dates if d not in screens_by_date]
    if missing:
        raise RuntimeError(f"GACOS 缺少 {len(missing)} 个日期的数据: {', '.join(missing[:4])}…")
    screen_stack = np.stack([screens_by_date[d].astype(np.float32) for d in stack.dates])
    k = _phase_scale(stack.meta)
    raw: dict[str, np.ndarray] = {}
    corrected: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for idx, pair in enumerate(pairs):
        if progress:
            progress(0.12 + 0.68 * idx / max(len(pairs), 1), f"GACOS 校正 {idx + 1}/{len(pairs)}")
        pid = str(pair["id"])
        phase = _load_pair_array(root, pid, "unwrapped.npy")
        mask = _load_pair_array(root, pid, "mask.npy")
        if phase is None or mask is None:
            continue
        mask = mask.astype(bool)
        sec, ref = int(pair["sec"]), int(pair["ref"])
        delay_mm = screen_stack[sec] - screen_stack[ref]
        phase_corr = phase - (k * delay_mm / 1000.0).astype(np.float32)
        phase_corr = np.where(mask, phase_corr, np.nan).astype(np.float32)
        np.save(root / "interferograms" / pid / "atmosphere_pair_mm.npy", delay_mm.astype(np.float32), allow_pickle=False)
        raw[pid] = phase
        corrected[pid] = phase_corr
        masks[pid] = mask
    summary: dict[str, Any] = {
        "method": "gacos",
        "source": info.get("source", "demo"),
        "simulated": bool(info.get("simulated", False)),
        "n_pairs": len(corrected),
        "n_screens": int(screen_stack.shape[0]),
        "mean_abs_screen_mm": float(np.nanmean(np.abs(screen_stack))),
        "screen_std_mm": float(np.nanstd(screen_stack)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": info.get("note", "GACOS LOS delay differenced per interferometric pair."),
        "warnings": list(info.get("warnings") or []),
        "missing_dates": list(info.get("missing_dates") or []),
    }
    if info.get("source") == "official":
        summary["directory"] = info.get("directory")
        summary["reference"] = "GACOS official products imported into the project"
    summary = _save_corrected(root, pairs, corrected, masks, raw, screen_stack, stack.dates, summary)
    truth = _truth_diagnostics(
        root,
        stack.meta,
        pairs,
        {**{f"raw_{k_}": v for k_, v in raw.items()}, **corrected},
        masks,
        shape,
        m,
    )
    if truth:
        summary["validation"] = truth
    if progress:
        progress(0.9, "生成大气诊断图")
    _diagnostic_figure(root, screen_stack, stack.dates, pairs, raw, corrected, masks, summary)
    atomic_write_json(root / "data" / "atmosphere" / "atmosphere.json", summary)
    if progress:
        progress(0.98, f"大气校正完成 · 相位离散度降低 {summary.get('phase_std_reduction_pct', 0):.1f}%")
    return summary


def _fit_pair_screen(
    phase: np.ndarray,
    dem: np.ndarray,
    coherence: np.ndarray,
    mask: np.ndarray,
    order: int = 1,
    iterations: int = 3,
    sigma: float = 2.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows, cols = phase.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    xn = xx / max(cols - 1, 1) - 0.5
    yn = yy / max(rows - 1, 1) - 0.5
    z = np.asarray(dem, dtype=np.float64)
    z = (z - np.nanmean(z)) / max(np.nanstd(z), 1e-9)
    cols_list = [np.ones(rows * cols), xn.ravel(), yn.ravel(), z.ravel()]
    if order >= 2:
        cols_list += [z.ravel() ** 2, (z * xn).ravel(), (z * yn).ravel()]
    if order >= 3:
        cols_list += [z.ravel() ** 3, (z.ravel() ** 2 * xn).ravel(), (z.ravel() ** 2 * yn).ravel()]
    A = np.column_stack(cols_list)
    good = mask & (coherence >= 0.45) & np.isfinite(phase)
    idx = np.flatnonzero(good.ravel())
    if idx.size < 150:
        return np.zeros_like(phase, dtype=np.float32), {"enabled": False, "reason": "高相干像元不足"}
    if idx.size > 40000:
        idx = np.random.default_rng(11).choice(idx, 40000, replace=False)
    y = phase.ravel()[idx]
    A = A[idx]
    w = np.ones(idx.size)
    coef = np.zeros(A.shape[1])
    for _ in range(max(1, iterations)):
        Aw = A * w[:, None]
        yw = y * w
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        resid = y - A @ coef
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
        mad = max(float(mad), 0.02)
        w = 1.0 / (1.0 + (resid / (max(1.0, sigma) * mad)) ** 2)
        w = np.clip(w, 0.02, 1.0)
    screen = (A_all := np.column_stack([c for c in cols_list])) @ coef
    screen = screen.reshape(rows, cols).astype(np.float32)
    return screen, {
        "enabled": True,
        "order": int(order),
        "elevation_slope_rad_per_sigma": float(coef[3]) if len(coef) > 3 else 0.0,
        "n_samples": int(idx.size),
        "coefficients": coef.tolist(),
    }


def correct_empirical(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    params = params or {}
    stack = load_stack(root)
    network = read_json(root / "data" / "network.json", {}) or {}
    pairs = network.get("pairs", [])
    if not pairs:
        raise RuntimeError("尚未生成干涉网络")
    m = max(1, int(params.get("multilook", 2)))
    dem = downsample_real(stack.dem, m)
    raw: dict[str, np.ndarray] = {}
    corrected: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    screens_info = []
    order = int(params.get("empirical_order", 1))
    for idx, pair in enumerate(pairs):
        if progress:
            progress(0.05 + 0.82 * idx / max(len(pairs), 1), f"经验大气校正 {idx + 1}/{len(pairs)}")
        pid = str(pair["id"])
        phase = _load_pair_array(root, pid, "unwrapped.npy")
        coh = _load_pair_array(root, pid, "coherence.npy")
        mask = _load_pair_array(root, pid, "mask.npy")
        if phase is None or coh is None or mask is None:
            continue
        mask = mask.astype(bool)
        d = dem[: phase.shape[0], : phase.shape[1]]
        if d.shape != phase.shape:
            d = np.pad(d, ((0, phase.shape[0] - d.shape[0]), (0, phase.shape[1] - d.shape[1])), mode="edge")
        screen, info = _fit_pair_screen(
            phase,
            d,
            coh,
            mask,
            order=order,
            iterations=int(params.get("empirical_iterations", 3)),
        )
        if not info.get("enabled") and not mask.any():
            info["reason"] = info.get("reason", "有效像元不足")
        corr = np.where(mask, phase - screen, np.nan).astype(np.float32)
        raw[pid] = phase
        corrected[pid] = corr
        masks[pid] = mask
        screens_info.append({"pair": pid, **info})
    summary: dict[str, Any] = {
        "method": "empirical",
        "source": "phase-elevation regression",
        "n_pairs": len(corrected),
        "order": order,
        "pairs": screens_info,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "逐干涉对稳健相位-高程回归；适用于无外部大气产品的场景，会吸收与地形相关的真实形变。",
    }
    summary = _save_corrected(root, pairs, corrected, masks, raw, None, stack.dates, summary)
    truth = _truth_diagnostics(
        root,
        stack.meta,
        pairs,
        {**{f"raw_{k_}": v for k_, v in raw.items()}, **corrected},
        masks,
        tuple(next(iter(raw.values())).shape) if raw else (stack.rows, stack.cols),
        m,
    )
    if truth:
        summary["validation"] = truth
    if progress:
        progress(0.9, "生成大气诊断图")
    _diagnostic_figure(root, None, stack.dates, pairs, raw, corrected, masks, summary)
    atomic_write_json(root / "data" / "atmosphere" / "atmosphere.json", summary)
    if progress:
        progress(0.98, f"经验大气校正完成 · 相位离散度降低 {summary.get('phase_std_reduction_pct', 0):.1f}%")
    return summary


def disable_atmosphere(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    network = read_json(root / "data" / "network.json", {}) or {}
    for pair in network.get("pairs", []):
        path = root / "interferograms" / str(pair["id"]) / "unwrapped_atmcorr.npy"
        if path.exists():
            path.unlink()
    summary = {
        "method": "off",
        "source": "",
        "n_pairs": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "未应用外部大气校正，反演阶段可使用时空高通近似。",
    }
    out = root / "data" / "atmosphere"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "atmosphere.json", summary)
    return summary


def run_atmosphere(
    project_root: Path,
    params: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    params = params or {}
    method = str(params.get("method", params.get("gacos_source", "gacos"))).lower()
    if method in ("gacos", "demo", "local", "online"):
        p = dict(params)
        p.setdefault("gacos_source", method if method != "gacos" else params.get("gacos_source", "demo"))
        return correct_gacos(project_root, p, progress)
    if method in ("empirical", "elevation", "stratified"):
        return correct_empirical(project_root, params, progress)
    if method in ("off", "none"):
        if progress:
            progress(0.5, "关闭外部大气校正")
        return disable_atmosphere(project_root)
    raise ValueError(f"未知大气校正方法: {method}")
