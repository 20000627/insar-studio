from pathlib import Path

path = Path(__file__).resolve().parents[1] / "insar_studio" / "core" / "inversion.py"
text = path.read_text(encoding="utf-8")
start = text.index("def _design(")
end = text.index("def _save_products(")
new = '''def _time_design(
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


def _reference_pixel(root: Path, mask: np.ndarray) -> tuple[int, int]:
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
) -> tuple[np.ndarray, np.ndarray]:
    p, rows, cols = phase.shape
    phase_flat = phase.reshape(p, -1)
    mask_flat = mask.reshape(p, -1)
    coverage = mask_flat.mean(axis=0)
    phi = np.where(mask_flat, phase_flat, 0.0).astype(np.float32)
    w = weights.astype(np.float64).copy()
    sqrt_w = np.sqrt(w)[:, None]
    extra_rows = G.shape[0] - p
    sqrt_w_full = np.vstack([sqrt_w, np.ones((extra_rows, 1), dtype=np.float64)])
    zeros_extra = np.zeros((extra_rows, phi.shape[1]), dtype=np.float32)
    X = np.zeros((n_dates, rows * cols), dtype=np.float32)
    if not (coverage >= 0.25).any():
        return X.reshape(n_dates, rows, cols), coverage.reshape(rows, cols)

    for it in range(max(1, iterations)):
        Gw = G * sqrt_w_full
        Y = np.vstack([phi * sqrt_w, zeros_extra])
        if it > 0:
            pred = A @ X
            miss = ~mask_flat
            Y[:p] = np.where(miss, pred * sqrt_w, Y[:p])
        X = (np.linalg.pinv(Gw) @ Y).astype(np.float32)
        model = (G[:p] @ X).astype(np.float32)
        resid = np.where(mask_flat, phase_flat - model, np.nan)
        med = np.nanmedian(np.abs(resid), axis=1)
        sigma = np.nanmedian(med) if np.isfinite(med).any() else 1.0
        sigma = max(float(sigma), 1e-3)
        ratio = med / (3.0 * sigma)
        new_w = np.clip(1.0 / (1.0 + ratio**2), 0.04, 1.0)
        w = weights * new_w
        sqrt_w = np.sqrt(w)[:, None]
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
    x_flat: np.ndarray,
    A: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Estimate one spatial APS screen per acquisition from pair residuals."""
    p, rows, cols = phase.shape
    model = (A @ x_flat).reshape(p, rows, cols)
    resid = np.where(mask, phase - model, np.nan)
    screens = np.zeros((n_dates, rows, cols), dtype=np.float32)
    counts = np.zeros((n_dates, rows, cols), dtype=np.float32)
    for k, pair in enumerate(pairs):
        screens[int(pair["sec"])] += np.nan_to_num(resid[k], nan=0.0)
        counts[int(pair["sec"])] += mask[k]
        screens[int(pair["ref"])] -= np.nan_to_num(resid[k], nan=0.0)
        counts[int(pair["ref"])] += mask[k]
    screens = np.where(counts > 0, screens / np.maximum(counts, 1e-6), 0.0)
    for i in range(n_dates):
        sm = nan_gaussian_filter(np.where(mask.any(axis=0), screens[i], np.nan), sigma)
        sm = np.nan_to_num(sm, nan=0.0)
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
        )
        if aps_it < n_aps:
            screens = _aps_screens(
                phase_corr,
                mask,
                pairs,
                n_dates,
                X.reshape(n_dates, -1),
                A,
                float(p.get("aps_sigma_px", 10.0)),
            )
            for k, pair in enumerate(pairs):
                phase_corr[k] = phase_corr[k] - (screens[int(pair["sec"])] - screens[int(pair["ref"])])

    X = X.copy()
    X -= X[0]
    disp = X * float(stack_meta["wavelength_m"]) / (4.0 * np.pi) * 1000.0
    model = (A @ X.reshape(n_dates, -1)).reshape(len(pairs), *phase.shape[1:])
    resid = np.where(mask, phase_corr - model, np.nan)
    tcoh = temporal_coherence(np.nan_to_num(resid, nan=0.0))
    valid = coverage >= 0.45
    if extra_mask is not None:
        valid &= extra_mask
    valid &= np.isfinite(disp).all(axis=0)
    valid &= np.isfinite(eps)
    disp[:, ~valid] = np.nan
    eps = np.where(valid, eps, np.nan)
    resid_std = np.sqrt(np.nanmean(resid**2, axis=0))
    resid_std = np.where(np.isfinite(resid_std), resid_std, 0.0)
    disp_std = resid_std * float(stack_meta["wavelength_m"]) / (4.0 * np.pi) * 1000.0
    vel, lin_rmse = _fit_velocity(np.nan_to_num(disp, nan=0.0), days, valid)
    vel = np.where(valid, vel, np.nan)

    rr, cc = _reference_pixel(root, valid)
    if np.isfinite(disp[0, rr, cc]):
        disp = disp - disp[0, rr, cc]

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


'''
path.write_text(text[:start] + new + text[end:], encoding="utf-8")
print("patched", len(new))
