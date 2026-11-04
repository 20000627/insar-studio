from pathlib import Path

p = Path(__file__).resolve().parents[1] / "insar_studio" / "core" / "inversion.py"
t = p.read_text(encoding="utf-8")
start = t.index("def _aps_screens(")
end = t.index("def invert_arrays(")
new = '''def _aps_screens(
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


'''
p.write_text(t[:start] + new + t[end:], encoding="utf-8")
print("patched")
