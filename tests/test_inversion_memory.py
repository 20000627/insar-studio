"""Regression tests for the memory-bounded inversion path."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from insar_studio.core.inversion import _aps_screens, _solve_weighted, _time_design


def _reference_solve_weighted(
    phase: np.ndarray,
    mask: np.ndarray,
    coh: np.ndarray,
    pairs: list[dict],
    G: np.ndarray,
    A: np.ndarray,
    weights: np.ndarray,
    n_dates: int,
    iterations: int,
    extra_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-refactor implementation, kept as the numerical reference."""
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
    zeros_extra = np.zeros((extra_rows, phi.shape[1]), dtype=np.float32)
    X = np.zeros((n_dates, rows * cols), dtype=np.float32)
    if not (coverage >= 0.25).any():
        return X.reshape(n_dates, rows, cols), coverage.reshape(rows, cols)
    for it in range(max(1, iterations)):
        Y = np.vstack([phi * sqrt_w, zeros_extra])
        if it > 0:
            pred = A @ X
            miss = ~mask_flat
            Y[:p] = np.where(miss, pred * sqrt_w, Y[:p])
        # Robust reweighting must also update the normal equations, not only
        # the observation vector.
        sqrt_w_full = np.vstack([sqrt_w, np.ones((extra_rows, phi.shape[1]), dtype=np.float64)])
        sw2 = (sqrt_w_full**2).astype(np.float64)
        X = np.empty((n_dates, phi.shape[1]), dtype=np.float32)
        for s in range(0, phi.shape[1], 4096):
            e = min(phi.shape[1], s + 4096)
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


def _small_problem(seed: int = 20261110):
    rng = np.random.default_rng(seed)
    n_dates = 7
    rows, cols = 9, 11
    pairs = []
    for ref in range(n_dates - 1):
        for sec in range(ref + 1, min(n_dates, ref + 4)):
            pairs.append({"ref": ref, "sec": sec, "id": f"{ref:02d}_{sec:02d}"})
    p = len(pairs)
    phase = rng.normal(scale=0.8, size=(p, rows, cols)).astype(np.float32)
    mask = rng.random((p, rows, cols)) > 0.25
    phase[~mask] = np.nan
    coh = rng.uniform(0.2, 1.0, size=(p, rows, cols)).astype(np.float32)
    weights = rng.uniform(0.05, 1.0, size=p)
    extra_weight = rng.uniform(0.2, 1.0, size=rows * cols)
    G, A = _time_design(pairs, n_dates, 0.25)
    # Fix the constant-gauge mode explicitly so the normal equations are well
    # conditioned; the production solver relies on X -= X[0] instead.
    G = np.vstack([G, np.full((1, n_dates), 1e-3)])
    return phase, mask, coh, pairs, G, A, weights, extra_weight, n_dates


def test_solve_weighted_matches_full_cube_reference():
    phase, mask, coh, pairs, G, A, weights, extra_weight, n_dates = _small_problem()
    expected, expected_coverage = _reference_solve_weighted(
        phase, mask, coh, pairs, G, A, weights, n_dates, 3, extra_weight
    )
    actual, coverage = _solve_weighted(
        phase, mask, coh, pairs, G, A, weights, n_dates, 3, extra_weight
    )
    assert np.allclose(actual, expected, atol=2e-5, rtol=1e-5)
    assert np.allclose(coverage, expected_coverage, atol=1e-12)


def test_solve_weighted_without_regularisation_rows():
    phase, mask, coh, pairs, G, A, weights, _, n_dates = _small_problem(seed=7)
    G = A
    expected, _ = _reference_solve_weighted(
        phase, mask, coh, pairs, G, A, weights, n_dates, 2, None
    )
    actual, _ = _solve_weighted(phase, mask, coh, pairs, G, A, weights, n_dates, 2, None)
    assert np.allclose(actual, expected, atol=2e-5, rtol=1e-5)


def test_aps_screens_blockwise_matches_direct_fit():
    rng = np.random.default_rng(11)
    n_dates, rows, cols = 10, 13, 15
    x_3d = rng.normal(size=(n_dates, rows, cols)).astype(np.float32)
    days = np.arange(n_dates, dtype=np.float64) * 12.0
    sigma = 2.0

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
    resid = (Xf - B @ (np.linalg.pinv(B) @ Xf)).reshape(n_dates, rows, cols)
    expected = np.zeros((n_dates, rows, cols), dtype=np.float32)
    for i in range(n_dates):
        sm = ndimage.gaussian_filter(
            np.nan_to_num(resid[i], nan=0.0), sigma=sigma, mode="nearest"
        )
        sm -= np.nanmedian(sm)
        expected[i] = sm.astype(np.float32)

    actual = _aps_screens(
        x_3d,
        np.ones((n_dates, rows, cols), dtype=bool),
        [],
        n_dates,
        x_3d,
        days,
        sigma,
    )
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
