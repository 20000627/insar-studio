"""DS phase-linking memory and block-equivalence regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from insar_studio.core import ds
from insar_studio.core.ds import (
    COVARIANCE_BUDGET_BYTES,
    _covariance_block_rows,
    pair_coherence,
    phase_link,
)


def _synthetic_inputs(n: int = 10, rows: int = 36, cols: int = 40):
    rng = np.random.default_rng(20261107)
    coreg = (
        rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))
    ).astype(np.complex64)
    meta = {
        "wavelength_m": 0.05546576,
        "incidence_deg": 38.5,
        "slant_range_m": 705000.0,
        "bperp_m": [0.0] * n,
        "_dem_full": np.zeros((rows, cols), dtype=np.float32),
    }
    mask = np.ones((rows, cols), dtype=bool)
    return coreg, meta, mask


def test_covariance_block_rows_stays_within_budget():
    block = _covariance_block_rows(60, 500, 500)

    assert 1 <= block < 500
    assert 60 * 60 * block * 500 * np.dtype(np.complex64).itemsize <= COVARIANCE_BUDGET_BYTES


def test_phase_link_blockwise_matches_full_matrix(monkeypatch):
    coreg, meta, mask = _synthetic_inputs()

    monkeypatch.setattr(ds, "_covariance_block_rows", lambda *args, **kwargs: 18)
    full = phase_link(coreg, meta, mask, multilook=2, patch=5, ref_index=0)

    monkeypatch.setattr(ds, "_covariance_block_rows", lambda *args, **kwargs: 5)
    blocked = phase_link(coreg, meta, mask, multilook=2, patch=5, ref_index=0)

    assert "coherence_matrix" not in full
    assert full["slc"].shape == (10, 18, 20)
    assert full["diag"].shape == (10, 18, 20)
    assert np.allclose(full["slc"], blocked["slc"], atol=1e-6)
    assert np.allclose(full["diag"], blocked["diag"], atol=1e-5)
    assert np.allclose(full["linked"], blocked["linked"], atol=1e-5)
    assert np.allclose(full["ds_coherence"], blocked["ds_coherence"], atol=1e-6)
    assert np.array_equal(full["base_mask"], blocked["base_mask"])


def test_pair_coherence_matches_normalised_covariance():
    coreg, meta, mask = _synthetic_inputs()
    result = phase_link(coreg, meta, mask, multilook=2, patch=5, ref_index=0)
    slc, diag = result["slc"], result["diag"]
    patch = result["patch"]

    pair = pair_coherence(slc, diag, 2, 5, patch)
    expected = ds._filter_covariance(slc[2] * np.conj(slc[5]), patch)
    expected /= np.maximum(np.sqrt(diag[2] * diag[5]), 1e-9)
    expected = np.clip(np.abs(expected), 0.0, 1.0).astype(np.float32)

    assert np.allclose(pair, expected, atol=1e-6)
    assert np.allclose(pair_coherence(slc, diag, 3, 3, patch), 1.0)


def test_phase_link_rejects_short_stack_and_empty_mask():
    coreg, meta, _ = _synthetic_inputs(n=5)
    with pytest.raises(ValueError, match="至少需要 6 个历元"):
        phase_link(coreg, meta, np.ones(coreg.shape[1:], dtype=bool))

    coreg, meta, mask = _synthetic_inputs()
    with pytest.raises(RuntimeError, match="有效像元不足"):
        phase_link(coreg, meta, np.zeros_like(mask))


def test_phase_link_rejects_mismatched_baselines():
    coreg, meta, mask = _synthetic_inputs()
    meta["bperp_m"] = [0.0] * (coreg.shape[0] - 1)

    with pytest.raises(ValueError, match="bperp_m 数量"):
        phase_link(coreg, meta, mask)
