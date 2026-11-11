"""PS amplitude-dispersion memory and low-coherence edge-case tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from insar_studio.core.io import Stack
from insar_studio.core.ps import amplitude_dispersion, select_ps
from insar_studio.core.utils import downsample_real


def _stack(n: int = 12, rows: int = 24, cols: int = 30) -> Stack:
    rng = np.random.default_rng(20261107)
    slc = (
        rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))
    ).astype(np.complex64)
    return Stack(
        root=Path("."),
        meta={},
        slc=slc,
        dem=np.zeros((rows, cols), dtype=np.float32),
        mask=np.ones((rows, cols), dtype=bool),
    )


def test_amplitude_dispersion_matches_reference_formula():
    stack = _stack()
    step = max(1, stack.n_dates // 12)
    amps = np.asarray([np.abs(stack.slc[i]) for i in range(0, stack.n_dates, step)], dtype=np.float32)
    mean = np.maximum(amps.mean(axis=0), 1e-6)
    expected = downsample_real(amps.std(axis=0) / mean, 2).astype(np.float32)

    result = amplitude_dispersion(stack, multilook=2)

    assert result.shape == expected.shape
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_amplitude_dispersion_blocking_is_equivalent():
    stack = _stack()
    default = amplitude_dispersion(stack, multilook=2)
    blocked = amplitude_dispersion(stack, multilook=2, block_rows=6)

    assert np.allclose(default, blocked, atol=1e-6)


def test_amplitude_dispersion_constant_amplitude_is_zero():
    stack = _stack()
    stack.slc[:] = 2.0 + 0.0j

    result = amplitude_dispersion(stack, multilook=1)

    assert np.allclose(result, 0.0)


def test_select_ps_all_nan_coherence_yields_no_candidates():
    stack = _stack()
    coherence = np.full((3, 24, 30), np.nan, dtype=np.float32)
    valid = np.ones((3, 24, 30), dtype=bool)

    result = select_ps(stack, coherence, valid, {"multilook": 1})

    assert result["stats"]["n_candidates"] == 0
    assert result["stats"]["relaxed"] is False
    assert np.isfinite(result["mean_coherence"]).all()
    assert not result["mask"].any()
