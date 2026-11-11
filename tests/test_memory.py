"""Memory-budget and block-wise multilook regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from insar_studio.core.memory import (
    MemoryBudgetError,
    available_memory_bytes,
    choose_block_rows,
    ensure_memory_budget,
    estimate_coregistration_peak,
    estimate_interferometry_peak,
    estimate_inversion_peak,
)
from insar_studio.core.utils import downsample_complex, downsample_complex_pair


def test_env_override_uses_binary_suffix(monkeypatch):
    monkeypatch.setenv("INSAR_MEMORY_BUDGET_BYTES", "8G")
    assert available_memory_bytes() == 8 * 1024**3
    monkeypatch.setenv("INSAR_MEMORY_BUDGET_BYTES", "512M")
    assert available_memory_bytes() == 512 * 1024**2
    monkeypatch.setenv("INSAR_MEMORY_BUDGET_BYTES", "1048576")
    assert available_memory_bytes() == 1048576


def test_invalid_override_falls_back_to_detection(monkeypatch):
    monkeypatch.setenv("INSAR_MEMORY_BUDGET_BYTES", "not-a-size")
    value = available_memory_bytes()
    assert value is None or value > 0


def test_choose_block_rows_honours_budget_and_multiple():
    block = choose_block_rows(1000, 1000, budget_bytes=10 * 1024**2, arrays=3, itemsize=8, multiple=2)
    assert block % 2 == 0
    assert block <= 1000
    assert block * 1000 * 3 * 8 <= 10 * 1024**2 + 1000 * 3 * 8
    # A single row is always returned instead of zero.
    assert choose_block_rows(100, 100, budget_bytes=1, multiple=3) == 3


def test_memory_guard_raises_with_actionable_message():
    available = 2 * 1024**3
    with pytest.raises(MemoryBudgetError) as excinfo:
        ensure_memory_budget(3 * 1024**3, "测试阶段", available=available)
    message = str(excinfo.value)
    assert "测试阶段" in message
    assert "INSAR_MEMORY_BUDGET_BYTES" in message
    assert "GiB" in message

    # Under the 90 % usable threshold, no error.
    ensure_memory_budget(1 * 1024**3, "测试阶段", available=available)


def test_estimates_grow_with_stack_size():
    assert estimate_coregistration_peak(100, 100) < estimate_coregistration_peak(200, 200)
    assert estimate_interferometry_peak(100, 100, 2) < estimate_interferometry_peak(400, 400, 2)
    assert estimate_inversion_peak(10, 10_000, 12) < estimate_inversion_peak(40, 100_000, 36)


def test_streaming_reference_pixel_matches_previous_formula():
    from insar_studio.core.interferometry import reference_pixel

    rng = np.random.default_rng(23)
    n, rows, cols = 9, 48, 56
    coreg = (rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))).astype(np.complex64)
    mask = np.ones((rows, cols), dtype=bool)
    step = max(1, n // 8)

    amp_stack = np.asarray([np.abs(coreg[i]) for i in range(0, n, step)], dtype=np.float32)
    mean_old = amp_stack.mean(axis=0)
    adi_old = amp_stack.std(axis=0) / np.maximum(mean_old, 1e-6)

    row, col, info = reference_pixel(coreg, mask, 1, ref_index=0)

    assert info["row"] == row and info["col"] == col
    # The streaming moments must reproduce the old per-date stack closely
    # enough that the selected stable pixel is unchanged.
    np.testing.assert_allclose(adi_old[row, col], info["adi"], rtol=1e-5, atol=1e-6)


def test_blockwise_pair_multilook_matches_full_operation():
    rng = np.random.default_rng(11)
    ref = (rng.normal(size=(37, 41)) + 1j * rng.normal(size=(37, 41))).astype(np.complex64)
    sec = (rng.normal(size=(37, 41)) + 1j * rng.normal(size=(37, 41))).astype(np.complex64)

    for factor in (1, 2, 3):
        ifg, ref_m, sec_m = downsample_complex_pair(ref, sec, factor, block_budget_bytes=1)
        expected_ifg = downsample_complex(sec * np.conj(ref), factor)
        expected_ref = downsample_complex(ref, factor)
        expected_sec = downsample_complex(sec, factor)
        assert ifg.shape == expected_ifg.shape
        assert ref_m.shape == expected_ref.shape
        assert sec_m.shape == expected_sec.shape
        np.testing.assert_allclose(ifg, expected_ifg, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_m, expected_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sec_m, expected_sec, rtol=1e-5, atol=1e-5)
