"""Ramp removal and parameter-validation tests."""

from __future__ import annotations

import numpy as np
import pytest

from insar_studio.core.interferometry import _remove_ramp
from insar_studio.params import ParamError, validate_params


def _quadratic_ramp(rows: int = 72, cols: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    xn = xx / (cols - 1) - 0.5
    yn = yy / (rows - 1) - 0.5
    return (
        0.40 * xn
        + 0.20 * yn
        + 0.80 * xn * xn
        - 0.55 * yn * yn
        + 0.35 * xn * yn
        + 0.03
    ).astype(np.float32)


def test_second_order_ramp_removal_removes_bilinear_and_quadratic_terms():
    phase = _quadratic_ramp()
    coherence = np.ones_like(phase, dtype=np.float32)
    valid = np.ones_like(phase, dtype=bool)
    valid[:2, :] = valid[-2:, :] = False
    valid[:, :2] = valid[:, -2:] = False

    out, info = _remove_ramp(phase, coherence, valid, order=2)

    assert info["removed"] is True
    assert info["order"] == 2
    assert set(info["terms"]) == {"x", "y", "intercept", "x2", "y2", "xy"}
    assert float(np.nanmax(np.abs(out[valid]))) < 1e-4


def test_first_order_ramp_removal_leaves_quadratic_residual():
    phase = _quadratic_ramp()
    coherence = np.ones_like(phase, dtype=np.float32)
    valid = np.ones_like(phase, dtype=bool)

    plane, plane_info = _remove_ramp(phase, coherence, valid, order=1)
    quadratic, quadratic_info = _remove_ramp(phase, coherence, valid, order=2)

    assert plane_info["order"] == 1
    assert quadratic_info["order"] == 2
    assert float(np.nanstd(plane[valid])) > 0.05
    assert float(np.nanstd(quadratic[valid])) < 1e-4


def test_ramp_order_is_validated():
    from insar_studio.config import DEFAULT_PROCESSING

    assert DEFAULT_PROCESSING["ramp_order"] == 1
    assert validate_params({"ramp_order": 1})["ramp_order"] == 1
    assert validate_params({"ramp_order": 2})["ramp_order"] == 2
    for bad in (0, 3, 1.5, "two"):
        with pytest.raises(ParamError):
            validate_params({"ramp_order": bad})


def test_blockwise_ramp_removal_matches_whole_image(monkeypatch):
    phase = _quadratic_ramp(rows=180, cols=220)
    coherence = np.ones_like(phase, dtype=np.float32)
    valid = np.ones_like(phase, dtype=bool)

    reference, _ = _remove_ramp(phase, coherence, valid, order=2)
    monkeypatch.setattr("insar_studio.core.interferometry.choose_block_rows", lambda *a, **k: 7)
    blocked, info = _remove_ramp(phase, coherence, valid, order=2)

    assert info["removed"] is True
    np.testing.assert_allclose(blocked, reference, rtol=0, atol=1e-5)
