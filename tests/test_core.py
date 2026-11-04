"""Unit tests for the numerical core.

These tests deliberately avoid the full processing pipeline so they stay fast;
the end-to-end route is covered by ``test_pipeline.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from insar_studio.core.filters import boxcar_filter, filter_complex, gaussian_filter_complex
from insar_studio.core.network import build_network, network_closure_candidates
from insar_studio.core.unwrapping import unwrap
from insar_studio.core.utils import wrap_phase


def _dates(n: int) -> list[str]:
    start = date(2023, 1, 3)
    return [(start + timedelta(days=12 * i)).isoformat() for i in range(n)]


def test_sbas_network_is_connected_and_within_baselines():
    dates = _dates(16)
    bperp = np.linspace(-120.0, 130.0, len(dates)).tolist()
    pairs, stats = build_network(
        dates,
        bperp,
        "sbas",
        {"max_temporal_baseline": 60, "max_perpendicular_baseline": 160, "max_pairs": 60},
    )

    assert stats["n_pairs"] == len(pairs)
    assert len(pairs) >= len(dates) - 1
    assert len(pairs) <= 60
    assert stats["connected_components"] == 1

    covered = {p["ref"] for p in pairs} | {p["sec"] for p in pairs}
    assert covered == set(range(len(dates)))
    assert all(p["ref"] < p["sec"] for p in pairs)
    assert all(p["dt_days"] <= 60 for p in pairs)
    assert all(abs(p["bperp_diff"]) <= 160 for p in pairs)


def test_ps_network_uses_a_star_and_adjacent_pairs():
    dates = _dates(14)
    bperp = np.linspace(-90.0, 90.0, len(dates)).tolist()
    pairs, stats = build_network(dates, bperp, "ps", {"ps_star_all": True})

    assert stats["connected_components"] == 1
    assert any(p["ref"] == 0 for p in pairs)
    assert any(p["sec"] - p["ref"] == 1 for p in pairs)
    # star pairs connect the reference to every other acquisition
    star_targets = {p["sec"] for p in pairs if p["ref"] == 0}
    assert star_targets == set(range(1, len(dates)))


def test_dinsar_network_contains_exactly_the_requested_pair():
    dates = _dates(20)
    bperp = np.linspace(-50.0, 50.0, len(dates)).tolist()
    pairs, stats = build_network(dates, bperp, "dinsar", {"dinsar_pair": [3, 17]})

    assert len(pairs) == 1
    assert stats["n_pairs"] == 1
    assert (pairs[0]["ref"], pairs[0]["sec"]) == (3, 17)
    assert pairs[0]["id"] == "03_17"
    assert pairs[0]["dt_days"] == 14 * 12


def test_network_closure_candidate_count():
    dates = _dates(8)
    bperp = np.linspace(-40.0, 40.0, len(dates)).tolist()
    pairs, _ = build_network(dates, bperp, "sbas", {"max_temporal_baseline": 36})
    assert network_closure_candidates(pairs) >= 0


def test_boxcar_and_gaussian_filters_reduce_phase_noise():
    rng = np.random.default_rng(7)
    clean = np.exp(1j * wrap_phase(np.linspace(0.0, 4.0, 60 * 70).reshape(60, 70)))
    noisy = clean * np.exp(1j * rng.normal(0.0, 0.6, clean.shape))
    noisy = noisy.astype(np.complex64)

    smoothed = boxcar_filter(noisy, size=3)
    gaussian = gaussian_filter_complex(noisy, sigma=1.5)
    goldstein = filter_complex(noisy, "goldstein", alpha=0.65, patch=16)

    for result in (smoothed, gaussian, goldstein):
        assert result.shape == noisy.shape
        assert np.isfinite(result).all()

    baseline = np.mean(np.abs(np.angle(noisy * np.conj(clean))))
    assert np.mean(np.abs(np.angle(smoothed * np.conj(clean)))) < baseline
    assert np.mean(np.abs(np.angle(gaussian * np.conj(clean)))) < baseline


def test_quality_guided_unwrap_recovers_a_smooth_ramp():
    rows, cols = 44, 52
    yy, xx = np.mgrid[0:rows, 0:cols]
    truth = 0.22 * xx + 0.16 * yy
    wrapped = wrap_phase(truth).astype(np.float32)
    coherence = np.full((rows, cols), 0.93, dtype=np.float32)
    valid = np.ones((rows, cols), dtype=bool)

    unwrapped, info = unwrap(
        wrapped,
        coherence,
        valid,
        method="quality",
        coherence_min=0.0,
    )

    assert info["method"] == "quality_guided"
    assert np.isfinite(unwrapped).all()
    offset = np.nanmedian(unwrapped - truth)
    residual = unwrapped - truth - offset
    assert np.nanstd(residual) < 0.35
    assert np.nanmax(np.abs(residual)) < 1.2


def test_unwrap_falls_back_when_snaphu_is_unavailable():
    rows, cols = 24, 28
    yy, xx = np.mgrid[0:rows, 0:cols]
    wrapped = wrap_phase(0.18 * xx - 0.11 * yy).astype(np.float32)
    coherence = np.full((rows, cols), 0.9, dtype=np.float32)

    unwrapped, info = unwrap(
        wrapped,
        coherence,
        np.ones((rows, cols), dtype=bool),
        method="snaphu",
        snaphu_exe="definitely-not-installed-snaphu",
    )

    assert info["method"] == "quality_guided"
    assert "fallback" in info
    assert np.isfinite(unwrapped).all()
