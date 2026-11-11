"""Project health diagnostics regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from insar_studio.health import diagnose_project


def _make_project(root: Path, bperp: list[float] | None, *, baseline_source: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(
        json.dumps(
            {
                "name": "health test",
                "technique": "sbas",
                "demo": False,
                "params": {},
                "steps_run": {},
            }
        ),
        encoding="utf-8",
    )
    stack = root / "data" / "stack"
    stack.mkdir(parents=True)
    n_dates, rows, cols = 6, 8, 10
    meta = {
        "n_dates": n_dates,
        "rows": rows,
        "cols": cols,
        "dates": [f"2023-01-{day:02d}" for day in range(1, 2 * n_dates, 2)],
        "wavelength_m": 0.05546576,
        "incidence_deg": 38.5,
    }
    if bperp is not None:
        meta["bperp_m"] = bperp
    if baseline_source is not None:
        meta["baseline_source"] = baseline_source
    (stack / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(stack / "slc.npy", np.ones((n_dates, rows, cols), dtype=np.complex64), allow_pickle=False)
    np.save(stack / "dem.npy", np.linspace(0, 100, rows * cols, dtype=np.float32).reshape(rows, cols), allow_pickle=False)
    mask = np.ones((rows, cols), dtype=bool)
    mask[:, :2] = False
    np.save(stack / "mask.npy", mask, allow_pickle=False)
    return root


def _baseline_check(health: dict) -> dict:
    return next(check for check in health["checks"] if check["name"] == "baseline")


def test_health_reads_bperp_m_and_warns_on_zero_baseline(tmp_path):
    root = _make_project(tmp_path / "project", [0.0] * 6)

    health = diagnose_project(root)
    check = _baseline_check(health)

    assert check["status"] == "warn"
    assert "垂直基线为 0" in check["detail"]


def test_health_reports_nonzero_bperp_m_as_ok(tmp_path):
    root = _make_project(tmp_path / "project", [-80.0, -40.0, 0.0, 25.0, 60.0, 95.0])

    health = diagnose_project(root)
    check = _baseline_check(health)

    assert check["status"] == "ok"
    assert "-80.0" in check["detail"]
    assert "bperp_m" not in check["detail"]


def test_health_marks_orbit_derived_baselines(tmp_path):
    root = _make_project(
        tmp_path / "project",
        [-80.0, -40.0, 0.0, 25.0, 60.0, 95.0],
        baseline_source="orbit",
    )

    check = _baseline_check(diagnose_project(root))

    assert check["status"] == "ok"
    assert "本地精密轨道" in check["detail"]


def test_health_warns_when_baseline_is_missing_or_mismatched(tmp_path):
    missing = diagnose_project(_make_project(tmp_path / "missing", None))
    assert _baseline_check(missing)["status"] == "warn"
    assert "缺少 bperp_m" in _baseline_check(missing)["detail"]

    mismatched = diagnose_project(_make_project(tmp_path / "mismatch", [1.0, 2.0]))
    assert _baseline_check(mismatched)["status"] == "warn"
    assert "不一致" in _baseline_check(mismatched)["detail"]
