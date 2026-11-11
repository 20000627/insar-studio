"""Sentinel-1 orbit parsing and baseline-estimation tests."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from insar_studio.core.orbit import (
    OrbitError,
    collect_safe_metadata,
    estimate_bperp,
    perpendicular_baselines,
    read_eof,
    read_safe_annotation,
)
from insar_studio.importer import import_stack
from insar_studio.project import ProjectStore

MU = 3.986004418e14
RADIUS = 7_071_000.0
OMEGA = math.sqrt(MU / RADIUS**3)
INCLINATION = math.radians(98.2)
EPOCH = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _state(seconds: float, drift_m_per_day: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    theta = OMEGA * seconds
    position = np.array(
        [
            RADIUS * math.cos(theta),
            RADIUS * math.sin(theta) * math.cos(INCLINATION),
            RADIUS * math.sin(theta) * math.sin(INCLINATION),
        ]
    )
    velocity = np.array(
        [
            -RADIUS * OMEGA * math.sin(theta),
            RADIUS * OMEGA * math.cos(theta) * math.cos(INCLINATION),
            RADIUS * OMEGA * math.cos(theta) * math.sin(INCLINATION),
        ]
    )
    position[2] += drift_m_per_day * seconds / 86400.0
    return position, velocity


def _write_eof(path: Path, days: float, *, drift: float = 0.0, step_s: int = 120) -> Path:
    lines = ["SENTINEL-1A ORBIT"]
    for seconds in range(0, int(days * 86400) + 1, step_s):
        position, velocity = _state(float(seconds), drift)
        stamp = EPOCH + timedelta(seconds=seconds)
        lines.append(
            stamp.strftime("%Y-%m-%dT%H:%M:%S.000")
            + " "
            + " ".join(f"{value:.6f}" for value in (*position, *velocity))
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_eof_parser_and_hermite_interpolation(tmp_path):
    path = _write_eof(tmp_path / "S1A_OPER_AUX_POEORB_TEST.EOF", days=1)
    orbit = read_eof(path)

    assert orbit.kind == "precise"
    assert orbit.start == EPOCH
    assert len(orbit.times) > 2

    when = EPOCH + timedelta(seconds=90)
    position, velocity = orbit.state(when)
    expected_position, expected_velocity = _state(90.0)
    assert np.linalg.norm(position - expected_position) < 0.01
    assert np.linalg.norm(velocity - expected_velocity) < 1e-4


def test_perpendicular_baselines_follow_cross_track_drift(tmp_path):
    path = _write_eof(
        tmp_path / "_synthetic_orbit.EOF",
        days=10,
        drift=100.0,
    )
    orbit = read_eof(path)
    times = [EPOCH + timedelta(days=day) for day in (0, 2, 5, 8)]
    baselines = perpendicular_baselines([orbit], times, ref_index=0)

    ref_position, ref_velocity = _state(0.0)
    normal = np.cross(ref_velocity, ref_position)
    normal /= np.linalg.norm(normal)
    expected = [
        float(np.dot(_state(day * 86400.0, 100.0)[0] - ref_position, normal))
        for day in (0, 2, 5, 8)
    ]
    assert baselines[0] == pytest.approx(0.0, abs=1e-6)
    assert baselines == pytest.approx(expected, abs=1e-3)


def test_estimate_bperp_requires_start_times(tmp_path):
    _write_eof(tmp_path / "S1A_OPER_AUX_POEORB_TEST.EOF", days=10)
    with pytest.raises(OrbitError):
        estimate_bperp(
            tmp_path,
            ["2023-01-01", "2023-01-03"],
            ["", ""],
            ref_index=0,
        )


def test_safe_annotation_reader(tmp_path):
    safe = tmp_path / "S1A_IW_SLC__1SDV_TEST.SAFE"
    measurement = safe / "measurement" / "s1a-iw1-slc-vv-20230103t052345-20230103t052412-001.xml.tiff"
    annotation_dir = safe / "annotation"
    measurement.parent.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    measurement.write_bytes(b"")
    annotation = annotation_dir / f"{measurement.stem}.xml"
    annotation.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<product>
  <imageAnnotation>
    <imageInformation>
      <productFirstLineUtcTime>2023-01-03T05:23:45.123456</productFirstLineUtcTime>
      <azimuthTimeInterval>0.002055556</azimuthTimeInterval>
      <slantRangeTime>0.0055</slantRangeTime>
      <rangePixelSpacing>2.329562</rangePixelSpacing>
      <azimuthPixelSpacing>13.968</azimuthPixelSpacing>
      <incidenceAngleMidSwath>38.9</incidenceAngleMidSwath>
    </imageInformation>
    <orbitNumber>45678</orbitNumber>
    <relativeOrbitNumber>42</relativeOrbitNumber>
    <pass>DESCENDING</pass>
  </imageAnnotation>
</product>
""",
        encoding="utf-8",
    )

    info = read_safe_annotation(annotation)
    assert info["start_time"] == datetime(2023, 1, 3, 5, 23, 45, 123456, tzinfo=timezone.utc)
    assert info["incidence_deg"] == pytest.approx(38.9)
    assert info["range_spacing_m"] == pytest.approx(2.329562)
    assert info["orbit_number"] == 45678
    assert info["relative_orbit"] == 42
    assert info["orbit_direction"] == "DESCENDING"

    collected = collect_safe_metadata([measurement])
    assert collected["start_times"][0] == info["start_time"]
    assert collected["incidence_deg"][0] == pytest.approx(38.9)


def test_importer_uses_local_orbit_for_bperp(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Orbit import", "dinsar")
    root = store.path(project["id"])

    source = tmp_path / "source"
    source.mkdir()
    n, rows, cols = 4, 12, 14
    rng = np.random.default_rng(7)
    slc = (rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))).astype(np.complex64)
    np.save(source / "slc.npy", slc)
    np.save(source / "dem.npy", np.zeros((rows, cols), dtype=np.float32))
    np.save(source / "mask.npy", np.ones((rows, cols), dtype=bool))

    days = [0, 2, 4, 6]
    meta = {
        "dates": [f"2023-01-{1 + day:02d}" for day in days],
        "start_times": [(EPOCH + timedelta(days=day)).isoformat() for day in days],
        "orbit_dir": str(tmp_path / "orbits"),
    }
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    orbits = tmp_path / "orbits"
    orbits.mkdir()
    _write_eof(orbits / "S1A_OPER_AUX_POEORB_TEST.EOF", days=10, drift=100.0)

    result = import_stack(root, source, technique="dinsar")

    assert result["baseline_source"] == "orbit"
    assert result["orbit"]["n_orbits"] == 1
    assert result["bperp_m"][0] == pytest.approx(0.0, abs=1e-3)
    assert max(abs(value) for value in result["bperp_m"]) > 50.0
    assert result["has_baselines"] is True
    assert any("轨道文件估计垂直基线" in warning for warning in result["import_warnings"])


def test_importer_warns_when_orbit_is_absent(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("No orbit", "dinsar")
    root = store.path(project["id"])

    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "slc.npy", np.ones((3, 10, 10), dtype=np.complex64))
    np.save(source / "mask.npy", np.ones((10, 10), dtype=bool))
    (source / "meta.json").write_text(
        json.dumps({"dates": ["2023-01-01", "2023-01-03", "2023-01-05"]}),
        encoding="utf-8",
    )

    result = import_stack(root, source, technique="dinsar")

    assert result["baseline_source"] == "none"
    assert result["bperp_m"] == [0.0, 0.0, 0.0]
    assert any("未找到可用的本地精密轨道" in warning for warning in result["import_warnings"])
