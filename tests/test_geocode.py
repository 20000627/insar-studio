"""Range-Doppler geocoding regression tests."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from insar_studio.core.geocode import (
    GeocodeError,
    MapGrid,
    RadarGeometry,
    build_lookup,
    derive_map_grid,
    ecef_to_geodetic,
    geocode_products,
    geocode_readiness,
    geodetic_to_ecef,
    map_grid,
    sample_product,
)
from insar_studio.core.geotiff import read_geotiff
from insar_studio.core.orbit import Orbit
from insar_studio.health import diagnose_project
from insar_studio.importer import import_stack
from insar_studio.project import ProjectStore

MU = 3.986004418e14
RADIUS = 7_071_000.0
OMEGA = math.sqrt(MU / RADIUS**3)
INCLINATION = math.radians(98.2)
EPOCH = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _state(seconds: float) -> tuple[np.ndarray, np.ndarray]:
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
    return position, velocity


def _synthetic_orbit(span_s: int = 600, step_s: int = 60) -> Orbit:
    times = np.arange(-span_s, span_s + 1, step_s, dtype=float)
    positions = []
    velocities = []
    for seconds in times:
        position, velocity = _state(float(seconds))
        positions.append(position)
        velocities.append(velocity)
    return Orbit(
        times=times,
        positions=np.array(positions),
        velocities=np.array(velocities),
        epoch=EPOCH,
    )


def _fine_geometry() -> RadarGeometry:
    """Small-ish scene with tens-of-metres pixels for round-trip tests."""
    return RadarGeometry(
        orbit=_synthetic_orbit(),
        first_line_time=EPOCH + timedelta(seconds=-30.0),
        azimuth_time_interval=0.002,
        slant_range_first=690_000.0,
        range_spacing=2.3,
        rows=30_000,
        cols=4_000,
    )


def _coarse_geometry() -> RadarGeometry:
    """80 x 100 scene used for grid derivation and sampling accuracy."""
    return RadarGeometry(
        orbit=_synthetic_orbit(),
        first_line_time=EPOCH + timedelta(seconds=-0.4),
        azimuth_time_interval=0.01,
        slant_range_first=850_000.0,
        range_spacing=230.0,
        rows=80,
        cols=100,
    )


def _write_geometry_stack(root: Path, geometry: RadarGeometry, **extra) -> dict:
    stack = root / "data" / "stack"
    stack.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "insar-studio/orbits/v1",
        "ref_index": 0,
        "dates": ["2023-01-01"],
        "orbits": [geometry.orbit.to_dict()],
    }
    (stack / "orbits.json").write_text(json.dumps(payload), encoding="utf-8")
    meta = {
        "rows": geometry.rows,
        "cols": geometry.cols,
        "ref_index": 0,
        "start_times": [geometry.first_line_time.isoformat()],
        "azimuth_time_interval": geometry.azimuth_time_interval,
        "range_spacing_m": geometry.range_spacing,
        "slant_range_time_s": 2.0 * geometry.slant_range_first / 299_792_458.0,
        "slant_range_source": "annotation",
        "incidence_deg": 38.5,
        "azimuth_spacing_m": 13.968,
        "wavelength_m": 0.05546576,
        "orbit": {"vectors_path": "data/stack/orbits.json", "source": "test"},
        **extra,
    }
    (stack / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_wgs84_geodetic_ecef_roundtrip():
    lon = np.array([-179.9, -73.5, 0.0, 106.52, 179.9])
    lat = np.array([-89.5, -45.0, 0.0, 29.6, 89.5])
    height = np.array([-50.0, 0.0, 1200.0, 3500.0, 9000.0])

    xyz = geodetic_to_ecef(lon, lat, height)
    lon2, lat2, height2 = ecef_to_geodetic(xyz)

    assert np.allclose(lon2, lon, atol=1e-9)
    assert np.allclose(lat2, lat, atol=1e-9)
    assert np.allclose(height2, height, atol=1e-3)
    radii = np.linalg.norm(xyz, axis=1)
    assert np.all(radii > 6.3e6)
    assert np.all(radii < 6.5e6)


def test_ground_to_radar_matches_independent_zero_doppler():
    geometry = _fine_geometry()
    lon, lat, height = 0.2, 0.1, 0.0
    target = geodetic_to_ecef(lon, lat, height)

    def doppler_seconds(seconds: float) -> float:
        position, velocity = geometry.orbit.interpolate(np.array([seconds]))
        return float(np.dot(target - position[0], velocity[0]))

    seconds = brentq(doppler_seconds, -600.0, 600.0, xtol=1e-10)
    position, _ = geometry.orbit.interpolate(np.array([seconds]))
    slant = float(np.linalg.norm(target - position[0]))
    expected_row = (seconds + 30.0) / geometry.azimuth_time_interval
    expected_col = (slant - geometry.slant_range_first) / geometry.range_spacing

    row, col, _, _ = geometry.ground_to_radar(
        np.array(lon), np.array(lat), np.array(height)
    )

    assert row.item() == pytest.approx(expected_row, abs=1e-4)
    assert col.item() == pytest.approx(expected_col, abs=1e-6)


def test_ground_radar_roundtrip_inside_scene():
    geometry = _fine_geometry()
    rows = np.array([15_000.0, 29_990.0])
    cols = np.array([1_500.0, 3_999.0])

    lon, lat = geometry.radar_to_ground(rows, cols, 0.0)
    row2, col2, _, _ = geometry.ground_to_radar(lon, lat, np.zeros_like(lon))

    assert np.isfinite(lon).all()
    assert np.isfinite(lat).all()
    assert np.allclose(row2, rows, atol=1e-3)
    assert np.allclose(col2, cols, atol=1e-3)


def test_map_grid_normalises_south_up_input():
    grid = MapGrid(
        rows=10,
        cols=20,
        corner_lon=100.0,
        corner_lat=30.0,
        lon_step=0.01,
        lat_step=0.02,
    )

    assert grid.lat_step == pytest.approx(-0.02)
    assert grid.corner_lat == pytest.approx(30.2)
    assert grid.bounds == pytest.approx((100.0, 30.0, 100.2, 30.2))
    lon, lat = grid.centers()
    assert lon.shape == (10, 20)
    assert lat.shape == (10, 20)
    assert lon[0, 0] == pytest.approx(100.005)
    assert lat[0, 0] == pytest.approx(30.2 - 0.01)

    with pytest.raises(GeocodeError):
        MapGrid(rows=0, cols=1, corner_lon=0, corner_lat=0, lon_step=1, lat_step=-1)
    with pytest.raises(GeocodeError):
        MapGrid(rows=1, cols=1, corner_lon=0, corner_lat=0, lon_step=0, lat_step=-1)


def test_derived_grid_covers_footprint_centre():
    geometry = _coarse_geometry()
    rows = np.arange(geometry.rows)
    cols = np.arange(geometry.cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dem = (900.0 * np.sin(rr / geometry.rows * np.pi) * np.cos(cc / geometry.cols * np.pi)).astype(np.float64)
    meta = {"incidence_deg": 38.5, "range_spacing_m": geometry.range_spacing, "azimuth_spacing_m": 13.968}

    grid = derive_map_grid(
        meta,
        geometry,
        dem,
        multilook=1,
        product_shape=(geometry.rows, geometry.cols),
    )

    center_lon, center_lat = geometry.radar_to_ground(
        np.array([40.0]), np.array([50.0]), np.array([0.0])
    )
    west, south, east, north = grid.bounds
    assert grid.rows > 1 and grid.cols > 1
    assert grid.lat_step < 0
    assert west < center_lon.item() < east
    assert south < center_lat.item() < north
    assert grid.to_dict()["crs"] == "EPSG:4326"


def test_map_grid_prefers_declared_grid_only_when_flagged():
    geometry = _coarse_geometry()
    declared = {
        "rows": 12,
        "cols": 14,
        "corner_lon": 100.0,
        "corner_lat": 30.0,
        "lon_step": 0.001,
        "lat_step": -0.001,
    }

    grid, notes = map_grid(
        {"geo_referenced": False},
        geometry,
        None,
        product_shape=(geometry.rows, geometry.cols),
        multilook=1,
        grid_meta=declared,
    )
    assert (grid.rows, grid.cols) != (12, 14)
    assert any("推导" in note for note in notes)

    grid, notes = map_grid(
        {"geo_referenced": True},
        geometry,
        None,
        product_shape=(geometry.rows, geometry.cols),
        multilook=1,
        grid_meta=declared,
    )
    assert (grid.rows, grid.cols) == (12, 14)
    assert grid.bounds == pytest.approx((100.0, 30.0 - 12 * 0.001, 100.0 + 14 * 0.001, 30.0))
    assert any("声明" in note for note in notes)

    grid, notes = map_grid(
        {"geo_referenced": True},
        geometry,
        None,
        product_shape=(geometry.rows, geometry.cols),
        multilook=1,
        grid_meta={"rows": 0},
    )
    assert (grid.rows, grid.cols) != (0, 0)
    assert any("无效" in note for note in notes)


def test_lookup_and_bilinear_sampling_accuracy():
    geometry = _coarse_geometry()
    rows, cols = geometry.rows, geometry.cols
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dem = (900.0 * np.sin(rr / rows * np.pi) * np.cos(cc / cols * np.pi)).astype(np.float64)
    lon_r, lat_r = geometry.radar_to_ground(rr, cc, dem)
    product = (
        10.0 * np.sin(np.deg2rad(lon_r * 20.0))
        + 5.0 * np.cos(np.deg2rad(lat_r * 20.0))
        + dem / 1000.0
    )
    product = np.where(np.isfinite(lon_r), product, np.nan)
    meta = {"incidence_deg": 38.5, "range_spacing_m": geometry.range_spacing, "azimuth_spacing_m": 13.968}

    grid = derive_map_grid(
        meta, geometry, dem, multilook=1, product_shape=(rows, cols)
    )
    lookup = build_lookup(
        geometry, grid, dem, multilook=1, product_shape=(rows, cols)
    )
    sampled = sample_product(product, lookup)
    lon_m, lat_m = grid.centers()
    expected = (
        10.0 * np.sin(np.deg2rad(lon_m * 20.0))
        + 5.0 * np.cos(np.deg2rad(lat_m * 20.0))
        + lookup["height_m"] / 1000.0
    )
    mask = np.isfinite(sampled) & np.isfinite(expected)
    errors = np.abs(sampled[mask] - expected[mask])

    assert 0.05 < lookup["coverage"] < 0.5
    assert lookup["iterations"] >= 1
    assert mask.sum() > 5_000
    assert float(np.median(errors)) < 5e-4
    assert float(np.percentile(errors, 95)) < 5e-3
    assert float(errors.max()) < 5e-2
    assert np.isnan(sampled[~lookup["valid"]]).all()


def test_lookup_block_and_thread_equivalence():
    geometry = _coarse_geometry()
    rows, cols = geometry.rows, geometry.cols
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dem = (900.0 * np.sin(rr / rows * np.pi) * np.cos(cc / cols * np.pi)).astype(np.float64)
    meta = {
        "incidence_deg": 38.5,
        "range_spacing_m": geometry.range_spacing,
        "azimuth_spacing_m": 13.968,
    }
    grid = derive_map_grid(meta, geometry, dem, multilook=1, product_shape=(rows, cols))

    full = build_lookup(
        geometry, grid, dem, multilook=1, product_shape=(rows, cols), block_rows=grid.rows
    )
    blocked = build_lookup(
        geometry, grid, dem, multilook=1, product_shape=(rows, cols), block_rows=7
    )
    threaded = build_lookup(
        geometry,
        grid,
        dem,
        multilook=1,
        product_shape=(rows, cols),
        block_rows=5,
        workers=3,
        max_block_pixels=1_000,
    )

    assert full["blocks"] == 1
    assert blocked["blocks"] > 1
    assert threaded["blocks"] > blocked["blocks"]
    assert threaded["workers"] == 3
    for candidate in (blocked, threaded):
        assert np.array_equal(full["valid"], candidate["valid"])
        assert np.allclose(full["row"], candidate["row"], atol=1e-3, equal_nan=True)
        assert np.allclose(full["col"], candidate["col"], atol=1e-3, equal_nan=True)
        assert np.allclose(full["height_m"], candidate["height_m"], atol=0.05)
        assert candidate["coverage"] == pytest.approx(full["coverage"], abs=1e-6)


def test_geocode_products_done_writes_geotiff_and_status(tmp_path):
    root = tmp_path / "project"
    geometry = _coarse_geometry()
    meta = _write_geometry_stack(root, geometry)
    lon_c, lat_c = geometry.radar_to_ground(
        np.array([40.0]), np.array([50.0]), np.array([0.0])
    )
    lon_c, lat_c = float(lon_c.item()), float(lat_c.item())
    step = 0.0004
    rows, cols = 10, 10
    grid_meta = {
        "rows": rows,
        "cols": cols,
        "corner_lon": lon_c - 0.5 * cols * step,
        "corner_lat": lat_c + 0.5 * rows * step,
        "lon_step": step,
        "lat_step": -step,
    }
    meta.update({"geo_referenced": True})
    product = np.linspace(0.0, 30.0, geometry.rows * geometry.cols, dtype=np.float32).reshape(
        geometry.rows, geometry.cols
    )

    info = geocode_products(
        root,
        meta,
        {"multilook": 1},
        product_shape=(geometry.rows, geometry.cols),
        grid_meta=grid_meta,
        dem=None,
        products={"velocity_mm_yr": product},
        technique="sbas",
    )

    assert info["status"] == "done"
    assert info["coverage"] > 0.95
    assert info["files"] == ["products/geotiff/geo/velocity_mm_yr.tif"]
    assert info["lookup_blocks"] >= 1
    assert info["lookup_block_rows"] >= 1
    assert info["lookup_workers"] >= 1
    assert info["lookup_seconds"] >= 0
    saved = json.loads((root / "products" / "geocode.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done"
    assert saved["grid"]["rows"] == rows
    assert saved["grid"]["bounds"] == pytest.approx(
        [lon_c - 0.5 * cols * step, lat_c - 0.5 * rows * step, lon_c + 0.5 * cols * step, lat_c + 0.5 * rows * step]
    )

    array, geo = read_geotiff(root / "products" / "geotiff" / "geo" / "velocity_mm_yr.tif")
    assert array.shape == (rows, cols)
    assert geo["crs"] == "EPSG:4326"
    assert geo["corner_lon"] == pytest.approx(grid_meta["corner_lon"])
    assert geo["corner_lat"] == pytest.approx(grid_meta["corner_lat"])
    assert geo["lon_step"] == pytest.approx(step)
    assert geo["lat_step"] == pytest.approx(-step)


def test_geocode_products_partial_on_derived_grid(tmp_path):
    root = tmp_path / "project"
    geometry = _coarse_geometry()
    meta = _write_geometry_stack(root, geometry)
    product = np.zeros((geometry.rows, geometry.cols), dtype=np.float32)

    info = geocode_products(
        root,
        meta,
        {"multilook": 1},
        product_shape=(geometry.rows, geometry.cols),
        grid_meta=None,
        dem=None,
        products={"velocity_mm_yr": product},
        technique="sbas",
    )

    assert info["status"] == "partial"
    assert 0.0 < info["coverage"] < 0.5
    assert info["files"] == ["products/geotiff/geo/velocity_mm_yr.tif"]
    assert (root / "products" / "geotiff" / "geo" / "velocity_mm_yr.tif").exists()


def test_geocode_products_unavailable_without_orbit(tmp_path):
    root = tmp_path / "project"
    (root / "products").mkdir(parents=True)
    meta = {"rows": 10, "cols": 12, "start_times": ["2023-01-01T00:00:00Z"]}

    info = geocode_products(
        root,
        meta,
        {"multilook": 1},
        product_shape=(10, 12),
        grid_meta=None,
        dem=None,
        products={"velocity_mm_yr": np.zeros((10, 12), dtype=np.float32)},
        technique="sbas",
    )

    assert info["status"] == "unavailable"
    assert info["files"] == []
    assert any("轨道" in reason for reason in info["reasons"])
    assert not (root / "products" / "geotiff" / "geo").exists()
    saved = json.loads((root / "products" / "geocode.json").read_text(encoding="utf-8"))
    assert saved["status"] == "unavailable"


def test_geocode_readiness_explains_missing_slant_range(tmp_path):
    geometry = _coarse_geometry()
    root = tmp_path / "project"
    meta = _write_geometry_stack(root, geometry)

    ready = geocode_readiness(meta, root)
    assert ready["ready"] is True
    assert ready["look_side"] == "right"
    assert ready["dem_available"] is False

    broken = dict(meta)
    broken.pop("slant_range_time_s")
    broken["slant_range_m"] = 850_000.0
    broken["slant_range_source"] = "default"
    not_ready = geocode_readiness(broken, root)
    assert not_ready["ready"] is False
    assert any("首列斜距" in reason for reason in not_ready["reasons"])


def test_importer_writes_orbit_vectors(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Orbit vectors", "dinsar")
    root = store.path(project["id"])

    source = tmp_path / "source"
    source.mkdir()
    n, rows, cols = 4, 12, 14
    rng = np.random.default_rng(11)
    slc = (rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))).astype(np.complex64)
    np.save(source / "slc.npy", slc)
    np.save(source / "dem.npy", np.zeros((rows, cols), dtype=np.float32))
    np.save(source / "mask.npy", np.ones((rows, cols), dtype=bool))

    days = [0, 2, 4, 6]
    meta = {
        "dates": [f"2023-01-{1 + day:02d}" for day in days],
        "start_times": [(EPOCH + timedelta(days=day)).isoformat() for day in days],
        "orbit_dir": str(tmp_path / "orbits"),
        "slant_range_time_s": 0.0055,
        "azimuth_time_interval": 0.002,
    }
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    orbits_dir = tmp_path / "orbits"
    orbits_dir.mkdir()
    lines = ["SENTINEL-1A ORBIT"]
    for seconds in range(0, 8 * 86400 + 1, 120):
        position, velocity = _state(float(seconds))
        stamp = EPOCH + timedelta(seconds=seconds)
        lines.append(
            stamp.strftime("%Y-%m-%dT%H:%M:%S.000")
            + " "
            + " ".join(f"{value:.6f}" for value in (*position, *velocity))
        )
    (orbits_dir / "S1A_OPER_AUX_POEORB_TEST.EOF").write_text("\n".join(lines), encoding="utf-8")

    result = import_stack(root, source, technique="dinsar")

    stack_meta = json.loads((root / "data" / "stack" / "meta.json").read_text(encoding="utf-8"))
    payload = json.loads((root / "data" / "stack" / "orbits.json").read_text(encoding="utf-8"))
    assert result["baseline_source"] == "orbit"
    assert stack_meta["orbit"]["vectors_path"] == "data/stack/orbits.json"
    assert stack_meta["orbit"]["n_vectors"] > 0
    assert stack_meta["slant_range_time_s"] == pytest.approx(0.0055)
    assert stack_meta["slant_range_source"] == "annotation"
    assert payload["schema"] == "insar-studio/orbits/v1"
    assert payload["ref_index"] == 0
    assert payload["dates"] == stack_meta["dates"]
    assert len(payload["orbits"]) == n
    assert len(payload["orbits"][0]["times"]) == len(payload["orbits"][0]["positions"])


def _make_health_project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(
        json.dumps(
            {
                "name": "geocode health",
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
    n, rows, cols = 6, 8, 10
    meta = {
        "n_dates": n,
        "rows": rows,
        "cols": cols,
        "dates": [f"2023-01-{day:02d}" for day in range(1, 2 * n, 2)],
        "wavelength_m": 0.05546576,
        "incidence_deg": 38.5,
        "bperp_m": [0.0, 40.0, 70.0, 95.0, 120.0, 150.0],
    }
    (stack / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(stack / "slc.npy", np.ones((n, rows, cols), dtype=np.complex64), allow_pickle=False)
    np.save(stack / "dem.npy", np.zeros((rows, cols), dtype=np.float32), allow_pickle=False)
    np.save(stack / "mask.npy", np.ones((rows, cols), dtype=bool), allow_pickle=False)
    return root


def test_health_reports_geocoding_status(tmp_path):
    root = _make_health_project(tmp_path / "project")
    health = diagnose_project(root)
    check = next(item for item in health["checks"] if item["name"] == "geocoding")
    assert check["status"] == "warn"
    assert "无法地理编码" in check["detail"]

    products = root / "products"
    products.mkdir()
    (products / "geocode.json").write_text(
        json.dumps(
            {
                "status": "done",
                "coverage": 0.83,
                "reasons": [],
                "warnings": [],
                "files": ["products/geotiff/geo/velocity_mm_yr.tif"],
            }
        ),
        encoding="utf-8",
    )
    health = diagnose_project(root)
    check = next(item for item in health["checks"] if item["name"] == "geocoding")
    assert check["status"] == "ok"
    assert "83.0%" in check["detail"]

    (products / "geocode.json").write_text(
        json.dumps(
            {
                "status": "unavailable",
                "coverage": None,
                "reasons": ["缺少参考影像的轨道状态向量（导入时未找到本地 EOF）"],
                "warnings": [],
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    health = diagnose_project(root)
    check = next(item for item in health["checks"] if item["name"] == "geocoding")
    assert check["status"] == "warn"
    assert "轨道" in check["detail"]
