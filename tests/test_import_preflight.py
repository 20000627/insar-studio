"""Import preflight, georeference warnings and pre-coregistered stacks."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile

import insar_studio.api as api_module
import insar_studio.importer as importer
import insar_studio.pipeline as pipeline
from insar_studio.health import diagnose_project
from insar_studio.importer import import_stack
from insar_studio.jobs import JobManager
from insar_studio.project import ProjectStore


def _make_source(
    tmp_path: Path,
    *,
    already_coregistered: bool = False,
    geo: bool = False,
    name: str = "source",
    n_dates: int = 8,
) -> Path:
    source = tmp_path / name
    source.mkdir(parents=True)
    rng = np.random.default_rng(11)
    rows, cols = 24, 28
    slc = (
        rng.normal(size=(n_dates, rows, cols))
        + 1j * rng.normal(size=(n_dates, rows, cols))
    ).astype(np.complex64)
    np.save(source / "slc.npy", slc, allow_pickle=False)
    np.save(
        source / "dem.npy",
        np.linspace(0, 250, rows * cols, dtype=np.float32).reshape(rows, cols),
        allow_pickle=False,
    )
    start = date(2023, 1, 3)
    meta = {
        "dates": [(start + timedelta(days=12 * i)).isoformat() for i in range(n_dates)],
        "bperp_m": list(np.linspace(-90.0, 110.0, n_dates)),
        "platform": "Sentinel-1A",
    }
    if already_coregistered:
        meta["already_coregistered"] = True
    if geo:
        meta.update(
            {
                "corner_lon": 106.52,
                "corner_lat": 29.63,
                "lon_step": 0.0004,
                "lat_step": -0.0004,
            }
        )
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return source


def _client(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(api_module, "ProjectStore", lambda: store)
    monkeypatch.setattr(api_module, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(api_module, "JobManager", lambda: JobManager(tmp_path / "jobs"))
    app = api_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_dry_run_preflight_reports_warnings_without_writing(tmp_path):
    source = _make_source(tmp_path / "src")
    root = tmp_path / "project"
    root.mkdir()

    meta = import_stack(root, source, dry_run=True)

    assert meta["dry_run"] is True
    assert meta["preflight"]["ok"] is True
    assert meta["preflight"]["warning_count"] == len(meta["import_warnings"])
    assert meta["preflight"]["warnings"] == meta["import_warnings"]
    assert not (root / "data").exists()
    assert not (root / "interferograms").exists()


def test_import_warns_when_georeference_is_missing(tmp_path):
    source = _make_source(tmp_path / "src", geo=False)
    meta = import_stack(tmp_path / "project", source, dry_run=True)

    assert meta["geo_referenced"] is False
    assert meta["geo_source"] == "none"
    assert any("地理参考" in warning for warning in meta["import_warnings"])


def test_import_reads_georeference_from_metadata(tmp_path):
    source = _make_source(tmp_path / "src", geo=True)
    meta = import_stack(tmp_path / "project", source, dry_run=True)

    assert meta["geo_referenced"] is True
    assert meta["geo_source"] == "metadata"
    assert meta["corner_lon"] == pytest.approx(106.52)
    assert not any("缺少 corner_lon" in warning for warning in meta["import_warnings"])


def test_missing_dates_are_marked_as_inferred(tmp_path):
    source = _make_source(tmp_path / "src", n_dates=4)
    (source / "meta.json").write_text(json.dumps({"bperp_m": [0, 1, 2, 3]}), encoding="utf-8")

    meta = import_stack(tmp_path / "project", source, dry_run=True)

    assert meta["dates_source"] == "inferred"
    assert meta["preflight"]["dates_inferred"] is True
    assert any("日期" in warning for warning in meta["import_warnings"])


def test_already_coregistered_import_writes_processed_stack(tmp_path):
    source = _make_source(tmp_path / "src", already_coregistered=True, geo=True)
    root = tmp_path / "project"

    meta = import_stack(root, source)

    assert meta["already_coregistered"] is True
    assert meta["coregistration_skipped"] is True
    coreg_array = root / "data" / "processed" / "slc_coreg.npy"
    assert coreg_array.exists()
    assert np.load(coreg_array, mmap_mode="r").shape == (8, 24, 28)
    coreg = json.loads((root / "data" / "processed" / "coreg.json").read_text(encoding="utf-8"))
    assert coreg["skipped"] is True
    assert coreg["model"] == "pre-coregistered"
    assert coreg["ref_index"] == meta["ref_index"]


def test_network_does_not_coregister_an_imported_prepared_stack(tmp_path, monkeypatch):
    source = _make_source(tmp_path / "src", already_coregistered=True, geo=True)
    store = ProjectStore(tmp_path / "projects")
    project = store.create("预配准项目", "sbas", source="import")
    root = store.path(project["id"])
    import_stack(root, source, technique="sbas")

    def forbidden(*args, **kwargs):
        raise AssertionError("预配准栈不应再次执行配准")

    monkeypatch.setattr(pipeline, "coregister_stack", forbidden)
    result = pipeline.run_network(
        root,
        "sbas",
        {
            "max_temporal_baseline": 72,
            "max_perpendicular_baseline": 180.0,
            "max_pairs": 80,
            "ref_index": 0,
        },
    )

    assert result["stats"]["n_pairs"] > 0
    assert (root / "data" / "processed" / "slc_coreg.npy").exists()


def test_explicit_coregistration_stage_is_a_noop_for_imported_stack(tmp_path):
    source = _make_source(tmp_path / "src", already_coregistered=True, geo=True)
    store = ProjectStore(tmp_path / "projects")
    project = store.create("预配准项目", "sbas", source="import")
    root = store.path(project["id"])
    import_stack(root, source, technique="sbas")
    before = (root / "data" / "processed" / "slc_coreg.npy").read_bytes()

    result = pipeline.run_coregistration(root, {"ref_index": 0})

    assert result["skipped"] is True
    assert result["model"] == "pre-coregistered"
    assert (root / "data" / "processed" / "slc_coreg.npy").read_bytes() == before


def test_health_warns_about_missing_geo_and_accepts_precoregistered_stack(tmp_path):
    source = _make_source(tmp_path / "src", already_coregistered=True, geo=False)
    store = ProjectStore(tmp_path / "projects")
    project = store.create("体检项目", "sbas", source="import")
    root = store.path(project["id"])
    import_stack(root, source, technique="sbas")

    health = diagnose_project(root)
    checks = {check["name"]: check for check in health["checks"]}

    assert checks["geo_reference"]["status"] == "warn"
    assert "像元坐标" in checks["geo_reference"]["detail"]
    assert checks["pre_coregistered"]["status"] == "ok"


def _slc(n_dates: int = 8, rows: int = 20, cols: int = 24, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_dates, rows, cols))
        + 1j * rng.normal(size=(n_dates, rows, cols))
    ).astype(np.complex64)


def _dates(n: int) -> list[str]:
    start = date(2023, 3, 1)
    return [(start + timedelta(days=12 * i)).isoformat() for i in range(n)]


def _forbidden(*args, **kwargs):
    raise AssertionError("轻量预检不应调用完整加载器")


def test_light_preflight_matches_deep_on_small_stack(tmp_path):
    source = _make_source(tmp_path / "src", geo=True)

    light = import_stack(tmp_path / "light", source, dry_run=True)
    deep = import_stack(tmp_path / "deep", source, dry_run=True, deep=True)

    assert light["preflight"]["validation"] == "full"
    assert light["preflight"]["sample_elements"] == light["preflight"]["total_elements"]
    assert light["preflight"]["sample_fraction"] == pytest.approx(1.0)
    assert light["import_warnings"] == deep["import_warnings"]
    for key in ("dates", "rows", "cols", "ref_index", "geo_referenced", "bperp_m"):
        assert light[key] == deep[key]
    assert not (tmp_path / "light" / "data").exists()
    assert not (tmp_path / "deep" / "data").exists()


def test_light_preflight_never_calls_full_loader(tmp_path, monkeypatch):
    source = _make_source(tmp_path / "src", geo=True)
    monkeypatch.setattr(importer, "_read_source", _forbidden)

    meta = import_stack(tmp_path / "project", source, dry_run=True)

    assert meta["preflight"]["validation"] == "full"
    assert meta["preflight"]["ok"] is True


def test_light_preflight_samples_large_stack(tmp_path, monkeypatch):
    path = tmp_path / "large.npy"
    stack = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.complex64, shape=(2, 1500, 1400)
    )
    stack[:] = np.complex64(1.0 + 0.5j)
    stack.flush()
    del stack
    monkeypatch.setattr(importer, "_read_source", _forbidden)

    meta = import_stack(tmp_path / "project", path, dry_run=True)
    pf = meta["preflight"]

    assert pf["validation"] == "sampled"
    assert 0 < pf["sample_elements"] < pf["total_elements"]
    assert 0.0 < pf["sample_fraction"] < 1.0
    assert (pf["rows"], pf["cols"], pf["n_dates"]) == (1500, 1400, 2)


def test_light_preflight_probes_npz_header(tmp_path, monkeypatch):
    source = tmp_path / "stack.npz"
    slc = _slc()
    np.savez(
        source,
        slc=slc,
        dem=np.linspace(0, 200, 20 * 24, dtype=np.float32).reshape(20, 24),
        mask=np.ones((20, 24), dtype=bool),
        dates=np.array(_dates(8)),
        meta=np.array(json.dumps({"bperp_m": list(np.linspace(-40, 40, 8))})),
    )
    deep = import_stack(tmp_path / "deep", source, dry_run=True, deep=True)
    monkeypatch.setattr(importer, "_read_source", _forbidden)
    light = import_stack(tmp_path / "light", source, dry_run=True)

    assert light["preflight"]["validation"] == "header"
    assert light["preflight"]["sample_elements"] == 0
    assert light["preflight"]["sample_fraction"] == 0.0
    assert deep["preflight"]["validation"] == "full"
    assert light["import_warnings"] == deep["import_warnings"]
    assert light["dates"] == deep["dates"] == _dates(8)
    assert light["bperp_m"] == deep["bperp_m"]


def test_light_preflight_probes_geotiff_headers(tmp_path, monkeypatch):
    from insar_studio.core.geotiff import write_geotiff

    source = tmp_path / "tiffs"
    source.mkdir()
    for name in _dates(4):
        write_geotiff(
            source / f"{name}.tif",
            np.ones((20, 24), dtype=np.float32),
            corner_lon=106.52,
            corner_lat=29.63,
            lon_step=0.0004,
            lat_step=-0.0004,
        )
    monkeypatch.setattr(importer, "read_geotiff", _forbidden)

    meta = import_stack(tmp_path / "project", source, dry_run=True)
    pf = meta["preflight"]

    assert pf["validation"] == "header"
    assert (pf["rows"], pf["cols"], pf["n_dates"]) == (20, 24, 4)
    assert meta["dates"] == _dates(4)
    assert meta["geo_referenced"] is True
    assert meta["corner_lon"] == pytest.approx(106.52)


def test_light_preflight_probes_envi_header(tmp_path, monkeypatch):
    slc = _slc(n_dates=4, rows=20, cols=24)
    slc.tofile(tmp_path / "stack.dat")
    (tmp_path / "stack.hdr").write_text(
        "ENVI\n"
        "samples = 24\n"
        "lines = 20\n"
        "bands = 4\n"
        "data type = 6\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        "data file = stack.dat\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(importer, "read_envi", _forbidden)

    meta = import_stack(tmp_path / "project", tmp_path / "stack.hdr", dry_run=True)
    pf = meta["preflight"]

    assert pf["validation"] == "header"
    assert (pf["rows"], pf["cols"], pf["n_dates"]) == (20, 24, 4)


def test_light_preflight_probes_safe_headers(tmp_path, monkeypatch):
    source = tmp_path / "safe"
    measurement = source / "measurement"
    measurement.mkdir(parents=True)
    for name in ("2023-01-01", "2023-01-13"):
        tifffile.imwrite(
            measurement / f"{name}.tiff",
            np.zeros((2, 20, 24), dtype=np.int16),
        )
    monkeypatch.setattr(importer, "_read_safe", _forbidden)

    meta = import_stack(tmp_path / "project", source, dry_run=True)
    pf = meta["preflight"]

    assert pf["validation"] == "header"
    assert (pf["rows"], pf["cols"], pf["n_dates"]) == (20, 24, 2)
    assert meta["dates"] == ["2023-01-01", "2023-01-13"]


@pytest.mark.parametrize(
    ("fill", "message"),
    [
        (0.0 + 0.0j, "全为零"),
        (np.nan + 1j * np.nan, "有限数"),
    ],
)
def test_light_preflight_rejects_degenerate_stacks(tmp_path, fill, message):
    source = tmp_path / "src"
    source.mkdir()
    np.save(
        source / "slc.npy",
        np.full((4, 16, 18), fill, dtype=np.complex64),
        allow_pickle=False,
    )
    (source / "meta.json").write_text(
        json.dumps({"dates": _dates(4)}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        import_stack(tmp_path / "project", source, dry_run=True)


def test_bounded_sample_is_deterministic_and_bounded():
    array = np.arange(1000 * 1000, dtype=np.float32).reshape(1000, 1000)

    first = importer._bounded_sample(array, 10_000)
    second = importer._bounded_sample(array, 10_000)

    assert first.size <= 10_000
    assert np.array_equal(first, second)
    assert first[0] == array.flat[0]
    assert first[-1] == array.flat[-1]
    full = importer._bounded_sample(array, array.size)
    assert full.size == array.size
    assert np.array_equal(full, array)


def test_api_preflight_deep_flag_switches_validation(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "深度预检", "technique": "dinsar", "source": "empty"},
    ).get_json()["project"]
    source = tmp_path / "stack.npz"
    np.savez(
        source,
        slc=_slc(n_dates=4),
        dates=np.array(_dates(4)),
        meta=np.array(json.dumps({"bperp_m": [0.0, 10.0, 20.0, 30.0]})),
    )

    light = client.post(
        f"/api/projects/{project['id']}/import/preflight",
        json={"path": str(source)},
    )
    assert light.status_code == 200
    light_pf = light.get_json()["preflight"]["preflight"]
    assert light_pf["validation"] == "header"

    deep = client.post(
        f"/api/projects/{project['id']}/import/preflight",
        json={"path": str(source), "deep": True},
    )
    assert deep.status_code == 200
    deep_pf = deep.get_json()["preflight"]["preflight"]
    assert deep_pf["validation"] == "full"
    assert deep_pf["sample_elements"] == deep_pf["total_elements"]
    assert not (store.path(project["id"]) / "data" / "stack" / "meta.json").exists()


def test_api_preflight_does_not_write_and_import_keeps_precoregistration(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "API 预检", "technique": "sbas", "source": "empty"},
    ).get_json()["project"]
    source = _make_source(tmp_path / "src", already_coregistered=True, geo=True)

    preflight = client.post(
        f"/api/projects/{project['id']}/import/preflight",
        json={"path": str(source)},
    )
    assert preflight.status_code == 200
    payload = preflight.get_json()["preflight"]
    assert payload["preflight"]["already_coregistered"] is True
    root = store.path(project["id"])
    assert not (root / "data" / "stack" / "meta.json").exists()

    imported = client.post(
        f"/api/projects/{project['id']}/import",
        json={"path": str(source), "meta": {"already_coregistered": True}},
    )
    assert imported.status_code == 200
    body = imported.get_json()["stack"]
    assert body["already_coregistered"] is True
    assert (root / "data" / "processed" / "slc_coreg.npy").exists()
    saved = store.get(project["id"])
    assert saved["stack"]["already_coregistered"] is True
