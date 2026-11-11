"""Project storage and local data import tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import insar_studio.project as project_module
from insar_studio.importer import import_stack
from insar_studio.params import ParamError, validate_params
from insar_studio.project import ProjectStore, purge_downstream


def test_path_validation_rejects_escapes(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    for bad in ("../escape", "a/b", "", "prj-1"):
        with pytest.raises(ValueError):
            store.path(bad)


def test_create_and_update_keeps_route_defaults(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("项目 A", "sbas")
    assert project["params"]["multilook"] == 2
    assert project["demo"] is True

    updated = store.update(project["id"], {"technique": "ps"})
    assert updated["technique"] == "ps"
    assert updated["params"]["multilook"] == 1
    assert updated["params"]["max_temporal_baseline"] == 96

    # Explicit user parameters survive a route switch.
    updated = store.update(project["id"], {"technique": "ds", "params": {"multilook": 3}})
    assert updated["params"]["multilook"] == 3
    assert updated["params"]["ds_patch"] == 9


def test_native_numpy_stack_import(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    rng = np.random.default_rng(3)
    slc = (
        rng.normal(size=(6, 32, 40)) + 1j * rng.normal(size=(6, 32, 40))
    ).astype(np.complex64)
    np.save(source / "slc.npy", slc, allow_pickle=False)
    np.save(source / "dem.npy", np.linspace(0, 300, 32 * 40, dtype=np.float32).reshape(32, 40))
    meta = {
        "dates": [
            "2023-02-01",
            "2023-02-13",
            "2023-02-25",
            "2023-03-09",
            "2023-03-21",
            "2023-04-02",
        ],
        "platform": "Sentinel-1A",
        "orbit_direction": "DESCENDING",
        "bperp_m": [-50, -20, 10, 45, 70, 90],
        "incidence_deg": 39.2,
    }
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    root = tmp_path / "project"
    root.mkdir()
    result = import_stack(root, source)

    assert result["n_dates"] == 6
    assert result["rows"] == 32 and result["cols"] == 40
    assert result["dates"][0] == "2023-02-01"
    assert result["orbit_direction"] == "DESCENDING"
    assert (root / "data" / "stack" / "slc.npy").exists()
    assert (root / "data" / "stack" / "meta.json").exists()

    loaded = np.load(root / "data" / "stack" / "slc.npy", allow_pickle=False)
    assert loaded.shape == (6, 32, 40)
    assert loaded.dtype == np.complex64


def test_import_npz_sorts_dates_and_remaps_reference(tmp_path):
    rng = np.random.default_rng(4)
    slc = (rng.normal(size=(4, 20, 20)) + 1j * rng.normal(size=(4, 20, 20))).astype(np.complex64)
    source = tmp_path / "stack.npz"
    np.savez(
        source,
        slc=slc,
        dem=np.zeros((20, 20), dtype=np.float32),
        mask=np.ones((20, 20), dtype=bool),
        dates=np.array(["2023-03-01", "2023-01-01", "2023-02-01", "2023-04-01"]),
        meta=np.array(json.dumps({"bperp_m": [0, 10, 20, 30], "ref_index": 0})),
    )

    root = tmp_path / "npz-project"
    root.mkdir()
    result = import_stack(root, source)

    assert result["dates"] == ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]
    assert result["bperp_m"] == [10.0, 20.0, 0.0, 30.0]
    assert result["ref_index"] == 2
    assert np.load(root / "data" / "stack" / "slc.npy").shape == (4, 20, 20)


def test_ds_patch_odd_check_only_applies_to_ds_route():
    # Legacy non-DS projects may carry an even value that is never used.
    assert validate_params({"ds_patch": 10}, technique="sbas")["ds_patch"] == 10
    assert validate_params({"ds_patch": 10}, technique="dinsar")["ds_patch"] == 10
    with pytest.raises(ParamError):
        validate_params({"ds_patch": 10}, technique="ds")
    assert validate_params({"ds_patch": 9}, technique="ds")["ds_patch"] == 9


def _write_marker(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_product_files_skips_directory_removed_during_scan(tmp_path, monkeypatch):
    """A job purge must not turn a project listing into a 404."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("并发扫描", "sbas")
    products = store.path(project["id"]) / "products"
    figures = products / "figures"
    figures.mkdir(parents=True)
    (figures / "velocity.png").write_bytes(b"gone")
    other = products / "other"
    other.mkdir()
    (other / "keep.json").write_text("{}", encoding="utf-8")

    real_scandir = project_module.os.scandir

    def flaky_scandir(path):
        if Path(path).name == "figures":
            raise FileNotFoundError(path)
        return real_scandir(path)

    monkeypatch.setattr(project_module.os, "scandir", flaky_scandir)

    files = store.product_files(project["id"])
    assert [item["name"] for item in files] == ["keep.json"]
    overview = store.overview(project["id"])
    assert overview["project"]["id"] == project["id"]
    assert overview["files"] == []
    with_files = store.overview(project["id"], include_files=True)
    assert [item["name"] for item in with_files["files"]] == ["keep.json"]


def test_purge_downstream_preserves_upstream_products(tmp_path):
    root = tmp_path / "prj_test"
    for relative in (
        "data/processed/marker.txt",
        "data/network.json",
        "interferograms/pair/unwrapped.npy",
        "data/atmosphere/atmosphere.json",
        "products/figures/network.png",
        "products/figures/atmosphere.png",
        "products/figures/coreg_coherence_gain.png",
        "products/figures/coreg_residual.png",
        "products/figures/ds_coherence.png",
        "products/figures/velocity.png",
        "products/inversion.json",
        "products/validation.json",
    ):
        _write_marker(root / relative)
    (root / "project.json").write_text(
        json.dumps(
            {
                "id": "prj_test",
                "steps_run": {
                    "coregistration": {"at": "x"},
                    "network": {"at": "x"},
                    "interferometry": {"at": "x"},
                    "atmosphere": {"at": "x"},
                    "inversion": {"at": "x"},
                    "pipeline": {"at": "x"},
                },
            }
        ),
        encoding="utf-8",
    )

    removed = purge_downstream(root, "interferometry")
    assert "interferograms" in removed
    assert "data/atmosphere" in removed
    assert (root / "data" / "processed" / "marker.txt").exists()
    assert (root / "data" / "network.json").exists()
    assert not (root / "interferograms").exists()
    assert not (root / "data" / "atmosphere").exists()
    # Network and coregistration figures are still valid; atmosphere and
    # inversion outputs belong to the stages being rebuilt.
    assert (root / "products" / "figures" / "network.png").exists()
    assert (root / "products" / "figures" / "coreg_coherence_gain.png").exists()
    assert (root / "products" / "figures" / "coreg_residual.png").exists()
    assert not (root / "products" / "figures" / "ds_coherence.png").exists()
    assert not (root / "products" / "figures" / "atmosphere.png").exists()
    assert not (root / "products" / "figures" / "velocity.png").exists()
    assert not (root / "products" / "inversion.json").exists()
    assert not (root / "products" / "validation.json").exists()

    steps = json.loads((root / "project.json").read_text(encoding="utf-8"))["steps_run"]
    assert "coregistration" in steps and "network" in steps
    assert not ({"interferometry", "atmosphere", "inversion", "pipeline"} & set(steps))


def test_purge_downstream_network_preserves_coregistration_figures(tmp_path):
    root = tmp_path / "prj_network"
    for relative in (
        "data/network.json",
        "interferograms/pair/unwrapped.npy",
        "data/atmosphere/atmosphere.json",
        "products/figures/network.png",
        "products/figures/atmosphere.png",
        "products/figures/coreg_coherence_gain.png",
        "products/figures/velocity.png",
        "products/inversion.json",
    ):
        _write_marker(root / relative)
    (root / "project.json").write_text(json.dumps({"steps_run": {}}), encoding="utf-8")

    purge_downstream(root, "network")

    assert not (root / "data" / "network.json").exists()
    assert not (root / "interferograms").exists()
    assert not (root / "data" / "atmosphere").exists()
    assert not (root / "products" / "figures" / "network.png").exists()
    assert not (root / "products" / "figures" / "atmosphere.png").exists()
    assert not (root / "products" / "figures" / "velocity.png").exists()
    assert not (root / "products" / "inversion.json").exists()
    assert (root / "products" / "figures" / "coreg_coherence_gain.png").exists()


def test_purge_downstream_atmosphere_preserves_network_figure(tmp_path):
    root = tmp_path / "prj_atmosphere"
    for relative in (
        "data/atmosphere/atmosphere.json",
        "products/figures/network.png",
        "products/figures/atmosphere.png",
        "products/figures/coreg_coherence_gain.png",
        "products/figures/ds_coherence.png",
        "products/figures/velocity.png",
        "products/inversion.json",
    ):
        _write_marker(root / relative)
    (root / "project.json").write_text(json.dumps({"steps_run": {}}), encoding="utf-8")

    purge_downstream(root, "atmosphere")

    assert not (root / "data" / "atmosphere").exists()
    assert not (root / "products" / "figures" / "atmosphere.png").exists()
    assert not (root / "products" / "figures" / "velocity.png").exists()
    assert not (root / "products" / "inversion.json").exists()
    assert (root / "products" / "figures" / "network.png").exists()
    assert (root / "products" / "figures" / "coreg_coherence_gain.png").exists()
    assert (root / "products" / "figures" / "ds_coherence.png").exists()


def test_purge_downstream_survives_concurrent_rewrite(tmp_path, monkeypatch):
    """A cache writer must not make products deletion fail with WinError 145."""
    root = tmp_path / "prj_race"
    products = root / "products"
    (products / "arrays").mkdir(parents=True)
    (products / "arrays" / "velocity_mm_yr.npy").write_bytes(b"old")
    (products / "validation.json").write_text("{}", encoding="utf-8")
    (root / "project.json").write_text(
        json.dumps(
            {
                "id": "prj_race",
                "steps_run": {"inversion": {"at": "x"}, "pipeline": {"at": "x"}},
            }
        ),
        encoding="utf-8",
    )

    real_rmtree = project_module.shutil.rmtree
    attempts = {"count": 0}

    def racing_rmtree(path, *args, **kwargs):
        attempts["count"] += 1
        staged = Path(path)
        if attempts["count"] == 1:
            # Simulate Windows: a writer recreated products/validation.json
            # after rmtree enumerated the tree.
            products.mkdir(exist_ok=True)
            (products / "validation.json").write_text('{"fresh": true}', encoding="utf-8")
            raise OSError(145, "目录不是空的", str(staged))
        # The retry must operate on the moved-aside tree, never the live path,
        # so a concurrent writer cannot repopulate the tree being deleted.
        assert staged != products
        assert staged.parent.name == ".trash"
        (products / "validation.json").write_text('{"fresh": true}', encoding="utf-8")
        real_rmtree(staged, *args, **kwargs)

    monkeypatch.setattr(project_module.shutil, "rmtree", racing_rmtree)

    removed = purge_downstream(root, "coregistration")

    assert removed == ["products"]
    assert attempts["count"] == 2
    assert json.loads((products / "validation.json").read_text(encoding="utf-8")) == {"fresh": True}
    assert not (root / ".trash").exists()
    steps = json.loads((root / "project.json").read_text(encoding="utf-8"))["steps_run"]
    assert "inversion" not in steps and "pipeline" not in steps


def test_purge_downstream_raises_after_persistent_lock(tmp_path, monkeypatch):
    """A real lock must fail the job instead of leaving stale products behind."""
    root = tmp_path / "prj_locked"
    products = root / "products"
    products.mkdir(parents=True)
    (products / "inversion.json").write_text("{}", encoding="utf-8")
    (root / "project.json").write_text(json.dumps({"steps_run": {}}), encoding="utf-8")

    def locked_rmtree(path, *args, **kwargs):
        raise OSError(145, "目录不是空的", str(path))

    monkeypatch.setattr(project_module, "_REMOVE_ATTEMPTS", 2)
    monkeypatch.setattr(project_module.shutil, "rmtree", locked_rmtree)

    with pytest.raises(OSError, match="目录不是空的"):
        purge_downstream(root, "coregistration")


def test_purge_downstream_inversion_preserves_upstream_figures(tmp_path):
    """Rerunning inversion must keep network/atmosphere/coreg diagnostics."""
    root = tmp_path / "prj_inversion"
    removed_artefacts = (
        "products/arrays/velocity_mm_yr.npy",
        "products/geotiff/velocity_mm_yr.tif",
        "products/csv/timeseries_points.csv",
        "products/figures/velocity.png",
        "products/figures/temporal_coherence.png",
        "products/figures/residual.png",
        "products/figures/dem_error.png",
        "products/figures/timeseries_peak.png",
        "products/report/report.md",
        "products/exports/result.zip",
        "products/inversion.json",
        "products/validation.json",
    )
    kept_artefacts = (
        "products/figures/network.png",
        "products/figures/atmosphere.png",
        "products/figures/coreg_coherence_gain.png",
        "products/figures/coreg_residual.png",
        "products/figures/ds_coherence.png",
    )
    for relative in removed_artefacts + kept_artefacts:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "project.json").write_text(
        json.dumps({"steps_run": {"inversion": {"at": "x"}, "pipeline": {"at": "x"}}}),
        encoding="utf-8",
    )

    purge_downstream(root, "inversion")

    for relative in removed_artefacts:
        assert not (root / relative).exists(), relative
    for relative in kept_artefacts:
        assert (root / relative).exists(), relative
