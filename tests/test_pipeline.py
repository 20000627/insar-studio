"""End-to-end tests of all four routes on compact synthetic demo datasets."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from insar_studio.core.synthetic import generate_demo_dataset
from insar_studio.health import diagnose_project
from insar_studio.pipeline import run_pipeline
from insar_studio.project import ProjectStore
from insar_studio.report import export_zip, generate_report, points_csv, series_csv
from insar_studio.validation import validate_demo


def _compact_demo_project(
    store: ProjectStore,
    technique: str,
    *,
    rows: int = 48,
    cols: int = 52,
    n_dates: int = 10,
) -> tuple[dict, Path]:
    """Create a demo project and replace the canonical stack with a small one.

    The compact stack keeps every processing branch (network, unwrapping,
    PS selection, DS phase linking) active while keeping the regression fast.
    """
    project = store.create(f"Pytest {technique}", technique, source="demo")
    root = store.path(project["id"])
    meta = generate_demo_dataset(root, rows=rows, cols=cols, n_dates=n_dates)
    project["stack"] = {
        "rows": meta["rows"],
        "cols": meta["cols"],
        "n_dates": meta["n_dates"],
        "dates": meta["dates"],
        "platform": meta["platform"],
        "source": "demo",
    }
    store.save(project)
    return store.get(project["id"]), root


def test_dinsar_pipeline_report_and_export(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create_demo("Pytest D-InSAR", "dinsar")
    root = store.path(project["id"])

    # The demo generator must create a deterministic 24-date stack.
    assert project["stack"]["n_dates"] == 24
    assert (root / "data" / "stack" / "meta.json").exists()
    assert (root / "data" / "stack" / "truth.npz").exists()

    params = {**project["params"], "dinsar_pair": "02_05"}
    store.update(project["id"], {"params": params})
    project = store.get(project["id"])

    result = run_pipeline(
        root,
        technique="dinsar",
        params=project["params"],
    )

    inversion = result["inversion"]
    assert result["technique"] == "dinsar"
    assert inversion["n_pairs"] == 1
    assert inversion["pair"]["id"] == "02_05"
    assert 0.0 < inversion["valid_fraction"] <= 1.0
    assert (root / "products" / "arrays" / "disp_pair_mm.npy").exists()
    assert (root / "products" / "figures" / "dinsar_displacement.png").exists()

    validation = validate_demo(root, force=True)
    assert validation is not None
    assert validation["technique"] == "dinsar"
    assert validation["displacement_rmse_mm"] < 6.0
    assert validation["grade"] in ("优秀", "良好")

    report = generate_report(root)
    assert report["docx"] is not None
    markdown_path = root / report["markdown"]
    docx_path = root / report["docx"]
    assert markdown_path.exists() and markdown_path.stat().st_size > 500
    assert docx_path.exists() and docx_path.stat().st_size > 10_000
    assert "InSAR" in markdown_path.read_text(encoding="utf-8")

    archive = export_zip(root)
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "project.json" in names
    assert "products/report/report.md" in names
    assert any(name.endswith("disp_pair_mm.npy") for name in names)
    assert archive.stat().st_size > 100_000

    # The engine records stage fingerprints; neither hidden defaults nor the
    # route-independent coregistration hash may make fresh products look stale.
    status = store.status(project["id"])
    assert all(not stage["stale"] for stage in status), status

    health = diagnose_project(root)
    assert health["summary"]["errors"] == 0
    assert health["grade"] in ("健康", "基本可用")

    rr, cc = inversion["reference_pixel"]
    pixel_csv = series_csv(root, int(rr), int(cc))
    assert pixel_csv.exists() and "date" in pixel_csv.read_text(encoding="utf-8").splitlines()[0]
    all_points = points_csv(root)
    assert all_points.exists() and all_points.stat().st_size > 100


@pytest.mark.parametrize(
    ("technique", "route_figure"),
    [
        ("sbas", None),
        ("ps", "ps_candidates.png"),
        ("ds", "ds_coherence.png"),
    ],
)
def test_multitemporal_routes_run_end_to_end(tmp_path, technique, route_figure):
    store = ProjectStore(tmp_path / "projects")
    project, root = _compact_demo_project(store, technique)

    result = run_pipeline(root, technique=technique, params=project["params"])

    assert result["technique"] == technique
    assert result["stages"] == [
        "coregistration",
        "network",
        "interferometry",
        "atmosphere",
        "inversion",
    ]

    inversion = result["inversion"]
    assert inversion["technique"] == technique
    assert inversion["n_dates"] == 10
    assert inversion["n_pairs"] > 0
    assert 0.0 < inversion["valid_fraction"] <= 1.0
    assert inversion["valid_pixels"] > 0

    assert (root / "data" / "network.json").exists()
    assert (root / "interferograms" / "summary.json").exists()
    assert (root / "data" / "atmosphere" / "atmosphere.json").exists()
    assert (root / "products" / "inversion.json").exists()
    for name in (
        "velocity_mm_yr",
        "disp_ts_mm",
        "temporal_coherence",
        "residual_mm",
        "valid",
        "dem_error_m",
    ):
        assert (root / "products" / "arrays" / f"{name}.npy").exists(), name
    assert (root / "products" / "geotiff" / "velocity_mm_yr.tif").exists()
    if route_figure:
        assert (root / "products" / "figures" / route_figure).exists()

    # Route-specific metadata proves the branch really ran instead of silently
    # falling back to the SBAS path.
    if technique == "ps":
        assert inversion["n_candidates"] > 0
        assert inversion["adi_max"] > 0
        assert inversion["coherence_min"] == project["params"]["ps_coherence_min"]
    if technique == "ds":
        assert inversion["phase_linking"] == "emi"
        assert inversion["ds_patch"] == project["params"]["ds_patch"]

    validation = validate_demo(root, force=True)
    assert validation is not None
    assert validation["technique"] == technique
    assert validation["displacement_rmse_mm"] < 12.0
    assert validation["velocity_rmse_mm_yr"] < 25.0
    assert validation["grade"] in ("优秀", "良好", "可用")

    status = store.status(project["id"])
    assert all(not stage["stale"] for stage in status), status

    health = diagnose_project(root)
    assert health["summary"]["errors"] == 0
    assert health["grade"] in ("健康", "基本可用")
