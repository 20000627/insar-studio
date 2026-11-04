"""End-to-end test of the D-InSAR route on the synthetic demo dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path

from insar_studio.health import diagnose_project
from insar_studio.pipeline import run_pipeline
from insar_studio.project import ProjectStore
from insar_studio.report import export_zip, generate_report, points_csv, series_csv
from insar_studio.validation import validate_demo


def test_dinsar_pipeline_report_and_export(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create_demo("Pytest D-InSAR", "dinsar")
    root = store.path(project["id"])

    # The demo generator must create a deterministic 24-date stack.
    assert project["stack"]["n_dates"] == 24
    assert (root / "data" / "stack" / "meta.json").exists()
    assert (root / "data" / "stack" / "truth.npz").exists()

    result = run_pipeline(
        root,
        technique="dinsar",
        params=project["params"],
    )

    inversion = result["inversion"]
    assert result["technique"] == "dinsar"
    assert inversion["n_pairs"] == 1
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
