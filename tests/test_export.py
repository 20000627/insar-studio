"""Export size estimation and slim archives (P2-4)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import insar_studio.api as api_module
from insar_studio.jobs import JobManager
from insar_studio.project import ProjectStore
from insar_studio.report import export_summary, export_zip


def _make_project(tmp_path: Path) -> tuple[ProjectStore, dict, Path]:
    store = ProjectStore(tmp_path / "projects")
    project = store.create("导出测试", "sbas", source="empty")
    root = store.path(project["id"])

    files = {
        "products/arrays/velocity_mm_yr.npy": b"n" * 2048,
        "products/report/report.md": b"# report\n",
        "products/exports/old.zip": b"old",
        "data/stack/slc.npy": b"s" * 4096,
        "data/stack/meta.json": json.dumps({"n_dates": 24}),
        "data/network.json": json.dumps({"pairs": []}),
        "data/atmosphere/atmosphere.json": json.dumps({"n_pairs": 0}),
        "interferograms/pair_1/unwrapped.npy": b"u" * 1024,
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return store, project, root


def test_export_summary_excludes_raw_data_from_results(tmp_path):
    _, _, root = _make_project(tmp_path)

    summary = export_summary(root)

    assert summary["recommended"] == "results"
    results = summary["modes"]["results"]
    full = summary["modes"]["full"]
    # results: report + velocity array; exports/old.zip must not be counted.
    assert results["files"] == 2
    assert results["bytes"] == 2048 + len(b"# report\n")
    # full adds stack, network, atmosphere and interferograms.
    assert full["files"] > results["files"]
    assert full["bytes"] > results["bytes"]


def test_results_zip_contains_only_deliverables(tmp_path):
    _, _, root = _make_project(tmp_path)

    archive = export_zip(root, mode="results")

    assert archive.name.endswith("_insar_results.zip")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "project.json" in names
    assert "products/report/report.md" in names
    assert any(name.endswith("velocity_mm_yr.npy") for name in names)
    assert not any(name.startswith("data/stack/") for name in names)
    assert not any(name.startswith("interferograms/") for name in names)
    assert not any("exports" in name.split("/") for name in names)


def test_full_zip_keeps_source_stack(tmp_path):
    _, _, root = _make_project(tmp_path)

    archive = export_zip(root, mode="full")

    assert archive.name.endswith("_insar.zip")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "data/stack/slc.npy" in names
    assert "data/network.json" in names
    assert "interferograms/pair_1/unwrapped.npy" in names
    assert not any("exports" in name.split("/") for name in names)


def test_unknown_export_mode_is_rejected(tmp_path):
    _, _, root = _make_project(tmp_path)

    with pytest.raises(ValueError):
        export_zip(root, mode="everything")


def _client(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(api_module, "ProjectStore", lambda: store)
    monkeypatch.setattr(api_module, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(api_module, "JobManager", lambda: JobManager(tmp_path / "jobs"))
    app = api_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_export_endpoints(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("API 导出", "sbas", source="empty")
    root = store.path(project["id"])
    report = root / "products" / "report" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# report\n", encoding="utf-8")

    summary = client.get(f"/api/projects/{project['id']}/export/summary")
    assert summary.status_code == 200
    assert summary.get_json()["modes"]["results"]["files"] == 1

    invalid = client.get(f"/api/projects/{project['id']}/export?mode=everything")
    assert invalid.status_code == 400

    slim = client.get(f"/api/projects/{project['id']}/export?mode=results")
    assert slim.status_code == 200
    assert slim.headers["Content-Type"].startswith("application/zip")
    assert "results" in slim.headers["Content-Disposition"]
