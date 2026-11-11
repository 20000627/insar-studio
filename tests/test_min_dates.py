"""Minimum acquisition count enforcement for every processing route."""

from __future__ import annotations

import json

import numpy as np
import pytest

from insar_studio.config import TECHNIQUES
from insar_studio.health import diagnose_project
from insar_studio.importer import import_stack
from insar_studio.params import ParamError, min_dates_for, require_min_dates
from insar_studio.pipeline import run_pipeline
from insar_studio.project import ProjectStore


@pytest.mark.parametrize(
    ("technique", "required"),
    [("dinsar", 2), ("ps", 8), ("sbas", 8), ("ds", 10)],
)
@pytest.mark.parametrize("n_dates", [2, 5, 6, 8, 10])
def test_min_dates_matrix(technique, required, n_dates):
    assert min_dates_for(technique) == required == TECHNIQUES[technique]["min_dates"]
    if n_dates >= required:
        assert require_min_dates(technique, n_dates) == required
    else:
        with pytest.raises(ParamError, match=rf"{required} 景影像.*{n_dates} 景"):
            require_min_dates(technique, n_dates)


def test_min_dates_skips_empty_project_and_rejects_unknown_route():
    assert require_min_dates("ps", None) == 8
    with pytest.raises(ParamError, match="未知技术路线"):
        min_dates_for("magic")


def _stack_source(tmp_path, n_dates: int):
    source = tmp_path / f"stack_{n_dates}.npy"
    rng = np.random.default_rng(11)
    slc = (
        rng.normal(size=(n_dates, 24, 30)) + 1j * rng.normal(size=(n_dates, 24, 30))
    ).astype(np.complex64)
    np.save(source, slc, allow_pickle=False)
    return source


@pytest.mark.parametrize(("technique", "n_dates"), [("ps", 5), ("sbas", 7), ("ds", 9)])
def test_import_rejects_stack_shorter_than_route_requires(tmp_path, technique, n_dates):
    source = _stack_source(tmp_path, n_dates)
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(ParamError, match="至少需要"):
        import_stack(root, source, technique=technique)

    assert not (root / "data" / "stack" / "slc.npy").exists()


def test_import_accepts_two_dates_for_dinsar(tmp_path):
    source = _stack_source(tmp_path, 2)
    root = tmp_path / "project"
    root.mkdir()

    result = import_stack(root, source, technique="dinsar")

    assert result["n_dates"] == 2
    assert (root / "data" / "stack" / "slc.npy").exists()


def test_route_switch_rejects_short_stack_but_keeps_previous_route(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("short stack", "dinsar", source="empty")
    root = store.path(project["id"])
    (root / "data" / "stack").mkdir(parents=True)
    (root / "data" / "stack" / "meta.json").write_text(
        json.dumps({"n_dates": 5}),
        encoding="utf-8",
    )

    assert store.update(project["id"], {"technique": "dinsar"})["technique"] == "dinsar"
    with pytest.raises(ParamError, match="PS-InSAR.*8 景"):
        store.update(project["id"], {"technique": "ps"})
    with pytest.raises(ParamError, match="SBAS-InSAR.*8 景"):
        store.update(project["id"], {"technique": "sbas"})
    with pytest.raises(ParamError, match="DS-InSAR.*10 景"):
        store.update(project["id"], {"technique": "ds"})
    assert store.get(project["id"])["technique"] == "dinsar"


def test_pipeline_rejects_short_stack_before_running(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("short pipeline", "ps", source="empty")
    root = store.path(project["id"])
    (root / "data" / "stack").mkdir(parents=True)
    (root / "data" / "stack" / "meta.json").write_text(
        json.dumps({"n_dates": 5}),
        encoding="utf-8",
    )

    with pytest.raises(ParamError, match="PS-InSAR.*8 景"):
        run_pipeline(root, technique="ps", params=project["params"])


def test_health_reports_route_date_shortage(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("health ds", "ds", source="empty")
    root = store.path(project["id"])
    stack = root / "data" / "stack"
    stack.mkdir(parents=True)
    (stack / "meta.json").write_text(
        json.dumps(
            {
                "n_dates": 6,
                "rows": 8,
                "cols": 10,
                "dates": [f"2023-01-{day:02d}" for day in range(1, 12, 2)],
            }
        ),
        encoding="utf-8",
    )
    np.save(stack / "slc.npy", np.ones((6, 8, 10), dtype=np.complex64), allow_pickle=False)
    np.save(stack / "dem.npy", np.zeros((8, 10), dtype=np.float32), allow_pickle=False)
    np.save(stack / "mask.npy", np.ones((8, 10), dtype=bool), allow_pickle=False)

    health = diagnose_project(root)
    check = next(item for item in health["checks"] if item["name"] == "route_dates")

    assert check["status"] == "error"
    assert "DS-InSAR" in check["detail"]
    assert "10" in check["detail"] and "6" in check["detail"]
