"""HTTP API smoke tests using an isolated temporary workspace."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import insar_studio.api as api_module
from insar_studio.jobs import JobManager
from insar_studio.project import ProjectStore


def _client(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(api_module, "ProjectStore", lambda: store)
    monkeypatch.setattr(api_module, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(api_module, "JobManager", lambda: JobManager(tmp_path / "jobs"))
    app = api_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_health_and_capabilities(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["ok"] is True

    capabilities = client.get("/api/capabilities")
    payload = capabilities.get_json()
    assert capabilities.status_code == 200
    assert set(payload["techniques"]) == {"sbas", "ps", "ds", "dinsar"}
    assert payload["truth"] is True
    assert payload["defaults"]["unwrap_method"] == "quality"


def test_project_lifecycle(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)

    created = client.post(
        "/api/projects",
        json={"name": "API 测试项目", "technique": "sbas", "source": "empty"},
    )
    assert created.status_code == 201
    project = created.get_json()["project"]
    project_id = project["id"]
    assert project["name"] == "API 测试项目"
    assert project["technique"] == "sbas"

    listed = client.get("/api/projects").get_json()["projects"]
    assert any(p["id"] == project_id for p in listed)

    overview = client.get(f"/api/projects/{project_id}")
    assert overview.status_code == 200
    assert overview.get_json()["project"]["id"] == project_id

    patched = client.patch(
        f"/api/projects/{project_id}",
        json={"name": "改名后的项目", "technique": "ps"},
    )
    assert patched.status_code == 200
    updated = patched.get_json()["project"]
    assert updated["name"] == "改名后的项目"
    assert updated["technique"] == "ps"
    assert updated["params"]["multilook"] == 1

    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200
    assert not store.path(project_id).exists()
    assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_create_rejects_unknown_technique(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    response = client.post("/api/projects", json={"name": "bad", "technique": "magic"})
    assert response.status_code == 400
    assert "技术路线" in response.get_json()["error"]


def test_import_into_missing_project_returns_404(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/projects/prj_missing/import",
        json={"path": str(tmp_path / "not-used")},
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == "项目不存在"


def test_route_switch_and_run_reject_short_stack(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("短栈", "dinsar", source="empty")
    stack = store.path(project["id"]) / "data" / "stack"
    stack.mkdir(parents=True)
    (stack / "meta.json").write_text(json.dumps({"n_dates": 5}), encoding="utf-8")

    switched = client.patch(
        f"/api/projects/{project['id']}",
        json={"technique": "ps"},
    )
    assert switched.status_code == 400
    assert "8 景" in switched.get_json()["error"]

    run = client.post(
        f"/api/projects/{project['id']}/run",
        json={"technique": "sbas", "params": {}},
    )
    assert run.status_code == 400
    assert "8 景" in run.get_json()["error"]


def test_series_before_processing_returns_404(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "空项目", "technique": "sbas", "source": "empty"},
    ).get_json()["project"]

    response = client.get(f"/api/projects/{project['id']}/series?row=1&col=1")
    assert response.status_code == 404
    assert "反演" in response.get_json()["error"]


def test_asset_path_traversal_is_rejected(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("路径测试", "sbas", source="import")
    response = client.get(f"/api/projects/{project['id']}/asset/..%2F..%2Fproject.json")
    assert response.status_code in (400, 404)


def test_asset_serves_existing_product(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("资源测试", "sbas", source="import")
    root = store.path(project["id"])
    target = root / "products" / "figures" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello insar", encoding="utf-8")

    response = client.get(f"/api/projects/{project['id']}/asset/products/figures/demo.txt")
    assert response.status_code == 200
    assert response.data == b"hello insar"


def test_route_switch_is_applied_inside_job(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "路线切换", "technique": "sbas", "source": "empty"},
    ).get_json()["project"]
    captured = {}

    def fake_pipeline(root, *, technique, params, steps, progress):
        captured.update(
            {"root": Path(root), "technique": technique, "params": dict(params), "steps": steps}
        )
        return {"technique": technique, "stages": []}

    monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline)
    response = client.post(
        f"/api/projects/{project['id']}/inversion",
        json={"technique": "ps", "params": {}},
    )
    assert response.status_code == 202
    job = response.get_json()["job"]

    for _ in range(100):
        payload = client.get(f"/api/jobs/{job['id']}").get_json()
        if payload["status"] in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.05)
    assert payload["status"] == "done", payload.get("error")
    assert payload["meta"]["route_changed"] is True

    updated = store.get(project["id"])
    assert updated["technique"] == "ps"
    assert captured["technique"] == "ps"
    assert captured["steps"] == ["network", "interferometry", "atmosphere", "inversion"]
    assert captured["params"]["multilook"] == 1
    assert captured["params"]["max_pairs"] == 140
    assert captured["params"]["ps_star_all"] is True


def test_active_job_blocks_health_and_second_submission(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "忙碌项目", "technique": "sbas", "source": "empty"},
    ).get_json()["project"]
    release = threading.Event()

    def slow_pipeline(root, *, technique, params, steps, progress):
        release.wait(5.0)
        return {"technique": technique}

    monkeypatch.setattr(api_module, "run_pipeline", slow_pipeline)
    response = client.post(
        f"/api/projects/{project['id']}/inversion",
        json={"technique": "sbas", "params": {}},
    )
    assert response.status_code == 202
    job = response.get_json()["job"]

    health = client.get(f"/api/projects/{project['id']}/health")
    assert health.status_code == 409
    payload = health.get_json()
    assert payload["job"]["id"] == job["id"]
    assert payload["error"].startswith("该项目已有任务")

    again = client.post(
        f"/api/projects/{project['id']}/inversion",
        json={"technique": "sbas", "params": {}},
    )
    assert again.status_code == 409
    assert again.get_json()["job"]["id"] == job["id"]

    release.set()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").get_json()
        if state["status"] in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.05)
    assert state["status"] == "done"


def test_results_skip_validation_while_job_runs(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    project = client.post(
        "/api/projects",
        json={"name": "忙碌成果", "technique": "sbas", "source": "empty"},
    ).get_json()["project"]
    release = threading.Event()
    validation_calls: list[str] = []

    def slow_pipeline(root, *, technique, params, steps, progress):
        release.wait(5.0)
        return {"technique": technique}

    def unexpected_validation(root, **kwargs):
        validation_calls.append(str(root))
        return {"technique": "sbas"}

    monkeypatch.setattr(api_module, "run_pipeline", slow_pipeline)
    monkeypatch.setattr(api_module, "validate_demo", unexpected_validation)
    started = client.post(
        f"/api/projects/{project['id']}/inversion",
        json={"technique": "sbas", "params": {}},
    )
    assert started.status_code == 202
    job = started.get_json()["job"]

    results = client.get(f"/api/projects/{project['id']}/results")
    assert results.status_code == 200
    assert results.get_json()["validation"] is None
    assert validation_calls == []

    release.set()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").get_json()
        if state["status"] in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.05)
    assert state["status"] == "done"
