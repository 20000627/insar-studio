"""API input validation and error semantics (P2-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import insar_studio.api as api_module
from insar_studio.jobs import JobManager
from insar_studio.project import InvalidProjectId, ProjectStore


def _client(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(api_module, "ProjectStore", lambda: store)
    monkeypatch.setattr(api_module, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(api_module, "JobManager", lambda: JobManager(tmp_path / "jobs"))
    app = api_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_invalid_project_id_is_client_error(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/api/projects/bad-id")
    assert response.status_code == 400
    assert "项目编号" in response.get_json()["error"]

    health = client.get("/api/projects/bad-id/health")
    assert health.status_code == 400

    run = client.post("/api/projects/bad-id/run", json={"technique": "sbas"})
    assert run.status_code == 400


def test_nonexistent_valid_project_id_stays_404(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/api/projects/prj_00000000_000000")

    assert response.status_code == 404
    assert response.get_json()["error"] == "项目不存在"


def test_store_path_raises_invalid_project_id(tmp_path):
    store = ProjectStore(tmp_path / "projects")

    with pytest.raises(InvalidProjectId):
        store.path("../escape")
    with pytest.raises(InvalidProjectId):
        store.path("bad-id")


def test_broken_json_does_not_create_project(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    before = {path.name for path in store.root.iterdir() if path.is_dir()}

    response = client.post(
        "/api/projects",
        data='{"name": "broken"',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "JSON" in response.get_json()["error"]
    after = {path.name for path in store.root.iterdir() if path.is_dir()}
    assert after == before


@pytest.mark.parametrize("payload", ["[]", '"just a string"', "42", "null"])
def test_json_body_must_be_object(tmp_path, monkeypatch, payload):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/projects",
        data=payload,
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "对象" in response.get_json()["error"]


def test_json_requires_json_content_type(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/projects",
        data=json.dumps({"name": "wrong type"}),
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert "application/json" in response.get_json()["error"]


def test_json_body_size_limit(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("INSAR_MAX_JSON_BYTES", "1K")

    response = client.post(
        "/api/projects",
        data=json.dumps({"name": "x" * 4096}),
        content_type="application/json",
    )

    assert response.status_code == 413
    assert "JSON" in response.get_json()["error"]


def test_empty_body_is_allowed_for_optional_json(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("空请求体", "sbas", source="empty")

    response = client.post(f"/api/projects/{project['id']}/reset")

    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_bad_json_on_run_does_not_start_job(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("坏 JSON", "sbas", source="empty")

    response = client.post(
        f"/api/projects/{project['id']}/run",
        data="{",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert client.get("/api/jobs").get_json()["jobs"] == []


def test_unknown_download_provider_is_400(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post("/api/download/search", json={"provider": "unknown"})

    assert response.status_code == 400
    assert "provider" in response.get_json()["error"]


def test_download_fetch_unknown_provider_is_400(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("下载项目", "sbas", source="empty")

    response = client.post(
        "/api/download/fetch",
        json={
            "project_id": project["id"],
            "url": "https://example.com/a.zip",
            "provider": "unknown",
        },
    )

    assert response.status_code == 400
    assert "provider" in response.get_json()["error"]


def test_download_upstream_failure_is_502(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("upstream timeout")

    monkeypatch.setattr(api_module.download_mod, "search", fail)

    response = client.post("/api/download/search", json={"provider": "copernicus"})

    assert response.status_code == 502
    assert "搜索失败" in response.get_json()["error"]
