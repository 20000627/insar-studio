"""Project list caching, filtering and pagination (P3-1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import insar_studio.api as api_module
import insar_studio.project as project_module
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


def _write_project(root: Path, project_id: str, name: str, technique: str = "sbas") -> Path:
    path = root / project_id
    path.mkdir(parents=True, exist_ok=True)
    project = {
        "schema": 2,
        "id": project_id,
        "name": name,
        "technique": technique,
        "source": "empty",
        "demo": False,
        "created_at": "2026-01-01 00:00:00",
        "updated_at": "2026-01-01 00:00:00",
        "params": {},
        "tags": [],
        "steps_run": {},
    }
    (path / "project.json").write_text(json.dumps(project), encoding="utf-8")
    return path


def test_list_page_filters_and_paginates(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    created = [
        store.create("Alpha 项目", "sbas", source="empty"),
        store.create("Beta 项目", "ps", source="empty"),
        store.create("Gamma 项目", "ds", source="empty"),
    ]

    first, total = store.list_page(limit=2)
    assert total == 3
    assert len(first) == 2
    assert len({row["id"] for row in first}) == 2

    second, total = store.list_page(offset=2, limit=2)
    assert total == 3
    assert len(second) == 1
    seen = {row["id"] for row in first} | {row["id"] for row in second}
    assert seen == {project["id"] for project in created}

    filtered, total = store.list_page(query="beta")
    assert total == 1
    assert filtered[0]["name"] == "Beta 项目"


def test_summary_cache_reuses_rows_and_invalidates(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("缓存项目", "sbas", source="empty")
    calls = {"count": 0}
    original = ProjectStore._summary_row

    def counting(self, item, path):
        calls["count"] += 1
        return original(self, item, path)

    monkeypatch.setattr(ProjectStore, "_summary_row", counting)

    rows, total = store.list_page()
    assert total == 1
    assert calls["count"] == 1

    # Unchanged files must hit the row cache instead of recomputing status/size.
    rows, total = store.list_page()
    assert total == 1
    assert calls["count"] == 1

    # A local write invalidates both the catalog and the computed row.
    store.update(project["id"], {"name": "改名后的项目"})
    rows, total = store.list_page()
    assert total == 1
    assert calls["count"] == 2
    assert rows[0]["name"] == "改名后的项目"


def test_catalog_cache_ttl_and_refresh(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    keep = store.create("已有项目", "sbas", source="empty")

    rows, total = store.list_page()
    assert total == 1
    assert rows[0]["id"] == keep["id"]

    # A project written by another process is invisible until the TTL expires
    # or the caller explicitly asks for a refresh.
    external = _write_project(store.root, "prj_external_0001", "外部项目")
    rows, total = store.list_page()
    assert total == 1

    rows, total = store.list_page(refresh=True)
    assert total == 2
    assert {row["id"] for row in rows} == {keep["id"], "prj_external_0001"}

    # Expiring the catalog timestamp makes the next scan see the new project
    # even without refresh.
    assert store._catalog_cache is not None
    store._catalog_cache = (
        time.monotonic() - project_module.PROJECT_LIST_TTL_SECONDS - 1,
        store._catalog_cache[1],
    )
    _write_project(store.root, "prj_external_0002", "第二个外部项目")
    rows, total = store.list_page()
    assert total == 3
    assert "prj_external_0002" in {row["id"] for row in rows}
    assert external.name == "prj_external_0001"


def test_row_cache_expires_with_ttl(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("旧名字", "sbas", source="empty")

    rows, _ = store.list_page()
    assert rows[0]["name"] == "旧名字"
    assert project["id"] in store._row_cache

    # Simulate an out-of-band project.json rewrite (engine/CLI process): both
    # caches must expire instead of pinning the old row forever.
    expired = time.monotonic() - project_module.PROJECT_LIST_TTL_SECONDS - 1
    store._catalog_cache = (expired, store._catalog_cache[1])
    store._row_cache[project["id"]] = (expired, store._row_cache[project["id"]][1])
    path = store.path(project["id"]) / "project.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["name"] = "新名字"
    path.write_text(json.dumps(data), encoding="utf-8")

    rows, total = store.list_page()
    assert total == 1
    assert rows[0]["name"] == "新名字"


def test_cache_drops_deleted_projects(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    keep = store.create("保留", "sbas", source="empty")
    drop = store.create("删除", "sbas", source="empty")

    store.list_page()
    assert drop["id"] in store._row_cache

    store.delete(drop["id"])
    rows, total = store.list_page()

    assert total == 1
    assert rows[0]["id"] == keep["id"]
    assert drop["id"] not in store._row_cache


def test_list_page_does_not_walk_project_trees(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("无扫描列表", "sbas", source="empty")
    products = store.path(project["id"]) / "products" / "figures"
    products.mkdir(parents=True)
    for index in range(20):
        (products / f"figure_{index:02d}.png").write_bytes(b"x")

    def forbidden(*args, **kwargs):
        raise AssertionError("项目列表不应递归扫描项目目录")

    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(ProjectStore, "product_files", forbidden)

    rows, total = store.list_page()
    assert total == 1
    assert rows[0]["id"] == project["id"]
    assert "size_bytes" not in rows[0]
    assert "size_human" not in rows[0]


def test_overview_omits_files_until_requested(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("按需文件列表", "sbas", source="empty")
    products = store.path(project["id"]) / "products" / "figures"
    products.mkdir(parents=True)
    (products / "velocity.png").write_bytes(b"image")
    (products / "velocity.tif").write_bytes(b"raster")

    overview = client.get(f"/api/projects/{project['id']}")
    assert overview.status_code == 200
    assert overview.get_json()["files"] == []

    results = client.get(f"/api/projects/{project['id']}/results")
    assert results.status_code == 200
    names = [item["name"] for item in results.get_json()["files"]]
    assert names == ["velocity.png", "velocity.tif"]


def test_api_project_list_pagination(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    for index in range(3):
        store.create(f"接口项目 {index}", "sbas", source="empty")

    response = client.get("/api/projects?limit=2")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 3
    assert len(payload["projects"]) == 2
    assert payload["has_more"] is True

    response = client.get("/api/projects?offset=2&limit=2")
    payload = response.get_json()
    assert payload["total"] == 3
    assert len(payload["projects"]) == 1
    assert payload["has_more"] is False

    response = client.get("/api/projects?q=接口项目 1")
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["projects"][0]["name"] == "接口项目 1"


def test_api_project_list_refresh_bypasses_cache(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    store.create("接口缓存", "sbas", source="empty")

    payload = client.get("/api/projects").get_json()
    assert payload["total"] == 1

    # Another ProjectStore instance (separate process) cannot invalidate this
    # one's cache; the explicit refresh parameter must rescan the directory.
    other = ProjectStore(store.root)
    other.create("其他进程项目", "sbas", source="empty")

    payload = client.get("/api/projects").get_json()
    assert payload["total"] == 1

    payload = client.get("/api/projects?refresh=1").get_json()
    assert payload["total"] == 2


def test_api_project_list_rejects_bad_pagination(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    for query in ("limit=0", "limit=abc", "offset=-1", "offset=x"):
        response = client.get(f"/api/projects?{query}")
        assert response.status_code == 400
        assert "error" in response.get_json()

    response = client.get("/api/projects?limit=9999")
    assert response.status_code == 200
    assert response.get_json()["limit"] == 500
