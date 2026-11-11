"""Official GACOS request, import and safety tests.

The official service is asynchronous: requests are submitted through its web
form and products arrive by email as ``tar.gz`` archives.  These tests use fake
sessions and synthetic archives; they never contact gacos.net.
"""

from __future__ import annotations

import io
import json
import stat
import tarfile
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import insar_studio.api as api_module
import insar_studio.core.gacos as gacos_module
from insar_studio.core.gacos import (
    GACOS_MAX_UPLOAD_BYTES,
    GacosImportError,
    GacosOfficialClient,
    GacosRequestError,
    gacos_official_status,
    import_gacos_path,
    import_gacos_url,
    load_screens,
    read_delay_file,
    submit_official_request,
    validate_download_url,
    validate_gacos_request,
)
from insar_studio.core.geotiff import write_geotiff
from insar_studio.core.synthetic import generate_demo_dataset
from insar_studio.core.utils import atomic_write_json, read_json
from insar_studio.jobs import JobManager
from insar_studio.project import ProjectStore


class _FakeResponse:
    def __init__(self, text: str, *, status: int = 200, url: str = "http://www.gacos.net/result.php?flag=MQ=="):
        self.text = text
        self.status_code = status
        self.url = url

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, data=None, headers=None, timeout=None, allow_redirects=True):
        self.calls.append(
            {
                "url": url,
                "data": dict(data or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        return self.response


class _RecordingClient:
    def __init__(self) -> None:
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        return {
            "ok": True,
            "message": "Thanks for using GACOS!",
            "final_url": "http://www.gacos.net/result.php?flag=MQ==",
        }


class _DownloadResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict | None = None):
        self.body = body
        self.status_code = status
        self.headers = headers or {
            "content-length": str(len(body)),
            "content-type": "application/octet-stream",
        }

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        return None


class _DownloadSession:
    def __init__(self, routes: dict[str, _DownloadResponse]):
        self.routes = routes
        self.urls: list[str] = []

    def get(self, url, stream=True, allow_redirects=False, timeout=None):
        self.urls.append(url)
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response
        return _DownloadResponse(b"not found", status=404)


def _date_list(count: int, start: str = "2023-01-03") -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=12 * i)).isoformat() for i in range(count)]


def _demo_project(tmp_path: Path, n_dates: int = 2) -> tuple[ProjectStore, str, Path]:
    store = ProjectStore(tmp_path / "projects")
    project = store.create("gacos-test", "sbas")
    root = store.path(project["id"])
    generate_demo_dataset(root, rows=16, cols=16, n_dates=n_dates, interval_days=12, start="2023-01-03")
    return store, project["id"], root


def _write_project_tif(root: Path, date_text: str, value_m: float = 2.3) -> Path:
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    rows = int(meta.get("rows") or 16)
    cols = int(meta.get("cols") or 16)
    path = root / "downloads" / f"{date_text.replace('-', '')}.ztd.tif"
    # A coarser grid covering the same footprint exercises resampling.
    write_geotiff(
        path,
        np.full((max(4, rows // 2), max(4, cols // 2)), value_m, dtype=np.float32),
        corner_lon=float(meta.get("corner_lon", 106.52)),
        corner_lat=float(meta.get("corner_lat", 29.63)),
        lon_step=float(meta.get("lon_step", 0.00042)) * 2,
        lat_step=float(meta.get("lat_step", -0.00036)) * 2,
    )
    return path


def _tar_gz(path: Path, files: dict[str, Path]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, source in files.items():
            archive.add(source, arcname=name)
    return path


def test_validate_request_normalises_and_rejects_oversize():
    request = validate_gacos_request(
        bbox={"W": 100, "S": 30, "E": 101, "N": 31},
        dates="2023-01-03, 20230115 20230115",
        email="user@example.com",
        hour="10",
        minute="30",
        output_format="GeoTIFF",
    )
    assert request.bbox == (100.0, 30.0, 101.0, 31.0)
    assert request.dates == ["2023-01-03", "2023-01-15"]
    assert request.hour == 10 and request.minute == 30
    assert request.output_format == "geotiff"

    with pytest.raises(GacosRequestError, match="最多 20 个日期"):
        validate_gacos_request(
            bbox=[100, 30, 101, 31],
            dates=_date_list(21),
            email="user@example.com",
        )
    with pytest.raises(GacosRequestError, match="邮箱"):
        validate_gacos_request(bbox=[100, 30, 101, 31], dates=["2023-01-03"], email="not-an-email")
    with pytest.raises(GacosRequestError, match="10"):
        validate_gacos_request(bbox=[100, 30, 111, 31], dates=["2023-01-03"], email="user@example.com")
    with pytest.raises(GacosRequestError, match="经度"):
        validate_gacos_request(bbox=[101, 30, 100, 31], dates=["2023-01-03"], email="user@example.com")


def test_official_client_submits_one_date_per_line_and_parses_success():
    session = _FakeSession(
        _FakeResponse(
            '<h3 class="text-center h31 map_div">Thanks for using GACOS! <br/></h3>'
            "We have received your request which is summarized below:"
        )
    )
    client = GacosOfficialClient(session=session)
    request = validate_gacos_request(
        bbox=[100.25, 30.5, 101.75, 31.25],
        dates=["2023-01-03", "2023-01-15"],
        email="user@example.com",
        hour=10,
        minute=5,
        output_format="binary",
    )
    result = client.submit(request)
    assert result["ok"] is True
    assert "Thanks for using GACOS" in result["message"]
    call = session.calls[0]
    assert call["data"]["date"] == "20230103\n20230115"
    assert call["data"]["type"] == "1"
    assert call["data"]["H"] == "10" and call["data"]["M"] == "5"
    assert call["data"]["N"] == "31.25" and call["data"]["W"] == "100.25"
    # Signed/query parameters are removed before the URL is shown or stored.
    assert result["final_url"] == "http://www.gacos.net/result.php?***"


def test_official_client_reports_html_error_message():
    session = _FakeSession(
        _FakeResponse(
            '<h3 class="h31">Wrong date format. Please keep YYYYMMDD format only.</h3>',
            url="http://www.gacos.net/result.php?flag=NQ==",
        )
    )
    client = GacosOfficialClient(session=session)
    request = validate_gacos_request(
        bbox=[100, 30, 101, 31],
        dates=["2023-01-03"],
        email="user@example.com",
    )
    with pytest.raises(GacosRequestError, match="Wrong date format"):
        client.submit(request)


def test_submit_splits_batches_and_never_returns_plain_email():
    client = _RecordingClient()
    result = submit_official_request(
        bbox=[100, 30, 101, 31],
        dates=_date_list(25),
        email="user@example.com",
        allow_split=True,
        client=client,
        delay_seconds=0,
    )
    assert result["ok"] is True
    assert result["n_batches"] == 2
    assert [len(item.dates) for item in client.requests] == [20, 5]
    assert result["email_masked"] == "u***@example.com"
    serialised = json.dumps(result, ensure_ascii=False)
    assert "user@example.com" not in serialised
    assert all("user@example.com" not in json.dumps(batch, ensure_ascii=False) for batch in result["batches"])

    with pytest.raises(GacosRequestError, match="确认允许拆分"):
        submit_official_request(
            bbox=[100, 30, 101, 31],
            dates=_date_list(21),
            email="user@example.com",
            client=client,
            delay_seconds=0,
        )


def test_import_official_tar_and_load_screens(tmp_path):
    _store, _pid, root = _demo_project(tmp_path)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = list(meta.get("dates") or [])
    archive = root / "gacos_batch.tar.gz"
    _tar_gz(
        archive,
        {
            f"gacos/{dates[0].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[0], 2.3),
            f"gacos/{dates[1].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[1], 2.4),
        },
    )
    result = import_gacos_path(root, archive)
    assert result["ok"] is True
    assert result["dates"] == sorted(dates)
    status = gacos_official_status(root)["official"]
    assert status["n_available"] == len(dates)
    assert status["missing_dates"] == []
    assert status["last_import"]["n_files"] == 2
    manifest = read_json(root / "data" / "gacos" / "official" / "manifest.json", {}) or {}
    assert "user@example.com" not in json.dumps(manifest, ensure_ascii=False)

    grids, info = load_screens(
        root,
        {"gacos_source": "official"},
        target_shape=(int(meta["rows"]), int(meta["cols"])),
    )
    assert info["source"] == "official"
    assert set(grids) == set(dates)
    assert np.nanmedian(grids[dates[0]]) == pytest.approx(2300.0, rel=1e-3)


def test_import_email_link_page_follows_archive_link(tmp_path, monkeypatch):
    _store, _pid, root = _demo_project(tmp_path)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = list(meta.get("dates") or [])
    archive = root / "email.tar.gz"
    _tar_gz(
        archive,
        {
            f"{dates[0].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[0]),
            f"{dates[1].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[1]),
        },
    )
    body = archive.read_bytes()
    page = (
        "<!DOCTYPE html><html><head><title>GACOS</title></head><body>"
        '<a href="https://download.example.com/gacos.tar.gz">Download GACOS</a>'
        "</body></html>"
    ).encode("utf-8")
    session = _DownloadSession(
        {
            "https://download.example.com/page": _DownloadResponse(page, headers={"content-length": str(len(page))}),
            "https://download.example.com/gacos.tar.gz": _DownloadResponse(body),
        }
    )
    # Avoid the real DNS lookup; host validation itself is covered separately.
    monkeypatch.setattr(gacos_module, "validate_download_url", lambda url: str(url))
    result = import_gacos_url(root, "https://download.example.com/page", session=session)
    assert result["ok"] is True
    assert result["dates"] == sorted(dates)
    assert len(session.urls) == 2
    assert session.urls[0].startswith("https://download.example.com/page")


def test_load_screens_reports_missing_official_dates(tmp_path):
    _store, _pid, root = _demo_project(tmp_path)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    first = str(meta["dates"][0])
    archive = root / "partial.tar.gz"
    _tar_gz(archive, {f"{first.replace('-', '')}.ztd.tif": _write_project_tif(root, first)})
    import_gacos_path(root, archive)
    with pytest.raises(RuntimeError, match="GACOS 缺少 1 个日期"):
        load_screens(root, {"gacos_source": "official"})


def test_identical_reimport_does_not_bump_manifest_time(tmp_path):
    _store, _pid, root = _demo_project(tmp_path)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = list(meta.get("dates") or [])
    archive = root / "batch.tar.gz"
    _tar_gz(
        archive,
        {
            f"{dates[0].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[0]),
            f"{dates[1].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[1]),
        },
    )
    import_gacos_path(root, archive)
    manifest_path = root / "data" / "gacos" / "official" / "manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    manifest["updated_at"] = "2000-01-01 00:00:00"
    atomic_write_json(manifest_path, manifest)
    import_gacos_path(root, archive)
    again = read_json(manifest_path, {}) or {}
    assert again["updated_at"] == "2000-01-01 00:00:00"
    assert again["imports"][-1]["changed"] is False


def test_atmosphere_becomes_stale_after_new_official_products(tmp_path):
    store, project_id, root = _demo_project(tmp_path)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = list(meta.get("dates") or [])
    archive = root / "batch.tar.gz"
    _tar_gz(
        archive,
        {
            f"{dates[0].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[0]),
            f"{dates[1].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[1]),
        },
    )
    import_gacos_path(root, archive)
    atomic_write_json(
        root / "data" / "atmosphere" / "atmosphere.json",
        {
            "method": "gacos",
            "source": "official",
            "created_at": "2000-01-01 00:00:00",
            "n_pairs": 1,
        },
    )
    steps = store.status(project_id)
    atmosphere = next(step for step in steps if step["key"] == "atmosphere")
    assert atmosphere["stale"] is True
    assert "GACOS 产品已更新" in atmosphere["detail"]


def test_tar_traversal_and_zip_symlink_are_rejected(tmp_path):
    _store, _pid, root = _demo_project(tmp_path)
    traversal = root / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as archive:
        payload = b"not a raster"
        info = tarfile.TarInfo("../evil.ztd.tif")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(GacosImportError, match="路径穿越"):
        import_gacos_path(root, traversal)
    assert not (root / "evil.ztd.tif").exists()

    symlink = root / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("20230103.ztd.tif")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../outside")
    with pytest.raises(GacosImportError, match="符号链接"):
        import_gacos_path(root, symlink)


def test_private_and_non_http_download_urls_are_rejected():
    for url in (
        "file:///etc/passwd",
        "ftp://example.com/a.tar.gz",
        "http://127.0.0.1/a.tar.gz",
        "http://localhost/a.tar.gz",
        "http://10.0.0.1/a.tar.gz",
        "http://[::1]/a.tar.gz",
    ):
        with pytest.raises(GacosImportError):
            validate_download_url(url)


def test_geotiff_bounds_and_ztd_reader(tmp_path):
    tif = tmp_path / "20230103.ztd.tif"
    write_geotiff(
        tif,
        np.full((4, 5), 2.5, dtype=np.float32),
        corner_lon=100.0,
        corner_lat=31.0,
        lon_step=0.1,
        lat_step=-0.2,
    )
    grid = read_delay_file(tif)
    assert grid.data_mm.shape == (4, 5)
    assert np.nanmedian(grid.data_mm) == pytest.approx(2500.0)
    assert grid.bounds == pytest.approx((100.0, 30.2, 100.5, 31.0))

    ztd = tmp_path / "20230115.ztd"
    np.asarray(np.full((3, 4), 2.0), dtype="<f4").tofile(ztd)
    (tmp_path / "20230115.rsc").write_text(
        "WIDTH=4\nFILE_LENGTH=3\nX_FIRST=100.0\nY_FIRST=31.0\nX_STEP=0.1\nY_STEP=-0.2\n",
        encoding="utf-8",
    )
    grid = read_delay_file(ztd)
    assert grid.data_mm.shape == (3, 4)
    assert np.nanmedian(grid.data_mm) == pytest.approx(2000.0)
    assert grid.bounds == pytest.approx((100.0, 30.4, 100.4, 31.0))


def _client(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "api-projects")
    monkeypatch.setattr(api_module, "ProjectStore", lambda: store)
    monkeypatch.setattr(api_module, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(api_module, "JobManager", lambda: JobManager(tmp_path / "jobs"))
    app = api_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_api_gacos_status_request_and_email_not_persisted(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("gacos-api", "sbas")
    root = store.path(project["id"])
    generate_demo_dataset(root, rows=12, cols=12, n_dates=2)

    status = client.get(f"/api/projects/{project['id']}/gacos")
    assert status.status_code == 200
    payload = status.get_json()
    assert payload["official"]["n_dates"] == 2
    assert payload["limits"]["max_dates"] == 20
    assert payload["defaults"]["bbox"] is not None

    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "partial": False,
            "message": "已提交 1 个 GACOS 申请，请查收邮箱。",
            "error": "",
            "n_dates": len(kwargs["dates"]),
            "n_batches": 1,
            "batches": [{"ok": True, "n_dates": len(kwargs["dates"]), "dates": kwargs["dates"]}],
            "email_masked": "u***@example.com",
            "request": {
                "bbox": list(kwargs["bbox"]),
                "dates": kwargs["dates"],
                "hour": kwargs["hour"],
                "minute": kwargs["minute"],
                "output_format": kwargs["output_format"],
            },
        }

    monkeypatch.setattr(api_module, "submit_official_request", fake_submit)
    response = client.post(
        f"/api/projects/{project['id']}/gacos/request",
        json={
            "bbox": [106.5, 29.6, 106.6, 29.7],
            "dates": ["2023-01-03", "2023-01-15"],
            "email": "user@example.com",
            "hour": 10,
            "minute": 30,
            "output_format": "geotiff",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert captured["email"] == "user@example.com"
    last = read_json(root / "data" / "gacos" / "official" / "last_request.json", {}) or {}
    serialised = json.dumps(last, ensure_ascii=False)
    assert "user@example.com" not in serialised
    assert last["email_masked"] == "u***@example.com"


def test_api_gacos_multipart_import_and_capabilities(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch)
    project = store.create("gacos-upload", "sbas")
    root = store.path(project["id"])
    generate_demo_dataset(root, rows=12, cols=12, n_dates=2)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = list(meta.get("dates") or [])

    archive = root / "upload.tar.gz"
    _tar_gz(
        archive,
        {
            f"{dates[0].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[0]),
            f"{dates[1].replace('-', '')}.ztd.tif": _write_project_tif(root, dates[1]),
        },
    )
    with open(archive, "rb") as handle:
        response = client.post(
            f"/api/projects/{project['id']}/gacos/import",
            data={"files": (io.BytesIO(handle.read()), "gacos.tar.gz")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["status"]["official"]["n_available"] == 2
    manifest = read_json(root / "data" / "gacos" / "official" / "manifest.json", {}) or {}
    assert "user@example.com" not in json.dumps(manifest, ensure_ascii=False)

    capabilities = client.get("/api/capabilities").get_json()
    assert capabilities["gacos"]["official"] is True
    assert capabilities["gacos"]["limits"]["max_dates"] == 20
    assert client.application.config["MAX_CONTENT_LENGTH"] == GACOS_MAX_UPLOAD_BYTES

    overview = client.get(f"/api/projects/{project['id']}").get_json()
    assert overview["gacos"]["official"]["n_available"] == 2
