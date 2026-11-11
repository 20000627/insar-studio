"""Credential redaction tests for project files, reports and job records."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

from insar_studio.core.utils import atomic_write_json
from insar_studio.jobs import JobManager
from insar_studio.params import public_params, redact_secrets
from insar_studio.pipeline import run_pipeline
from insar_studio.project import ProjectStore
from insar_studio.report import export_zip, generate_report

TOKEN = "P0-7-token-9f3a"
PASSWORD = "P0-7-password-7c1d"
EMAIL = "P0-7-email@example.com"
USER = "P0-7-user"


def _secret_params() -> dict:
    return {
        "gacos_token": TOKEN,
        "gacos_email": EMAIL,
        "username": USER,
        "password": PASSWORD,
        "extra": {"api_token": TOKEN, "keep": "visible"},
    }


def test_redact_secrets_masks_nested_credentials():
    payload = {
        "params": {
            "gacos_token": TOKEN,
            "gacos_email": EMAIL,
            "multilook": 2,
            "extra": {"password": PASSWORD, "keep": "visible"},
        },
        "items": [{"authorization": "Bearer abc", "keep": 1}],
    }

    out = redact_secrets(payload)
    text = json.dumps(out, ensure_ascii=False)

    assert out["params"]["gacos_token"] == "***"
    assert out["params"]["gacos_email"] == "***"
    assert out["params"]["multilook"] == 2
    assert out["params"]["extra"] == {"password": "***", "keep": "visible"}
    assert out["items"][0] == {"authorization": "***", "keep": 1}
    for secret in (TOKEN, EMAIL, PASSWORD, "Bearer abc"):
        assert secret not in text


def test_public_params_recursively_drops_credentials():
    out = public_params(
        {
            "gacos_token": TOKEN,
            "multilook": 2,
            "extra": {"password": PASSWORD, "keep": "visible"},
            "items": [{"token": TOKEN, "keep": 1}],
        }
    )

    assert "gacos_token" not in out
    assert out["multilook"] == 2
    assert out["extra"] == {"keep": "visible"}
    assert out["items"] == [{"keep": 1}]


def test_atomic_write_json_never_persists_plaintext_credentials(tmp_path):
    path = tmp_path / "nested" / "summary.json"
    atomic_write_json(path, {"params": _secret_params(), "ok": True})

    raw = path.read_text(encoding="utf-8")
    for secret in (TOKEN, PASSWORD, EMAIL, USER):
        assert secret not in raw
    payload = json.loads(raw)
    assert payload["params"]["gacos_token"] == "***"
    assert payload["params"]["extra"] == {"api_token": "***", "keep": "visible"}


def test_pipeline_outputs_and_zip_are_free_of_plaintext_credentials(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create_demo("P0-7 secrets", "dinsar")
    root = store.path(project["id"])
    params = {**project["params"], "dinsar_pair": [0, 5], **_secret_params()}

    result = run_pipeline(root, technique="dinsar", params=params)
    assert result["technique"] == "dinsar"

    expected = [
        root / "data" / "processed" / "coreg.json",
        root / "interferograms" / "summary.json",
        root / "products" / "inversion.json",
    ]
    for path in expected:
        assert path.exists(), path
        raw = path.read_text(encoding="utf-8")
        for secret in (TOKEN, PASSWORD, EMAIL, USER):
            assert secret not in raw, f"{secret} leaked into {path}"

    for path in root.rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        for secret in (TOKEN, PASSWORD, EMAIL, USER):
            assert secret not in raw, f"{secret} leaked into {path}"

    report = generate_report(root)
    markdown = (root / report["markdown"]).read_text(encoding="utf-8")
    for secret in (TOKEN, PASSWORD, EMAIL, USER):
        assert secret not in markdown

    archive = export_zip(root)
    text_suffixes = {".json", ".md", ".csv", ".txt", ".geojson"}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in text_suffixes:
                text = zf.read(name).decode("utf-8", errors="ignore")
                for secret in (TOKEN, PASSWORD, EMAIL, USER):
                    assert secret not in text, f"{secret} leaked into ZIP member {name}"


def test_job_result_masks_credentials_in_memory_and_on_disk(tmp_path):
    manager = JobManager(tmp_path / "jobs")

    def work(ctx):
        ctx.progress(0.5, "处理中")
        return {"params": _secret_params(), "value": 1}

    job = manager.submit("prj_secret_test", "test", work, label="凭据测试", exclusive=False)
    for _ in range(200):
        payload = manager.get(job["id"])
        if payload["status"] in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.02)

    assert payload["status"] == "done", payload.get("error")
    assert payload["result"]["params"]["gacos_token"] == "***"
    assert payload["result"]["params"]["extra"] == {"api_token": "***", "keep": "visible"}

    raw = (tmp_path / "jobs" / f"{job['id']}.json").read_text(encoding="utf-8")
    for secret in (TOKEN, PASSWORD, EMAIL, USER):
        assert secret not in raw
