"""Job retention, cancellation and global concurrency (P2-3)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from insar_studio.jobs import JobManager


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_global_concurrency_limit_and_queueing(tmp_path):
    manager = JobManager(tmp_path / "jobs", max_concurrent=2)
    release = threading.Event()
    lock = threading.Lock()
    running = 0
    peak = 0

    def work(ctx):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        release.wait(5.0)
        with lock:
            running -= 1
        return {}

    jobs = [manager.submit(f"p{index}", "quick", work) for index in range(3)]

    assert _wait_for(lambda: peak == 2)
    statuses = [manager.get(job["id"])["status"] for job in jobs]
    assert statuses.count("running") == 2
    assert statuses.count("queued") == 1

    release.set()
    for job in jobs:
        manager.wait(job["id"], timeout=10.0)
        assert manager.get(job["id"])["status"] == "done"
    assert peak == 2


def test_cancelled_queued_job_never_runs(tmp_path):
    manager = JobManager(tmp_path / "jobs", max_concurrent=1)
    release = threading.Event()
    started = threading.Event()
    second_runs: list[str] = []

    def first(ctx):
        started.set()
        release.wait(5.0)
        return {}

    def second(ctx):
        second_runs.append(ctx.job_id)
        return {}

    first_job = manager.submit("p1", "quick", first)
    assert started.wait(5.0)
    queued = manager.submit("p2", "quick", second)
    assert manager.get(queued["id"])["status"] == "queued"

    assert manager.cancel(queued["id"]) is True
    release.set()
    manager.wait(first_job["id"], timeout=10.0)
    manager.wait(queued["id"], timeout=10.0)

    assert manager.get(queued["id"])["status"] == "cancelled"
    assert second_runs == []


def test_runtime_pruning_keeps_newest_and_drops_files(tmp_path):
    manager = JobManager(tmp_path / "jobs", max_concurrent=2)
    manager.MAX_JOBS = 3
    job_ids = []

    for index in range(6):
        job = manager.submit(f"p{index}", "quick", lambda ctx: {})
        job_ids.append(job["id"])
        manager.wait(job["id"], timeout=10.0)
        assert manager.get(job["id"])["status"] == "done"

    assert len(manager._jobs) <= 3
    assert set(job_ids[-3:]) <= set(manager._jobs)
    for stale in job_ids[:3]:
        assert stale not in manager._jobs
        assert not (manager.root / f"{stale}.json").exists()
    assert manager._cancel == {}


def test_pruning_never_evicts_an_active_job(tmp_path):
    manager = JobManager(tmp_path / "jobs", max_concurrent=2)
    manager.MAX_JOBS = 1
    release = threading.Event()

    def slow(ctx):
        release.wait(5.0)
        return {}

    active = manager.submit("p1", "slow", slow)
    assert _wait_for(lambda: manager.get(active["id"])["status"] == "running")

    quick_finished = threading.Event()

    def quick(ctx):
        quick_finished.set()
        return {}

    quick = manager.submit("p2", "quick", quick)
    assert quick_finished.wait(5.0)
    assert _wait_for(lambda: quick["id"] not in manager._jobs)

    assert active["id"] in manager._jobs
    assert manager.get(active["id"])["status"] == "running"

    release.set()
    manager.wait(active["id"], timeout=10.0)


def test_startup_prunes_old_job_files(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    ids = [f"job_0000{index}" for index in range(5)]
    for index, job_id in enumerate(ids):
        payload = {
            "id": job_id,
            "project_id": f"p{index}",
            "kind": "quick",
            "status": "done",
            "created_at": f"2026-01-0{index + 1} 00:00:00",
            "updated_at": f"2026-01-0{index + 1} 00:00:00",
        }
        (jobs_dir / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(JobManager, "MAX_JOBS", 3)
    manager = JobManager(jobs_dir)

    assert len(manager._jobs) == 3
    remaining = sorted(path.stem for path in jobs_dir.glob("job_*.json"))
    assert remaining == ids[-3:]


def test_concurrency_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("INSAR_MAX_CONCURRENT_JOBS", "3")
    assert JobManager(tmp_path / "a").max_concurrent == 3

    monkeypatch.setenv("INSAR_MAX_CONCURRENT_JOBS", "999")
    assert JobManager(tmp_path / "b").max_concurrent == 32

    monkeypatch.setenv("INSAR_MAX_CONCURRENT_JOBS", "not-a-number")
    assert JobManager(tmp_path / "c").max_concurrent == 2
