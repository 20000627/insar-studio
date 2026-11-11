"""Background job manager for long-running InSAR steps.

Jobs are executed in daemon threads and mirrored to JSON files under
``workspace/jobs`` so that the browser can reconnect after a page reload.  The
engine stays synchronous; the manager only provides progress, logging and
cooperative cancellation.

Only one exclusive job may run per project at a time.  This prevents two
pipelines from writing the same interferogram or product files concurrently,
which is otherwise a silent source of corrupt results.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import WORKSPACE_DIR
from .core.utils import atomic_write_json, json_safe, read_json
from .params import redact_secrets


FINISHED_STATUSES = ("done", "failed", "cancelled", "interrupted")
ACTIVE_STATUSES = ("queued", "running", "cancelling")


def _max_concurrent_from_env(default: int = 2) -> int:
    raw = os.environ.get("INSAR_MAX_CONCURRENT_JOBS", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value < 1:
        return default
    return min(value, 32)


class JobCancelled(RuntimeError):
    pass


class JobBusyError(RuntimeError):
    """Raised when a project already has an active exclusive job."""

    def __init__(self, job: dict[str, Any]) -> None:
        self.job = job
        super().__init__(f"项目已有任务在运行：{job.get('label') or job.get('kind') or job.get('id')}")


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str, cancel_event: threading.Event) -> None:
        self.manager = manager
        self.job_id = job_id
        self._cancel_event = cancel_event

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def progress(self, value: float, message: str = "") -> None:
        self.check()
        self.manager.update(self.job_id, progress=max(0.0, min(1.0, float(value))), stage=message)
        if message:
            self.manager.log(self.job_id, message)

    def log(self, message: str, level: str = "info") -> None:
        self.manager.log(self.job_id, message, level)

    def check(self) -> None:
        if self.cancelled:
            raise JobCancelled("任务已取消")


class JobManager:
    MAX_JOBS = 300

    def __init__(self, jobs_dir: Path | None = None, *, max_concurrent: int | None = None) -> None:
        self.root = Path(jobs_dir or (WORKSPACE_DIR / "jobs"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = (
            max(1, min(int(max_concurrent), 32))
            if max_concurrent is not None
            else _max_concurrent_from_env()
        )
        self._slots = threading.BoundedSemaphore(self.max_concurrent)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._load_existing()

    def _prune_runtime(self) -> None:
        """Bound in-memory jobs and delete files for evicted records.

        The initial implementation only trimmed ``job_*.json`` at startup while
        ``_jobs`` and ``_cancel`` grew forever.  Keep active jobs regardless of
        the limit; evict the oldest finished ones in insertion order so a long
        running server stays bounded.
        """
        with self._lock:
            excess = len(self._jobs) - self.MAX_JOBS
            if excess > 0:
                finished = [
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.get("status") in FINISHED_STATUSES
                ]
                for job_id in finished[:excess]:
                    self._jobs.pop(job_id, None)
                    self._cancel.pop(job_id, None)
            # Remove files whose records are no longer tracked (evicted or
            # unreadable).  Active and retained jobs stay untouched.
            for path in self.root.glob("job_*.json"):
                if path.stem not in self._jobs:
                    try:
                        path.unlink()
                    except OSError:
                        pass

    def _load_existing(self) -> None:
        # Oldest first: insertion order then drives runtime pruning.
        paths = sorted(
            self.root.glob("job_*.json"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        for path in paths:
            data = read_json(path)
            if not isinstance(data, dict) or not data.get("id"):
                continue
            if data.get("status") in ("queued", "running", "cancelling"):
                data["status"] = "interrupted"
                data["stage"] = "服务重启，任务中断"
                data["error"] = "服务重启，任务中断"
                data["finished_at"] = data.get("updated_at", "")
                atomic_write_json(path, data)
            self._jobs[data["id"]] = data
        self._prune_runtime()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _persist(self, job: dict[str, Any]) -> None:
        data = dict(job)
        data["logs"] = data.get("logs", [])[-300:]
        atomic_write_json(self._path(job["id"]), data)

    def submit(
        self,
        project_id: str,
        kind: str,
        func: Callable[[JobContext], Any],
        *,
        label: str = "",
        meta: dict[str, Any] | None = None,
        exclusive: bool = True,
    ) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        job = {
            "id": job_id,
            "project_id": project_id,
            "kind": kind,
            "label": label or kind,
            "status": "queued",
            "progress": 0.0,
            "stage": "排队中",
            "created_at": now,
            "updated_at": now,
            "started_at": "",
            "finished_at": "",
            "cancel_requested_at": "",
            "logs": [],
            "result": None,
            "error": "",
            "meta": meta or {},
            "exclusive": bool(exclusive),
        }
        cancel_event = threading.Event()
        ctx = JobContext(self, job_id, cancel_event)
        with self._lock:
            if exclusive:
                active = self.active(project_id)
                if active:
                    raise JobBusyError(active[0])
            self._jobs[job_id] = job
            self._cancel[job_id] = cancel_event
            self._persist(job)
            thread = threading.Thread(target=self._run, args=(job_id, func, ctx), daemon=True, name=job_id)
            self._threads[job_id] = thread
            thread.start()
        return self.get(job_id, include_logs=False)

    def _run(self, job_id: str, func: Callable[[JobContext], Any], ctx: JobContext) -> None:
        acquired = False
        try:
            # Wait for a global execution slot.  Queued jobs keep their thread
            # alive but do no work; cancellation is checked while waiting.
            while not ctx.cancelled:
                if self._slots.acquire(timeout=0.2):
                    acquired = True
                    break
            if ctx.cancelled:
                self.update(job_id, status="cancelled", stage="已取消", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                return
            self.update(job_id, status="running", started_at=time.strftime("%Y-%m-%d %H:%M:%S"), stage="开始")
            result = func(ctx)
            ctx.check()
            self.update(
                job_id,
                status="done",
                progress=1.0,
                stage="完成",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                result=redact_secrets(json_safe(result)),
            )
        except JobCancelled:
            self.update(job_id, status="cancelled", stage="已取消", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            self.log(job_id, tb, "error")
            self.update(
                job_id,
                status="failed",
                stage="失败",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        finally:
            if acquired:
                self._slots.release()
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancel.pop(job_id, None)
            self._prune_runtime()

    def update(self, job_id: str, **patch: Any) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.get("status") in ("done", "failed", "cancelled", "interrupted") and any(
                key in patch for key in ("status", "progress", "stage")
            ):
                # Late progress callbacks must not resurrect a finished job.
                return dict(job)
            job.update(patch)
            job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._persist(job)
            return dict(job)

    def log(self, job_id: str, message: str, level: str = "info") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if job.get("status") in ("done", "failed", "cancelled", "interrupted"):
                return
            logs = job.setdefault("logs", [])
            last = logs[-1]["message"] if logs else None
            if last == message:
                return
            logs.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "level": level,
                    "message": str(message)[:2000],
                }
            )
            job["logs"] = logs[-300:]
            job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._persist(job)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            cancel = self._cancel.get(job_id)
            if not job or not cancel:
                return False
            if job.get("status") in ("done", "failed", "cancelled", "interrupted"):
                return False
            cancel.set()
            job["status"] = "cancelling"
            job["stage"] = "正在取消…"
            job["cancel_requested_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._persist(job)
            return True

    def get(self, job_id: str, *, include_logs: bool = True) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise FileNotFoundError("任务不存在")
            data = dict(job)
            if not include_logs:
                data["logs"] = data.get("logs", [])[-6:]
            return data

    def list(self, project_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values() if not project_id or j.get("project_id") == project_id]
        jobs.sort(key=lambda j: (j.get("created_at", ""), j.get("id", "")), reverse=True)
        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 30
        for job in jobs:
            job["logs"] = job.get("logs", [])[-3:]
        return jobs[:limit]

    def active(self, project_id: str) -> list[dict[str, Any]]:
        return [
            j
            for j in self.list(project_id, limit=50)
            if j.get("status") in ("queued", "running", "cancelling")
        ]

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread:
            thread.join(timeout)
        return self.get(job_id)
