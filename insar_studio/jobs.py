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

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import WORKSPACE_DIR
from .core.utils import atomic_write_json, json_safe, read_json


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

    def __init__(self, jobs_dir: Path | None = None) -> None:
        self.root = Path(jobs_dir or (WORKSPACE_DIR / "jobs"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._prune_old_files()
        self._load_existing()

    def _prune_old_files(self) -> None:
        files = sorted(self.root.glob("job_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[self.MAX_JOBS :]:
            try:
                path.unlink()
            except OSError:
                pass

    def _load_existing(self) -> None:
        for path in sorted(self.root.glob("job_*.json")):
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
        try:
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
                result=json_safe(result),
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
            with self._lock:
                self._threads.pop(job_id, None)

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
