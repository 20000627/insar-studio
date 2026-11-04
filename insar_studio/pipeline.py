"""Pipeline orchestration.

Every step is independent and can be re-run.  The orchestration layer only
maps a global progress range onto the individual processors and records which
steps have been executed in the project metadata.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .core.atmosphere import run_atmosphere
from .core.coregistration import coregister_stack
from .core.interferometry import process_interferograms
from .core.inversion import run_inversion
from .core.io import load_stack
from .core.network import build_network, persist_network
from .core.utils import atomic_write_json, read_json
from .params import fingerprint, merge_params, validate_params, with_atmosphere_defaults
from .project import purge_downstream


class ProgressMapper:
    def __init__(self, callback: Callable[[float, str], None] | None, lo: float, hi: float) -> None:
        self.callback = callback
        self.lo = lo
        self.hi = hi

    def __call__(self, value: float, message: str = "") -> None:
        if self.callback:
            self.callback(self.lo + (self.hi - self.lo) * max(0.0, min(1.0, float(value))), message)


def _mark_step(
    root: Path,
    step: str,
    result: Any = None,
    *,
    params: dict[str, Any] | None = None,
    technique: str | None = None,
) -> None:
    path = root / "project.json"
    project = read_json(path, {}) or {}
    steps = project.setdefault("steps_run", {})
    steps[step] = {"at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if params is not None:
        steps[step]["params_hash"] = fingerprint(params, stage=step, technique=technique)
    if result is not None:
        try:
            if isinstance(result, dict):
                steps[step]["summary"] = {
                    key: result.get(key)
                    for key in ("n_pairs", "velocity_median_mm_yr", "mean_coherence", "method", "technique", "n_dates")
                    if key in result
                }
        except Exception:
            pass
    project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(path, project)


def run_coregistration(root: Path, params: dict[str, Any], progress=None) -> dict[str, Any]:
    purge_downstream(root, "coregistration")
    result = coregister_stack(root, params, progress=progress)
    _mark_step(root, "coregistration", result, params=params)
    return result


def run_network(root: Path, technique: str, params: dict[str, Any], progress=None) -> dict[str, Any]:
    stack = load_stack(root)
    # A new network invalidates every interferogram and product derived from
    # the previous pair list.  Purge before rebuilding while jobs are exclusive.
    purge_downstream(root, "network")
    coreg_path = root / "data" / "processed" / "slc_coreg.npy"
    coreg_done = coreg_path.exists()
    net_progress = progress
    project = read_json(root / "project.json", {}) or {}
    coreg_hash = (project.get("steps_run", {}).get("coregistration") or {}).get("params_hash")
    coreg_stale = bool(coreg_hash) and coreg_hash != fingerprint(params, stage="coregistration")
    if not coreg_done or coreg_stale:
        # The wizard exposes network as the first processing step after data
        # loading.  Coregistration is a required precursor, so run it here
        # instead of failing later in interferometry.  Re-run it when the
        # parameters that produced the coregistered stack have changed.
        if progress:
            reason = "配准参数已变更，重新执行 SLC 配准" if coreg_done else "检测到尚未配准，先执行 SLC 配准"
            progress(0.02, reason)

        def _coreg_progress(value: float, message: str = "") -> None:
            if progress:
                progress(0.04 + 0.56 * max(0.0, min(1.0, float(value))), message)

        coreg_result = coregister_stack(root, params, progress=_coreg_progress)
        _mark_step(root, "coregistration", coreg_result, params=params)
        if progress:
            progress(0.62, "配准完成，开始构建网络")

        def _net_progress(value: float, message: str = "") -> None:
            if progress:
                progress(0.64 + 0.34 * max(0.0, min(1.0, float(value))), message)

        net_progress = _net_progress
    if net_progress:
        net_progress(0.1, "构建基线网络")
    pairs, stats = build_network(
        stack.dates,
        stack.meta.get("bperp_m", [0.0] * stack.n_dates),
        technique=technique,
        params=params,
        ref_index=int(params.get("ref_index", stack.meta.get("ref_index", 0))),
    )
    if not pairs:
        raise RuntimeError("当前参数下没有可用干涉对，请放宽时间/垂直基线阈值")
    if net_progress:
        net_progress(0.75, f"保留 {len(pairs)} 个干涉对")
    payload = persist_network(root, pairs, stats)
    _mark_step(root, "network", stats, params=params, technique=technique)
    return payload


def run_interferometry(root: Path, technique: str, params: dict[str, Any], progress=None) -> dict[str, Any]:
    purge_downstream(root, "interferometry")
    if technique == "ds":
        from .core.ds import run_ds_interferometry

        summary = run_ds_interferometry(root, params, progress=progress)
    else:
        summary = process_interferograms(root, params, progress=progress, save_figures=False)
    _mark_step(root, "interferometry", summary, params=params, technique=technique)
    return summary


def _atmosphere_params(root: Path, project: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    p = dict(params)
    method = str(p.get("atmosphere_method", "")).lower()
    if not method:
        method = "gacos" if project.get("demo") else "empirical"
    p["method"] = method
    if method in ("gacos", "demo"):
        source = str(p.get("gacos_source", "demo" if project.get("demo") else "local"))
        p["gacos_source"] = source
        p["method"] = "gacos"
    return p


def run_atmosphere_step(root: Path, project: dict[str, Any], params: dict[str, Any], progress=None) -> dict[str, Any]:
    purge_downstream(root, "atmosphere")
    summary = run_atmosphere(root, _atmosphere_params(root, project, params), progress=progress)
    _mark_step(root, "atmosphere", summary, params=params)
    return summary


def run_inversion_step(root: Path, project: dict[str, Any], technique: str, params: dict[str, Any], progress=None) -> dict[str, Any]:
    purge_downstream(root, "inversion")
    summary = run_inversion(root, technique, params, progress=progress)
    _mark_step(root, "inversion", summary, params=params, technique=technique)
    return summary


def stage_ready(root: Path, stage: str) -> bool:
    if stage == "network":
        network = read_json(root / "data" / "network.json", {}) or {}
        return bool(network.get("pairs"))
    if stage == "interferometry":
        summary = read_json(root / "interferograms" / "summary.json", {}) or {}
        return bool(summary.get("n_pairs"))
    if stage == "atmosphere":
        return (root / "data" / "atmosphere" / "atmosphere.json").exists()
    if stage == "inversion":
        return (root / "products" / "inversion.json").exists()
    return False


def _expand_steps(root: Path, requested: list[str] | None) -> list[str]:
    """Add hard prerequisites so a single-step run never gets stuck.

    Atmosphere correction is intentionally *not* a prerequisite of inversion:
    the inversion can legitimately run on raw unwrapped phase.  Coregistration
    is handled by ``run_network`` itself.
    """
    wanted = set(requested or ["coregistration", "network", "interferometry", "atmosphere", "inversion"])
    order = ["coregistration", "network", "interferometry", "atmosphere", "inversion"]
    if "atmosphere" in wanted and not stage_ready(root, "interferometry"):
        wanted.add("interferometry")
    if "inversion" in wanted and not stage_ready(root, "interferometry"):
        wanted.add("interferometry")
    if "interferometry" in wanted and not stage_ready(root, "network"):
        wanted.add("network")
    return [stage for stage in order if stage in wanted]


def run_pipeline(
    project_root: Path,
    *,
    technique: str = "sbas",
    params: dict[str, Any] | None = None,
    steps: list[str] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run coregistration → network → interferometry → atmosphere → inversion."""
    root = Path(project_root)
    project = read_json(root / "project.json", {}) or {}
    technique = (technique or project.get("technique", "sbas")).lower()
    # Normalise once at the boundary.  Stage fingerprints must see the same
    # numeric types and default keys regardless of whether the caller is the
    # HTTP API, the CLI, or a direct Python invocation.
    p = validate_params(
        merge_params(project.get("params", {}), params or {}),
        technique=technique,
    )
    p = with_atmosphere_defaults(p, demo=bool(project.get("demo")))
    stages = _expand_steps(root, steps)
    if not stages:
        raise RuntimeError("没有可运行的阶段")
    outputs: dict[str, Any] = {}
    t0 = time.time()
    n = max(1, len(stages))
    for idx, stage in enumerate(stages):
        lo = 0.02 + 0.96 * idx / n
        hi = 0.02 + 0.96 * (idx + 1) / n
        mapper = ProgressMapper(progress, lo, hi)
        mapper(0.0, f"阶段 {idx + 1}/{n}：{stage}")
        if stage == "coregistration":
            outputs["coregistration"] = run_coregistration(root, p, mapper)
        elif stage == "network":
            outputs["network"] = run_network(root, technique, p, mapper)
        elif stage == "interferometry":
            outputs["interferometry"] = run_interferometry(root, technique, p, mapper)
        elif stage == "atmosphere":
            outputs["atmosphere"] = run_atmosphere_step(root, project, p, mapper)
        elif stage == "inversion":
            outputs["inversion"] = run_inversion_step(root, project, technique, p, mapper)
    outputs["elapsed_s"] = round(time.time() - t0, 2)
    outputs["technique"] = technique
    outputs["stages"] = stages
    _mark_step(root, "pipeline", outputs, params=p, technique=technique)
    return outputs
