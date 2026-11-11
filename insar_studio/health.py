"""Project self-diagnostics.

The health check is intentionally read-only.  It walks the project directory,
validates the most important shapes and metadata, and returns a short,
actionable report that the web UI can render without loading large arrays.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import TECHNIQUES
from .core.geocode import geocode_readiness
from .core.utils import format_bytes, read_json
from .params import (
    fingerprint,
    merge_params,
    min_dates_for,
    validate_params,
    with_atmosphere_defaults,
)


def _check(name: str, label: str, status: str, detail: str, action: str | None = None) -> dict[str, Any]:
    return {"name": name, "label": label, "status": status, "detail": detail, "action": action}


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    try:
        return tuple(int(v) for v in np.load(path, mmap_mode="r", allow_pickle=False).shape)
    except Exception:
        return None


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "是"}


def _add_geocoding_check(add, root: Path, meta: dict[str, Any]) -> None:
    status = read_json(root / "products" / "geocode.json", {})
    if isinstance(status, dict) and status:
        state = str(status.get("status", "")).lower()
        coverage = status.get("coverage")
        if state in ("done", "partial"):
            detail = "已输出地图坐标 GeoTIFF"
            if coverage is not None:
                try:
                    detail += f" · 覆盖 {float(coverage):.1%}"
                except (TypeError, ValueError):
                    pass
            if state == "partial":
                detail += "；地图网格与雷达覆盖仅部分相交"
            add(_check("geocoding", "地理编码", "ok" if state == "done" else "warn", detail))
            return
        reasons = status.get("reasons") or []
        detail = "未输出地图坐标 GeoTIFF"
        if reasons:
            detail += f"：{reasons[0]}"
        add(_check("geocoding", "地理编码", "warn", detail, "data"))
        return
    ready = geocode_readiness(meta, root)
    if ready.get("ready"):
        add(_check("geocoding", "地理编码", "ok", "轨道与时间元数据齐全，反演后可输出地图坐标 GeoTIFF"))
    else:
        reasons = ready.get("reasons") or ["元数据不足"]
        add(_check("geocoding", "地理编码", "warn", f"无法地理编码：{reasons[0]}；成果仅在雷达坐标", "data"))


def diagnose_project(root: Path) -> dict[str, Any]:
    root = Path(root)
    project = read_json(root / "project.json", {})
    if not isinstance(project, dict) or not project:
        raise FileNotFoundError("项目不存在")
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    technique = str(project.get("technique", "sbas")).lower()
    params = _dict(project.get("params"))
    steps = _dict(project.get("steps_run"))

    def add(check: dict[str, Any]) -> None:
        checks.append(check)
        if check["status"] in ("warn", "error"):
            issues.append(check)

    # ---------------------------------------------------------------- files
    stack_dir = root / "data" / "stack"
    meta = _dict(read_json(stack_dir / "meta.json", {}))
    slc_file = stack_dir / "slc.npy"
    dem_file = stack_dir / "dem.npy"
    mask_file = stack_dir / "mask.npy"
    missing = [f.name for f in (slc_file, dem_file, mask_file) if not f.exists()]
    if missing:
        add(_check("stack_files", "数据文件", "error", f"缺少 {', '.join(missing)}", "data"))
    else:
        add(_check("stack_files", "数据文件", "ok", "slc / dem / mask 均存在"))

    if not meta:
        add(_check("stack_meta", "数据元信息", "error", "缺少 meta.json", "data"))
    else:
        try:
            n_dates = int(meta.get("n_dates") or 0)
            rows = int(meta.get("rows") or 0)
            cols = int(meta.get("cols") or 0)
        except (TypeError, ValueError):
            n_dates = rows = cols = 0
        dates = meta.get("dates")
        dates = dates if isinstance(dates, list) else []
        if n_dates < 2:
            add(_check("stack_meta", "数据元信息", "error", f"仅 {n_dates} 景影像，至少需要 2 景", "data"))
        elif rows < 2 or cols < 2:
            add(_check("stack_meta", "数据元信息", "error", "影像尺寸异常", "data"))
        elif len(dates) != n_dates:
            add(_check("stack_meta", "数据元信息", "error", "日期数量与影像数量不一致", "data"))
        elif len(set(dates)) != len(dates):
            add(_check("stack_meta", "数据元信息", "warn", "日期存在重复值", "data"))
        elif list(dates) != sorted(dates):
            add(_check("stack_meta", "数据元信息", "warn", "日期未按升序排列", "data"))
        else:
            add(_check("stack_meta", "数据元信息", "ok", f"{n_dates} 景 · {rows} × {cols} 像元"))

        try:
            required_dates = min_dates_for(technique)
        except Exception:
            required_dates = None
        if n_dates and required_dates is not None:
            route_label = str(TECHNIQUES.get(technique, {}).get("name") or technique.upper())
            if n_dates < required_dates:
                add(
                    _check(
                        "route_dates",
                        "路线影像数",
                        "error",
                        f"{route_label} 至少需要 {required_dates} 景影像，当前仅 {n_dates} 景",
                        "data",
                    )
                )
            else:
                add(
                    _check(
                        "route_dates",
                        "路线影像数",
                        "ok",
                        f"{route_label} 需要 ≥ {required_dates} 景，当前 {n_dates} 景",
                    )
                )

        if n_dates and len(dates) == n_dates:
            shape = _npy_shape(slc_file) if slc_file.exists() else None
            expected = (n_dates, rows, cols)
            if shape and shape != expected:
                add(_check("stack_shape", "数据形状", "error", f"SLC 形状 {shape}，元信息期望 {expected}", "data"))
            elif shape:
                add(_check("stack_shape", "数据形状", "ok", "形状一致"))

        for key, label, default in (("wavelength_m", "波长", 0.0555), ("incidence_deg", "入射角", 39.0)):
            try:
                value = float(meta.get(key, default))
            except (TypeError, ValueError):
                value = 0
            if not np.isfinite(value) or value <= 0:
                add(_check(f"meta_{key}", f"元信息 · {label}", "error", f"{key} 无效", "data"))

        if n_dates:
            raw_bperp = meta.get("bperp_m")
            legacy = False
            if raw_bperp is None:
                # Older imported metadata used ``bperp``.  Keep reading it for
                # compatibility, but never let a missing/non-canonical field
                # silently disable this check.
                raw_bperp = meta.get("bperp")
                legacy = raw_bperp is not None
            try:
                bperp = np.asarray(raw_bperp if raw_bperp is not None else [], dtype=float).ravel()
                if bperp.size == 0:
                    add(_check("baseline", "垂直基线", "warn", "缺少 bperp_m，DEM 误差与基线筛选不可用", "data"))
                elif bperp.size != n_dates:
                    add(
                        _check(
                            "baseline",
                            "垂直基线",
                            "warn",
                            f"bperp_m 数量为 {bperp.size}，与影像数量 {n_dates} 不一致",
                            "data",
                        )
                    )
                elif not np.all(np.isfinite(bperp)):
                    add(_check("baseline", "垂直基线", "warn", "bperp_m 含非有限值", "data"))
                elif float(np.nanmax(np.abs(bperp))) < 1e-6:
                    add(_check("baseline", "垂直基线", "warn", "所有垂直基线为 0，DEM 误差无法估计", "data"))
                else:
                    suffix = "（兼容旧字段 bperp）" if legacy else ""
                    source = str(meta.get("baseline_source", "")).lower()
                    if source == "orbit":
                        suffix += "，由本地精密轨道估计"
                    add(
                        _check(
                            "baseline",
                            "垂直基线",
                            "ok",
                            f"范围 {float(np.nanmin(bperp)):+.1f} – {float(np.nanmax(bperp)):+.1f} m{suffix}",
                        )
                    )
            except (TypeError, ValueError):
                add(_check("baseline", "垂直基线", "warn", "垂直基线格式异常", "data"))

    if meta:
        explicit_geo = meta.get("geo_referenced")
        if explicit_geo is None:
            try:
                geo_ok = (
                    abs(float(meta.get("corner_lon", 0.0))) > 1e-9
                    or abs(float(meta.get("corner_lat", 0.0))) > 1e-9
                )
            except (TypeError, ValueError):
                geo_ok = False
        else:
            geo_ok = _flag(explicit_geo)
        if geo_ok:
            try:
                west = float(meta.get("corner_lon", 0.0))
                north = float(meta.get("corner_lat", 0.0))
                add(
                    _check(
                        "geo_reference",
                        "地理参考",
                        "ok",
                        f"左上角 {west:.5f}, {north:.5f} · {meta.get('crs', 'EPSG:4326')}",
                    )
                )
            except (TypeError, ValueError):
                add(_check("geo_reference", "地理参考", "warn", "地理参考字段无法解析", "data"))
        else:
            add(
                _check(
                    "geo_reference",
                    "地理参考",
                    "warn",
                    "缺少 corner_lon/corner_lat，成果按像元坐标输出，无法直接叠加到 GIS",
                    "data",
                )
            )
        if str(meta.get("dates_source", "")).lower() == "inferred":
            add(
                _check(
                    "dates_source",
                    "日期完整性",
                    "warn",
                    "部分日期由系统按 12 天间隔推断，时间基线可能失真",
                    "data",
                )
            )
        if _flag(meta.get("already_coregistered")):
            coreg_file = root / "data" / "processed" / "slc_coreg.npy"
            if coreg_file.exists():
                add(
                    _check(
                        "pre_coregistered",
                        "预配准栈",
                        "ok",
                        "导入时已标记为已配准，流水线跳过二次配准",
                    )
                )
            else:
                add(
                    _check(
                        "pre_coregistered",
                        "预配准栈",
                        "error",
                        "标记为已配准，但缺少 data/processed/slc_coreg.npy，流水线会重新配准",
                        "data",
                    )
                )
        _add_geocoding_check(add, root, meta)

    if dem_file.exists():
        try:
            dem = np.load(dem_file, mmap_mode="r", allow_pickle=False)
            if dem.size and float(np.nanstd(dem)) < 1e-9:
                add(_check("dem", "DEM", "warn", "DEM 为常数，地形相位与 DEM 误差不可用"))
            else:
                add(_check("dem", "DEM", "ok", "DEM 有效"))
        except Exception:
            add(_check("dem", "DEM", "error", "DEM 读取失败", "data"))
    if mask_file.exists():
        try:
            mask = np.load(mask_file, mmap_mode="r", allow_pickle=False).astype(bool)
            fraction = float(mask.mean()) if mask.size else 0.0
            if fraction < 0.05:
                add(_check("mask", "有效区掩膜", "warn", f"有效像元仅 {fraction:.1%}", "data"))
            elif fraction > 0.995:
                add(_check("mask", "有效区掩膜", "warn", "掩膜几乎全为有效，建议检查水体/阴影掩膜"))
            else:
                add(_check("mask", "有效区掩膜", "ok", f"有效像元 {fraction:.1%}"))
        except Exception:
            add(_check("mask", "有效区掩膜", "error", "掩膜读取失败", "data"))

    # ------------------------------------------------------- stage fingerprints
    stale_stages: list[str] = []
    try:
        effective = validate_params(
            merge_params(params, {}),
            technique=technique,
        )
        effective = with_atmosphere_defaults(effective, demo=bool(project.get("demo")))
        for stage, action in (
            ("coregistration", "network"),
            ("network", "network"),
            ("interferometry", "interferometry"),
            ("atmosphere", "atmosphere"),
            ("inversion", "inversion"),
        ):
            record = steps.get(stage) if isinstance(steps, dict) else None
            stored = record.get("params_hash") if isinstance(record, dict) else None
            if not stored:
                continue
            stage_technique = None if stage in ("coregistration", "atmosphere") else technique
            current = fingerprint(effective, stage=stage, technique=stage_technique)
            if str(stored) != current:
                stale_stages.append(action)
    except Exception:
        pass
    if stale_stages:
        # Preserve execution order and avoid duplicate actions.
        unique = list(dict.fromkeys(stale_stages))
        add(
            _check(
                "params_sync",
                "参数一致性",
                "warn",
                f"{len(unique)} 个阶段使用了不同的参数，需重跑：{', '.join(unique)}",
                unique[0],
            )
        )
    elif steps:
        add(_check("params_sync", "参数一致性", "ok", "已记录的阶段参数一致"))

    # -------------------------------------------------------- coreg / network
    network = _dict(read_json(root / "data" / "network.json", {}))
    pairs = network.get("pairs", []) if isinstance(network, dict) else []
    if not isinstance(pairs, list):
        pairs = []
    coreg_file = root / "data" / "processed" / "slc_coreg.npy"
    if pairs:
        if not coreg_file.exists():
            add(_check("coreg", "配准", "error", "存在干涉网络但缺少配准影像", "network"))
        else:
            add(_check("coreg", "配准", "ok", "配准影像可用"))
        bad = []
        for pair in pairs:
            try:
                ref, sec = int(pair["ref"]), int(pair["sec"])
                if not (0 <= ref < len(meta.get("dates", [])) and 0 <= sec < len(meta.get("dates", []))):
                    bad.append(str(pair.get("id", "?")))
            except Exception:
                bad.append(str(pair.get("id", "?")))
        if bad:
            add(_check("network", "干涉网络", "error", f"{len(bad)} 个干涉对索引越界", "network"))
        else:
            add(_check("network", "干涉网络", "ok", f"{len(pairs)} 对 · 索引有效"))
        if isinstance(steps.get("network"), dict) and steps["network"].get("params_hash") and params:
            # The richer stale computation lives in ProjectStore.status(); here
            # we only report that a network exists so health stays read-only.
            pass
    else:
        add(_check("network", "干涉网络", "info", "尚未构建（数据就绪后运行网络）", "network"))

    # ------------------------------------------------------- interferograms
    summary = _dict(read_json(root / "interferograms" / "summary.json", {}))
    if pairs:
        if not summary:
            add(_check("interferograms", "干涉图", "warn", "网络已构建，但还没有干涉处理摘要", "interferometry"))
        else:
            try:
                n_done = int(summary.get("n_pairs") or 0)
            except (TypeError, ValueError):
                n_done = 0
            if n_done < 1:
                add(_check("interferograms", "干涉图", "error", "干涉处理没有生成有效产品", "interferometry"))
            else:
                add(_check("interferograms", "干涉图", "ok", f"{n_done} 对干涉图"))
            missing_pairs = []
            shape_bad = []
            expected_shape = None
            try:
                m = max(1, int(params.get("multilook", 2)))
            except (TypeError, ValueError):
                m = 2
            if meta.get("rows") and meta.get("cols"):
                try:
                    expected_shape = (int(meta["rows"]) // m, int(meta["cols"]) // m)
                except (TypeError, ValueError):
                    expected_shape = None
            for pair in pairs:
                if not isinstance(pair, dict):
                    missing_pairs.append(str(pair))
                    continue
                base = root / "interferograms" / str(pair.get("id", ""))
                for name in ("unwrapped.npy", "coherence.npy", "mask.npy"):
                    if not (base / name).exists():
                        missing_pairs.append(str(pair.get("id", "?/" + name)))
                if expected_shape and expected_shape[0] > 0 and (base / "unwrapped.npy").exists():
                    shape = _npy_shape(base / "unwrapped.npy")
                    if shape and shape != expected_shape:
                        shape_bad.append(str(pair.get("id", "?")))
            if missing_pairs:
                add(_check("ifg_files", "干涉产品完整性", "error", f"{len(missing_pairs)} 个文件缺失（首个：{missing_pairs[0]}）", "interferometry"))
            elif shape_bad:
                add(_check("ifg_shape", "干涉图尺寸", "error", f"{len(shape_bad)} 对尺寸与多视参数不一致", "interferometry"))
            else:
                add(_check("ifg_files", "干涉产品完整性", "ok", "关键数组齐全"))
    else:
        add(_check("interferograms", "干涉图", "info", "等待网络构建", "network"))

    # ---------------------------------------------------------- atmosphere
    atm = _dict(read_json(root / "data" / "atmosphere" / "atmosphere.json", {}))
    if atm:
        method = str(atm.get("method", "off"))
        corrected = list((root / "interferograms").glob("*/unwrapped_atmcorr.npy"))
        if method != "off" and not corrected:
            add(_check("atmosphere", "大气校正", "warn", f"方法为 {method}，但未找到校正数组", "atmosphere"))
        else:
            add(_check("atmosphere", "大气校正", "ok", f"{method} · {len(corrected)} 个校正数组"))
    elif summary:
        add(_check("atmosphere", "大气校正", "info", "尚未运行（反演可直接使用原始相位）", "atmosphere"))

    # ------------------------------------------------------------- products
    inv = _dict(read_json(root / "products" / "inversion.json", {}))
    if inv:
        if str(inv.get("technique", "")).lower() != technique:
            add(_check("inversion", "反演成果", "warn", f"成果路线为 {inv.get('technique')}，项目路线为 {technique}", "inversion"))
        else:
            try:
                n_valid = int(inv.get("valid_pixels") or 0)
            except (TypeError, ValueError):
                n_valid = 0
            add(_check("inversion", "反演成果", "ok", f"{technique.upper()} · {n_valid} 个有效像元"))
        arrays = root / "products" / "arrays"
        valid_file = arrays / "valid.npy"
        if not valid_file.exists():
            add(_check("product_files", "成果数组", "error", "缺少 valid.npy", "inversion"))
        else:
            valid_shape = _npy_shape(valid_file)
            for name in ("disp_ts_mm.npy", "velocity_mm_yr.npy", "temporal_coherence.npy", "residual_mm.npy", "dem_error_m.npy"):
                if not (arrays / name).exists():
                    add(_check("product_files", "成果数组", "error", f"缺少 {name}", "inversion"))
                    break
            else:
                add(_check("product_files", "成果数组", "ok", "速率、时序、相干与残差数组齐全"))
            if valid_shape:
                try:
                    valid = np.load(valid_file, mmap_mode="r", allow_pickle=False)
                    if int(valid.sum()) == 0:
                        add(_check("valid", "有效成果", "error", "没有有效像元", "inversion"))
                    else:
                        add(_check("valid", "有效成果", "ok", f"{int(valid.sum())} 个有效像元"))
                except Exception:
                    add(_check("valid", "有效成果", "error", "valid.npy 读取失败", "inversion"))
            ref = inv.get("reference_pixel")
            ref = ref if isinstance(ref, (list, tuple)) else []
            if len(ref) == 2 and valid_shape:
                try:
                    rr, cc = int(ref[0]), int(ref[1])
                    valid = np.load(valid_file, mmap_mode="r", allow_pickle=False)
                    in_bounds = 0 <= rr < valid.shape[0] and 0 <= cc < valid.shape[1]
                    if not in_bounds:
                        add(_check("reference", "参考像元", "error", "参考像元超出成果范围", "inversion"))
                    elif not bool(valid[rr, cc]):
                        add(_check("reference", "参考像元", "warn", f"参考像元 ({rr}, {cc}) 不在有效掩膜内", "inversion"))
                    else:
                        add(_check("reference", "参考像元", "ok", f"row {rr} / col {cc}"))
                except Exception:
                    add(_check("reference", "参考像元", "warn", "参考像元无法验证", "inversion"))
    elif summary:
        add(_check("inversion", "反演成果", "info", "干涉已就绪，可运行反演", "inversion"))
    else:
        add(_check("inversion", "反演成果", "info", "等待前置步骤"))

    report = root / "products" / "report" / "report.md"
    if inv and not report.exists():
        add(_check("report", "处理报告", "info", "尚未生成报告（不影响数据处理）", "results"))
    elif report.exists():
        add(_check("report", "处理报告", "ok", "已生成"))

    # ----------------------------------------------------------- disk / size
    try:
        usage = shutil.disk_usage(root)
        free = int(usage.free)
        if free < 200 * 1024 * 1024:
            add(_check("disk", "磁盘空间", "error", f"仅剩 {format_bytes(free)}", None))
        elif free < 2 * 1024 * 1024 * 1024:
            add(_check("disk", "磁盘空间", "warn", f"仅剩 {format_bytes(free)}", None))
        else:
            add(_check("disk", "磁盘空间", "ok", f"可用 {format_bytes(free)}"))
    except OSError:
        free = 0
    size = _dir_size(root)

    errors = [issue for issue in issues if issue["status"] == "error"]
    warnings = [issue for issue in issues if issue["status"] == "warn"]
    if errors:
        grade = "需要处理"
    elif warnings:
        grade = "基本可用"
    else:
        grade = "健康"

    if not meta:
        next_step = "data"
    elif not pairs:
        next_step = "network"
    elif not summary:
        next_step = "interferometry"
    elif not inv:
        next_step = "inversion"
    else:
        next_step = "results"
    if inv and report.exists():
        next_step = "results"
    if stale_stages:
        # A mismatching upstream fingerprint invalidates everything after it.
        next_step = list(dict.fromkeys(stale_stages))[0]

    return {
        "ok": not errors,
        "grade": grade,
        "technique": technique,
        "next_step": next_step,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": len(checks),
            "size_bytes": size,
            "size_human": format_bytes(size),
            "free_bytes": free,
            "free_human": format_bytes(free),
        },
        "issues": issues,
        "checks": checks,
    }
