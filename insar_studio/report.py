"""Deliverable report and archive export."""

from __future__ import annotations

import csv
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .core.utils import format_bytes, read_json
from .params import public_params
from .validation import validate_demo

#: File types kept in an export.  Everything under ``products`` matches these,
#: while caches and editor backups are skipped.
_EXPORT_SUFFIXES = (".npy", ".tif", ".tiff", ".png", ".csv", ".json", ".md", ".docx")

#: ``results`` is the small, shareable package; ``full`` keeps the source stack
#: and all intermediates for archival or reprocessing on another machine.
EXPORT_MODES = ("results", "full")
_EXPORT_ROOTS = {
    "results": ("products",),
    "full": (
        "products",
        "data/stack",
        "data/network.json",
        "data/processed/coreg.json",
        "data/atmosphere",
        "interferograms",
    ),
}


def _iter_export_files(root: Path, mode: str):
    """Yield files that belong in an export, skipping the export directory."""
    for folder in _EXPORT_ROOTS[mode]:
        path = root / folder
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else (item for item in path.rglob("*") if item.is_file())
        for item in candidates:
            rel = item.relative_to(root)
            if "exports" in rel.parts:
                continue
            if item.suffix.lower() not in _EXPORT_SUFFIXES:
                continue
            yield item


def export_summary(root: Path) -> dict[str, Any]:
    """Return the file count and uncompressed size of each export mode."""
    root = Path(root)
    modes: dict[str, Any] = {}
    for mode in EXPORT_MODES:
        files = list(_iter_export_files(root, mode))
        total = 0
        for item in files:
            try:
                total += item.stat().st_size
            except OSError:
                continue
        modes[mode] = {
            "files": len(files),
            "bytes": total,
            "human": format_bytes(total),
            "label": "仅成果" if mode == "results" else "完整包（含原始栈）",
        }
    return {
        "modes": modes,
        "recommended": "results",
        "note": "ZIP 实际大小通常小于未压缩合计；导出在请求内同步生成。",
    }


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def _project_context(root: Path) -> dict[str, Any]:
    project = read_json(root / "project.json", {}) or {}
    stack = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    network = read_json(root / "data" / "network.json", {}) or {}
    ifg = read_json(root / "interferograms" / "summary.json", {}) or {}
    atm = read_json(root / "data" / "atmosphere" / "atmosphere.json", {}) or {}
    inv = read_json(root / "products" / "inversion.json", {}) or {}
    coreg = read_json(root / "data" / "processed" / "coreg.json", {}) or {}
    validation = validate_demo(root)
    return {
        "project": project,
        "stack": stack,
        "network": network,
        "ifg": ifg,
        "atm": atm,
        "inv": inv,
        "coreg": coreg,
        "validation": validation or {},
    }


def build_markdown(root: Path) -> str:
    root = Path(root)
    c = _project_context(root)
    project, stack, network, ifg, atm, inv, coreg, val = (
        c["project"],
        c["stack"],
        c["network"],
        c["ifg"],
        c["atm"],
        c["inv"],
        c["coreg"],
        c["validation"],
    )
    dates = stack.get("dates", [])
    lines: list[str] = []
    lines.append(f"# {project.get('name', 'InSAR 项目')} · 处理报告")
    lines.append("")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 项目概况")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("| --- | --- |")
    lines.append(f"| 技术路线 | {str(project.get('technique', 'sbas')).upper()} |")
    lines.append(f"| 数据来源 | {'合成演示数据' if project.get('demo') else project.get('source', '导入数据')} |")
    lines.append(f"| 平台 | {stack.get('platform', '—')} |")
    lines.append(f"| 轨道 | {stack.get('orbit_direction', '—')} / {stack.get('relative_orbit', '—')} |")
    lines.append(f"| 影像数量 | {stack.get('n_dates', 0)} 景 |")
    if dates:
        lines.append(f"| 时间范围 | {dates[0]} — {dates[-1]} |")
    lines.append(f"| 影像尺寸 | {stack.get('rows', '—')} × {stack.get('cols', '—')} px |")
    lines.append(f"| 波长 / 入射角 | {_fmt(stack.get('wavelength_m'), 5, ' m')} / {_fmt(stack.get('incidence_deg'), 1, '°')} |")
    lines.append("")
    lines.append("## 2. 处理链与参数")
    lines.append("")
    lines.append("| 阶段 | 状态 / 关键结果 |")
    lines.append("| --- | --- |")
    lines.append(f"| 配准 | {coreg.get('model', '—')} · 相干性 {_fmt(coreg.get('coherence_before'), 2)} → {_fmt(coreg.get('coherence_after'), 2)} |")
    lines.append(f"| 网络 | {network.get('stats', {}).get('n_pairs', '—')} 对 · 冗余度 {_fmt(network.get('stats', {}).get('redundancy'), 2)} |")
    lines.append(f"| 干涉 | {ifg.get('n_pairs', '—')} 对 · 平均相干性 {_fmt(ifg.get('mean_coherence'), 2)} |")
    lines.append(f"| 大气 | {atm.get('method', '—')} · 相位离散度降低 {_fmt(atm.get('phase_std_reduction_pct'), 1, '%')} |")
    lines.append(f"| 反演 | {str(inv.get('technique', '—')).upper()} · 有效像元 {inv.get('valid_pixels', '—')} |")
    geo = inv.get("geocoding") if isinstance(inv.get("geocoding"), dict) else {}
    geo_status = str(geo.get("status", "")).lower()
    if geo_status in {"done", "partial"}:
        geo_detail = "已输出地图坐标 GeoTIFF"
        if geo_status == "partial":
            geo_detail = "部分覆盖 · 已输出地图坐标 GeoTIFF"
        if geo.get("coverage") is not None:
            try:
                geo_detail += f" · 覆盖 {float(geo['coverage']):.1%}"
            except (TypeError, ValueError):
                pass
        if geo.get("files"):
            geo_detail += f" · {len(geo['files'])} 个文件"
    else:
        reasons = geo.get("reasons") or ["尚未运行地理编码"]
        geo_detail = f"不可用 · {reasons[0]}"
    lines.append(f"| 地理编码 | {geo_detail} |")
    lines.append("")
    params = {**public_params(project.get("params", {})), **public_params(inv.get("params", {}))}
    lines.append("关键参数：")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps({k: params.get(k) for k in sorted(params)}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 3. 主要成果")
    lines.append("")
    lines.append(f"- 空间参考像元：row {inv.get('reference_pixel', ['—', '—'])[0]} / col {inv.get('reference_pixel', ['—', '—'])[1]}")
    lines.append("")
    if str(inv.get("technique", "")).lower() == "dinsar":
        lines.append(f"- 干涉对：{inv.get('pair', {}).get('ref_date', '—')} → {inv.get('pair', {}).get('sec_date', '—')}")
        lines.append(f"- 视线向位移中位数：{_fmt(inv.get('displacement_median_mm'), 2, ' mm')}")
        lines.append(f"- 5%–95% 分位：{_fmt(inv.get('displacement_p05_mm'), 2, ' / ')}{_fmt(inv.get('displacement_p95_mm'), 2, ' mm')}")
        lines.append(f"- 相干性中位数：{_fmt(inv.get('coherence_median'), 2)}")
    else:
        lines.append(f"- 视线向形变速率中位数：{_fmt(inv.get('velocity_median_mm_yr'), 2, ' mm/yr')}")
        lines.append(f"- 速率 5%–95% 分位：{_fmt(inv.get('velocity_p05_mm_yr'), 2, ' / ')}{_fmt(inv.get('velocity_p95_mm_yr'), 2, ' mm/yr')}")
        lines.append(f"- 时序相干性中位数：{_fmt(inv.get('temporal_coherence_median'), 2)}")
        lines.append(f"- 反演残差中位数：{_fmt(inv.get('residual_median_mm'), 2, ' mm')}")
        lines.append(f"- 有效像元比例：{_fmt(100 * inv.get('valid_fraction', 0), 1, '%')}")
    lines.append("")
    for title, rel in (
        ("形变速率 / 位移", "figures/velocity.png" if str(inv.get("technique", "")).lower() != "dinsar" else "figures/dinsar_displacement.png"),
        ("时序相干性", "figures/temporal_coherence.png"),
        ("干涉网络", "figures/network.png"),
        ("大气诊断", "figures/atmosphere.png"),
    ):
        if (root / "products" / rel).exists():
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}](../{rel})")
            lines.append("")
    if val:
        lines.append("## 4. 合成真值校验")
        lines.append("")
        lines.append(val.get("note", ""))
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        for key, label, unit, digits in (
            ("displacement_rmse_mm", "时序位移 RMSE", " mm", 2),
            ("displacement_bias_mm", "时序位移偏差", " mm", 2),
            ("velocity_rmse_mm_yr", "速率 RMSE", " mm/yr", 2),
            ("velocity_correlation", "速率空间相关", "", 3),
            ("dem_correlation", "DEM 误差空间相关", "", 2),
        ):
            if key in val:
                lines.append(f"| {label} | {_fmt(val.get(key), digits, unit)} |")
        if val.get("atmosphere"):
            a = val["atmosphere"]
            lines.append(f"| 大气校正前成对残差 | {_fmt(a.get('pair_residual_before_mm'), 2, ' mm')} |")
            lines.append(f"| 大气校正后成对残差 | {_fmt(a.get('pair_residual_after_mm'), 2, ' mm')} |")
        lines.append(f"| 综合评价 | {val.get('grade', '—')} |")
        lines.append("")
    lines.append("## 5. 使用限制")
    lines.append("")
    lines.append("- 本报告由 InSAR Studio 自动生成；结果质量取决于数据质量、网络设计、解缠与大气校正。")
    lines.append("- 演示项目使用合成数据，真值校验只用于验证软件链路，不能外推为真实场景精度。")
    lines.append("- 在线 GACOS、Copernicus/ASF 下载适配器需要在联网环境配置账号；本地产品路线更适合业务化运行。")
    lines.append("- 地理编码依赖参考轨道和完整几何元数据；缺少时成果只保留雷达坐标，不会伪造经纬度。")
    lines.append("- 速率与位移均为视线向（LOS），未做升降轨分解或 GNSS 基准校正。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("InSAR Studio · 本地处理，数据不出工作区。")
    return "\n".join(lines)


def build_docx(root: Path, markdown: str | None = None) -> Path | None:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except Exception:
        return None
    root = Path(root)
    c = _project_context(root)
    project, stack, network, ifg, atm, inv, coreg, val = (
        c["project"],
        c["stack"],
        c["network"],
        c["ifg"],
        c["atm"],
        c["inv"],
        c["coreg"],
        c["validation"],
    )
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10)
    title = doc.add_heading(project.get("name", "InSAR 项目"), level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph(f"处理报告 · 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_heading("1. 项目概况", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    rows = [
        ("技术路线", str(project.get("technique", "sbas")).upper()),
        ("数据来源", "合成演示数据" if project.get("demo") else project.get("source", "导入数据")),
        ("平台 / 轨道", f"{stack.get('platform', '—')} / {stack.get('orbit_direction', '—')}"),
        ("影像数量", str(stack.get("n_dates", "—"))),
        ("时间范围", f"{stack.get('dates', ['—'])[0]} — {stack.get('dates', ['—'])[-1]}" if stack.get("dates") else "—"),
        ("影像尺寸", f"{stack.get('rows', '—')} × {stack.get('cols', '—')} px"),
    ]
    for key, value in rows:
        cells = table.add_row().cells
        cells[0].text = key
        cells[1].text = str(value)
    doc.add_heading("2. 处理结果", level=1)
    if str(inv.get("technique", "")).lower() == "dinsar":
        doc.add_paragraph(f"视线向位移中位数 {_fmt(inv.get('displacement_median_mm'), 2, ' mm')}，相干性中位数 {_fmt(inv.get('coherence_median'), 2)}。")
    else:
        doc.add_paragraph(
            f"视线向形变速率中位数 {_fmt(inv.get('velocity_median_mm_yr'), 2, ' mm/yr')}；"
            f"时序相干性中位数 {_fmt(inv.get('temporal_coherence_median'), 2)}；"
            f"有效像元 {_fmt(100 * inv.get('valid_fraction', 0), 1, '%')}。"
        )
    for rel in (
        "figures/velocity.png" if str(inv.get("technique", "")).lower() != "dinsar" else "figures/dinsar_displacement.png",
        "figures/temporal_coherence.png",
        "figures/network.png",
        "figures/atmosphere.png",
    ):
        path = root / "products" / rel
        if path.exists():
            try:
                doc.add_picture(str(path), width=Inches(6.2))
            except Exception:
                pass
    doc.add_heading("3. 参数", level=1)
    doc.add_paragraph(
        f"配准 {coreg.get('model', '—')}；网络 {network.get('stats', {}).get('n_pairs', '—')} 对；"
        f"干涉 {ifg.get('n_pairs', '—')} 对；大气 {atm.get('method', '—')}；反演 {str(inv.get('technique', '—')).upper()}。"
    )
    if val:
        doc.add_heading("4. 合成真值校验", level=1)
        doc.add_paragraph(val.get("note", ""))
        doc.add_paragraph(
            f"时序位移 RMSE {_fmt(val.get('displacement_rmse_mm'), 2, ' mm')}；"
            f"速率 RMSE {_fmt(val.get('velocity_rmse_mm_yr'), 2, ' mm/yr')}；"
            f"综合评价 {val.get('grade', '—')}。"
        )
    doc.add_heading("5. 使用限制", level=1)
    doc.add_paragraph("结果均为视线向，未经 GNSS 基准校正；真实数据需自行核验解缠、大气与轨道误差。")
    out = root / "products" / "report" / "report.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def series_csv(root: Path, row: int, col: int) -> Path:
    """Write an inspectable CSV for one pixel and return its path."""
    root = Path(root)
    inv = read_json(root / "products" / "inversion.json", {}) or {}
    if not inv:
        raise ValueError("尚未生成反演产品")
    arrays = root / "products" / "arrays"
    technique = str(inv.get("technique", "sbas")).lower()
    if technique == "dinsar":
        disp = np.load(arrays / "disp_pair_mm.npy", allow_pickle=False)
        valid = np.load(arrays / "valid.npy", allow_pickle=False).astype(bool)
        if not (0 <= row < disp.shape[0] and 0 <= col < disp.shape[1]):
            raise ValueError("像元超出范围")
        tcoh = np.load(arrays / "temporal_coherence.npy", allow_pickle=False)
        pair = inv.get("pair", {})
        dates = [pair.get("ref_date"), pair.get("sec_date")]
        values = [0.0, float(disp[row, col])]
        velocity = ""
        tvals = ["", None if not np.isfinite(tcoh[row, col]) else float(tcoh[row, col])]
        valid_values = ["", bool(valid[row, col])]
        residual = ["", ""]
    else:
        disp = np.load(arrays / "disp_ts_mm.npy", allow_pickle=False)
        valid = np.load(arrays / "valid.npy", allow_pickle=False).astype(bool)
        if not (0 <= row < disp.shape[1] and 0 <= col < disp.shape[2]):
            raise ValueError("像元超出范围")
        meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
        dates = meta.get("dates", [])
        values = disp[:, row, col].tolist()
        vel_arr = np.load(arrays / "velocity_mm_yr.npy", allow_pickle=False)
        velocity = None if not np.isfinite(vel_arr[row, col]) else float(vel_arr[row, col])
        tcoh = np.load(arrays / "temporal_coherence.npy", allow_pickle=False)[row, col]
        resid = np.load(arrays / "residual_mm.npy", allow_pickle=False)[row, col]
        tvals = [None if not np.isfinite(tcoh) else float(tcoh)] * len(values)
        valid_values = [bool(valid[row, col])] * len(values)
        residual = [None if not np.isfinite(resid) else float(resid)] * len(values)

    out_dir = root / "products" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"series_r{int(row)}_c{int(col)}.csv"
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "disp_mm", "velocity_mm_yr", "temporal_coherence", "residual_mm", "valid"])
        for idx, date_value in enumerate(dates):
            writer.writerow(
                [
                    date_value,
                    values[idx] if idx < len(values) else "",
                    velocity,
                    tvals[idx] if idx < len(tvals) else "",
                    residual[idx] if idx < len(residual) else "",
                    valid_values[idx] if idx < len(valid_values) else "",
                ]
            )
    os.replace(tmp, target)
    return target


def points_csv(root: Path) -> Path:
    """Export all valid pixels as a compact, GIS-friendly CSV."""
    root = Path(root)
    inv = read_json(root / "products" / "inversion.json", {}) or {}
    if not inv:
        raise ValueError("尚未生成反演产品")
    arrays = root / "products" / "arrays"
    technique = str(inv.get("technique", "sbas")).lower()
    valid = np.load(arrays / "valid.npy", allow_pickle=False).astype(bool)
    rows, cols = np.nonzero(valid)
    grid = inv.get("grid", {})
    corner_lon = float(grid.get("corner_lon", 0.0))
    corner_lat = float(grid.get("corner_lat", 0.0))
    lon_step = float(grid.get("lon_step", 1.0))
    lat_step = float(grid.get("lat_step", -1.0))
    lon = corner_lon + (cols.astype(np.float64) + 0.5) * lon_step
    lat = corner_lat + (rows.astype(np.float64) + 0.5) * lat_step
    out_dir = root / "products" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "points.csv"
    tmp = target.with_name(target.name + ".tmp")
    if rows.size == 0:
        with tmp.open("w", encoding="utf-8-sig") as fh:
            fh.write(
                "row,col,lon,lat,velocity_mm_yr,displacement_mm,temporal_coherence,residual_mm,dem_error_m\n"
            )
        os.replace(tmp, target)
        return target

    if technique == "dinsar":
        disp = np.load(arrays / "disp_pair_mm.npy", allow_pickle=False)
        velocity = np.full(disp.shape, np.nan, dtype=np.float32)
        final_disp = disp
    else:
        disp_ts = np.load(arrays / "disp_ts_mm.npy", mmap_mode="r", allow_pickle=False)
        velocity = np.load(arrays / "velocity_mm_yr.npy", allow_pickle=False)
        final_disp = np.asarray(disp_ts[-1], dtype=np.float32)
    tcoh = np.load(arrays / "temporal_coherence.npy", allow_pickle=False)
    residual = np.load(arrays / "residual_mm.npy", allow_pickle=False)
    dem = np.load(arrays / "dem_error_m.npy", allow_pickle=False)

    data = np.column_stack(
        [
            rows.astype(np.float64),
            cols.astype(np.float64),
            lon,
            lat,
            np.asarray(velocity[rows, cols], dtype=np.float64),
            np.asarray(final_disp[rows, cols], dtype=np.float64),
            np.asarray(tcoh[rows, cols], dtype=np.float64),
            np.asarray(residual[rows, cols], dtype=np.float64),
            np.asarray(dem[rows, cols], dtype=np.float64),
        ]
    )
    with tmp.open("w", encoding="utf-8-sig") as fh:
        fh.write(
            "row,col,lon,lat,velocity_mm_yr,displacement_mm,temporal_coherence,residual_mm,dem_error_m\n"
        )
        np.savetxt(
            fh,
            data,
            delimiter=",",
            fmt=["%d", "%d", "%.7f", "%.7f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f"],
        )
    os.replace(tmp, target)
    return target


def generate_report(root: Path) -> dict[str, Any]:
    root = Path(root)
    md = build_markdown(root)
    out = root / "products" / "report"
    out.mkdir(parents=True, exist_ok=True)
    md_path = _atomic_text(out / "report.md", md)
    docx_path = build_docx(root, md)
    return {
        "markdown": str(md_path.relative_to(root)).replace("\\", "/"),
        "docx": str(docx_path.relative_to(root)).replace("\\", "/") if docx_path else None,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def export_zip(root: Path, *, mode: str = "full") -> Path:
    root = Path(root)
    if mode not in EXPORT_MODES:
        raise ValueError(f"未知导出模式: {mode}")
    out_dir = root / "products" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{root.name}_insar" + ("_results" if mode == "results" else "")
    target = out_dir / f"{stem}.zip"
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(root / "project.json", "project.json")
        for item in _iter_export_files(root, mode):
            zf.write(item, str(item.relative_to(root)))
    try:
        os.replace(tmp, target)
    except OSError:
        # If a viewer holds the previous ZIP open on Windows, keep a unique
        # copy instead of failing the download.
        target = out_dir / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        os.replace(tmp, target)
    return target
