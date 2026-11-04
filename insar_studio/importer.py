"""Import local SLC stacks and prepare a project data directory.

Supported inputs
----------------
* a ``.npy`` complex stack with shape ``(dates, rows, cols)``;
* a ``.npz`` bundle with ``slc`` plus optional ``dem`` / ``mask`` / ``dates``;
* a directory containing ``slc.npy`` plus optional ``dem.npy`` / ``mask.npy``
  and ``meta.json`` (the native layout);
* an ENVI ``.hdr`` complex stack;
* a directory of complex GeoTIFF scenes;
* a Sentinel-1 SAFE folder (experimental, single swath, complex int16 TIFF).

Orbit/baseline metadata is taken from the source when present; otherwise
perpendicular baselines are set to zero and DEM-error estimation is disabled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_AZIMUTH_SPACING, DEFAULT_INCIDENCE, DEFAULT_RANGE_SPACING, S1_WAVELENGTH
from .core.geotiff import read_geotiff
from .core.io import read_envi
from .core.utils import atomic_write_json, read_json, resample_to_shape


def _dates_from_names(names: list[str]) -> list[str]:
    out = []
    for name in names:
        m = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", name)
        if m:
            try:
                out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat())
            except ValueError:
                out.append("")
        else:
            out.append("")
    return out


def _fill_dates(dates: list[str], n: int) -> list[str]:
    cleaned: list[str] = []
    for value in dates[:n]:
        try:
            cleaned.append(date.fromisoformat(str(value)[:10]).isoformat())
        except (TypeError, ValueError):
            cleaned.append("")
    dated = [d for d in cleaned if d]
    if len(dated) >= n:
        return dated[:n]
    start = date.fromisoformat(dated[0]) if dated else date(2023, 1, 1)
    out = []
    for i in range(n):
        if i < len(cleaned) and cleaned[i]:
            out.append(cleaned[i])
        else:
            out.append((start + timedelta(days=12 * i)).isoformat())
    return out


def _read_native(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    meta = {}
    for name in ("meta.json", "stack.json"):
        p = source / name
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                meta = loaded if isinstance(loaded, dict) else {}
            except Exception:
                meta = {}
            break
    slc_rel = meta.get("slc_path") or meta.get("slc") or "slc.npy"
    slc_path = source / slc_rel
    if not slc_path.exists():
        candidates = list(source.glob("*.npy"))
        if len(candidates) == 1:
            slc_path = candidates[0]
        else:
            raise FileNotFoundError(f"目录中找不到 SLC 栈: {source}")
    slc = np.load(slc_path, allow_pickle=False)
    dem = None
    for key in ("dem_path", "dem"):
        rel = meta.get(key)
        if rel and (source / rel).exists():
            dem = _read_raster(source / rel)
            break
    if dem is None:
        for name in ("dem.npy", "dem.tif", "dem.tiff"):
            if (source / name).exists():
                dem = _read_raster(source / name)
                break
    mask = None
    for name in ("mask.npy", "mask.tif", "mask.tiff"):
        if (source / name).exists():
            mask = np.asarray(_read_raster(source / name), dtype=bool)
            break
    return slc, dem, mask, meta


def _read_raster(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr, _ = read_geotiff(path)
        return arr
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in (".hdr", ".dat", ".img"):
        hdr = path if path.suffix.lower() == ".hdr" else path.with_suffix(".hdr")
        arr, _ = read_envi(hdr)
        return arr
    raise ValueError(f"不支持的栅格格式: {path.suffix}")


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        slc_key = "slc" if "slc" in keys else ("stack" if "stack" in keys else "")
        if not slc_key:
            raise ValueError("NPZ 中缺少 slc 或 stack 数组")
        slc = data[slc_key]
        dem = data["dem"] if "dem" in keys else None
        mask = data["mask"] if "mask" in keys else None
        meta: dict[str, Any] = {}
        for key in ("meta", "meta_json", "stack"):
            if key in keys and key != slc_key and data[key].dtype.kind in ("U", "S"):
                try:
                    value = data[key].item()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    parsed = json.loads(str(value))
                    meta = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    meta = {}
                break
        dates: list[str] = []
        if "dates" in keys:
            dates = [
                (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v))
                for v in np.asarray(data["dates"]).ravel().tolist()
            ]
        elif meta.get("dates"):
            dates = [str(v) for v in meta["dates"]]
        if "bperp_m" in keys and "bperp_m" not in meta:
            meta["bperp_m"] = np.asarray(data["bperp_m"]).ravel().tolist()
    return slc, dem, mask, meta, dates


def _clear_derived(root: Path) -> None:
    for rel in ("interferograms", "products", "data/processed", "data/atmosphere"):
        target = root / rel
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    network = root / "data" / "network.json"
    if network.exists():
        try:
            network.unlink()
        except OSError:
            pass
    project_file = root / "project.json"
    project = read_json(project_file, {}) or {}
    if project:
        steps = project.get("steps_run", {}) or {}
        project["steps_run"] = {k: v for k, v in steps.items() if k in ("project", "data")}
        atomic_write_json(project_file, project)


def _write_stack_atomic(root: Path, slc: np.ndarray, dem: np.ndarray, mask: np.ndarray, meta: dict[str, Any]) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stack_dir = data_dir / "stack"
    tmp = data_dir / f"stack.tmp.{uuid.uuid4().hex[:8]}"
    backup = data_dir / f"stack.old.{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        np.save(tmp / "slc.npy", slc.astype(np.complex64, copy=False), allow_pickle=False)
        np.save(tmp / "dem.npy", dem.astype(np.float32, copy=False), allow_pickle=False)
        np.save(tmp / "mask.npy", mask.astype(bool, copy=False), allow_pickle=False)
        atomic_write_json(tmp / "meta.json", meta)
        if stack_dir.exists():
            os.replace(stack_dir, backup)
        os.replace(tmp, stack_dir)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if backup.exists() and not stack_dir.exists():
            try:
                os.replace(backup, stack_dir)
            except OSError:
                pass
        raise
    return stack_dir


def _read_safe(source: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Best-effort single-swath SAFE reader (experimental)."""
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("读取 SAFE 需要 tifffile") from exc
    measurement = source / "measurement"
    if not measurement.exists():
        # allow a directory of .SAFE folders
        safes = sorted([p for p in source.iterdir() if p.is_dir() and p.suffix.lower() == ".safe"])
        if safes:
            tiffs = []
            for safe in safes:
                tiffs.extend(sorted((safe / "measurement").glob("*.tiff")))
        else:
            raise FileNotFoundError("未找到 SAFE measurement 目录")
    else:
        tiffs = sorted(measurement.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError("SAFE 中没有 measurement TIFF")
    arrays = []
    dates = _dates_from_names([p.name for p in tiffs])
    shapes = []
    for p in tiffs:
        arr = tifffile.imread(p)
        if arr.ndim == 3 and arr.shape[0] == 2:  # I/Q bands
            arr = arr[0].astype(np.float32) + 1j * arr[1].astype(np.float32)
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            arr = arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)
        if not np.iscomplexobj(arr):
            arr = arr.astype(np.float32).astype(np.complex64)
        arrays.append(arr.astype(np.complex64))
        shapes.append(arr.shape)
    if len(set(shapes)) != 1:
        h = min(s[0] for s in shapes)
        w = min(s[1] for s in shapes)
        arrays = [a[:h, :w] for a in arrays]
    return np.stack(arrays), _fill_dates(dates, len(arrays)), {"source": "sentinel-1-safe", "note": "Best-effort SAFE import; verify orbit/baseline metadata."}


def import_stack(
    project_root: Path,
    source: Path | str,
    *,
    dem_path: Path | str | None = None,
    meta_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"数据源不存在: {source}")
    if meta_overrides is not None and not isinstance(meta_overrides, dict):
        raise ValueError("meta_overrides 必须是对象")
    warnings: list[str] = []
    dates: list[str] = []
    meta: dict[str, Any] = {}
    if source.is_file():
        if source.suffix.lower() == ".npy":
            slc = np.load(source, allow_pickle=False)
            dem = None
            mask = None
            dates = _dates_from_names([source.stem])
        elif source.suffix.lower() == ".npz":
            slc, dem, mask, meta, dates = _read_npz(source)
        elif source.suffix.lower() in (".tif", ".tiff"):
            slc = _read_raster(source)
            dem = None
            mask = None
        elif source.suffix.lower() in (".hdr", ".dat", ".img"):
            slc, _ = read_envi(source if source.suffix.lower() == ".hdr" else source.with_suffix(".hdr"))
            dem = None
            mask = None
        else:
            raise ValueError("不支持的数据文件格式")
    else:
        if (source / "measurement").exists() or any(p.suffix.lower() == ".safe" for p in source.iterdir() if p.is_dir()):
            slc, dates, meta = _read_safe(source)
            dem = None
            mask = None
            warnings.append("SAFE 导入为实验性功能，请核对轨道与基线元数据。")
        elif (source / "stack.json").exists() or (source / "meta.json").exists() or (source / "slc.npy").exists():
            slc, dem, mask, meta = _read_native(source)
            dates = list(meta.get("dates", []))
        else:
            tifs = sorted(source.glob("*.tif")) + sorted(source.glob("*.tiff"))
            if tifs:
                arrays = [_read_raster(p) for p in tifs]
                shapes = {a.shape for a in arrays}
                if len(shapes) > 1:
                    h = min(a.shape[0] for a in arrays)
                    w = min(a.shape[1] for a in arrays)
                    arrays = [a[:h, :w] for a in arrays]
                    warnings.append("输入影像尺寸不一致，已按最小公共范围裁剪。")
                slc = np.stack([a.astype(np.complex64) for a in arrays])
                dates = _dates_from_names([p.stem for p in tifs])
                dem = None
                mask = None
            else:
                raise ValueError("无法识别数据目录，请提供 stack.json、slc.npy、NPZ、GeoTIFF 或 SAFE")
    slc = np.asarray(slc)
    if slc.ndim == 2:
        slc = slc[None, ...]
    if slc.ndim != 3:
        raise ValueError("SLC 栈必须是 (日期, 行, 列) 三维数组")
    n, rows, cols = (int(v) for v in slc.shape)
    if n < 2:
        raise ValueError("至少需要 2 景影像才能构建干涉网络")
    if rows < 4 or cols < 4:
        raise ValueError("影像尺寸过小，无法处理")
    if not np.iscomplexobj(slc):
        warnings.append("输入不是复数 SLC，已按零相位实数数据转换，结果只能作为演示。")
        slc = slc.astype(np.float32).astype(np.complex64)
    finite_fraction = float(np.isfinite(slc).mean())
    if finite_fraction < 0.5:
        raise ValueError(f"SLC 中仅 {finite_fraction:.1%} 的值为有限数，请检查输入")
    if finite_fraction < 0.999:
        warnings.append(f"SLC 含 {1 - finite_fraction:.2%} 的非有限值，已在相干性/掩膜阶段剔除。")
    if np.nanmax(np.abs(slc)) == 0:
        raise ValueError("SLC 全为零，无法处理")

    dates = _fill_dates(dates, n)
    if sum(1 for value in dates if value) < n:
        warnings.append("部分日期缺失，已按 12 天间隔自动补齐。")
    try:
        parsed_dates = [date.fromisoformat(value) for value in dates]
    except ValueError as exc:
        raise ValueError("日期格式必须为 YYYY-MM-DD") from exc
    if len(set(dates)) != n:
        raise ValueError("日期存在重复，无法建立唯一时间轴")
    order = sorted(range(n), key=lambda idx: parsed_dates[idx])
    if order != list(range(n)):
        slc = slc[order]
        dates = [dates[idx] for idx in order]
        warnings.append("影像日期未按升序排列，已自动重排。")

    if dem_path:
        dem = _read_raster(Path(dem_path))
    if dem is None:
        dem = np.zeros((rows, cols), dtype=np.float32)
        warnings.append("未提供 DEM，地形相位与 DEM 误差估计将不可用。")
    else:
        dem = np.asarray(dem, dtype=np.float32)
        if dem.shape != (rows, cols):
            dem = resample_to_shape(dem, (rows, cols)).astype(np.float32)
            warnings.append("DEM 尺寸与 SLC 不一致，已重采样。")
        if not np.isfinite(dem).all():
            dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
            warnings.append("DEM 含非有限值，已替换为 0。")
        if float(np.nanstd(dem)) < 1e-9:
            warnings.append("DEM 为常数，地形相位将为零。")

    if mask is None:
        amp = np.abs(slc[0])
        lo, hi = np.percentile(amp, [3, 99])
        mask = amp > max(lo, hi * 0.08)
        # keep a two-pixel border out of the solution
        mask[:2, :] = mask[-2:, :] = False
        mask[:, :2] = mask[:, -2:] = False
        warnings.append("未提供掩膜，已根据首景振幅自动生成。")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (rows, cols):
        mask = resample_to_shape(mask.astype(np.float32), (rows, cols)) > 0.5
        warnings.append("掩膜尺寸与 SLC 不一致，已重采样。")
    if not mask.any():
        raise ValueError("掩膜内没有有效像元")

    source_meta = {**meta, **(meta_overrides or {})}
    d0 = date.fromisoformat(dates[0])
    sampling = [float((date.fromisoformat(value) - d0).days) for value in dates]
    try:
        bperp = np.asarray(source_meta.get("bperp_m") or [0.0] * n, dtype=float).ravel()
        if bperp.size != n:
            bperp = np.zeros(n, dtype=float)
            warnings.append("垂直基线数量与影像数量不一致，已置零。")
        elif order != list(range(n)):
            bperp = bperp[order]
        bperp = [float(value) if np.isfinite(value) else 0.0 for value in bperp]
    except (TypeError, ValueError):
        bperp = [0.0] * n
        warnings.append("垂直基线格式异常，已置零。")

    def _number(key: str, default: float) -> float:
        try:
            value = float(source_meta.get(key, default))
            if not np.isfinite(value):
                raise ValueError
            return value
        except (TypeError, ValueError):
            warnings.append(f"元信息 {key} 无效，已使用默认值 {default}。")
            return float(default)

    old_ref = source_meta.get("ref_index", 0)
    try:
        old_ref = int(old_ref)
    except (TypeError, ValueError):
        old_ref = 0
    if not 0 <= old_ref < n:
        old_ref = 0
    new_ref = order.index(old_ref)

    out_meta = {
        "schema": "insar-studio/stack/v1",
        "name": str(source_meta.get("name") or f"Imported · {source.name}")[:200],
        "source": str(source_meta.get("source", "import")),
        "demo": False,
        "created_at": date.today().isoformat(),
        "platform": str(source_meta.get("platform", "Sentinel-1")),
        "orbit_direction": str(source_meta.get("orbit_direction", "UNKNOWN")),
        "relative_orbit": source_meta.get("relative_orbit"),
        "rows": rows,
        "cols": cols,
        "n_dates": n,
        "dates": dates,
        "sampling_days": sampling,
        "wavelength_m": _number("wavelength_m", S1_WAVELENGTH),
        "incidence_deg": _number("incidence_deg", DEFAULT_INCIDENCE),
        "range_spacing_m": _number("range_spacing_m", DEFAULT_RANGE_SPACING),
        "azimuth_spacing_m": _number("azimuth_spacing_m", DEFAULT_AZIMUTH_SPACING),
        "slant_range_m": _number("slant_range_m", 705000.0),
        "bperp_m": bperp,
        "slc_path": "data/stack/slc.npy",
        "dem_path": "data/stack/dem.npy",
        "mask_path": "data/stack/mask.npy",
        "corner_lon": _number("corner_lon", 0.0),
        "corner_lat": _number("corner_lat", 0.0),
        "lon_step": _number("lon_step", 0.0004),
        "lat_step": _number("lat_step", -0.0004),
        "crs": str(source_meta.get("crs", "EPSG:4326")),
        "ref_index": int(new_ref),
        "notes": str(source_meta.get("notes", "Imported local stack."))[:2000],
        "has_baselines": bool(np.any(np.asarray(bperp) != 0)),
        "import_warnings": warnings,
    }
    _write_stack_atomic(root, slc, dem, mask, out_meta)
    _clear_derived(root)
    return out_meta
