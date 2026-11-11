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
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_AZIMUTH_SPACING, DEFAULT_INCIDENCE, DEFAULT_RANGE_SPACING, S1_WAVELENGTH
from .core.geotiff import probe_geotiff, read_geotiff
from .core.io import probe_envi, read_envi
from .core.orbit import OrbitError, collect_safe_metadata, estimate_bperp
from .core.utils import atomic_write_json, read_json, resample_to_shape
from .params import require_min_dates


#: Bounded pixel budget for the lightweight import preflight.  A 4M-pixel
#: complex64 sample is 32 MiB, small enough to keep preflight cheap while
#: still catching all-zero stacks and grossly non-finite input.
PREFLIGHT_SAMPLE_ELEMENTS = 4_000_000


@dataclass
class _Source:
    """A source stack, either fully loaded or probed from headers/samples."""

    shape: tuple[int, int, int]
    dtype: np.dtype
    meta: dict[str, Any]
    dates: list[str]
    geo: dict[str, Any]
    geo_source: str
    dem: np.ndarray | None
    mask: np.ndarray | None
    warnings: list[str] = field(default_factory=list)
    slc: np.ndarray | None = None
    validation: str = "full"
    sample_elements: int = 0
    total_elements: int = 0
    finite_fraction: float | None = None
    max_amplitude: float | None = None

    @property
    def sampled(self) -> bool:
        return self.validation == "sampled"


def _npy_header(fp: Any) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError(f"不支持的 .npy 格式版本: {version}")
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("不支持 object dtype 的 .npy 数据")
    return tuple(int(v) for v in shape), dtype


def _bounded_sample(array: np.ndarray, budget: int) -> np.ndarray:
    """Take a deterministic sample of at most ~``budget`` elements.

    The sample is assembled from evenly spaced *contiguous* blocks so a large
    file on disk is not paged in end-to-end by a fine stride.  Small arrays are
    returned whole.  This is a quality estimate for the preflight only; the
    real import still validates every element.
    """
    total = int(array.size)
    budget = max(1, int(budget))
    if total <= budget or total == 0:
        return np.asarray(array)
    if not array.flags.c_contiguous:
        # Rare fallback: strided slice for non-contiguous inputs.
        factor = (total / budget) ** (1.0 / max(1, array.ndim))
        step = max(1, int(math.ceil(factor)))
        sample = np.asarray(array[tuple(slice(None, None, step) for _ in array.shape)])
        return sample
    flat = array.reshape(-1)
    blocks = min(8, budget)
    block = max(1, budget // blocks)
    starts = np.linspace(0, max(0, total - block), num=blocks, dtype=np.int64)
    return np.concatenate([np.asarray(flat[int(p): int(p) + block]) for p in starts])


def _sample_stats(array: np.ndarray) -> tuple[float, float, int, int]:
    """Return ``(finite_fraction, max_abs, sampled_pixels, total_pixels)``."""
    total = int(array.size)
    if total == 0:
        return 0.0, 0.0, 0, 0
    sample = _bounded_sample(array, PREFLIGHT_SAMPLE_ELEMENTS)
    finite = np.isfinite(sample)
    finite_count = int(finite.sum())
    finite_fraction = finite_count / sample.size if sample.size else 0.0
    max_amplitude = float(np.abs(sample[finite]).max()) if finite_count else 0.0
    return finite_fraction, max_amplitude, int(sample.size), total


def _probe_npy(path: Path) -> tuple[tuple[int, ...], np.dtype, float, float, int, int]:
    """Read a ``.npy`` header and a bounded sample without loading the stack."""
    with path.open("rb") as fp:
        shape, dtype = _npy_header(fp)
    mmap = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        finite_fraction, max_amplitude, sampled, total = _sample_stats(mmap)
    finally:
        del mmap
    return shape, dtype, finite_fraction, max_amplitude, sampled, total


def _native_paths(source: Path) -> tuple[dict[str, Any], Path, Path | None, Path | None]:
    """Locate the metadata, SLC, DEM and mask files of a native stack."""
    meta: dict[str, Any] = {}
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
    dem_path: Path | None = None
    for key in ("dem_path", "dem"):
        rel = meta.get(key)
        if rel and (source / rel).exists():
            dem_path = source / rel
            break
    if dem_path is None:
        for name in ("dem.npy", "dem.tif", "dem.tiff"):
            if (source / name).exists():
                dem_path = source / name
                break
    mask_path: Path | None = None
    for name in ("mask.npy", "mask.tif", "mask.tiff"):
        if (source / name).exists():
            mask_path = source / name
            break
    return meta, slc_path, dem_path, mask_path


def _read_native(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    meta, slc_path, dem_path, mask_path = _native_paths(source)
    slc = np.load(slc_path, allow_pickle=False)
    dem = _read_raster(dem_path) if dem_path is not None else None
    mask = np.asarray(_read_raster(mask_path), dtype=bool) if mask_path is not None else None
    return slc, dem, mask, meta


def _safe_tiffs(source: Path) -> list[Path]:
    measurement = source / "measurement"
    if measurement.exists():
        # allow a directory of .SAFE folders
        tiffs = sorted(measurement.glob("*.tiff"))
    else:
        safes = sorted([p for p in source.iterdir() if p.is_dir() and p.suffix.lower() == ".safe"])
        if safes:
            tiffs = []
            for safe in safes:
                tiffs.extend(sorted((safe / "measurement").glob("*.tiff")))
        else:
            raise FileNotFoundError("未找到 SAFE measurement 目录")
    if not tiffs:
        raise FileNotFoundError("SAFE 中没有 measurement TIFF")
    return tiffs


def _safe_meta(source: Path, tiffs: list[Path]) -> dict[str, Any]:
    annotation = collect_safe_metadata(tiffs)
    starts = [
        value.isoformat() if value is not None else ""
        for value in annotation["start_times"]
    ]

    def _first(key: str) -> Any:
        for value in annotation.get(key, []):
            if value not in (None, "", 0):
                return value
        return None

    return {
        "source": "sentinel-1-safe",
        "note": "Best-effort SAFE import; verify orbit/baseline metadata.",
        "start_times": starts,
        "incidence_deg": _first("incidence_deg"),
        "range_spacing_m": _first("range_spacing_m"),
        "azimuth_spacing_m": _first("azimuth_spacing_m"),
        "azimuth_time_interval": _first("azimuth_time_interval"),
        "slant_range_time_s": _first("slant_range_time_s"),
        "range_spacing_source": "annotation",
        "azimuth_spacing_source": "annotation",
        "incidence_source": "annotation",
        "slant_range_source": "annotation" if _first("slant_range_time_s") else "default",
        "orbit_number": _first("orbit_number"),
        "relative_orbit": _first("relative_orbit"),
        "orbit_direction": (_first("orbit_direction") or "").upper() or None,
        "orbit_dir": str(source),
    }


def _truthy(value: Any) -> bool:
    """Interpret common metadata spellings for a boolean flag."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "是", "已配准"}


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


def _npz_meta(data: Any) -> tuple[dict[str, Any], list[str], str]:
    """Read NPZ metadata without touching the SLC member."""
    keys = set(data.files)
    slc_key = "slc" if "slc" in keys else ("stack" if "stack" in keys else "")
    if not slc_key:
        raise ValueError("NPZ 中缺少 slc 或 stack 数组")
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
    return meta, dates, slc_key


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        meta, dates, slc_key = _npz_meta(data)
        keys = set(data.files)
        slc = data[slc_key]
        dem = data["dem"] if "dem" in keys else None
        mask = data["mask"] if "mask" in keys else None
    return slc, dem, mask, meta, dates


def _probe_npz(path: Path) -> tuple[tuple[int, ...], np.dtype, dict[str, Any], list[str], np.ndarray | None, np.ndarray | None]:
    """Read the NPZ index, metadata and the SLC header without loading the stack."""
    with np.load(path, allow_pickle=False) as data:
        meta, dates, slc_key = _npz_meta(data)
        keys = set(data.files)
        dem = data["dem"] if "dem" in keys else None
        mask = data["mask"] if "mask" in keys else None
        with data.zip.open(f"{slc_key}.npy") as fp:
            shape, dtype = _npy_header(fp)
    return shape, dtype, meta, dates, dem, mask


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


def _write_stack_atomic(
    root: Path,
    slc: np.ndarray,
    dem: np.ndarray,
    mask: np.ndarray,
    meta: dict[str, Any],
    *,
    orbit_payload: dict[str, Any] | None = None,
) -> Path:
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
        if orbit_payload:
            atomic_write_json(tmp / "orbits.json", orbit_payload)
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


def _write_precoregistered(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Expose an imported, already-coregistered SLC stack to the engine.

    The source stack is hard-linked when possible so a one-gigabyte import does
    not silently become two gigabytes.  A normal copy is used on filesystems
    that do not support hard links.
    """
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    source = root / "data" / "stack" / "slc.npy"
    target = processed / "slc_coreg.npy"
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    ref_index = int(meta.get("ref_index", 0))
    dates = list(meta.get("dates", []))
    result = {
        "ref_index": ref_index,
        "ref_date": dates[ref_index] if 0 <= ref_index < len(dates) else None,
        "model": "pre-coregistered",
        "skipped": True,
        "skip_reason": "导入时已标记为已配准，未执行二次配准",
        "n_dates": int(meta.get("n_dates", len(dates))),
        "transforms": [
            {"index": index, "date": date_value, "model": "pre-coregistered"}
            for index, date_value in enumerate(dates)
        ],
        "shifts": [[0.0, 0.0] for _ in dates],
        "coherence_before": None,
        "coherence_after": None,
        "coherence_gain": 0.0,
        "stack_path": "data/processed/slc_coreg.npy",
        "params": {},
    }
    atomic_write_json(processed / "coreg.json", result)
    return result


def _read_safe(source: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Best-effort single-swath SAFE reader (experimental)."""
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("读取 SAFE 需要 tifffile") from exc
    tiffs = _safe_tiffs(source)
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
    meta = _safe_meta(source, tiffs)
    return np.stack(arrays), _fill_dates(dates, len(arrays)), meta


def _detect_directory_source(source: Path) -> str:
    if (source / "measurement").exists() or any(
        p.suffix.lower() == ".safe" for p in source.iterdir() if p.is_dir()
    ):
        return "safe"
    if (source / "stack.json").exists() or (source / "meta.json").exists() or (source / "slc.npy").exists():
        return "native"
    if sorted(source.glob("*.tif")) or sorted(source.glob("*.tiff")):
        return "geotiff"
    raise ValueError("无法识别数据目录，请提供 stack.json、slc.npy、NPZ、GeoTIFF 或 SAFE")


def _normalise_source(loaded: _Source) -> _Source:
    shape = tuple(int(v) for v in loaded.shape)
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
        if loaded.slc is not None:
            loaded.slc = np.asarray(loaded.slc)[None, ...]
    if len(shape) != 3:
        raise ValueError("SLC 栈必须是 (日期, 行, 列) 三维数组")
    loaded.shape = (shape[0], shape[1], shape[2])
    loaded.total_elements = int(np.prod(loaded.shape))
    return loaded


def _probe_source(source: Path) -> _Source:
    """Inspect a source stack from headers plus a bounded data sample.

    The probe is deliberately cheap: it never loads the whole SLC stack.  It
    validates structure and metadata, estimates data quality from an evenly
    strided sample, and lets the caller show a confirmation dialog.  The
    authoritative full validation still runs inside the real import.
    """
    warnings: list[str] = []
    dates: list[str] = []
    meta: dict[str, Any] = {}
    geo: dict[str, Any] = {}
    geo_source = "none"
    dem: np.ndarray | None = None
    mask: np.ndarray | None = None
    finite_fraction: float | None = None
    max_amplitude: float | None = None
    sample_elements = 0
    total_elements = 0
    validation = "header"

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix == ".npy":
            shape, dtype, finite_fraction, max_amplitude, sample_elements, total_elements = _probe_npy(source)
            validation = "full" if total_elements and sample_elements >= total_elements else "sampled"
            dates = _dates_from_names([source.stem])
        elif suffix == ".npz":
            shape, dtype, meta, dates, dem, mask = _probe_npz(source)
        elif suffix in (".tif", ".tiff"):
            shape, dtype, geo = probe_geotiff(source)
            if "corner_lon" in geo and "corner_lat" in geo:
                geo_source = "geotiff"
        elif suffix in (".hdr", ".dat", ".img"):
            hdr = source if suffix == ".hdr" else source.with_suffix(".hdr")
            shape, dtype, _, _ = probe_envi(hdr)
        else:
            raise ValueError("不支持的数据文件格式")
    else:
        kind = _detect_directory_source(source)
        if kind == "safe":
            import tifffile

            tiffs = _safe_tiffs(source)
            scenes: list[tuple[int, int]] = []
            for path in tiffs:
                with tifffile.TiffFile(path) as tif:
                    page_shape = tuple(int(v) for v in tif.pages[0].shape)
                if len(page_shape) == 3 and page_shape[0] == 2:
                    scene = (page_shape[1], page_shape[2])
                elif len(page_shape) == 3 and page_shape[-1] == 2:
                    scene = (page_shape[0], page_shape[1])
                else:
                    scene = page_shape
                if len(scene) != 2:
                    raise ValueError("SAFE measurement TIFF 必须是二维影像")
                scenes.append((scene[0], scene[1]))
            if len(set(scenes)) > 1:
                warnings.append("输入影像尺寸不一致，已按最小公共范围裁剪。")
            shape = (len(tiffs), min(s[0] for s in scenes), min(s[1] for s in scenes))
            dtype = np.dtype(np.complex64)
            meta = _safe_meta(source, tiffs)
            dates = _dates_from_names([p.name for p in tiffs])
            warnings.append("SAFE 导入为实验性功能，请核对轨道与基线元数据。")
        elif kind == "native":
            meta, slc_path, dem_path, mask_path = _native_paths(source)
            shape, dtype, finite_fraction, max_amplitude, sample_elements, total_elements = _probe_npy(slc_path)
            validation = "full" if total_elements and sample_elements >= total_elements else "sampled"
            dates = list(meta.get("dates", []))
            if dem_path is not None:
                dem = _read_raster(dem_path)
            if mask_path is not None:
                mask = np.asarray(_read_raster(mask_path), dtype=bool)
        else:
            tifs = sorted(source.glob("*.tif")) + sorted(source.glob("*.tiff"))
            scenes = []
            first_geo: dict[str, Any] = {}
            for index, path in enumerate(tifs):
                scene_shape, _, scene_geo = probe_geotiff(path)
                if len(scene_shape) != 2:
                    raise ValueError("GeoTIFF 目录中的每个文件必须是单景二维影像")
                scenes.append((scene_shape[0], scene_shape[1]))
                if index == 0:
                    first_geo = scene_geo
            if len(set(scenes)) > 1:
                warnings.append("输入影像尺寸不一致，已按最小公共范围裁剪。")
            shape = (len(tifs), min(s[0] for s in scenes), min(s[1] for s in scenes))
            dtype = np.dtype(np.complex64)
            dates = _dates_from_names([p.stem for p in tifs])
            geo = {
                k: v
                for k, v in first_geo.items()
                if k in ("corner_lon", "corner_lat", "lon_step", "lat_step", "epsg")
            }
            if "corner_lon" in geo and "corner_lat" in geo:
                geo_source = "geotiff"

    loaded = _Source(
        shape=tuple(int(v) for v in shape),
        dtype=np.dtype(dtype),
        meta=meta,
        dates=dates,
        geo=geo,
        geo_source=geo_source,
        dem=dem,
        mask=mask,
        warnings=warnings,
        validation=validation,
        sample_elements=sample_elements,
        total_elements=total_elements,
        finite_fraction=finite_fraction,
        max_amplitude=max_amplitude,
    )
    return _normalise_source(loaded)


def _read_source(source: Path) -> _Source:
    """Load a source stack completely for the real import."""
    warnings: list[str] = []
    dates: list[str] = []
    meta: dict[str, Any] = {}
    geo: dict[str, Any] = {}
    geo_source = "none"
    dem: np.ndarray | None = None
    mask: np.ndarray | None = None
    if source.is_file():
        suffix = source.suffix.lower()
        if suffix == ".npy":
            slc = np.load(source, allow_pickle=False)
            dates = _dates_from_names([source.stem])
        elif suffix == ".npz":
            slc, dem, mask, meta, dates = _read_npz(source)
        elif suffix in (".tif", ".tiff"):
            slc, geo = read_geotiff(source)
            if "corner_lon" in geo and "corner_lat" in geo:
                geo_source = "geotiff"
        elif suffix in (".hdr", ".dat", ".img"):
            slc, _ = read_envi(source if suffix == ".hdr" else source.with_suffix(".hdr"))
        else:
            raise ValueError("不支持的数据文件格式")
    else:
        kind = _detect_directory_source(source)
        if kind == "safe":
            slc, dates, meta = _read_safe(source)
            warnings.append("SAFE 导入为实验性功能，请核对轨道与基线元数据。")
        elif kind == "native":
            slc, dem, mask, meta = _read_native(source)
            dates = list(meta.get("dates", []))
        else:
            tifs = sorted(source.glob("*.tif")) + sorted(source.glob("*.tiff"))
            arrays = [_read_raster(p) for p in tifs]
            shapes = {a.shape for a in arrays}
            if len(shapes) > 1:
                h = min(a.shape[0] for a in arrays)
                w = min(a.shape[1] for a in arrays)
                arrays = [a[:h, :w] for a in arrays]
                warnings.append("输入影像尺寸不一致，已按最小公共范围裁剪。")
            slc = np.stack([a.astype(np.complex64) for a in arrays])
            dates = _dates_from_names([p.stem for p in tifs])
            # Pick up georeferencing from the first scene without overriding
            # an explicit meta.json.
            _, first_geo = read_geotiff(tifs[0])
            geo = {k: v for k, v in first_geo.items() if k in ("corner_lon", "corner_lat", "lon_step", "lat_step", "epsg")}
            if "corner_lon" in geo and "corner_lat" in geo:
                geo_source = "geotiff"

    slc = np.asarray(slc)
    if slc.ndim == 2:
        slc = slc[None, ...]
    loaded = _Source(
        shape=tuple(int(v) for v in slc.shape),
        dtype=slc.dtype,
        meta=meta,
        dates=dates,
        geo=geo,
        geo_source=geo_source,
        dem=dem,
        mask=mask,
        warnings=warnings,
        slc=slc,
        validation="full",
    )
    return _normalise_source(loaded)


def import_stack(
    project_root: Path,
    source: Path | str,
    *,
    dem_path: Path | str | None = None,
    meta_overrides: dict[str, Any] | None = None,
    technique: str | None = None,
    dry_run: bool = False,
    deep: bool = False,
) -> dict[str, Any]:
    root = Path(project_root)
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"数据源不存在: {source}")
    if meta_overrides is not None and not isinstance(meta_overrides, dict):
        raise ValueError("meta_overrides 必须是对象")
    if dry_run and not deep:
        loaded = _probe_source(source)
    else:
        loaded = _read_source(source)
    warnings = list(loaded.warnings)
    dates = list(loaded.dates)
    meta = loaded.meta
    geo = loaded.geo
    geo_source = loaded.geo_source
    slc = loaded.slc
    start_times: list[Any] = list(meta.get("start_times") or [])
    n, rows, cols = loaded.shape
    if n < 2:
        raise ValueError("至少需要 2 景影像才能构建干涉网络")
    if technique:
        # Reject too-short stacks before writing anything: the caller can then
        # clean up the half-created project without leaving stale data behind.
        require_min_dates(technique, n, action="导入该技术路线的数据")
    if rows < 4 or cols < 4:
        raise ValueError("影像尺寸过小，无法处理")
    if loaded.dtype.kind != "c":
        warnings.append("输入不是复数 SLC，已按零相位实数数据转换，结果只能作为演示。")
        if slc is not None:
            slc = slc.astype(np.float32).astype(np.complex64)
    finite_fraction = loaded.finite_fraction
    if finite_fraction is None and slc is not None:
        finite_fraction = float(np.isfinite(slc).mean())
        loaded.sample_elements = loaded.total_elements
    if finite_fraction is not None:
        if finite_fraction < 0.5:
            raise ValueError(f"SLC 中仅 {finite_fraction:.1%} 的值为有限数，请检查输入")
        if finite_fraction < 0.999:
            prefix = "抽样检查：" if loaded.sampled else ""
            warnings.append(f"{prefix}SLC 含 {1 - finite_fraction:.2%} 的非有限值，已在相干性/掩膜阶段剔除。")
    if loaded.max_amplitude is None and slc is not None:
        loaded.max_amplitude = float(np.nanmax(np.abs(slc)))
    if loaded.max_amplitude is not None and loaded.max_amplitude == 0:
        raise ValueError("SLC 全为零，无法处理")

    provided_dates = sum(1 for value in dates if str(value).strip())
    dates = _fill_dates(dates, n)
    dates_inferred = provided_dates < n
    if dates_inferred:
        warnings.append(
            f"仅有 {provided_dates}/{n} 个日期可用，缺失日期已按 12 天间隔自动补齐；时间基线可能失真。"
        )
    try:
        parsed_dates = [date.fromisoformat(value) for value in dates]
    except ValueError as exc:
        raise ValueError("日期格式必须为 YYYY-MM-DD") from exc
    if len(set(dates)) != n:
        raise ValueError("日期存在重复，无法建立唯一时间轴")
    order = sorted(range(n), key=lambda idx: parsed_dates[idx])
    if order != list(range(n)):
        if slc is not None:
            slc = slc[order]
        dates = [dates[idx] for idx in order]
        if len(start_times) == n:
            start_times = [start_times[idx] for idx in order]
        warnings.append("影像日期未按升序排列，已自动重排。")

    if dem_path:
        dem = _read_raster(Path(dem_path))
    else:
        dem = loaded.dem
    if dem is None:
        dem = np.zeros((rows, cols), dtype=np.float32)
        warnings.append("未提供 DEM，地形相位与 DEM 误差估计将不可用。")
    else:
        dem = np.asarray(dem, dtype=np.float32)
        if dem.shape != (rows, cols):
            if slc is not None:
                dem = resample_to_shape(dem, (rows, cols)).astype(np.float32)
            warnings.append("DEM 尺寸与 SLC 不一致，已重采样。")
        if not np.isfinite(dem).all():
            dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
            warnings.append("DEM 含非有限值，已替换为 0。")
        if float(np.nanstd(dem)) < 1e-9:
            warnings.append("DEM 为常数，地形相位将为零。")

    mask = loaded.mask
    if mask is None:
        if slc is not None:
            amp = np.abs(slc[0])
            lo, hi = np.percentile(amp, [3, 99])
            mask = amp > max(lo, hi * 0.08)
            # keep a two-pixel border out of the solution
            mask[:2, :] = mask[-2:, :] = False
            mask[:, :2] = mask[:, -2:] = False
        else:
            mask = np.ones((rows, cols), dtype=bool)
        warnings.append("未提供掩膜，已根据首景振幅自动生成。")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (rows, cols):
        if slc is not None:
            mask = resample_to_shape(mask.astype(np.float32), (rows, cols)) > 0.5
        warnings.append("掩膜尺寸与 SLC 不一致，已重采样。")
    if not mask.any():
        raise ValueError("掩膜内没有有效像元")

    source_meta = {**geo, **meta, **(meta_overrides or {})}
    if (meta_overrides or {}).get("start_times"):
        start_times = list((meta_overrides or {}).get("start_times") or [])
        if len(start_times) == n and order != list(range(n)):
            start_times = [start_times[idx] for idx in order]
    explicit_geo = _truthy(source_meta.get("geo_referenced"))
    raw_lon = source_meta.get("corner_lon")
    raw_lat = source_meta.get("corner_lat")
    try:
        corner_nonzero = (
            raw_lon is not None
            and raw_lat is not None
            and (abs(float(raw_lon)) > 1e-9 or abs(float(raw_lat)) > 1e-9)
        )
    except (TypeError, ValueError):
        corner_nonzero = False
    geo_referenced = bool(explicit_geo or corner_nonzero)
    if geo_referenced:
        if geo_source == "none":
            geo_source = "metadata"
        for key, default in (("lon_step", 0.0004), ("lat_step", -0.0004)):
            if source_meta.get(key) is None:
                warnings.append(f"缺少 {key}，已使用默认值 {default}；地理坐标换算可能不准确。")
    else:
        warnings.append(
            "未提供地理参考（corner_lon/corner_lat），成果按像元坐标输出，无法直接叠加到 GIS；"
            "如坐标恰好为 0,0，请在 meta.json 中设置 geo_referenced=true。"
        )
    already_coregistered = any(
        _truthy(source_meta.get(key))
        for key in ("already_coregistered", "coregistered", "pre_coregistered")
    )

    d0 = date.fromisoformat(dates[0])
    sampling = [float((date.fromisoformat(value) - d0).days) for value in dates]
    old_ref = source_meta.get("ref_index", 0)
    try:
        old_ref = int(old_ref)
    except (TypeError, ValueError):
        old_ref = 0
    if not 0 <= old_ref < n:
        old_ref = 0
    new_ref = order.index(old_ref)

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

    has_baselines = bool(np.any(np.asarray(bperp, dtype=float) != 0))
    baseline_source = "metadata" if has_baselines else "none"
    orbit_info: dict[str, Any] | None = None
    orbit_payload: dict[str, Any] | None = None
    if not has_baselines:
        orbit_candidates: list[Path] = []
        configured_orbit = source_meta.get("orbit_dir")
        if configured_orbit:
            orbit_candidates.append(Path(str(configured_orbit)))
        base = source if source.is_dir() else source.parent
        orbit_candidates.extend([base, base.parent])
        seen: set[str] = set()
        for candidate in orbit_candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.exists():
                continue
            if len(start_times) != n:
                warnings.append(
                    "未提供每景成像时间 start_times，无法用本地精密轨道估计垂直基线。"
                )
                break
            try:
                estimated, orbit_info = estimate_bperp(
                    candidate,
                    dates,
                    start_times,
                    ref_index=new_ref,
                    n_rows=rows,
                    azimuth_time_interval=source_meta.get("azimuth_time_interval"),
                    include_series=True,
                )
            except (FileNotFoundError, OrbitError):
                continue
            series = orbit_info.pop("series", None) if orbit_info else None
            if series:
                orbit_payload = {
                    "schema": "insar-studio/orbits/v1",
                    "ref_index": int(new_ref),
                    "dates": list(dates),
                    "orbits": series,
                }
                orbit_info["vectors_path"] = "data/stack/orbits.json"
                orbit_info["n_vectors"] = int(sum(len(item.get("times", [])) for item in series))
            bperp = estimated
            has_baselines = bool(np.any(np.asarray(bperp, dtype=float) != 0))
            baseline_source = "orbit"
            warnings.append(
                f"已由本地轨道文件估计垂直基线（{orbit_info.get('n_orbits', 0)} 个轨道文件，"
                f"{'精密' if 'precise' in orbit_info.get('orbit_kind', []) else '可用'}轨道）。"
            )
            break

    if not has_baselines:
        if source_meta.get("bperp_m") or source_meta.get("bperp"):
            warnings.append("垂直基线全为 0，DEM 误差估计不可用。")
        else:
            warnings.append(
                "未提供垂直基线 bperp_m，且未找到可用的本地精密轨道，已置零；"
                "DEM 误差估计与基线筛选不可用。"
            )

    def _number(key: str, default: float) -> float:
        try:
            value = float(source_meta.get(key, default))
            if not np.isfinite(value):
                raise ValueError
            return value
        except (TypeError, ValueError):
            warnings.append(f"元信息 {key} 无效，已使用默认值 {default}。")
            return float(default)

    raw_slant_time = source_meta.get("slant_range_time_s")
    slant_range_time_s: float | None = None
    if raw_slant_time not in (None, ""):
        try:
            candidate = float(raw_slant_time)
            if np.isfinite(candidate) and candidate > 0:
                slant_range_time_s = candidate
        except (TypeError, ValueError):
            pass
    if slant_range_time_s is not None:
        slant_range_m = 299_792_458.0 * slant_range_time_s / 2.0
        slant_range_source = "annotation"
    else:
        slant_range_m = _number("slant_range_m", 705000.0)
        slant_range_source = "provided" if source_meta.get("slant_range_m") not in (None, "") else "default"

    def _source(key: str) -> str:
        return "provided" if source_meta.get(key) not in (None, "") else "default"

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
        "incidence_source": _source("incidence_deg"),
        "range_spacing_m": _number("range_spacing_m", DEFAULT_RANGE_SPACING),
        "range_spacing_source": _source("range_spacing_m"),
        "azimuth_spacing_m": _number("azimuth_spacing_m", DEFAULT_AZIMUTH_SPACING),
        "azimuth_spacing_source": _source("azimuth_spacing_m"),
        "slant_range_m": slant_range_m,
        "slant_range_time_s": slant_range_time_s,
        "slant_range_source": slant_range_source,
        "bperp_m": bperp,
        "baseline_source": baseline_source,
        "start_times": [
            value if isinstance(value, str) else value.isoformat() if hasattr(value, "isoformat") else str(value)
            for value in start_times
        ],
        "orbit": orbit_info,
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
        "has_baselines": has_baselines,
        "already_coregistered": bool(already_coregistered),
        "coregistration_skipped": bool(already_coregistered),
        "geo_referenced": bool(geo_referenced),
        "geo_source": geo_source,
        "dates_source": "inferred" if dates_inferred else "provided",
        "import_warnings": warnings,
    }
    out_meta["preflight"] = {
        "ok": True,
        "n_dates": n,
        "rows": rows,
        "cols": cols,
        "technique": technique,
        "already_coregistered": bool(already_coregistered),
        "geo_referenced": bool(geo_referenced),
        "dates_inferred": bool(dates_inferred),
        "has_baselines": has_baselines,
        "validation": loaded.validation,
        "sample_elements": int(loaded.sample_elements if loaded.validation != "header" else 0),
        "total_elements": int(loaded.total_elements),
        "sample_fraction": (
            float(loaded.sample_elements / loaded.total_elements)
            if loaded.validation != "header" and loaded.total_elements
            else 0.0
        ),
        "warning_count": len(warnings),
        "warnings": list(warnings),
    }
    if dry_run:
        result = dict(out_meta)
        result["dry_run"] = True
        return result
    if slc is None:
        raise RuntimeError("导入预检必须使用 dry_run=True")
    _write_stack_atomic(root, slc, dem, mask, out_meta, orbit_payload=orbit_payload)
    _clear_derived(root)
    if already_coregistered:
        out_meta["coregistration"] = _write_precoregistered(root, out_meta)
    return out_meta
