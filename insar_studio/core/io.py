"""Stack I/O: NumPy, ENVI and GeoTIFF.

The engine keeps the SLC stack as a memory-mapped complex64 array so that
real stacks larger than RAM remain workable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geotiff import read_geotiff
from .utils import atomic_write_json, read_json


class DataError(RuntimeError):
    """Raised when a dataset cannot be read or is inconsistent."""


def safe_join(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise DataError("路径越界：只允许访问项目目录内的文件")
    return target


@dataclass
class Stack:
    root: Path
    meta: dict[str, Any]
    slc: np.ndarray
    dem: np.ndarray
    mask: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.slc.shape[1])

    @property
    def cols(self) -> int:
        return int(self.slc.shape[2])

    @property
    def n_dates(self) -> int:
        return int(self.slc.shape[0])

    @property
    def dates(self) -> list[str]:
        return list(self.meta["dates"])

    def amplitude(self, index: int = 0) -> np.ndarray:
        return np.abs(self.slc[index]).astype(np.float32)

    def phase(self, index: int = 0) -> np.ndarray:
        return np.angle(self.slc[index]).astype(np.float32)

    def close(self) -> None:
        if isinstance(self.slc, np.memmap):
            self.slc._mmap.close()  # type: ignore[attr-defined]


_DATA_TYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def _parse_envi_header(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]
    if not lines or lines[0].lower() != "envi":
        raise DataError(f"{path.name} 不是 ENVI 头文件")
    out: dict[str, str] = {}
    for line in lines[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip().lower()] = value.strip()
    return out


def probe_envi(hdr_path: Path | str) -> tuple[tuple[int, int, int], np.dtype, str, int]:
    """Return ENVI shape/dtype/interleave/byte-order without reading pixels."""
    hdr_path = Path(hdr_path)
    header = _parse_envi_header(hdr_path)
    try:
        samples = int(header["samples"])
        lines = int(header["lines"])
        bands = int(header.get("bands", "1"))
    except (KeyError, ValueError) as exc:
        raise DataError(f"ENVI 头缺少 samples/lines/bands: {hdr_path.name}") from exc
    dtype = _DATA_TYPES.get(int(header.get("data type", "4")))
    if dtype is None:
        raise DataError(f"不支持的 ENVI 数据类型: {header.get('data type')}")
    interleave = header.get("interleave", "bsq").lower()
    if interleave not in ("bsq", "bil", "bip"):
        raise DataError(f"不支持的 ENVI interleave: {interleave}")
    byte_order = int(header.get("byte order", "0"))
    return (bands, lines, samples), np.dtype(dtype), interleave, byte_order


def read_envi(hdr_path: Path | str) -> tuple[np.ndarray, dict[str, str]]:
    hdr_path = Path(hdr_path)
    header = _parse_envi_header(hdr_path)
    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header.get("bands", "1"))
    dtype = _DATA_TYPES.get(int(header.get("data type", "4")))
    if dtype is None:
        raise DataError(f"不支持的 ENVI 数据类型: {header.get('data type')}")
    interleave = header.get("interleave", "bsq").lower()
    byte_order = int(header.get("byte order", "0"))
    data_path = hdr_path.with_suffix("")
    if "data file" in header:
        data_path = hdr_path.parent / header["data file"]
    raw = np.fromfile(data_path, dtype=dtype)
    expected = samples * lines * bands
    if raw.size < expected:
        raise DataError(f"ENVI 数据长度不足: {raw.size} < {expected}")
    raw = raw[:expected]
    if byte_order == 1 and dtype.itemsize > 1:
        raw = raw.byteswap()
    if interleave == "bsq":
        arr = raw.reshape(bands, lines, samples)
    elif interleave == "bil":
        arr = raw.reshape(lines, bands, samples).transpose(1, 0, 2)
    elif interleave == "bip":
        arr = raw.reshape(lines, samples, bands).transpose(2, 0, 1)
    else:
        raise DataError(f"不支持的 ENVI interleave: {interleave}")
    return arr, header


def read_complex_image(path: Path | str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix in (".tif", ".tiff"):
        arr, _ = read_geotiff(path)
    elif suffix in (".hdr", ".dat", ".img", ".raw"):
        hdr = path if suffix == ".hdr" else path.with_suffix(".hdr")
        if not hdr.exists():
            raise DataError(f"找不到 ENVI 头文件: {hdr}")
        arr, _ = read_envi(hdr)
    else:
        raise DataError(f"不支持的影像格式: {path.suffix}")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)
    # A real raster can only provide amplitude; keep it as real-part phase 0
    return arr.astype(np.float32).astype(np.complex64)


def load_stack(project_root: Path) -> Stack:
    project_root = Path(project_root)
    meta_path = project_root / "data" / "stack" / "meta.json"
    meta = read_json(meta_path)
    if not meta:
        raise DataError("项目尚未包含 SLC 数据栈，请先在「数据」页创建演示数据或导入数据")
    slc_path = safe_join(project_root, meta["slc_path"])
    if not slc_path.exists():
        raise DataError(f"SLC 数据文件缺失: {slc_path.name}")
    slc = np.load(slc_path, mmap_mode="r", allow_pickle=False)
    if slc.ndim != 3:
        raise DataError("SLC 栈必须是 (日期, 行, 列) 三维数组")
    if slc.shape[0] != len(meta.get("dates", [])):
        raise DataError("SLC 日期数与元数据不一致")
    dem = np.load(safe_join(project_root, meta["dem_path"]), allow_pickle=False) if meta.get("dem_path") else np.zeros(slc.shape[1:], dtype=np.float32)
    if dem.shape != slc.shape[1:]:
        raise DataError("DEM 尺寸与 SLC 不一致")
    mask_path = meta.get("mask_path")
    if mask_path and safe_join(project_root, mask_path).exists():
        mask = np.load(safe_join(project_root, mask_path), allow_pickle=False).astype(bool)
    else:
        mask = np.ones(slc.shape[1:], dtype=bool)
    if mask.shape != slc.shape[1:]:
        raise DataError("有效掩膜尺寸与 SLC 不一致")
    return Stack(project_root, meta, slc, dem.astype(np.float32), mask)


def preview_amplitude(stack: Stack, index: int = 0) -> np.ndarray:
    amp = stack.amplitude(index)
    finite = amp[np.isfinite(amp)]
    if finite.size == 0:
        return np.zeros_like(amp)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((amp - lo) / (hi - lo), 0, 1).astype(np.float32)


def export_complex_geotiff(path: Path, data: np.ndarray, meta: dict[str, Any]) -> Path:
    """Write a complex SLC as a two-band (I, Q) float32 GeoTIFF."""
    from .geotiff import write_geotiff

    iq = np.stack([data.real.astype(np.float32), data.imag.astype(np.float32)], axis=0)
    # write_geotiff expects 2-D; use tifffile directly for the two bands
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    extratags = [
        (33550, "d", 3, (abs(float(meta.get("lon_step", 0.0004))), abs(float(meta.get("lat_step", -0.0004))), 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(meta.get("corner_lon", 0.0)), float(meta.get("corner_lat", 0.0)), 0.0), False),
        (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326), False),
    ]
    tifffile.imwrite(path, iq, photometric="minisblack", metadata=None, extratags=extratags, compression="zlib")
    return path
