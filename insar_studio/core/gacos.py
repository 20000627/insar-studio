"""GACOS atmospheric delay products: local import, demo and online adapter.

The online client is intentionally protocol-agnostic: the GACOS service has
changed endpoints over time, so the base URL and token are configuration
values rather than hard-coded assumptions.  Local ``.ztd``/``.rsc`` and
GeoTIFF products are fully supported and are the recommended route for
operational use.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from scipy import ndimage

from .geotiff import read_geotiff
from .io import load_stack, safe_join
from .utils import atomic_write_json, read_json

ZTD_UNIT_HINT = "GACOS ZTD is converted to LOS delay in millimetres"


@dataclass
class DelayGrid:
    data_mm: np.ndarray
    bounds: tuple[float, float, float, float] | None = None
    source: str = ""


def _parse_rsc(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def read_ztd(path: Path) -> DelayGrid:
    """Read a GACOS binary ``.ztd`` file and its ``.rsc`` metadata."""
    path = Path(path)
    rsc = _parse_rsc(path.with_suffix(".rsc"))
    if not rsc:
        rsc = _parse_rsc(path.with_suffix(".rsc.txt"))
    rows = int(float(rsc.get("file_length", rsc.get("lines", 0))))
    cols = int(float(rsc.get("width", rsc.get("samples", 0))))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"无法从 .rsc 读取行列数: {path.name}")
    raw = np.fromfile(path, dtype=">f4")
    if raw.size < rows * cols:
        raw = np.fromfile(path, dtype="<f4")
    if raw.size < rows * cols:
        raise ValueError(f"GACOS 文件长度不足: {path.name}")
    data = raw[: rows * cols].reshape(rows, cols).astype(np.float64)
    # GACOS values are usually metres; some portals ship millimetres.
    med = float(np.nanmedian(np.abs(data)))
    if med > 50:
        data = data / 1000.0
    data_mm = data * 1000.0
    bounds = None
    if all(k in rsc for k in ("x_first", "y_first", "x_step", "y_step")):
        x0 = float(rsc["x_first"])
        y0 = float(rsc["y_first"])
        dx = float(rsc["x_step"])
        dy = float(rsc["y_step"])
        bounds = (x0, y0 + dy * rows, x0 + dx * cols, y0)
        if bounds[1] > bounds[3]:
            bounds = (bounds[0], bounds[3], bounds[2], bounds[1])
    return DelayGrid(data_mm=data_mm, bounds=bounds, source=str(path))


def read_delay_file(path: Path) -> DelayGrid:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".ztd", ".dat", ".bin"):
        return read_ztd(path)
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False).astype(np.float64)
        return DelayGrid(data_mm=data, source=str(path))
    if suffix in (".tif", ".tiff"):
        data, geo = read_geotiff(path)
        data = data.astype(np.float64)
        med = float(np.nanmedian(np.abs(data)))
        if med < 10:  # metres
            data *= 1000.0
        bounds = None
        if all(k in geo for k in ("corner_lon", "corner_lat", "lon_step", "lat_step")):
            bounds = (
                float(geo["corner_lon"]),
                float(geo["corner_lat"]) + float(geo["lat_step"]) * data.shape[0],
                float(geo["corner_lon"]) + float(geo["lon_step"]) * data.shape[1],
                float(geo["corner_lat"]),
            )
        return DelayGrid(data_mm=data, bounds=bounds, source=str(path))
    raise ValueError(f"不支持的 GACOS/大气产品格式: {path.suffix}")


def _resample_grid(grid: DelayGrid, shape: tuple[int, int], target_bounds: tuple[float, float, float, float] | None) -> np.ndarray:
    data = grid.data_mm
    if data.shape == shape:
        return data.astype(np.float32)
    if grid.bounds and target_bounds:
        west, south, east, north = target_bounds
        rows, cols = shape
        lon = np.linspace(west + (east - west) / (2 * cols), east - (east - west) / (2 * cols), cols)
        lat = np.linspace(north - (north - south) / (2 * rows), south + (north - south) / (2 * rows), rows)
        gw, gs, ge, gn = grid.bounds
        x = (lon - gw) / max(ge - gw, 1e-9) * (data.shape[1] - 1)
        y = (gn - lat) / max(gn - gs, 1e-9) * (data.shape[0] - 1)
        xx, yy = np.meshgrid(x, y)
        return ndimage.map_coordinates(data, [yy, xx], order=1, mode="nearest").astype(np.float32)
    zoom = (shape[0] / data.shape[0], shape[1] / data.shape[1])
    out = ndimage.zoom(data, zoom, order=1, mode="nearest")
    if out.shape != shape:
        out = out[: shape[0], : shape[1]]
        out = np.pad(out, ((0, max(0, shape[0] - out.shape[0])), (0, max(0, shape[1] - out.shape[1]))), mode="edge")
    return out.astype(np.float32)


def _date_tokens(date_str: str) -> list[str]:
    d = str(date_str)[:10]
    return [d.replace("-", ""), d]


def find_local_screens(directory: Path, dates: Iterable[str]) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"GACOS 目录不存在: {directory}")
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in (".ztd", ".tif", ".tiff", ".npy")]
    found: dict[str, Path] = {}
    for date in dates:
        for token in _date_tokens(date):
            match = [p for p in files if token in p.name]
            if match:
                found[str(date)[:10]] = sorted(match, key=lambda p: len(p.name))[0]
                break
    return found


def prepare_demo_gacos(root: Path) -> dict[str, Any]:
    """Write the synthetic atmosphere as per-date 'GACOS' screens."""
    root = Path(root)
    stack = load_stack(root)
    truth_path = root / "data" / "stack" / "truth.npz"
    if not truth_path.exists():
        raise FileNotFoundError("当前项目不是演示数据，无法生成模拟 GACOS")
    with np.load(truth_path) as truth:
        atm = truth["atmosphere_mm"].astype(np.float32)
    out = root / "data" / "gacos" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(stack.dates):
        np.save(out / f"{d.replace('-', '')}.npy", atm[i], allow_pickle=False)
    meta = {
        "source": "demo",
        "simulated": True,
        "unit": "mm LOS delay",
        "dates": stack.dates,
        "rows": stack.rows,
        "cols": stack.cols,
        "bounds": [
            float(stack.meta.get("corner_lon", 0.0)),
            float(stack.meta.get("corner_lat", 0.0)) + stack.rows * float(stack.meta.get("lat_step", -0.0004)),
            float(stack.meta.get("corner_lon", 0.0)) + stack.cols * float(stack.meta.get("lon_step", 0.0004)),
            float(stack.meta.get("corner_lat", 0.0)),
        ],
        "note": "Synthetic GACOS for end-to-end demonstration. Not a real weather-model product.",
    }
    atomic_write_json(out / "meta.json", meta)
    return meta


def load_screens(
    root: Path,
    params: dict[str, Any],
    *,
    target_shape: tuple[int, int] | None = None,
    target_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(root)
    stack = load_stack(root)
    source = str(params.get("gacos_source", "demo")).lower()
    shape = target_shape or (stack.rows, stack.cols)
    meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    if target_bounds is None:
        west = float(meta.get("corner_lon", 0.0))
        north = float(meta.get("corner_lat", 0.0))
        target_bounds = (
            west,
            north + shape[0] * float(meta.get("lat_step", -0.0004)),
            west + shape[1] * float(meta.get("lon_step", 0.0004)),
            north,
        )
    if source == "demo":
        demo_dir = root / "data" / "gacos" / "demo"
        if not (demo_dir / "meta.json").exists():
            prepare_demo_gacos(root)
        grids = {}
        for d in stack.dates:
            path = demo_dir / f"{d.replace('-', '')}.npy"
            if path.exists():
                grids[d] = _resample_grid(DelayGrid(data_mm=np.load(path)), shape, target_bounds)
        return grids, {"source": "demo", "simulated": True, "n_dates": len(grids), "unit": "mm"}

    directory = params.get("gacos_dir")
    if source == "local" and directory:
        found = find_local_screens(Path(directory), stack.dates)
        grids = {}
        for d, path in found.items():
            grid = read_delay_file(path)
            grids[d] = _resample_grid(grid, shape, target_bounds)
        return grids, {"source": "local", "directory": str(directory), "n_dates": len(grids), "unit": "mm"}

    if source == "online":
        client = GacosClient(
            base_url=str(params.get("gacos_url", "")),
            email=str(params.get("gacos_email", "")),
            token=str(params.get("gacos_token", "")),
            timeout=int(params.get("gacos_timeout", 60)),
        )
        grids = client.fetch_for_stack(root, stack, params)
        return grids, {"source": "online", "n_dates": len(grids), "unit": "mm"}

    raise ValueError("未配置 GACOS 数据源（可选 demo / local / online）")


class GacosClient:
    """Small, configurable client for GACOS-style delay products.

    Different GACOS deployments expose different endpoints.  Set ``base_url``
    to the service you are licensed to use; the client posts a JSON request
    and accepts either a JSON list of URLs or a binary/text response.
    Credentials are kept in memory only and are never written to disk.
    """

    def __init__(self, *, base_url: str, email: str = "", token: str = "", timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.token = token
        self.timeout = timeout

    def status(self) -> dict[str, Any]:
        if not self.base_url:
            return {"ok": False, "message": "未配置 GACOS 服务地址"}
        try:
            r = requests.get(self.base_url, timeout=min(self.timeout, 15))
            return {"ok": r.status_code < 500, "status": r.status_code, "message": r.reason}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def fetch_for_stack(self, root: Path, stack: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
        if not self.base_url:
            raise ValueError("未配置 GACOS 服务地址")
        bbox = [
            float(stack.meta.get("corner_lon", 0.0)),
            float(stack.meta.get("corner_lat", 0.0)) + stack.rows * float(stack.meta.get("lat_step", -0.0004)),
            float(stack.meta.get("corner_lon", 0.0)) + stack.cols * float(stack.meta.get("lon_step", 0.0004)),
            float(stack.meta.get("corner_lat", 0.0)),
        ]
        payload = {
            "email": self.email,
            "token": self.token,
            "bbox": bbox,
            "dates": stack.dates,
            "dem": params.get("gacos_dem", "SRTM"),
            "format": "ztd",
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        cache = root / "data" / "gacos" / "online"
        cache.mkdir(parents=True, exist_ok=True)
        content_type = r.headers.get("content-type", "")
        grids: dict[str, np.ndarray] = {}
        if "application/json" in content_type:
            body = r.json()
            items = body.get("products", body if isinstance(body, list) else [])
            for item in items:
                date = str(item.get("date", ""))[:10]
                url = item.get("url")
                if not date or not url:
                    continue
                data = requests.get(url, headers=headers, timeout=self.timeout)
                data.raise_for_status()
                suffix = Path(url).suffix or ".ztd"
                path = cache / f"{date.replace('-', '')}{suffix}"
                path.write_bytes(data.content)
                grid = read_delay_file(path)
                grids[date] = grid.data_mm
        else:
            # single product response: cache as today's screen
            path = cache / f"{stack.dates[0].replace('-', '')}.ztd"
            path.write_bytes(r.content)
            grid = read_delay_file(path)
            grids[stack.dates[0]] = grid.data_mm
        return grids
