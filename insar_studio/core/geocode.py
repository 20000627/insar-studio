"""Range-Doppler geocoding of radar-geometry products.

The geocoder uses the reference acquisition's orbit state vectors and the
timing stored in ``data/stack/meta.json`` to solve the zero-Doppler (or
constant-Doppler-centroid) geometry.  For every map pixel it finds the radar
row/column by Newton iteration, optionally refining the terrain height from the
radar-geometry DEM, and then samples the radar product with bilinear
interpolation.

The implementation is deliberately conservative.  When the orbit, first-line
time, azimuth interval, range spacing or first-sample slant range is missing,
it refuses to geocode and writes an explicit reason instead of emitting a
GeoTIFF with plausible-looking but wrong coordinates.  It does not handle TOPS
bursts, deramping, map projections other than EPSG:4326, or antimeridian
crossings.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from ..config import S1_WAVELENGTH
from .geotiff import write_geotiff
from .io import safe_join
from .memory import choose_block_rows
from .orbit import Orbit, OrbitError, parse_time
from .utils import atomic_write_json, read_json

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
SPEED_OF_LIGHT = 299_792_458.0


class GeocodeError(ValueError):
    """Raised when geocoding metadata or geometry is inconsistent."""


def geodetic_to_ecef(
    lon_deg: np.ndarray | float,
    lat_deg: np.ndarray | float,
    height_m: np.ndarray | float,
) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF metres."""
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    height = np.asarray(height_m, dtype=np.float64)
    lon, lat, height = np.broadcast_arrays(lon, lat, height)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (radius + height) * cos_lat * np.cos(lon)
    y = (radius + height) * cos_lat * np.sin(lon)
    z = (radius * (1.0 - WGS84_E2) + height) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF metres to WGS-84 longitude, latitude and ellipsoid height."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    height = np.zeros_like(lat)
    for _ in range(8):
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = np.where(
            np.abs(cos_lat) > 1e-10,
            p / np.where(np.abs(cos_lat) > 1e-10, cos_lat, 1.0) - radius,
            z / np.where(np.abs(sin_lat) > 1e-10, sin_lat, 1.0) - radius * (1.0 - WGS84_E2),
        )
        denominator = p * (1.0 - WGS84_E2 * radius / np.maximum(radius + height, 1.0))
        updated = np.arctan2(z, denominator)
        if np.nanmax(np.abs(updated - lat)) < 1e-12:
            lat = updated
            break
        lat = updated
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = np.where(
        np.abs(cos_lat) > 1e-10,
        p / np.where(np.abs(cos_lat) > 1e-10, cos_lat, 1.0) - radius,
        z / np.where(np.abs(sin_lat) > 1e-10, sin_lat, 1.0) - radius * (1.0 - WGS84_E2),
    )
    return np.rad2deg(lon), np.rad2deg(lat), height


def ellipsoid_height(xyz: np.ndarray) -> np.ndarray:
    return ecef_to_geodetic(xyz)[2]


@dataclass
class MapGrid:
    """North-up WGS-84 raster grid.

    ``corner_lon``/``corner_lat`` refer to the upper-left outer corner of the
    first pixel, matching :func:`insar_studio.core.geotiff.write_geotiff`.
    """

    rows: int
    cols: int
    corner_lon: float
    corner_lat: float
    lon_step: float
    lat_step: float

    def __post_init__(self) -> None:
        self.rows = int(self.rows)
        self.cols = int(self.cols)
        if self.rows <= 0 or self.cols <= 0:
            raise GeocodeError("地图网格尺寸必须为正")
        if not math.isfinite(self.lon_step) or self.lon_step == 0:
            raise GeocodeError("地图网格经度步长无效")
        if not math.isfinite(self.lat_step) or self.lat_step == 0:
            raise GeocodeError("地图网格纬度步长无效")
        if self.lat_step > 0:
            self.corner_lat = self.corner_lat + self.rows * self.lat_step
            self.lat_step = -self.lat_step

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        west = self.corner_lon
        north = self.corner_lat
        east = west + self.cols * self.lon_step
        south = north + self.rows * self.lat_step
        return west, min(south, north), east, max(south, north)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        lon = self.corner_lon + (np.arange(self.cols) + 0.5) * self.lon_step
        lat = self.corner_lat + (np.arange(self.rows) + 0.5) * self.lat_step
        return np.meshgrid(lon, lat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "corner_lon": float(self.corner_lon),
            "corner_lat": float(self.corner_lat),
            "lon_step": float(self.lon_step),
            "lat_step": float(self.lat_step),
            "bounds": [float(value) for value in self.bounds],
            "crs": "EPSG:4326",
        }


@dataclass
class RadarGeometry:
    """Reference acquisition geometry used by the range-Doppler solver."""

    orbit: Orbit
    first_line_time: datetime
    azimuth_time_interval: float
    slant_range_first: float
    range_spacing: float
    rows: int
    cols: int
    look_side: str = "right"
    doppler_centroid_hz: float = 0.0
    wavelength_m: float = S1_WAVELENGTH

    def __post_init__(self) -> None:
        self.rows = int(self.rows)
        self.cols = int(self.cols)
        if self.rows <= 0 or self.cols <= 0:
            raise GeocodeError("雷达网格尺寸必须为正")
        if not math.isfinite(self.azimuth_time_interval) or self.azimuth_time_interval <= 0:
            raise GeocodeError("方位时间间隔必须为正")
        if not math.isfinite(self.range_spacing) or self.range_spacing <= 0:
            raise GeocodeError("距离向像元间距必须为正")
        if not math.isfinite(self.slant_range_first) or self.slant_range_first <= 0:
            raise GeocodeError("首列斜距必须为正")
        side = str(self.look_side).lower()
        if side not in ("right", "left"):
            raise GeocodeError("look_side 必须是 right 或 left")
        self.look_side = side

    def time_for_row(self, row: float | np.ndarray) -> datetime | np.ndarray:
        seconds = np.asarray(row, dtype=np.float64) * self.azimuth_time_interval
        if np.ndim(seconds) == 0:
            return self.first_line_time + timedelta(seconds=float(seconds))
        base = self.first_line_time.replace(tzinfo=timezone.utc)
        return np.array(
            [base + timedelta(seconds=float(value)) for value in seconds.ravel()],
            dtype=object,
        ).reshape(seconds.shape)

    def range_for_col(self, col: float | np.ndarray) -> float | np.ndarray:
        return self.slant_range_first + np.asarray(col, dtype=np.float64) * self.range_spacing

    def _scene_seconds(self, row: np.ndarray) -> np.ndarray:
        first = (self.first_line_time - self.orbit.epoch).total_seconds()
        return first + np.asarray(row, dtype=np.float64) * self.azimuth_time_interval

    def ground_to_radar(
        self,
        lon_deg: np.ndarray,
        lat_deg: np.ndarray,
        height_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve map coordinates to radar row/column.

        Returns ``(row, col, seconds, slant_range)`` in original radar pixel
        coordinates.  ``row`` and ``col`` are NaN where the solution does not
        converge inside the scene or the orbit coverage.
        """
        lon, lat, height = np.broadcast_arrays(
            np.asarray(lon_deg, dtype=np.float64),
            np.asarray(lat_deg, dtype=np.float64),
            np.asarray(height_m, dtype=np.float64),
        )
        shape = lon.shape
        target = geodetic_to_ecef(lon, lat, height).reshape(-1, 3)
        if target.size == 0:
            empty = np.zeros(shape, dtype=np.float64)
            return empty, empty.copy(), empty.copy(), empty.copy()
        finite = np.all(np.isfinite(target), axis=1)
        if not finite.any():
            empty = np.full(target.shape[0], np.nan)
            return (
                empty.reshape(shape),
                empty.copy().reshape(shape),
                empty.copy().reshape(shape),
                empty.copy().reshape(shape),
            )
        # Replace non-finite targets with a harmless point; their result is
        # masked out below.  This keeps the vectorised Newton loop finite.
        target = np.where(finite[:, None], target, np.array([WGS84_A, 0.0, 0.0]))

        first = (self.first_line_time - self.orbit.epoch).total_seconds()
        seconds = np.full(target.shape[0], first + 0.5 * self.rows * self.azimuth_time_interval)
        low, high = float(self.orbit.times[0]), float(self.orbit.times[-1])
        residual = np.full(target.shape[0], np.inf)
        for _ in range(8):
            clipped = np.clip(seconds, low, high)
            position, velocity = self.orbit.interpolate(clipped, chunk=100_000)
            delta = target - position
            slant = np.linalg.norm(delta, axis=1)
            doppler_term = 0.5 * self.wavelength_m * self.doppler_centroid_hz * slant
            value = np.einsum("ij,ij->i", delta, velocity) + doppler_term
            derivative = -np.einsum("ij,ij->i", velocity, velocity)
            derivative = np.where(np.abs(derivative) < 1.0, -1.0, derivative)
            step = np.clip(value / derivative, -120.0, 120.0)
            seconds = np.clip(seconds - step, low, high)
            residual = np.abs(value) / np.maximum(np.linalg.norm(velocity, axis=1), 1.0)
            if (
                np.max(np.abs(step[finite])) < 1e-5
                and np.max(residual[finite]) < 0.05
            ):
                break

        position, _ = self.orbit.interpolate(seconds, chunk=100_000)
        slant_range = np.linalg.norm(target - position, axis=1)
        row = (seconds - first) / self.azimuth_time_interval
        col = (slant_range - self.slant_range_first) / self.range_spacing
        valid = (
            finite
            & np.isfinite(row)
            & np.isfinite(col)
            & np.isfinite(residual)
            & (residual < 1.0)
            & (row >= -0.5)
            & (row <= self.rows - 0.5)
            & (col >= -0.5)
            & (col <= self.cols - 0.5)
        )
        row = np.where(valid, row, np.nan).reshape(shape)
        col = np.where(valid, col, np.nan).reshape(shape)
        seconds = np.where(valid, seconds, np.nan).reshape(shape)
        slant_range = np.where(valid, slant_range, np.nan).reshape(shape)
        return row, col, seconds, slant_range

    def radar_to_ground(
        self,
        row: np.ndarray,
        col: np.ndarray,
        height_m: np.ndarray | float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intersect range spheres with the WGS-84 ellipsoid at a given height.

        The along-track direction defines a zero-Doppler plane.  Inside that
        plane the look vector is rotated by an angle ``alpha`` from nadir; a
        bisection finds the ellipsoid intersection on the configured look side.
        """
        row, col, height = np.broadcast_arrays(
            np.asarray(row, dtype=np.float64),
            np.asarray(col, dtype=np.float64),
            np.asarray(height_m, dtype=np.float64),
        )
        shape = row.shape
        flat_row = row.ravel()
        flat_col = col.ravel()
        flat_height = height.ravel()
        finite = np.isfinite(flat_row) & np.isfinite(flat_col) & np.isfinite(flat_height)
        if not finite.any():
            empty = np.full(shape, np.nan, dtype=np.float64)
            return empty, empty.copy()
        safe_row = np.where(finite, flat_row, 0.0)
        safe_col = np.where(finite, flat_col, 0.0)
        safe_height = np.where(finite, flat_height, 0.0)
        seconds = self._scene_seconds(safe_row)
        low, high = float(self.orbit.times[0]), float(self.orbit.times[-1])
        inside = (seconds >= low - 1e-6) & (seconds <= high + 1e-6)
        seconds = np.clip(seconds, low, high)
        position, velocity = self.orbit.interpolate(seconds, chunk=100_000)
        slant = self.slant_range_first + safe_col * self.range_spacing

        speed = np.linalg.norm(velocity, axis=1)
        along = velocity / np.maximum(speed[:, None], 1e-9)
        radial = position - np.einsum("ij,ij->i", position, along)[:, None] * along
        radial_norm = np.linalg.norm(radial, axis=1)
        e1 = radial / np.maximum(radial_norm[:, None], 1e-9)
        cross_track = np.cross(along, e1)
        cross_norm = np.linalg.norm(cross_track, axis=1)
        e2 = cross_track / np.maximum(cross_norm[:, None], 1e-9)
        side = 1.0 if self.look_side == "right" else -1.0

        def point(alpha: np.ndarray) -> np.ndarray:
            direction = -np.cos(alpha)[:, None] * e1 + side * np.sin(alpha)[:, None] * e2
            return position + slant[:, None] * direction

        lower = np.zeros_like(seconds)
        upper = np.full_like(seconds, 0.5 * np.pi)
        f_lower = ellipsoid_height(point(lower)) - safe_height
        f_upper = ellipsoid_height(point(upper)) - safe_height
        valid = (
            finite
            & inside
            & np.isfinite(f_lower)
            & np.isfinite(f_upper)
            & (f_lower <= 0)
            & (f_upper >= 0)
        )
        for _ in range(26):
            middle = 0.5 * (lower + upper)
            value = ellipsoid_height(point(middle)) - safe_height
            lower = np.where(valid & (value < 0), middle, lower)
            upper = np.where(valid & (value >= 0), middle, upper)
        alpha = 0.5 * (lower + upper)
        ground = point(alpha)
        lon, lat, _ = ecef_to_geodetic(ground)
        return (
            np.where(valid, lon, np.nan).reshape(shape),
            np.where(valid, lat, np.nan).reshape(shape),
        )


def _geometry_reasons(meta: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    orbit_info = meta.get("orbit")
    if not isinstance(orbit_info, dict) or not orbit_info.get("vectors_path"):
        reasons.append("缺少参考影像的轨道状态向量（导入时未找到本地 EOF）")
        return None, reasons, warnings
    try:
        rows = int(meta["rows"])
        cols = int(meta["cols"])
    except (KeyError, TypeError, ValueError):
        reasons.append("缺少雷达影像行列数")
        return None, reasons, warnings
    raw_start = meta.get("start_times")
    if not isinstance(raw_start, list) or not raw_start:
        reasons.append("缺少每景成像时间 start_times")
        return None, reasons, warnings
    try:
        ref_index = int(meta.get("ref_index", 0))
    except (TypeError, ValueError):
        ref_index = 0
    if not 0 <= ref_index < len(raw_start):
        reasons.append("参考影像索引 ref_index 超出 start_times 范围")
        return None, reasons, warnings
    try:
        first_line = parse_time(str(raw_start[ref_index]))
    except OrbitError:
        reasons.append("参考影像的 start_times 无法解析")
        return None, reasons, warnings
    try:
        azimuth_interval = float(meta.get("azimuth_time_interval"))
    except (TypeError, ValueError):
        azimuth_interval = 0.0
    if not math.isfinite(azimuth_interval) or azimuth_interval <= 0:
        reasons.append("缺少有效的方位时间间隔 azimuth_time_interval")
        return None, reasons, warnings
    try:
        range_spacing = float(meta.get("range_spacing_m"))
    except (TypeError, ValueError):
        range_spacing = 0.0
    if not math.isfinite(range_spacing) or range_spacing <= 0:
        reasons.append("缺少有效的距离向像元间距 range_spacing_m")
        return None, reasons, warnings

    slant_time = meta.get("slant_range_time_s")
    slant_range = 0.0
    if slant_time not in (None, ""):
        try:
            value = float(slant_time)
            if math.isfinite(value) and value > 0:
                slant_range = SPEED_OF_LIGHT * value / 2.0
        except (TypeError, ValueError):
            pass
    if slant_range <= 0:
        source = str(meta.get("slant_range_source", "default")).lower()
        try:
            value = float(meta.get("slant_range_m", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if source == "default" or not math.isfinite(value) or value <= 0:
            reasons.append("缺少首列斜距：需要 slant_range_time_s，或显式提供 slant_range_m")
            return None, reasons, warnings
        slant_range = value

    try:
        doppler = float(meta.get("doppler_centroid_hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        doppler = 0.0
    if not math.isfinite(doppler):
        doppler = 0.0
    if abs(doppler) < 1e-9:
        warnings.append("未提供多普勒中心频率，按零多普勒几何处理")
    else:
        warnings.append(f"多普勒中心按常数 {doppler:.1f} Hz 处理")

    orbit_path = safe_join(Path(meta.get("__project_root__", ".")), str(orbit_info["vectors_path"]))
    try:
        payload = read_json(orbit_path, {})
    except Exception:
        payload = {}
    records = payload.get("orbits") if isinstance(payload, dict) else None
    if not isinstance(records, list) or ref_index >= len(records):
        reasons.append("轨道状态向量文件缺失或与参考影像不对应")
        return None, reasons, warnings
    try:
        orbit = Orbit.from_dict(records[ref_index])
    except (OrbitError, TypeError, KeyError, ValueError):
        reasons.append("轨道状态向量记录无法解析")
        return None, reasons, warnings

    scene_end = first_line + timedelta(seconds=max(0, rows - 1) * azimuth_interval)
    if not orbit.covers(first_line) or not orbit.covers(scene_end):
        reasons.append("轨道未覆盖完整成像时间，拒绝地理编码")
        return None, reasons, warnings

    geometry = RadarGeometry(
        orbit=orbit,
        first_line_time=first_line,
        azimuth_time_interval=azimuth_interval,
        slant_range_first=slant_range,
        range_spacing=range_spacing,
        rows=rows,
        cols=cols,
        look_side=str(meta.get("look_side", "right")),
        doppler_centroid_hz=doppler,
        wavelength_m=float(meta.get("wavelength_m", S1_WAVELENGTH) or S1_WAVELENGTH),
    )
    return geometry, reasons, warnings


def load_geometry(meta: dict[str, Any], project_root: Path) -> tuple[RadarGeometry | None, list[str], list[str]]:
    """Build reference geometry from stack metadata, or explain why not."""
    decorated = dict(meta)
    decorated["__project_root__"] = str(project_root)
    return _geometry_reasons(decorated)


def _sample_dem(dem: np.ndarray | None, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    if dem is None:
        return np.zeros_like(row, dtype=np.float64)
    data = np.asarray(dem, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        return np.zeros_like(row, dtype=np.float64)
    rr = np.clip(np.nan_to_num(row, nan=0.0), 0, data.shape[0] - 1)
    cc = np.clip(np.nan_to_num(col, nan=0.0), 0, data.shape[1] - 1)
    values = ndimage.map_coordinates(data, [rr, cc], order=1, mode="nearest")
    return np.where(np.isfinite(values), values, 0.0).astype(np.float64)


#: Per-block working-set budget for the range-Doppler lookup.  The solver's
#: largest transient is a ``(pixels, 8, 6)`` interpolation buffer (~384 bytes
#: per pixel) plus its intermediates, so budget roughly 512 bytes per pixel.
GEOCODE_BLOCK_BUDGET_BYTES = 16 * 1024 * 1024
GEOCODE_BLOCK_BYTES_PER_PIXEL = 512
GEOCODE_MAX_BLOCK_PIXELS = 100_000
GEOCODE_THREAD_THRESHOLD = 250_000
GEOCODE_MAX_WORKERS = 4


def build_lookup(
    geometry: RadarGeometry,
    grid: MapGrid,
    dem: np.ndarray | None,
    *,
    multilook: int,
    product_shape: tuple[int, int],
    max_iterations: int = 3,
    tolerance_m: float = 0.5,
    block_rows: int | None = None,
    workers: int | None = None,
    max_block_pixels: int = GEOCODE_MAX_BLOCK_PIXELS,
) -> dict[str, Any]:
    """Resolve every map pixel to a product-space radar coordinate.

    The solve is independent per pixel, so the grid is processed in row blocks.
    ``block_rows=None`` derives a block height from a 32 MiB working-set
    budget; ``workers=None`` uses up to four threads once the grid is large
    enough to amortise thread startup.  Outputs are float32 arrays, which keeps
    a 20-megapixel lookup near 260 MB instead of roughly 500 MB in float64.
    """
    map_rows, map_cols = int(grid.rows), int(grid.cols)
    rows, cols = int(product_shape[0]), int(product_shape[1])
    m = max(1, int(multilook))
    offset = 0.5 * (m - 1)

    total_pixels = map_rows * map_cols
    if block_rows is None:
        block_rows = choose_block_rows(
            map_rows,
            map_cols,
            budget_bytes=GEOCODE_BLOCK_BUDGET_BYTES,
            arrays=GEOCODE_BLOCK_BYTES_PER_PIXEL // 8,
            itemsize=8,
        )
        block_rows = min(block_rows, max(1, int(max_block_pixels) // max(1, map_cols)))
    block_rows = max(1, min(map_rows, int(block_rows)))
    blocks = [
        (start, min(map_rows, start + block_rows))
        for start in range(0, map_rows, block_rows)
    ]
    if workers is None:
        cpus = os.cpu_count() or 1
        workers = 1 if total_pixels < GEOCODE_THREAD_THRESHOLD else min(
            GEOCODE_MAX_WORKERS, cpus, len(blocks)
        )
    workers = max(1, min(int(workers), len(blocks)))

    row_out = np.full((map_rows, map_cols), np.nan, dtype=np.float32)
    col_out = np.full((map_rows, map_cols), np.nan, dtype=np.float32)
    height_out = np.zeros((map_rows, map_cols), dtype=np.float32)
    valid_out = np.zeros((map_rows, map_cols), dtype=bool)

    lon_axis = grid.corner_lon + (np.arange(map_cols) + 0.5) * grid.lon_step
    lat_axis = grid.corner_lat + (np.arange(map_rows) + 0.5) * grid.lat_step

    def solve_block(bounds: tuple[int, int]):
        start, stop = bounds
        block_shape = (stop - start, map_cols)
        lon = np.broadcast_to(lon_axis[None, :], block_shape)
        lat = np.broadcast_to(lat_axis[start:stop, None], block_shape)
        height = np.zeros(block_shape, dtype=np.float64)
        last_delta = float("inf")
        used = 0
        for iteration in range(1, max_iterations + 1):
            row, col, _, _ = geometry.ground_to_radar(lon, lat, height)
            inside = np.isfinite(row) & np.isfinite(col)
            sampled = _sample_dem(dem, row, col)
            updated = np.where(inside, sampled, height)
            last_delta = (
                float(np.nanmax(np.abs(updated - height))) if inside.any() else float("inf")
            )
            height = updated
            used = iteration
            if not np.isfinite(last_delta) or last_delta < tolerance_m:
                break
        row, col, _, _ = geometry.ground_to_radar(lon, lat, height)
        product_row = (row - offset) / m
        product_col = (col - offset) / m
        valid = (
            np.isfinite(product_row)
            & np.isfinite(product_col)
            & (product_row >= -0.5)
            & (product_row <= rows - 0.5)
            & (product_col >= -0.5)
            & (product_col <= cols - 0.5)
        )
        return (
            product_row.astype(np.float32),
            product_col.astype(np.float32),
            height.astype(np.float32),
            valid,
            used,
            last_delta,
        )

    started = time.perf_counter()
    used = 0
    finite_deltas: list[float] = []

    def store_result(
        bounds: tuple[int, int],
        result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float],
    ) -> None:
        nonlocal used
        start, stop = bounds
        block_row, block_col, block_height, block_valid, block_used, delta = result
        row_out[start:stop] = block_row
        col_out[start:stop] = block_col
        height_out[start:stop] = block_height
        valid_out[start:stop] = block_valid
        used = max(used, int(block_used))
        if np.isfinite(delta):
            finite_deltas.append(float(delta))

    if workers > 1 and len(blocks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Keep only a small number of blocks in flight.  Submitting every
            # block at once lets completed float32 results pile up while the
            # first (slowest) block is still running; bounded submission keeps
            # the transient working set close to ``workers`` blocks.
            block_iter = iter(blocks)
            pending: dict[Any, tuple[int, int]] = {}
            limit = max(1, workers * 2)
            for bounds in block_iter:
                pending[pool.submit(solve_block, bounds)] = bounds
                if len(pending) >= limit:
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    bounds = pending.pop(future)
                    store_result(bounds, future.result())
                    follow_up = next(block_iter, None)
                    if follow_up is not None:
                        pending[pool.submit(solve_block, follow_up)] = follow_up
    else:
        for bounds in blocks:
            store_result(bounds, solve_block(bounds))
    last_delta = max(finite_deltas) if finite_deltas else float("inf")
    elapsed = time.perf_counter() - started

    return {
        "row": row_out,
        "col": col_out,
        "height_m": height_out,
        "valid": valid_out,
        "iterations": used,
        "max_height_update_m": last_delta,
        "coverage": float(valid_out.mean()) if valid_out.size else 0.0,
        "block_rows": block_rows,
        "blocks": len(blocks),
        "workers": workers,
        "elapsed_s": elapsed,
    }


def sample_product(data: np.ndarray, lookup: dict[str, Any]) -> np.ndarray:
    """Bilinear-sample a radar product at the resolved map pixels."""
    array = np.asarray(data, dtype=np.float64)
    valid = lookup["valid"]
    output = np.full(valid.shape, np.nan, dtype=np.float32)
    if array.ndim != 2 or array.size == 0 or not valid.any():
        return output
    row = np.clip(lookup["row"][valid], 0, array.shape[0] - 1)
    col = np.clip(lookup["col"][valid], 0, array.shape[1] - 1)
    values = ndimage.map_coordinates(array, [row, col], order=1, mode="nearest")
    output[valid] = values.astype(np.float32)
    return output


def derive_map_grid(
    meta: dict[str, Any],
    geometry: RadarGeometry,
    dem: np.ndarray | None,
    *,
    multilook: int,
    product_shape: tuple[int, int],
    max_pixels: int = 20_000_000,
) -> MapGrid:
    """Create a north-up grid covering the radar footprint."""
    rows_used = max(1, int(product_shape[0]) * max(1, int(multilook)))
    cols_used = max(1, int(product_shape[1]) * max(1, int(multilook)))
    edge = np.linspace(0, 1, 17)
    top = np.stack([np.zeros_like(edge), edge * (cols_used - 1)], axis=1)
    bottom = np.stack([np.full_like(edge, rows_used - 1), edge * (cols_used - 1)], axis=1)
    left = np.stack([edge * (rows_used - 1), np.zeros_like(edge)], axis=1)
    right = np.stack([edge * (rows_used - 1), np.full_like(edge, cols_used - 1)], axis=1)
    radar = np.concatenate([top, bottom, left, right], axis=0)
    heights = _sample_dem(dem, radar[:, 0], radar[:, 1])
    lon, lat = geometry.radar_to_ground(radar[:, 0], radar[:, 1], heights)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if finite.sum() < 4:
        raise GeocodeError("无法由轨道推导地面覆盖范围")
    lon = lon[finite]
    lat = lat[finite]
    if float(lon.max() - lon.min()) > 180.0:
        raise GeocodeError("轨道覆盖范围跨越 180° 经线，请显式提供 corner_lon/corner_lat 网格")
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    center_lat = 0.5 * (south + north)
    incidence = float(meta.get("incidence_deg", 38.5))
    incidence = min(85.0, max(5.0, incidence))
    ground_range = float(meta.get("range_spacing_m", 2.329562)) * max(1, int(multilook)) / math.sin(math.radians(incidence))
    azimuth = float(meta.get("azimuth_spacing_m", 13.968)) * max(1, int(multilook))
    lon_step = max(1e-7, ground_range / (111_320.0 * max(0.05, math.cos(math.radians(center_lat)))))
    lat_step = max(1e-7, azimuth / 110_540.0)
    padding = 2
    rows = int(math.ceil((north - south) / lat_step)) + 2 * padding + 1
    cols = int(math.ceil((east - west) / lon_step)) + 2 * padding + 1
    if rows * cols > max_pixels:
        scale = math.sqrt((rows * cols) / max_pixels)
        lon_step *= scale
        lat_step *= scale
        rows = int(math.ceil((north - south) / lat_step)) + 2 * padding + 1
        cols = int(math.ceil((east - west) / lon_step)) + 2 * padding + 1
    return MapGrid(
        rows=rows,
        cols=cols,
        corner_lon=west - padding * lon_step,
        corner_lat=north + padding * lat_step,
        lon_step=lon_step,
        lat_step=-lat_step,
    )


def map_grid(
    meta: dict[str, Any],
    geometry: RadarGeometry,
    dem: np.ndarray | None,
    *,
    product_shape: tuple[int, int],
    multilook: int,
    grid_meta: dict[str, Any] | None = None,
) -> tuple[MapGrid, list[str]]:
    """Use the declared grid when it is real, otherwise derive one."""
    notes: list[str] = []
    if meta.get("geo_referenced") and grid_meta:
        try:
            grid = MapGrid(
                rows=int(grid_meta["rows"]),
                cols=int(grid_meta["cols"]),
                corner_lon=float(grid_meta["corner_lon"]),
                corner_lat=float(grid_meta["corner_lat"]),
                lon_step=float(grid_meta["lon_step"]),
                lat_step=float(grid_meta["lat_step"]),
            )
            notes.append("使用 meta 中声明的经纬度网格")
            return grid, notes
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"声明的经纬度网格无效，改用轨道推导：{exc}")
    grid = derive_map_grid(
        meta,
        geometry,
        dem,
        multilook=multilook,
        product_shape=product_shape,
    )
    notes.append("由参考轨道和 DEM 推导北向上经纬度网格（近似）")
    return grid, notes


def geocode_products(
    root: Path,
    stack_meta: dict[str, Any],
    params: dict[str, Any],
    *,
    product_shape: tuple[int, int],
    grid_meta: dict[str, Any] | None,
    dem: np.ndarray | None,
    products: dict[str, np.ndarray],
    technique: str,
) -> dict[str, Any]:
    """Geocode key products and persist a machine-readable status file.

    Geocoding is a post-processing step: failures are recorded in
    ``products/geocode.json`` and never invalidate the radar-grid products.
    """
    root = Path(root)
    products_dir = root / "products"
    geo_dir = products_dir / "geotiff" / "geo"
    m = max(1, int(params.get("multilook", 2)))
    info: dict[str, Any] = {
        "schema": "insar-studio/geocode/v1",
        "status": "unavailable",
        "method": "range-doppler",
        "technique": technique,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reasons": [],
        "warnings": [],
        "files": [],
    }
    try:
        geometry, reasons, warnings = load_geometry(stack_meta, root)
        info["warnings"].extend(warnings)
        if geometry is None:
            info["reasons"] = reasons
            atomic_write_json(products_dir / "geocode.json", info)
            return info
        grid, notes = map_grid(
            stack_meta,
            geometry,
            dem,
            product_shape=product_shape,
            multilook=m,
            grid_meta=grid_meta,
        )
        info["warnings"].extend(notes)
        lookup = build_lookup(
            geometry,
            grid,
            dem,
            multilook=m,
            product_shape=product_shape,
        )
        info["grid"] = grid.to_dict()
        info["coverage"] = float(lookup["coverage"])
        info["valid_pixels"] = int(lookup["valid"].sum())
        info["dem_iterations"] = int(lookup["iterations"])
        info["max_height_update_m"] = float(lookup["max_height_update_m"])
        info["lookup_blocks"] = int(lookup.get("blocks", 1))
        info["lookup_block_rows"] = int(lookup.get("block_rows", grid.rows))
        info["lookup_workers"] = int(lookup.get("workers", 1))
        info["lookup_seconds"] = round(float(lookup.get("elapsed_s", 0.0)), 3)
        info["doppler_centroid_hz"] = float(geometry.doppler_centroid_hz)
        info["look_side"] = geometry.look_side
        coverage = float(lookup["coverage"])
        if coverage <= 0:
            info["status"] = "failed"
            info["reasons"] = ["地图网格与雷达覆盖范围没有交集"]
            atomic_write_json(products_dir / "geocode.json", info)
            return info
        info["status"] = "done" if coverage >= 0.5 else "partial"
        for name, data in products.items():
            array = np.asarray(data)
            if array.ndim != 2 or tuple(array.shape) != tuple(product_shape):
                info["warnings"].append(f"跳过 {name}：形状与反演网格不一致")
                continue
            sampled = sample_product(array, lookup)
            target = geo_dir / f"{name}.tif"
            write_geotiff(
                target,
                sampled,
                corner_lon=grid.corner_lon,
                corner_lat=grid.corner_lat,
                lon_step=grid.lon_step,
                lat_step=grid.lat_step,
            )
            info["files"].append(str(target.relative_to(root)).replace("\\", "/"))
        atomic_write_json(products_dir / "geocode.json", info)
        return info
    except Exception as exc:  # geocoding must not break a completed inversion
        info["status"] = "failed"
        info["reasons"] = [f"{type(exc).__name__}: {exc}"]
        try:
            atomic_write_json(products_dir / "geocode.json", info)
        except Exception:
            pass
        return info


def geocode_readiness(meta: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Return whether the stack metadata is sufficient for geocoding."""
    try:
        geometry, reasons, warnings = load_geometry(meta, project_root)
    except Exception as exc:
        return {
            "ready": False,
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
    dem_path = project_root / "data" / "stack" / "dem.npy"
    return {
        "ready": geometry is not None,
        "reasons": reasons,
        "warnings": warnings,
        "dem_available": dem_path.exists(),
        "orbit_source": (meta.get("orbit") or {}).get("source") if isinstance(meta.get("orbit"), dict) else None,
        "look_side": geometry.look_side if geometry else None,
    }
