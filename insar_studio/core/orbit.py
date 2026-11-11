"""Sentinel-1 orbit handling and perpendicular-baseline estimation.

The module deliberately keeps the orbit reader small and dependency-free.  It
accepts the Copernicus Sentinel-1 EOF files (precise ``POEORB`` or restituted
``RESORB``) and the minimal SAFE annotation fields needed to place each
acquisition on the orbit.  Baselines are computed in the cross-track plane of
the reference orbit, which is the same convention used by the network and DEM
error stages.

Only the orbit metadata is handled here.  Burst stitching, deramping and TOPS
coregistration remain outside this project; a stack must still be imported
already coregistered.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import numpy as np


class OrbitError(ValueError):
    """Raised when an orbit or annotation file cannot be interpreted."""


_ISO_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
)


def parse_time(value: str) -> datetime:
    """Parse the UTC timestamps used by Sentinel-1 products."""
    text = str(value).strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise OrbitError(f"无法解析时间: {value}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


@dataclass
class Orbit:
    """A time-ordered set of ECEF state vectors."""

    times: np.ndarray  # seconds relative to ``epoch``
    positions: np.ndarray  # metres, shape (N, 3)
    velocities: np.ndarray  # metres/second, shape (N, 3)
    epoch: datetime
    source: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        if self.times.ndim != 1 or self.positions.shape != (self.times.size, 3):
            raise OrbitError("轨道状态向量维度不一致")
        if self.velocities.shape != self.positions.shape:
            raise OrbitError("轨道位置与速度数量不一致")
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.positions = self.positions[order]
        self.velocities = self.velocities[order]
        if self.times.size < 2:
            raise OrbitError("轨道至少需要两个状态向量")

    @property
    def start(self) -> datetime:
        return self.epoch + timedelta(seconds=float(self.times[0]))

    @property
    def end(self) -> datetime:
        return self.epoch + timedelta(seconds=float(self.times[-1]))

    def covers(self, when: datetime) -> bool:
        seconds = (when - self.epoch).total_seconds()
        return bool(self.times[0] - 1e-6 <= seconds <= self.times[-1] + 1e-6)

    def state(self, when: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity with local barycentric polynomials.

        Sentinel-1 EOF files normally provide 60-second state vectors.  A
        degree-8 local polynomial keeps interpolation error well below a metre
        without requiring SciPy's spline objects.
        """
        seconds = (when - self.epoch).total_seconds()
        if seconds < self.times[0] - 1e-6 or seconds > self.times[-1] + 1e-6:
            raise OrbitError(
                f"时间 {_iso(when)} 超出轨道覆盖范围 "
                f"{_iso(self.start)} – {_iso(self.end)}"
            )
        index = int(np.searchsorted(self.times, seconds, side="right") - 1)
        index = max(0, min(index, self.times.size - 1))
        if abs(seconds - float(self.times[index])) <= 1e-6:
            return self.positions[index].copy(), self.velocities[index].copy()
        count = min(8, self.times.size)
        start = max(0, min(index - count // 2, self.times.size - count))
        nodes = self.times[start : start + count]
        values = np.concatenate(
            [self.positions[start : start + count], self.velocities[start : start + count]],
            axis=1,
        )
        # Barycentric weights for the local nodes.
        delta = nodes[:, None] - nodes[None, :]
        np.fill_diagonal(delta, 1.0)
        weights = 1.0 / np.prod(delta, axis=1)
        diff = seconds - nodes
        if np.any(np.abs(diff) <= 1e-9):
            exact = int(np.argmin(np.abs(diff)))
            interpolated = values[exact]
        else:
            ratio = weights / diff
            interpolated = ratio @ values / ratio.sum()
        return interpolated[:3].copy(), interpolated[3:].copy()

    def interpolate(
        self,
        seconds: np.ndarray | Iterable[float],
        *,
        chunk: int = 50_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised position/velocity interpolation for many query times.

        The same degree-8 barycentric polynomial as :meth:`state` is used.
        Query times outside the orbit are rejected rather than silently
        clamped, because a clamped geolocation would place a pixel on the
        wrong part of the orbit.
        """
        query = np.asarray(seconds, dtype=np.float64)
        shape = query.shape
        flat = query.ravel()
        if flat.size == 0:
            return np.empty(shape + (3,)), np.empty(shape + (3,))
        if not np.isfinite(flat).all():
            raise OrbitError("轨道查询时间含非有限值")
        if np.nanmin(flat) < self.times[0] - 1e-6 or np.nanmax(flat) > self.times[-1] + 1e-6:
            raise OrbitError(
                f"查询时间超出轨道覆盖范围 {_iso(self.start)} – {_iso(self.end)}"
            )
        count = min(8, self.times.size)
        windows = np.lib.stride_tricks.sliding_window_view(self.times, count)
        delta = windows[:, :, None] - windows[:, None, :]
        diagonal = np.arange(count)
        delta[:, diagonal, diagonal] = 1.0
        weights = 1.0 / np.prod(delta, axis=2)
        values_all = np.concatenate([self.positions, self.velocities], axis=1)
        n_windows = windows.shape[0]
        out = np.empty((flat.size, 6), dtype=np.float64)
        for begin in range(0, flat.size, max(1, int(chunk))):
            block = flat[begin : begin + max(1, int(chunk))]
            index = np.searchsorted(self.times, block, side="right") - 1
            index = np.clip(index - count // 2, 0, n_windows - 1)
            offsets = np.arange(count)
            nodes = self.times[index[:, None] + offsets]
            diff = block[:, None] - nodes
            exact = np.abs(diff) <= 1e-9
            safe_diff = np.where(exact, 1.0, diff)
            ratio = weights[index] / safe_diff
            values = values_all[index[:, None] + offsets]
            interpolated = np.einsum("nc,nck->nk", ratio, values) / ratio.sum(axis=1, keepdims=True)
            hit = exact.any(axis=1)
            if np.any(hit):
                nearest = np.argmin(np.abs(diff[hit]), axis=1)
                interpolated[hit] = values[hit, nearest]
            out[begin : begin + block.size] = interpolated
        return out[:, :3].reshape(shape + (3,)), out[:, 3:].reshape(shape + (3,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": _iso(self.epoch),
            "times": [float(value) for value in self.times],
            "positions": [[float(value) for value in row] for row in self.positions],
            "velocities": [[float(value) for value in row] for row in self.velocities],
            "kind": self.kind,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Orbit":
        try:
            return cls(
                times=np.asarray(data["times"], dtype=np.float64),
                positions=np.asarray(data["positions"], dtype=np.float64),
                velocities=np.asarray(data["velocities"], dtype=np.float64),
                epoch=parse_time(str(data["epoch"])),
                source=str(data.get("source", "")),
                kind=str(data.get("kind", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OrbitError("轨道状态向量记录不完整") from exc


def read_eof(path: Path) -> Orbit:
    """Read a Sentinel-1 EOF orbit file.

    The reader accepts both the ``SENTINEL-1A ORBIT`` header form and plain
    tables with an ISO timestamp followed by X, Y, Z, VX, VY, VZ columns.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OrbitError(f"无法读取轨道文件: {path}") from exc
    rows: list[tuple[datetime, list[float]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ISO_RE.search(stripped)
        if not match:
            continue
        tail = stripped[match.end() :]
        numbers = [float(v) for v in _NUMBER_RE.findall(tail)]
        if len(numbers) < 6:
            continue
        rows.append((parse_time(match.group(1)), numbers[:6]))
    if len(rows) < 2:
        raise OrbitError(f"轨道文件 {path.name} 中没有足够的状态向量")
    rows.sort(key=lambda item: item[0])
    epoch = rows[0][0]
    times = np.asarray([(stamp - epoch).total_seconds() for stamp, _ in rows])
    values = np.asarray([values for _, values in rows], dtype=np.float64)
    kind = "precise" if "POEORB" in path.name.upper() else (
        "restituted" if "RESORB" in path.name.upper() else "orbit"
    )
    return Orbit(
        times=times,
        positions=values[:, :3],
        velocities=values[:, 3:6],
        epoch=epoch,
        source=str(path),
        kind=kind,
    )


def read_orbit_directory(directory: Path | str) -> list[Orbit]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"轨道目录不存在: {directory}")
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in (".eof", ".orb", ".txt")
    )
    orbits: list[Orbit] = []
    for path in paths:
        try:
            orbits.append(read_eof(path))
        except OrbitError:
            continue
    if not orbits:
        raise OrbitError(f"轨道目录中没有可用的 EOF 文件: {directory}")
    return orbits


def select_orbit(orbits: Iterable[Orbit], when: datetime) -> Orbit:
    """Pick the best orbit covering ``when`` (precise before restituted)."""
    candidates = [orbit for orbit in orbits if orbit.covers(when)]
    if not candidates:
        raise OrbitError(f"没有轨道覆盖成像时间 {_iso(when)}")
    candidates.sort(key=lambda orbit: (orbit.kind != "precise", -orbit.times.size))
    return candidates[0]


def perpendicular_baselines(
    orbits: Iterable[Orbit],
    times: Iterable[datetime],
    *,
    ref_index: int = 0,
) -> list[float]:
    """Compute perpendicular baselines relative to one reference acquisition.

    The cross-track normal is taken from the reference orbit:
    ``n = (v_ref x r_ref) / |v_ref x r_ref|`` and ``B_perp = (r_i - r_ref) . n``.
    The sign therefore follows the satellite's flight direction, not the
    ascending/descending label.
    """
    orbits = list(orbits)
    stamps = list(times)
    if not stamps:
        return []
    ref_index = int(ref_index)
    if not 0 <= ref_index < len(stamps):
        ref_index = 0
    ref_orbit = select_orbit(orbits, stamps[ref_index])
    ref_position, ref_velocity = ref_orbit.state(stamps[ref_index])
    normal = np.cross(ref_velocity, ref_position)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 0:
        raise OrbitError("无法从参考轨道构造垂直基线法向量")
    normal /= norm
    out: list[float] = []
    for stamp in stamps:
        orbit = select_orbit(orbits, stamp)
        position, _ = orbit.state(stamp)
        out.append(float(np.dot(position - ref_position, normal)))
    return out


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_text(root: ElementTree.Element, names: set[str]) -> str:
    for element in root.iter():
        if _local_name(element.tag) in names and element.text:
            return element.text.strip()
    return ""


def read_safe_annotation(path: Path | str) -> dict[str, Any]:
    """Extract the fields needed for baseline and geometry from a SAFE XML."""
    path = Path(path)
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise OrbitError(f"无法读取 SAFE annotation: {path.name}") from exc

    def number(name: str, default: float | None = None) -> float | None:
        text = _first_text(root, {name})
        if not text:
            return default
        try:
            value = float(text)
        except ValueError:
            return default
        return value if math.isfinite(value) else default

    start_text = _first_text(root, {"productFirstLineUtcTime", "firstLineUtcTime"})
    start = parse_time(start_text) if start_text else None
    azimuth_interval = number("azimuthTimeInterval")
    range_time = number("slantRangeTime")
    range_spacing = number("rangePixelSpacing")
    azimuth_spacing = number("azimuthPixelSpacing")
    incidence = number("incidenceAngleMidSwath")
    orbit_number = number("orbitNumber")
    relative_orbit = number("relativeOrbitNumber")
    pass_name = _first_text(root, {"pass"}).upper()
    return {
        "start_time": start,
        "azimuth_time_interval": azimuth_interval,
        "slant_range_time_s": range_time,
        "range_spacing_m": range_spacing,
        "azimuth_spacing_m": azimuth_spacing,
        "incidence_deg": incidence,
        "orbit_number": int(orbit_number) if orbit_number is not None else None,
        "relative_orbit": int(relative_orbit) if relative_orbit is not None else None,
        "orbit_direction": pass_name or None,
        "source": str(path),
    }


def safe_annotation_for(measurement: Path) -> Path | None:
    """Find the annotation XML matching a measurement TIFF."""
    measurement = Path(measurement)
    safe = measurement.parent.parent
    annotation = safe / "annotation"
    if not annotation.exists():
        return None
    stem = measurement.stem.replace("-slc-", "-slc-")
    # Sentinel-1 names differ only by the .tiff extension and occasional
    # polarisation suffixes; prefer an exact stem match, then a date match.
    exact = annotation / f"{stem}.xml"
    if exact.exists():
        return exact
    date_match = re.search(r"(20\d{2})(\d{2})(\d{2})t", measurement.name, re.I)
    if not date_match:
        return None
    token = "".join(date_match.groups())
    candidates = sorted(annotation.glob(f"*{token}*.xml"))
    return candidates[0] if candidates else None


def collect_safe_metadata(measurement_paths: Iterable[Path]) -> dict[str, Any]:
    """Collect per-scene SAFE annotation values in input order."""
    starts: list[datetime | None] = []
    fields: dict[str, list[float | None]] = {
        "incidence_deg": [],
        "range_spacing_m": [],
        "azimuth_spacing_m": [],
        "azimuth_time_interval": [],
        "slant_range_time_s": [],
    }
    orbit_numbers: list[int | None] = []
    relative_orbits: list[int | None] = []
    directions: list[str | None] = []
    for measurement in measurement_paths:
        annotation = safe_annotation_for(measurement)
        if annotation is None:
            starts.append(None)
            for values in fields.values():
                values.append(None)
            orbit_numbers.append(None)
            relative_orbits.append(None)
            directions.append(None)
            continue
        try:
            info = read_safe_annotation(annotation)
        except OrbitError:
            starts.append(None)
            for values in fields.values():
                values.append(None)
            orbit_numbers.append(None)
            relative_orbits.append(None)
            directions.append(None)
            continue
        starts.append(info["start_time"])
        for key, values in fields.items():
            values.append(info.get(key))
        orbit_numbers.append(info.get("orbit_number"))
        relative_orbits.append(info.get("relative_orbit"))
        directions.append(info.get("orbit_direction"))
    return {
        "start_times": starts,
        "incidence_deg": fields["incidence_deg"],
        "range_spacing_m": fields["range_spacing_m"],
        "azimuth_spacing_m": fields["azimuth_spacing_m"],
        "azimuth_time_interval": fields["azimuth_time_interval"],
        "slant_range_time_s": fields["slant_range_time_s"],
        "orbit_number": orbit_numbers,
        "relative_orbit": relative_orbits,
        "orbit_direction": [value for value in directions if value],
    }


def _clean_list(values: Iterable[Any], n: int) -> list[Any]:
    out = list(values)
    if len(out) < n:
        out.extend([None] * (n - len(out)))
    return out[:n]


def estimate_bperp(
    orbit_dir: Path | str,
    dates: Iterable[str],
    start_times: Iterable[Any],
    *,
    ref_index: int = 0,
    n_rows: int | None = None,
    azimuth_time_interval: float | None = None,
    include_series: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    """Estimate per-date perpendicular baselines from local EOF files.

    ``start_times`` may contain ISO strings or ``datetime`` objects.  When
    ``n_rows`` and ``azimuth_time_interval`` are known, the scene-centre time
    is used; otherwise the first-line time is a good approximation for the
    slowly varying cross-track baseline.
    """
    dates = [str(value)[:10] for value in dates]
    n = len(dates)
    stamps: list[datetime] = []
    missing: list[int] = []
    for index, value in enumerate(_clean_list(start_times, n)):
        if value in (None, ""):
            missing.append(index)
            continue
        try:
            stamp = value if isinstance(value, datetime) else parse_time(str(value))
        except OrbitError:
            missing.append(index)
            continue
        if n_rows and azimuth_time_interval:
            stamp = stamp + timedelta(seconds=0.5 * int(n_rows) * float(azimuth_time_interval))
        stamps.append(stamp)
    if len(stamps) != n or missing:
        raise OrbitError(
            "缺少每景成像时间，无法由轨道估计垂直基线"
            + (f"（缺 {len(missing)} 景）" if missing else "")
        )
    orbits = read_orbit_directory(orbit_dir)
    baselines = perpendicular_baselines(orbits, stamps, ref_index=ref_index)
    used = [select_orbit(orbits, stamp) for stamp in stamps]
    info = {
        "source": str(orbit_dir),
        "n_orbits": len(orbits),
        "orbit_kind": sorted({orbit.kind for orbit in used}),
        "n_dates": n,
        "ref_index": int(ref_index),
        "bperp_min_m": float(np.min(baselines)) if baselines else 0.0,
        "bperp_max_m": float(np.max(baselines)) if baselines else 0.0,
    }
    if include_series:
        info["series"] = [orbit.to_dict() for orbit in used]
    return [float(value) for value in baselines], info
