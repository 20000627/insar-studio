"""GACOS atmospheric delay products: demo, official, local and online adapters.

The official GACOS service (http://www.gacos.net) is asynchronous.  A request is
submitted through its web form, the products arrive by email as ``tar.gz``
archives, and the user then imports either the email link or the downloaded
archive into the project.  This module implements that workflow without ever
persisting the recipient email address.  The legacy ``online`` adapter remains
available for deployments that expose a synchronous JSON API.

All imported products live under ``data/gacos/official`` inside the project.
Archives are extracted with explicit path-traversal, symlink, member-count and
uncompressed-size checks because they arrive from an external service.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import ipaddress
import math
import os
import re
import shutil
import socket
import stat
import tarfile
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import tifffile
from scipy import ndimage

from .geotiff import read_geotiff
from .io import load_stack
from .utils import atomic_write_json, read_json

ZTD_UNIT_HINT = "GACOS ZTD is converted to LOS delay in millimetres"

#: Official GACOS form and public constraints.
GACOS_FORM_URL = "http://www.gacos.net/M/action_page.php"
GACOS_HOME_URL = "http://www.gacos.net/"
GACOS_MAX_DATES = 20
GACOS_MAX_BATCHES = 3
GACOS_MAX_SPAN_DEG = 10.0
GACOS_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
GACOS_MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
GACOS_MAX_EXTRACT_BYTES = 2 * 1024 * 1024 * 1024
GACOS_MAX_MEMBERS = 4000
GACOS_MAX_IMPORT_FILES = 500
GACOS_TIMEOUT = 90
GACOS_DOWNLOAD_TIMEOUT = 300

GACOS_PRODUCT_SUFFIXES = {".tif", ".tiff", ".ztd", ".dat", ".bin", ".npy"}
GACOS_METADATA_SUFFIXES = {".rsc"}
GACOS_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

GACOS_FORMATS = {
    "geotiff": {
        "label": "GeoTIFF（推荐）",
        "type": "2",
        "note": "文件更小，带地理参考；GACOS 默认格式。",
    },
    "binary": {
        "label": "Binary grid",
        "type": "1",
        "note": "GACOS 二进制格网，需同时保留 .rsc 元数据。",
    },
}

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_DATE_DASH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_COMPACT_RE = re.compile(r"^\d{8}$")


class GacosError(RuntimeError):
    """Base class for user-facing GACOS failures."""


class GacosRequestError(GacosError):
    """The official request could not be validated or submitted."""


class GacosImportError(GacosError):
    """A product archive or download could not be imported safely."""


@dataclass
class DelayGrid:
    data_mm: np.ndarray
    bounds: tuple[float, float, float, float] | None = None
    source: str = ""


@dataclass
class GacosRequest:
    bbox: tuple[float, float, float, float]
    dates: list[str]
    email: str
    hour: int = 0
    minute: int = 0
    output_format: str = "geotiff"


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
    expected = rows * cols
    # The official product is little-endian 4-byte float.  Some third-party
    # mirrors ship big-endian data, so try both and prefer the plausible one
    # instead of silently interpreting denormals as a delay screen.
    candidates: list[tuple[int, np.ndarray, float]] = []
    for dtype in ("<f4", ">f4"):
        raw = np.fromfile(path, dtype=dtype)
        if raw.size < expected:
            continue
        data = raw[:expected].reshape(rows, cols).astype(np.float64)
        med = float(np.nanmedian(np.abs(data)))
        if np.isfinite(med) and med > 0:
            plausible = 0 if 0.05 <= med <= 1e5 else 1
            candidates.append((plausible, data, med))
    if not candidates:
        raise ValueError(f"GACOS 文件长度不足或数值无效: {path.name}")
    candidates.sort(key=lambda item: item[0])
    data, med = candidates[0][1], candidates[0][2]
    # GACOS values are usually metres; some portals ship millimetres.
    if med > 50:
        data = data / 1000.0
    data_mm = data * 1000.0
    bounds = None
    if all(k in rsc for k in ("x_first", "y_first", "x_step", "y_step")):
        x0 = float(rsc["x_first"])
        y0 = float(rsc["y_first"])
        dx = float(rsc["x_step"])
        dy = float(rsc["y_step"])
        west, east = sorted((x0, x0 + dx * cols))
        south, north = sorted((y0, y0 + dy * rows))
        bounds = (west, south, east, north)
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
        if data.ndim != 2:
            raise ValueError(f"GACOS GeoTIFF 必须是二维格网: {path.name}")
        med = float(np.nanmedian(np.abs(data)))
        if med < 10:  # metres
            data *= 1000.0
        bounds = None
        if all(k in geo for k in ("corner_lon", "corner_lat", "lon_step", "lat_step")):
            x0 = float(geo["corner_lon"])
            y0 = float(geo["corner_lat"])
            dx = float(geo["lon_step"])
            dy = float(geo["lat_step"])
            west, east = sorted((x0, x0 + dx * data.shape[1]))
            south, north = sorted((y0, y0 + dy * data.shape[0]))
            bounds = (west, south, east, north)
        return DelayGrid(data_mm=data, bounds=bounds, source=str(path))
    raise ValueError(f"不支持的 GACOS/大气产品格式: {path.suffix}")


def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return not (ae <= bw or aw >= be or an <= bs or as_ >= bn)


def _bounds_cover(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return aw <= bw and ae >= be and as_ <= bs and an >= bn


def _resample_grid(
    grid: DelayGrid,
    shape: tuple[int, int],
    target_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    data = grid.data_mm
    if data.shape == shape:
        return data.astype(np.float32)
    if grid.bounds and target_bounds:
        if not _bounds_intersect(grid.bounds, target_bounds):
            raise ValueError(
                "GACOS 产品覆盖范围与项目范围不重叠："
                f"产品 {tuple(round(v, 4) for v in grid.bounds)}，"
                f"项目 {tuple(round(v, 4) for v in target_bounds)}"
            )
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
        out = np.pad(
            out,
            ((0, max(0, shape[0] - out.shape[0])), (0, max(0, shape[1] - out.shape[1]))),
            mode="edge",
        )
    return out.astype(np.float32)


def _date_tokens(date_str: str) -> list[str]:
    d = str(date_str)[:10]
    return [d.replace("-", ""), d]


def find_local_screens(directory: Path, dates: Iterable[str]) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"GACOS 目录不存在: {directory}")
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in GACOS_PRODUCT_SUFFIXES
    ]
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


def _normalise_date(value: Any) -> str:
    text = str(value).strip()
    if _DATE_DASH_RE.fullmatch(text) or _DATE_COMPACT_RE.fullmatch(text):
        try:
            return datetime.strptime(text, "%Y-%m-%d" if "-" in text else "%Y%m%d").date().isoformat()
        except ValueError as exc:
            raise GacosRequestError(f"日期无效: {value}") from exc
    raise GacosRequestError(f"日期格式无效: {value}（应为 YYYY-MM-DD 或 YYYYMMDD）")


def _normalise_dates(dates: Any) -> list[str]:
    if isinstance(dates, str):
        parts = [p for p in re.split(r"[,;\s]+", dates.strip()) if p]
    elif isinstance(dates, (list, tuple, set)):
        parts = list(dates)
    else:
        raise GacosRequestError("dates 必须是日期数组或逗号/空白分隔的字符串")
    if not parts:
        raise GacosRequestError("至少需要一个日期")
    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        d = _normalise_date(item)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return sorted(out)


def _normalise_bbox(bbox: Any) -> tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        lower = {str(k).lower(): v for k, v in bbox.items()}
        try:
            bbox = [
                lower.get("w", lower.get("west")),
                lower.get("s", lower.get("south")),
                lower.get("e", lower.get("east")),
                lower.get("n", lower.get("north")),
            ]
        except Exception as exc:
            raise GacosRequestError("bbox 必须包含 W/S/E/N 四个数值") from exc
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise GacosRequestError("bbox 必须是 [W, S, E, N] 四个数值")
    try:
        west, south, east, north = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise GacosRequestError("bbox 必须是 [W, S, E, N] 四个数值") from exc
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise GacosRequestError("bbox 必须是有限数值")
    if not (-180.0 <= west < east <= 180.0):
        raise GacosRequestError("经度范围必须是 -180 ≤ W < E ≤ 180，且不支持跨 180° 经线")
    if not (-90.0 <= south < north <= 90.0):
        raise GacosRequestError("纬度范围必须是 -90 ≤ S < N ≤ 90")
    if east - west > GACOS_MAX_SPAN_DEG + 1e-9 or north - south > GACOS_MAX_SPAN_DEG + 1e-9:
        raise GacosRequestError(
            f"GACOS 官方服务单次范围最大 {GACOS_MAX_SPAN_DEG:g}° × {GACOS_MAX_SPAN_DEG:g}°，"
            f"当前为 {east - west:.3f}° × {north - south:.3f}°"
        )
    return (west, south, east, north)


def validate_gacos_request(
    *,
    bbox: Any,
    dates: Any,
    email: Any,
    hour: Any = 0,
    minute: Any = 0,
    output_format: Any = "geotiff",
    max_dates: int | None = GACOS_MAX_DATES,
) -> GacosRequest:
    """Validate one official request.

    ``max_dates=None`` lets the caller submit more than one batch; the caller is
    then responsible for splitting the dates.  Direct single-request callers
    keep the official 20-date limit.
    """
    bbox_tuple = _normalise_bbox(bbox)
    date_list = _normalise_dates(dates)
    if max_dates is not None and len(date_list) > max_dates:
        raise GacosRequestError(
            f"GACOS 官方服务单次最多 {max_dates} 个日期，当前 {len(date_list)} 个；"
            "请拆分为多批申请"
        )
    email_text = str(email or "").strip()
    if len(email_text) > 254 or not _EMAIL_RE.fullmatch(email_text):
        raise GacosRequestError("请填写有效的接收邮箱")
    if "\r" in email_text or "\n" in email_text:
        raise GacosRequestError("邮箱不能包含换行")
    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except (TypeError, ValueError) as exc:
        raise GacosRequestError("UTC 时和分必须是整数") from exc
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        raise GacosRequestError("UTC 时必须在 0–23，分必须在 0–59")
    fmt = str(output_format or "geotiff").lower()
    if fmt not in GACOS_FORMATS:
        raise GacosRequestError("输出格式必须是 geotiff 或 binary")
    return GacosRequest(
        bbox=bbox_tuple,
        dates=date_list,
        email=email_text,
        hour=hour_int,
        minute=minute_int,
        output_format=fmt,
    )


def mask_email(email: str) -> str:
    local, sep, domain = str(email or "").partition("@")
    if not sep or not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def redact_url(url: str) -> str:
    """Drop query strings and fragments, which may contain signed tokens."""
    try:
        parsed = urlparse(str(url))
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" + ("?***" if parsed.query else "")
    except Exception:
        pass
    return "***"


def _format_coord(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _extract_html_message(text: str) -> str:
    for pattern in (
        r'<h3[^>]*class="[^"]*h31[^"]*"[^>]*>(.*?)</h3>',
        r"<h3[^>]*>(.*?)</h3>",
    ):
        match = re.search(pattern, text, re.S | re.I)
        if match:
            plain = re.sub(r"<[^>]+>", " ", match.group(1))
            plain = html_lib.unescape(plain)
            return re.sub(r"\s+", " ", plain).strip()
    return ""


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "InSAR-Studio/1.0 (local research workstation; GACOS import)",
            "Referer": GACOS_HOME_URL,
        }
    )
    return session


class GacosOfficialClient:
    """Submit one GACOS request through the official web form.

    The client deliberately accepts an injected ``requests.Session`` so tests
    never touch the real service.  It parses the result page instead of trying
    to download anything: GACOS always answers asynchronously by email.
    """

    def __init__(
        self,
        *,
        form_url: str = GACOS_FORM_URL,
        timeout: int = GACOS_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.form_url = str(form_url)
        self.timeout = max(10, int(timeout))
        self.session = session or _session()

    def submit(self, request: GacosRequest) -> dict[str, Any]:
        data = {
            "N": _format_coord(request.bbox[3]),
            "W": _format_coord(request.bbox[0]),
            "E": _format_coord(request.bbox[2]),
            "S": _format_coord(request.bbox[1]),
            "H": str(request.hour),
            "M": str(request.minute),
            "date": "\n".join(d.replace("-", "") for d in request.dates),
            "type": GACOS_FORMATS[request.output_format]["type"],
            "email": request.email,
        }
        try:
            response = self.session.post(
                self.form_url,
                data=data,
                headers={"Referer": GACOS_HOME_URL},
                timeout=(10, self.timeout),
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise GacosRequestError(
                "无法连接 GACOS 官方服务（http://www.gacos.net），请检查网络，"
                "或改用官网手动申请后导入 tar.gz"
            ) from exc
        if response.status_code >= 400:
            raise GacosRequestError(f"GACOS 官方服务返回 HTTP {response.status_code}")
        text = response.text or ""
        message = _extract_html_message(text)
        lower = text.lower()
        if (
            "thanks for using gacos" in lower
            or "we have received your request" in lower
            or "flag=mq==" in str(getattr(response, "url", "")).lower()
        ):
            return {
                "ok": True,
                "message": message or "Thanks for using GACOS!",
                "final_url": redact_url(str(getattr(response, "url", "") or self.form_url)),
                "http_status": int(response.status_code),
                "n_dates": len(request.dates),
            }
        if message:
            raise GacosRequestError(f"GACOS 官方服务拒绝了申请：{message}")
        raise GacosRequestError(
            "GACOS 官方服务返回了无法识别的响应；请稍后重试，或打开官网手动申请"
        )


def submit_official_request(
    *,
    bbox: Any,
    dates: Any,
    email: Any,
    hour: Any = 0,
    minute: Any = 0,
    output_format: Any = "geotiff",
    allow_split: bool = False,
    timeout: int = GACOS_TIMEOUT,
    client: GacosOfficialClient | None = None,
    delay_seconds: float = 2.0,
    max_batches: int = GACOS_MAX_BATCHES,
) -> dict[str, Any]:
    """Validate and submit an official request, optionally in 20-date batches."""
    request = validate_gacos_request(
        bbox=bbox,
        dates=dates,
        email=email,
        hour=hour,
        minute=minute,
        output_format=output_format,
        max_dates=None,
    )
    chunks = [request.dates[i : i + GACOS_MAX_DATES] for i in range(0, len(request.dates), GACOS_MAX_DATES)]
    if len(chunks) > max_batches:
        raise GacosRequestError(
            f"当前 {len(request.dates)} 个日期需要 {len(chunks)} 批申请，超过单次最多 {max_batches} 批；"
            "请减少日期或分多次处理"
        )
    if len(chunks) > 1 and not allow_split:
        raise GacosRequestError(
            f"官方服务单次最多 {GACOS_MAX_DATES} 个日期；当前 {len(request.dates)} 个，"
            "请确认允许拆分为多批申请"
        )
    submitter = client or GacosOfficialClient(timeout=timeout)
    batches: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        batch_request = replace(request, dates=chunk)
        try:
            result = submitter.submit(batch_request)
            batches.append(
                {
                    "ok": True,
                    "n_dates": len(chunk),
                    "dates": chunk,
                    "message": str(result.get("message", "")),
                    "final_url": redact_url(str(result.get("final_url", ""))),
                }
            )
        except GacosRequestError as exc:
            batches.append(
                {
                    "ok": False,
                    "n_dates": len(chunk),
                    "dates": chunk,
                    "error": str(exc),
                }
            )
        if index < len(chunks) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)
    ok_count = sum(1 for batch in batches if batch["ok"])
    errors = [batch["error"] for batch in batches if not batch["ok"]]
    if ok_count == len(batches):
        message = (
            f"已提交 {ok_count} 个 GACOS 申请，请查收邮箱。"
            "官方处理可能需要数分钟到数小时；下载链接或 tar.gz 到手后再导入。"
        )
    elif ok_count:
        message = f"已提交 {ok_count}/{len(batches)} 个申请；失败批次：{'；'.join(errors)}"
    else:
        message = f"所有 GACOS 申请均失败：{'；'.join(errors)}"
    return {
        "ok": ok_count == len(batches),
        "partial": 0 < ok_count < len(batches),
        "message": message,
        "error": "" if ok_count == len(batches) else "；".join(errors),
        "n_dates": len(request.dates),
        "n_batches": len(batches),
        "batches": batches,
        "email_masked": mask_email(request.email),
        "request": {
            "bbox": list(request.bbox),
            "dates": request.dates,
            "hour": request.hour,
            "minute": request.minute,
            "output_format": request.output_format,
        },
    }


# ---------------------------------------------------------------------------
# Official product import
# ---------------------------------------------------------------------------


def _safe_member_path(name: str) -> Path:
    text = str(name).replace("\\", "/")
    if "\x00" in text:
        raise GacosImportError("压缩包包含非法文件名")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise GacosImportError(f"压缩包包含路径穿越条目: {name}")
    parts = [part for part in pure.parts if part not in ("", ".")]
    if not parts:
        raise GacosImportError(f"压缩包包含空文件名: {name}")
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise GacosImportError(f"压缩包包含绝对路径: {name}")
    return Path(*parts)


def _ensure_under(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != base_resolved and base_resolved not in candidate_resolved.parents:
        raise GacosImportError("压缩包条目越出临时目录")
    return candidate_resolved


def _detect_source_kind(path: Path) -> str:
    """Detect archives by magic first, then fall back to the extension."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        head = b""
    if head[:2] == b"\x1f\x8b":
        return "tar"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    if head[:6] == b"\x93NUMPY":
        return "npy"
    suffix = path.suffix.lower()
    if suffix in (".tar.gz", ".tgz", ".tar"):
        return "tar"
    if suffix == ".zip":
        return "zip"
    if suffix == ".ztd":
        return "ztd"
    if suffix in (".dat", ".bin"):
        return "ztd"
    if suffix in (".tif", ".tiff"):
        return "tif"
    if suffix == ".npy":
        return "npy"
    return "unknown"


def _extract_tar(archive: Path, destination: Path) -> None:
    total = 0
    count = 0
    try:
        with tarfile.open(archive, mode="r:*") as tf:
            for member in tf:
                count += 1
                if count > GACOS_MAX_MEMBERS:
                    raise GacosImportError(f"压缩包条目超过 {GACOS_MAX_MEMBERS} 个")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise GacosImportError(f"压缩包包含不允许的条目类型: {member.name}")
                rel = _safe_member_path(member.name)
                suffix = rel.suffix.lower()
                if suffix not in GACOS_PRODUCT_SUFFIXES | GACOS_METADATA_SUFFIXES:
                    continue
                total += int(member.size)
                if total > GACOS_MAX_EXTRACT_BYTES:
                    raise GacosImportError("压缩包展开后超过 2 GiB 上限")
                target = _ensure_under(destination, destination / rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    raise GacosImportError(f"无法读取压缩包条目: {member.name}")
                with source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
    except (tarfile.TarError, EOFError, OSError) as exc:
        if isinstance(exc, GacosImportError):
            raise
        raise GacosImportError(f"无法解压 GACOS tar 包: {exc}") from exc


def _extract_zip(archive: Path, destination: Path) -> None:
    total = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            if len(infos) > GACOS_MAX_MEMBERS:
                raise GacosImportError(f"压缩包条目超过 {GACOS_MAX_MEMBERS} 个")
            for info in infos:
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise GacosImportError(f"压缩包包含符号链接: {info.filename}")
                if info.flag_bits & 0x1:
                    raise GacosImportError("不支持加密的 GACOS zip 包")
                rel = _safe_member_path(info.filename)
                suffix = rel.suffix.lower()
                if suffix not in GACOS_PRODUCT_SUFFIXES | GACOS_METADATA_SUFFIXES:
                    continue
                total += int(info.file_size)
                if total > GACOS_MAX_EXTRACT_BYTES:
                    raise GacosImportError("压缩包展开后超过 2 GiB 上限")
                target = _ensure_under(destination, destination / rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
    except (zipfile.BadZipFile, OSError) as exc:
        raise GacosImportError(f"无法解压 GACOS zip 包: {exc}") from exc


def _is_supported_source(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in GACOS_PRODUCT_SUFFIXES | GACOS_METADATA_SUFFIXES:
        return True
    lowered = path.name.lower()
    return lowered.endswith(GACOS_ARCHIVE_SUFFIXES)


def _probe_product(path: Path) -> None:
    """Cheaply validate a product before copying it into the project."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                if len(page.shape) != 2 or min(page.shape) < 1:
                    raise ValueError("不是二维单波段格网")
                if page.dtype is None or not np.issubdtype(np.dtype(page.dtype), np.number):
                    raise ValueError("像元不是数值类型")
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError(f"无法读取 GeoTIFF: {exc}") from exc
        return
    if suffix in (".ztd", ".dat", ".bin"):
        rsc = _parse_rsc(path.with_suffix(".rsc"))
        if not rsc:
            rsc = _parse_rsc(path.with_suffix(".rsc.txt"))
        rows = int(float(rsc.get("file_length", rsc.get("lines", 0)) or 0))
        cols = int(float(rsc.get("width", rsc.get("samples", 0)) or 0))
        if rows <= 0 or cols <= 0:
            raise ValueError("缺少可用的 .rsc 行列数")
        if path.stat().st_size < rows * cols * 4:
            raise ValueError("文件长度小于 .rsc 声明的格网大小")
        return
    if suffix == ".npy":
        try:
            data = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取 .npy: {exc}") from exc
        if data.ndim != 2 or min(data.shape) < 1:
            raise ValueError(".npy 必须是二维格网")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(".npy 像元不是数值类型")
        return
    if suffix in GACOS_METADATA_SUFFIXES:
        return
    raise ValueError(f"不支持的格式: {path.suffix}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + f".part-{uuid.uuid4().hex[:8]}")
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _merge_product_files(
    root: Path,
    files: Iterable[Path],
    *,
    source_label: str,
    archive: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    candidates = [Path(p) for p in files if Path(p).is_file()]
    products = [p for p in candidates if p.suffix.lower() in GACOS_PRODUCT_SUFFIXES]
    metadata = [p for p in candidates if p.suffix.lower() in GACOS_METADATA_SUFFIXES]
    if not products:
        raise GacosImportError("压缩包或目录中没有 GACOS 产品文件（.tif/.ztd/.npy/.dat/.bin）")
    warnings: list[str] = []
    valid: list[Path] = []
    for path in products:
        try:
            _probe_product(path)
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")
        else:
            valid.append(path)
    if not valid:
        raise GacosImportError("没有可读取的 GACOS 产品文件；" + "；".join(warnings[:4]))
    product_dir = root / "data" / "gacos" / "official" / "products"
    product_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    changed = False
    for source in valid + metadata:
        destination = product_dir / source.name
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            copied.append(destination)
            continue
        if destination.exists():
            warnings.append(f"{source.name}: 同名文件已覆盖")
        _atomic_copy(source, destination)
        copied.append(destination)
        changed = True
    stack_meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    dates = [str(d)[:10] for d in (stack_meta.get("dates") or [])]
    found = find_local_screens(product_dir, dates) if dates else {}
    manifest_path = root / "data" / "gacos" / "official" / "manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "source": source_label,
        "archive": archive.name if archive else None,
        "archive_sha256": _sha256(archive) if archive and archive.is_file() else None,
        "n_files": len(copied),
        "dates": sorted(found),
        "warnings": warnings,
        "changed": changed,
        "created_at": now,
    }
    imports = list(manifest.get("imports") or [])
    imports.append(record)
    manifest.update(
        {
            "source": "official",
            # Re-importing byte-identical files must not invalidate an existing
            # atmosphere product; only genuinely new products bump updated_at.
            "updated_at": now if (changed or not manifest.get("updated_at")) else manifest.get("updated_at"),
            "imports": imports[-20:],
            "files": [
                {"name": p.name, "size": p.stat().st_size}
                for p in sorted(product_dir.iterdir())
                if p.is_file()
            ],
        }
    )
    atomic_write_json(manifest_path, manifest)
    if not found:
        warnings.append("未从文件名识别出项目日期；请确认产品文件名包含 YYYYMMDD 或 YYYY-MM-DD")
    return {
        "ok": True,
        "source": source_label,
        "n_files": len(copied),
        "dates": sorted(found),
        "warnings": warnings,
        "manifest": str(manifest_path),
    }


def _import_single_source(root: Path, source: Path) -> dict[str, Any]:
    source = Path(source)
    if not source.is_file():
        raise GacosImportError(f"不是可导入的文件: {source}")
    kind = _detect_source_kind(source)
    if kind in ("tar", "zip"):
        with tempfile.TemporaryDirectory(prefix="gacos_extract_", dir=str(_cache_root(root))) as temp:
            temp_dir = Path(temp)
            if kind == "tar":
                _extract_tar(source, temp_dir)
            else:
                _extract_zip(source, temp_dir)
            files = [p for p in temp_dir.rglob("*") if p.is_file()]
            if not files:
                raise GacosImportError("压缩包中没有可导入的 GACOS 产品文件")
            return _merge_product_files(
                root,
                files,
                source_label=f"压缩包 {source.name}",
                archive=source,
            )
    suffix = source.suffix.lower()
    if kind in ("tif", "npy", "ztd") or suffix in GACOS_PRODUCT_SUFFIXES:
        if suffix not in GACOS_PRODUCT_SUFFIXES:
            raise GacosImportError(
                f"产品文件缺少可识别的扩展名: {source.name}；请保留 .tif/.tiff/.ztd/.npy 后缀"
            )
        return _merge_product_files(root, [source], source_label=f"文件 {source.name}")
    if suffix in GACOS_METADATA_SUFFIXES:
        raise GacosImportError(f"{source.name} 只是元数据，需要同时提供 .ztd/.tif/.npy 产品")
    raise GacosImportError(f"无法识别的 GACOS 文件格式: {source.name}")


def import_gacos_path(root: Path, source: Path) -> dict[str, Any]:
    """Import a local archive, product file, or directory of GACOS products."""
    root = Path(root)
    source = Path(source).expanduser()
    if not source.exists():
        raise GacosImportError(f"GACOS 本地路径不存在: {source}")
    if source.is_dir():
        direct = [
            p
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in GACOS_PRODUCT_SUFFIXES | GACOS_METADATA_SUFFIXES
        ]
        if direct:
            return _merge_product_files(root, direct[:GACOS_MAX_IMPORT_FILES], source_label=f"目录 {source.name}")
        archives = [p for p in source.iterdir() if p.is_file() and _is_supported_source(p)]
        if not archives:
            raise GacosImportError("目录中没有 GACOS 产品文件或 tar.gz/zip 压缩包")
        results = [_import_single_source(root, p) for p in archives[:GACOS_MAX_IMPORT_FILES]]
        return _combine_import_results(results, source_label=f"目录 {source.name}")
    return _import_single_source(root, source)


def _combine_import_results(results: list[dict[str, Any]], *, source_label: str) -> dict[str, Any]:
    dates: set[str] = set()
    warnings: list[str] = []
    n_files = 0
    for result in results:
        dates.update(result.get("dates") or [])
        warnings.extend(result.get("warnings") or [])
        n_files += int(result.get("n_files") or 0)
    return {
        "ok": True,
        "source": source_label,
        "n_files": n_files,
        "dates": sorted(dates),
        "warnings": warnings,
        "manifest": str(results[-1].get("manifest", "")) if results else "",
    }


def _cache_root(root: Path) -> Path:
    path = Path(root) / "data" / "gacos" / ".tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_public_host(hostname: str, port: int | None, scheme: str) -> None:
    host = str(hostname or "").strip().lower().rstrip(".")
    if not host:
        raise GacosImportError("下载地址缺少主机名")
    if host == "localhost" or host.endswith(".localhost"):
        raise GacosImportError("下载地址不能指向本机")
    try:
        infos = socket.getaddrinfo(
            host,
            port or (443 if scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise GacosImportError(f"无法解析下载地址主机: {host}") from exc
    for info in infos:
        raw_ip = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            raise GacosImportError(f"下载地址解析到无效 IP: {raw_ip}")
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise GacosImportError(f"下载地址解析到非公网地址，已拒绝: {host}")


def validate_download_url(url: str) -> str:
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https"):
        raise GacosImportError("下载地址必须以 http:// 或 https:// 开头")
    if not parsed.hostname:
        raise GacosImportError("下载地址缺少主机名")
    _ensure_public_host(parsed.hostname, parsed.port, parsed.scheme)
    return parsed.geturl()


def _safe_download(
    url: str,
    destination: Path,
    *,
    timeout: int = GACOS_DOWNLOAD_TIMEOUT,
    max_bytes: int = GACOS_MAX_DOWNLOAD_BYTES,
    session: requests.Session | None = None,
) -> tuple[str, str]:
    session = session or _session()
    current = validate_download_url(url)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(6):
        try:
            response = session.get(
                current,
                stream=True,
                allow_redirects=False,
                timeout=(10, max(10, int(timeout))),
            )
        except requests.RequestException as exc:
            raise GacosImportError(f"下载 GACOS 产品失败: {exc}") from exc
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise GacosImportError("下载地址返回重定向，但缺少 Location")
                current = validate_download_url(urljoin(current, location))
                continue
            if response.status_code != 200:
                raise GacosImportError(f"下载 GACOS 产品失败: HTTP {response.status_code}")
            length = int(response.headers.get("content-length") or 0)
            if length > max_bytes:
                raise GacosImportError(f"GACOS 产品超过下载上限 {max_bytes // (1024 * 1024)} MiB")
            temp = destination.with_name(destination.name + ".part")
            done = 0
            try:
                with open(temp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        done += len(chunk)
                        if done > max_bytes:
                            raise GacosImportError(
                                f"GACOS 产品超过下载上限 {max_bytes // (1024 * 1024)} MiB"
                            )
                        fh.write(chunk)
                os.replace(temp, destination)
            finally:
                if temp.exists():
                    temp.unlink(missing_ok=True)
            return current, str(response.headers.get("content-type", ""))
        finally:
            response.close()
    raise GacosImportError("下载地址重定向次数过多")


def _is_html_file(path: Path) -> bool:
    try:
        head = Path(path).read_bytes()[:2048].lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<head" in head[:800]


def _download_links_from_html(text: str, base_url: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", text, re.I):
        links.append(html_lib.unescape(match.group(1)).strip())
    for match in re.finditer(r"""https?://[^\s"'<>]+""", text, re.I):
        links.append(html_lib.unescape(match.group(0)).strip())
    out: list[str] = []
    seen: set[str] = set()
    for raw in links:
        if not raw:
            continue
        link = urljoin(base_url, raw)
        if urlparse(link).scheme not in ("http", "https"):
            continue
        lowered = link.lower()
        if not any(
            lowered.endswith(suffix) or suffix in lowered
            for suffix in (".tar.gz", ".tgz", ".zip", ".tif", ".tiff", ".ztd", ".npy", ".dat", ".bin")
        ):
            continue
        if link not in seen:
            seen.add(link)
            out.append(link)
        if len(out) >= 20:
            break
    return out


def _import_downloaded_file(
    root: Path,
    downloaded: Path,
    *,
    final_url: str,
) -> dict[str, Any]:
    kind = _detect_source_kind(downloaded)
    if kind == "unknown":
        suffix = Path(urlparse(final_url).path).suffix.lower()
        if suffix in GACOS_PRODUCT_SUFFIXES:
            kind = suffix.lstrip(".")
    if kind in ("tar", "zip"):
        with tempfile.TemporaryDirectory(prefix="gacos_download_", dir=str(_cache_root(root))) as temp:
            temp_dir = Path(temp)
            if kind == "tar":
                _extract_tar(downloaded, temp_dir)
            else:
                _extract_zip(downloaded, temp_dir)
            files = [p for p in temp_dir.rglob("*") if p.is_file()]
            if not files:
                raise GacosImportError("下载的压缩包中没有 GACOS 产品文件")
            return _merge_product_files(
                root,
                files,
                source_label=f"下载 {redact_url(final_url)}",
                archive=downloaded,
            )
    if kind in ("tif", "npy", "ztd") or downloaded.suffix.lower() in GACOS_PRODUCT_SUFFIXES:
        suffix = Path(urlparse(final_url).path).suffix.lower()
        if suffix not in GACOS_PRODUCT_SUFFIXES:
            suffix = {"tif": ".tif", "npy": ".npy", "ztd": ".ztd"}.get(kind, downloaded.suffix.lower())
        if downloaded.suffix.lower() != suffix:
            renamed = downloaded.with_name(downloaded.stem + suffix)
            os.replace(downloaded, renamed)
            downloaded = renamed
        return _merge_product_files(
            root,
            [downloaded],
            source_label=f"下载 {redact_url(final_url)}",
        )
    raise GacosImportError(
        "下载内容不是可识别的 GACOS 产品；请在浏览器中打开邮件链接，下载 tar.gz 后再导入"
    )


def import_gacos_url(
    root: Path,
    url: str,
    *,
    timeout: int = GACOS_DOWNLOAD_TIMEOUT,
    session: requests.Session | None = None,
    max_bytes: int = GACOS_MAX_DOWNLOAD_BYTES,
    _depth: int = 0,
) -> dict[str, Any]:
    """Download and import a product URL, or follow an HTML link page."""
    if _depth > 1:
        raise GacosImportError("下载链接嵌套层级过深")
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="gacos_url_", dir=str(_cache_root(root))) as temp:
        downloaded = Path(temp) / "download.bin"
        final_url, _content_type = _safe_download(
            url,
            downloaded,
            timeout=timeout,
            max_bytes=max_bytes,
            session=session,
        )
        if _is_html_file(downloaded):
            text = downloaded.read_text(encoding="utf-8", errors="replace")
            links = _download_links_from_html(text, final_url)
            if not links:
                raise GacosImportError(
                    "下载链接返回的是网页，未找到 tar.gz/GeoTIFF 直链；"
                    "请在浏览器中打开邮件链接并下载，然后导入本地文件"
                )
            results: list[dict[str, Any]] = []
            errors: list[str] = []
            for link in links[:10]:
                try:
                    results.append(
                        import_gacos_url(
                            root,
                            link,
                            timeout=timeout,
                            session=session,
                            max_bytes=max_bytes,
                            _depth=_depth + 1,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{redact_url(link)}: {exc}")
            if not results:
                raise GacosImportError("邮件页面中的下载链接均失败；" + "；".join(errors[:4]))
            combined = _combine_import_results(results, source_label=f"邮件页面 {redact_url(final_url)}")
            if errors:
                combined.setdefault("warnings", []).extend(errors)
            return combined
        return _import_downloaded_file(root, downloaded, final_url=final_url)


def import_gacos_sources(
    root: Path,
    *,
    urls: Iterable[str] | None = None,
    paths: Iterable[Path | str] | None = None,
    timeout: int = GACOS_DOWNLOAD_TIMEOUT,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Import several URLs/paths, continuing after individual failures."""
    imported: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for url in [str(u).strip() for u in (urls or []) if str(u).strip()]:
        try:
            imported.append(import_gacos_url(root, url, timeout=timeout, session=session))
        except Exception as exc:
            errors.append({"source": redact_url(url), "error": str(exc)})
    for path in [Path(p) for p in (paths or [])]:
        try:
            imported.append(import_gacos_path(root, path))
        except Exception as exc:
            errors.append({"source": str(path), "error": str(exc)})
    return {
        "ok": bool(imported) and not errors,
        "imported": imported,
        "errors": errors,
        "status": gacos_official_status(root),
    }


# ---------------------------------------------------------------------------
# Status and request defaults
# ---------------------------------------------------------------------------


def _stack_dates(meta: dict[str, Any]) -> list[str]:
    return [str(d)[:10] for d in (meta.get("dates") or []) if str(d).strip()]


def _bbox_from_meta(meta: dict[str, Any]) -> tuple[list[float] | None, str]:
    try:
        rows = int(meta.get("rows") or 0)
        cols = int(meta.get("cols") or 0)
        west0 = float(meta["corner_lon"])
        north0 = float(meta["corner_lat"])
        dx = float(meta["lon_step"])
        dy = float(meta["lat_step"])
    except (KeyError, TypeError, ValueError):
        return None, "数据元数据缺少 corner_lon/corner_lat/lon_step/lat_step"
    if rows <= 0 or cols <= 0 or dx == 0 or dy == 0:
        return None, "数据行列数或像元间距无效"
    if abs(west0) < 1e-9 and abs(north0) < 1e-9 and not bool(meta.get("geo_referenced")):
        return None, "数据未标记地理参考，角点为 (0, 0)"
    east0 = west0 + cols * dx
    south0 = north0 + rows * dy
    west, east = sorted((west0, east0))
    south, north = sorted((south0, north0))
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        return None, "数据角点超出 WGS-84 合法范围"
    crs = str(meta.get("crs") or "EPSG:4326").upper()
    if "4326" not in crs:
        return None, f"数据坐标系为 {crs}，GACOS 只接受 WGS-84 经纬度"
    return [west, south, east, north], "data/stack/meta.json"


def _time_from_meta(meta: dict[str, Any]) -> tuple[int, int, str, int]:
    minutes: list[int] = []
    for value in meta.get("start_times") or []:
        match = re.search(r"T(\d{2}):(\d{2})", str(value))
        if match:
            minutes.append(int(match.group(1)) * 60 + int(match.group(2)))
    if not minutes:
        return 0, 0, "default", 0
    minutes.sort()
    median = minutes[len(minutes) // 2]
    return median // 60, median % 60, "start_times", len(minutes)


def gacos_request_defaults(
    root: Path,
    *,
    dates: list[str] | None = None,
    stack_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    meta = stack_meta if stack_meta is not None else (read_json(root / "data" / "stack" / "meta.json", {}) or {})
    date_list = list(dates) if dates is not None else _stack_dates(meta)
    bbox, bbox_source = _bbox_from_meta(meta)
    hour, minute, time_source, _time_count = _time_from_meta(meta)
    warnings: list[str] = []
    if bbox is None:
        warnings.append(f"无法自动确定申请范围：{bbox_source}；请在申请时手动填写 W/S/E/N")
    else:
        span_w = bbox[2] - bbox[0]
        span_h = bbox[3] - bbox[1]
        if span_w > GACOS_MAX_SPAN_DEG or span_h > GACOS_MAX_SPAN_DEG:
            warnings.append(
                f"项目范围 {span_w:.2f}° × {span_h:.2f}° 超过官方单次 {GACOS_MAX_SPAN_DEG:g}° 上限；"
                "请缩小范围或分块申请"
            )
    if not date_list:
        warnings.append("项目没有日期列表，无法提交 GACOS 申请")
    n_batches = math.ceil(len(date_list) / GACOS_MAX_DATES) if date_list else 0
    if n_batches > 1:
        warnings.append(
            f"官方单次最多 {GACOS_MAX_DATES} 个日期；当前 {len(date_list)} 景将拆成 {n_batches} 次申请"
        )
    if n_batches > GACOS_MAX_BATCHES:
        warnings.append(
            f"当前日期需要 {n_batches} 批，超过最多 {GACOS_MAX_BATCHES} 批；请减少日期或分多次处理"
        )
    if time_source == "default":
        warnings.append("未找到成像时刻 start_times，默认 00:00 UTC；提交前可手动修改")
    times = []
    for value in meta.get("start_times") or []:
        match = re.search(r"T(\d{2}):(\d{2})", str(value))
        if match:
            times.append(int(match.group(1)) * 60 + int(match.group(2)))
    if times and max(times) - min(times) > 120:
        warnings.append("成像时刻在栈内相差超过 2 小时，请确认是否属于同一轨道")
    return {
        "dates": date_list,
        "bbox": bbox,
        "bbox_source": bbox_source,
        "hour": hour,
        "minute": minute,
        "time_source": time_source,
        "n_dates": len(date_list),
        "n_batches": n_batches,
        "max_dates": GACOS_MAX_DATES,
        "max_batches": GACOS_MAX_BATCHES,
        "max_span_deg": GACOS_MAX_SPAN_DEG,
        "warnings": warnings,
    }


def gacos_official_status(root: Path, dates: list[str] | None = None) -> dict[str, Any]:
    root = Path(root)
    stack_meta = read_json(root / "data" / "stack" / "meta.json", {}) or {}
    date_list = list(dates) if dates is not None else _stack_dates(stack_meta)
    official_dir = root / "data" / "gacos" / "official"
    product_dir = official_dir / "products"
    found = find_local_screens(product_dir, date_list) if product_dir.exists() and date_list else {}
    missing = [d for d in date_list if d not in found]
    manifest = read_json(official_dir / "manifest.json", {}) or {}
    last_request = read_json(official_dir / "last_request.json", {}) or {}
    all_products = (
        [
            p
            for p in product_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in GACOS_PRODUCT_SUFFIXES
        ]
        if product_dir.exists()
        else []
    )
    defaults = gacos_request_defaults(root, dates=date_list, stack_meta=stack_meta)
    warnings = list(defaults.get("warnings") or [])
    if product_dir.exists() and date_list and len(all_products) > len(found):
        warnings.append(
            f"产品目录中有 {len(all_products) - len(found)} 个文件未能匹配到项目日期；"
            "请检查文件名是否包含 YYYYMMDD 或 YYYY-MM-DD"
        )
    imports = list(manifest.get("imports") or [])
    return {
        "official": {
            "dir": str(product_dir),
            "exists": product_dir.exists(),
            "n_dates": len(date_list),
            "n_available": len(found),
            "coverage": (len(found) / len(date_list)) if date_list else 0.0,
            "available_dates": sorted(found),
            "missing_dates": missing,
            "files": [
                {
                    "date": d,
                    "name": path.name,
                    "size": path.stat().st_size,
                }
                for d, path in sorted(found.items())
            ],
            "n_product_files": len(all_products),
            "updated_at": manifest.get("updated_at"),
            "last_import": imports[-1] if imports else None,
            "last_request": last_request or None,
            "warnings": warnings,
        },
        "defaults": defaults,
        "limits": {
            "max_dates": GACOS_MAX_DATES,
            "max_batches": GACOS_MAX_BATCHES,
            "max_span_deg": GACOS_MAX_SPAN_DEG,
            "max_upload_bytes": GACOS_MAX_UPLOAD_BYTES,
            "max_download_bytes": GACOS_MAX_DOWNLOAD_BYTES,
        },
        "formats": GACOS_FORMATS,
        "form_url": GACOS_FORM_URL,
    }


# ---------------------------------------------------------------------------
# GACOS loading used by the atmosphere stage
# ---------------------------------------------------------------------------


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
    warnings: list[str] = []
    info: dict[str, Any]
    grids: dict[str, np.ndarray] = {}

    if source == "demo":
        demo_dir = root / "data" / "gacos" / "demo"
        if not (demo_dir / "meta.json").exists():
            try:
                prepare_demo_gacos(root)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "当前项目没有合成真值，无法使用模拟 GACOS；请改用官方产品、本地文件或经验高程"
                ) from exc
        for d in stack.dates:
            path = demo_dir / f"{d.replace('-', '')}.npy"
            if path.exists():
                grids[d] = _resample_grid(DelayGrid(data_mm=np.load(path)), shape, target_bounds)
        info = {
            "source": "demo",
            "simulated": True,
            "n_dates": len(grids),
            "unit": "mm",
            "note": "Synthetic GACOS for end-to-end demonstration. Not a real weather-model product.",
        }
    elif source == "official":
        directory = root / "data" / "gacos" / "official" / "products"
        if not directory.exists():
            raise RuntimeError(
                "尚未导入官方 GACOS 产品：请先在「大气」页提交申请，"
                "收到邮件后导入下载链接或 tar.gz"
            )
        found = find_local_screens(directory, stack.dates)
        warnings.extend(_load_grids(found, shape, target_bounds, grids))
        info = {
            "source": "official",
            "directory": str(directory),
            "n_dates": len(grids),
            "unit": "mm",
            "note": "GACOS official products imported into the project.",
        }
    elif source == "local":
        directory = params.get("gacos_dir")
        if not directory:
            raise RuntimeError(
                "未配置 GACOS 本地目录：请在「大气」页填写包含 .ztd/.rsc、GeoTIFF 或 .npy 的目录"
            )
        found = find_local_screens(Path(str(directory)), stack.dates)
        warnings.extend(_load_grids(found, shape, target_bounds, grids))
        info = {
            "source": "local",
            "directory": str(directory),
            "n_dates": len(grids),
            "unit": "mm",
        }
    elif source == "online":
        client = GacosClient(
            base_url=str(params.get("gacos_url", "")),
            email=str(params.get("gacos_email", "")),
            token=str(params.get("gacos_token", "")),
            timeout=int(params.get("gacos_timeout", 60)),
        )
        grids = client.fetch_for_stack(root, stack, params)
        info = {
            "source": "online",
            "n_dates": len(grids),
            "unit": "mm",
            "note": "Custom synchronous GACOS adapter; not the official asynchronous service.",
        }
    else:
        raise ValueError("未配置 GACOS 数据源（可选 demo / official / local / online）")

    missing = [d for d in stack.dates if d not in grids]
    info["missing_dates"] = missing
    info["warnings"] = warnings
    if missing:
        if source == "official":
            hint = "请在「大气」页补齐这些日期的官方产品（邮件链接或 tar.gz）后重试"
        elif source == "local":
            hint = "请检查文件名是否包含 YYYYMMDD 或 YYYY-MM-DD，且目录中包含 .ztd/.rsc、GeoTIFF 或 .npy"
        elif source == "demo":
            hint = "请重新生成演示数据，或改用其他大气数据源"
        else:
            hint = "请检查自定义 GACOS 服务是否返回了所有日期"
        raise RuntimeError(
            f"GACOS 缺少 {len(missing)} 个日期的数据: {', '.join(missing[:6])}"
            f"{'…' if len(missing) > 6 else ''}。{hint}"
        )
    return grids, info


def _load_grids(
    found: dict[str, Path],
    shape: tuple[int, int],
    target_bounds: tuple[float, float, float, float] | None,
    grids: dict[str, np.ndarray],
) -> list[str]:
    warnings: list[str] = []
    for date_str, path in sorted(found.items()):
        grid = read_delay_file(path)
        if grid.bounds and target_bounds:
            if not _bounds_intersect(grid.bounds, target_bounds):
                raise ValueError(
                    f"{path.name} 的覆盖范围与项目范围不重叠："
                    f"{tuple(round(v, 4) for v in grid.bounds)} vs "
                    f"{tuple(round(v, 4) for v in target_bounds)}"
                )
            if not _bounds_cover(grid.bounds, target_bounds):
                warnings.append(f"{path.name} 未完全覆盖项目范围，边缘像元使用最近邻外推")
        grids[date_str] = _resample_grid(grid, shape, target_bounds)
    return warnings


# ---------------------------------------------------------------------------
# Legacy custom synchronous adapter
# ---------------------------------------------------------------------------


class GacosClient:
    """Small, configurable client for custom synchronous GACOS-style APIs.

    This is *not* the official GACOS service.  Different deployments expose
    different endpoints; set ``base_url`` to the service you are licensed to
    use.  The client posts a JSON request and accepts either a JSON list of
    URLs or a binary/text response.  Credentials are kept in memory only and
    are never written to disk.
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
            response = requests.get(self.base_url, timeout=min(self.timeout, 15))
            return {"ok": response.status_code < 500, "status": response.status_code, "message": response.reason}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def fetch_for_stack(self, root: Path, stack: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
        if not self.base_url:
            raise ValueError("未配置自定义 GACOS 服务地址")
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
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        cache = root / "data" / "gacos" / "online"
        cache.mkdir(parents=True, exist_ok=True)
        content_type = response.headers.get("content-type", "")
        grids: dict[str, np.ndarray] = {}
        if "application/json" in content_type:
            body = response.json()
            items = body.get("products", body if isinstance(body, list) else [])
            for item in items:
                date_str = str(item.get("date", ""))[:10]
                url = item.get("url")
                if not date_str or not url:
                    continue
                data = requests.get(url, headers=headers, timeout=self.timeout)
                data.raise_for_status()
                suffix = Path(urlparse(str(url)).path).suffix or ".ztd"
                path = cache / f"{date_str.replace('-', '')}{suffix}"
                path.write_bytes(data.content)
                grid = read_delay_file(path)
                grids[date_str] = grid.data_mm
        else:
            # single product response: cache as today's screen
            path = cache / f"{stack.dates[0].replace('-', '')}.ztd"
            path.write_bytes(response.content)
            grid = read_delay_file(path)
            grids[stack.dates[0]] = grid.data_mm
        return grids
