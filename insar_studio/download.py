"""Experimental Sentinel-1 search and download adapters.

The adapters are intentionally thin: credentials are supplied per request and
never written to the project.  Downloaded products are staged under
``data/downloads``; production stacks are best prepared with SNAP/ISCE and
then imported through the stack importer, because orbit, burst and baseline
metadata need careful handling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import requests

COP_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
COP_SEARCH = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COP_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ASF_SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"

PROVIDER_ALIASES = {"copernicus": "copernicus", "cdse": "copernicus", "asf": "asf"}


def validate_provider(provider: str) -> str:
    """Return the canonical provider name or raise ``ValueError`` for a 400."""
    key = str(provider or "").strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError("provider 必须是 copernicus 或 asf")
    return PROVIDER_ALIASES[key]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "InSAR-Studio/1.0 (local research workstation)"})
    return s


def copernicus_token(username: str, password: str, timeout: int = 30) -> str:
    r = _session().post(
        COP_TOKEN,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


def _bbox_polygon(bbox: list[float]) -> str:
    west, south, east, north = [float(x) for x in bbox]
    return (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )


def search_copernicus(
    *,
    bbox: list[float],
    start: str,
    end: str,
    max_results: int = 20,
    username: str = "",
    password: str = "",
    timeout: int = 40,
) -> list[dict[str, Any]]:
    token = copernicus_token(username, password, timeout) if username and password else ""
    flt = (
        "Collection/Name eq 'SENTINEL-1' "
        "and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'SLC') "
        f"and ContentDate/Start ge {start}T00:00:00.000Z "
        f"and ContentDate/End le {end}T23:59:59.000Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{_bbox_polygon(bbox)}')"
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = _session().get(
        COP_SEARCH,
        params={"$filter": flt, "$top": max_results, "$orderby": "ContentDate/Start desc"},
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    values = r.json().get("value", [])
    out = []
    for item in values:
        out.append(
            {
                "provider": "copernicus",
                "id": item.get("Id"),
                "name": item.get("Name"),
                "date": (item.get("ContentDate") or {}).get("Start", "")[:10],
                "size": (item.get("ContentLength") or 0),
                "url": COP_DOWNLOAD.format(product_id=item.get("Id")),
                "requires_auth": True,
                "footprint": item.get("Footprint"),
            }
        )
    return out


def search_asf(
    *,
    bbox: list[float],
    start: str,
    end: str,
    max_results: int = 20,
    timeout: int = 40,
) -> list[dict[str, Any]]:
    west, south, east, north = [float(x) for x in bbox]
    r = _session().get(
        ASF_SEARCH,
        params={
            "platform": "Sentinel-1",
            "processingLevel": "SLC",
            "intersectsWith": f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))",
            "start": start,
            "end": end,
            "output": "json",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    out = []
    for item in data[:max_results]:
        url = item.get("downloadUrl") or item.get("url") or ""
        out.append(
            {
                "provider": "asf",
                "id": item.get("granuleName") or item.get("sceneName") or item.get("fileID"),
                "name": item.get("granuleName") or item.get("sceneName") or item.get("fileID"),
                "date": (item.get("startTime") or "")[:10],
                "size": item.get("bytes") or item.get("size") or 0,
                "url": url,
                "requires_auth": True,
                "footprint": item.get("footprint"),
            }
        )
    return out


def search(
    provider: str,
    *,
    bbox: list[float],
    start: str,
    end: str,
    max_results: int = 20,
    username: str = "",
    password: str = "",
) -> list[dict[str, Any]]:
    provider = validate_provider(provider)
    if provider == "copernicus":
        return search_copernicus(
            bbox=bbox,
            start=start,
            end=end,
            max_results=max_results,
            username=username,
            password=password,
        )
    if provider == "asf":
        return search_asf(bbox=bbox, start=start, end=end, max_results=max_results)
    raise ValueError("provider 必须是 copernicus 或 asf")


def download(
    url: str,
    dest: Path,
    *,
    provider: str = "copernicus",
    username: str = "",
    password: str = "",
    progress: Callable[[float, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    timeout: int = 120,
) -> Path:
    """Stream a product to disk, reporting bytes and relative progress."""
    provider = validate_provider(provider)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = ""
    auth = None
    if provider == "copernicus":
        if username and password:
            token = copernicus_token(username, password)
    elif provider == "asf":
        if username and password:
            auth = (username, password)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with _session().get(url, headers=headers, auth=auth, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            done = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if cancel and cancel():
                    raise RuntimeError("下载已取消")
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress((done / total) if total else 0.0, done)
        tmp.replace(dest)
    return dest


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "sentinel1_product")
    return cleaned[-180:]


def provider_help() -> dict[str, Any]:
    return {
        "copernicus": {
            "label": "Copernicus Data Space",
            "auth": "邮箱 + 密码（仅用于本次请求，不落盘）",
            "note": "OData 搜索与 $value 下载；大文件建议使用官方客户端。",
        },
        "asf": {
            "label": "ASF DAAC",
            "auth": "Earthdata 用户名 + 密码",
            "note": "搜索公开；SLC 下载受 Earthdata 登录限制。",
        },
        "local": {
            "label": "本地导入",
            "auth": "无需账号",
            "note": "推荐：用 SNAP/ISCE 完成 burst、轨道与配准后导入栈。",
        },
    }
