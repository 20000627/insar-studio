"""Minimal GeoTIFF reader/writer with WGS-84 georeferencing.

Only the tags needed by InSAR Studio are emitted.  Files are plain,
standards-compliant GeoTIFFs and open in QGIS / ArcGIS / SNAP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# GeoTIFF tag numbers
MODEL_PIXEL_SCALE = 33550
MODEL_TIEPOINT = 33922
GEO_KEY_DIRECTORY = 34735

# GeoKey IDs
GT_MODEL_TYPE = 1024
GT_RASTER_TYPE = 1025
GEOGRAPHIC_TYPE = 2048


def _geokeys(epsg: int = 4326) -> tuple[int, ...]:
    # version 1.1.0, 3 keys: model type, raster type, geographic CRS
    return (
        1, 1, 0, 3,
        GT_MODEL_TYPE, 0, 1, 2,
        GT_RASTER_TYPE, 0, 1, 1,
        GEOGRAPHIC_TYPE, 0, 1, int(epsg),
    )


def write_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    corner_lon: float,
    corner_lat: float,
    lon_step: float,
    lat_step: float,
    epsg: int = 4326,
    compress: bool = True,
) -> Path:
    """Write a single-band float32/float64 GeoTIFF.

    ``corner_lon``/``corner_lat`` refer to the upper-left *outer* corner of the
    first pixel.  ``lat_step`` is normally negative (north-up raster).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.dtype == np.complex64 or data.dtype == np.complex128:
        raise TypeError("GeoTIFF export supports real-valued products only")
    extratags = [
        (MODEL_PIXEL_SCALE, "d", 3, (abs(float(lon_step)), abs(float(lat_step)), 0.0), False),
        (MODEL_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, float(corner_lon), float(corner_lat), 0.0), False),
        (GEO_KEY_DIRECTORY, "H", len(_geokeys(epsg)), _geokeys(epsg), False),
    ]
    tifffile.imwrite(
        path,
        data.astype(np.float32),
        photometric="minisblack",
        metadata=None,
        extratags=extratags,
        compression="zlib" if compress else None,
    )
    return path


def read_geotiff(path: Path, *, mmap: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band GeoTIFF and its approximate WGS-84 georeference."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = {tag.name: tag.value for tag in page.tags.values()}
        geo: dict[str, Any] = {"crs": "EPSG:4326"}
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale and tie:
            geo["lon_step"] = float(scale[0])
            geo["lat_step"] = -float(scale[1])
            geo["corner_lon"] = float(tie[3])
            geo["corner_lat"] = float(tie[4])
        keys = tags.get("GeoKeyDirectoryTag")
        if keys:
            # crude extraction of GeographicTypeGeoKey
            arr_keys = list(keys)
            for i in range(4, len(arr_keys), 4):
                if arr_keys[i] == GEOGRAPHIC_TYPE:
                    geo["epsg"] = int(arr_keys[i + 3])
    if arr.ndim == 3 and arr.shape[-1] == 2 and arr.dtype != np.complex64:
        arr = arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if mmap:
        arr = np.asarray(arr)
    return arr, geo


def geotiff_bounds(meta: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) from stack metadata."""
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    west = float(meta.get("corner_lon", 0.0))
    north = float(meta.get("corner_lat", 0.0))
    lon_step = float(meta.get("lon_step", 0.0004))
    lat_step = float(meta.get("lat_step", -0.0004))
    east = west + cols * lon_step
    south = north + rows * lat_step
    return west, min(south, north), east, max(south, north)
