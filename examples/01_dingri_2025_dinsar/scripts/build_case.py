#!/usr/bin/env python
"""Rebuild the compact Dingri 2025 D-InSAR replay stack.

The script downloads (or accepts) the public GAMMA unwrapped GeoTIFF from
Zenodo record 14836721, crops the coseismic lobe, removes no-data holes with a
nearest-neighbour fill, applies a gentle Gaussian smoothing, downsamples by 3,
and writes a two-scene complex SLC stack that InSAR Studio can import.

The phase is deliberately divided by ``PHASE_SCALE`` so the built-in unwrapper
does not lose cycles on a ~60-fringe coseismic field.  The workbench output must
be multiplied by ``PHASE_SCALE`` to recover the published LOS values.  This is
a *replay* stack derived from a published product, not raw Sentinel-1 SLC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
DATA_DIR = CASE_DIR / "data"
SOURCE_DIR = CASE_DIR / "source"

SOURCE_URL = (
    "https://zenodo.org/api/records/14836721/files/"
    "ASC_20250105_20250117.adf_quad.unw.geo.tif/content"
)
SOURCE_NAME = "ASC_20250105_20250117.adf_quad.unw.geo.tif"
SOURCE_MD5 = "894d51cb912ff24ece4af8931d8ee57c"
SOURCE_DOI = "10.5281/zenodo.14836721"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_CITATION = (
    "Geodetic Observations and Seismogenic Structures of the 2025 Mw 7.0 "
    "Dingri Earthquake: The Largest Normal Faulting Event in the Southern "
    "Tibet Rift (Zenodo record 14836721)"
)

# Crop selected around the main coseismic lobe (row, col, size in source pixels).
CROP_ROW = 569
CROP_COL = 1288
CROP_SIZE = 1500
GAUSSIAN_SIGMA = 10.0
DOWNSAMPLE = 3
PHASE_SCALE = 0.1
WAVELENGTH_M = 0.055465763
INCIDENCE_DEG = 38.0
REFERENCE_ROW = 24
REFERENCE_COL = 63


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url}")
    with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    print(f"已保存 {destination} ({destination.stat().st_size / 1024 / 1024:.2f} MiB)")


def _read_source(path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        phase = page.asarray().astype(np.float64)
        tags = {tag.name: tag.value for tag in page.tags.values()}
    scale = tags.get("ModelPixelScaleTag")
    tie = tags.get("ModelTiepointTag")
    if not scale or not tie:
        raise SystemExit("GeoTIFF 缺少 ModelPixelScaleTag / ModelTiepointTag，无法保留地理参考")
    return phase, list(scale), list(tie)


def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    rows = array.shape[0] // factor * factor
    cols = array.shape[1] // factor * factor
    view = array[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    return view.mean(axis=(1, 3))


def build(source: Path, out_dir: Path) -> dict:
    if not source.exists():
        raise SystemExit(f"源文件不存在: {source}")
    digest = _md5(source)
    if digest != SOURCE_MD5:
        raise SystemExit(
            f"源文件 MD5 不匹配：期望 {SOURCE_MD5}，实际 {digest}。"
            "请重新下载，不要使用被截断的文件。"
        )

    phase, pixel_scale, tiepoint = _read_source(source)
    rows, cols = phase.shape
    r0 = int(CROP_ROW)
    c0 = int(CROP_COL)
    size = int(CROP_SIZE)
    if r0 < 0 or c0 < 0 or r0 + size > rows or c0 + size > cols:
        raise SystemExit("裁剪窗口超出源影像范围")
    crop = phase[r0 : r0 + size, c0 : c0 + size]

    valid = np.isfinite(crop) & (crop != 0)
    if valid.mean() < 0.2:
        raise SystemExit("裁剪区域有效像元过少，源文件可能不是预期的解缠产品")
    nearest = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = crop[tuple(nearest)]
    smooth = ndimage.gaussian_filter(filled, sigma=float(GAUSSIAN_SIGMA))
    reduced = _block_mean(smooth, int(DOWNSAMPLE))
    phase_replay = (reduced * PHASE_SCALE).astype(np.float32)

    height, width = reduced.shape
    yy, xx = np.mgrid[0:height, 0:width]
    amplitude = (
        0.86
        + 0.10 * np.cos(2.0 * np.pi * xx / 53.0)
        + 0.05 * np.cos(2.0 * np.pi * yy / 71.0)
    ).astype(np.float32)
    amplitude = np.clip(amplitude, 0.55, 1.25)

    slc = np.empty((2, height, width), dtype=np.complex64)
    slc[0] = amplitude.astype(np.complex64)
    slc[1] = (amplitude * np.exp(1j * phase_replay)).astype(np.complex64)

    mask = np.ones((height, width), dtype=bool)
    mask[:2, :] = mask[-2:, :] = False
    mask[:, :2] = mask[:, -2:] = False

    los_mm = (reduced * WAVELENGTH_M / (4.0 * np.pi) * 1000.0).astype(np.float32)

    lon0 = float(tiepoint[3]) + c0 * float(pixel_scale[0])
    lat0 = float(tiepoint[4]) - r0 * float(pixel_scale[1])
    lon_step = float(pixel_scale[0]) * int(DOWNSAMPLE)
    lat_step = -float(pixel_scale[1]) * int(DOWNSAMPLE)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    np.save(out_dir / "slc.npy", slc, allow_pickle=False)
    np.save(out_dir / "dem.npy", np.zeros((height, width), dtype=np.float32), allow_pickle=False)
    np.save(out_dir / "mask.npy", mask, allow_pickle=False)
    np.save(out_dir / "reference_los_mm.npy", los_mm, allow_pickle=False)

    meta = {
        "schema": "insar-studio/stack/v1",
        "name": "2025-01-07 Dingri Mw 7.0 earthquake — derived D-InSAR replay stack",
        "source": "public-unwrapped-product-replay",
        "demo": False,
        "platform": "Sentinel-1A",
        "orbit_direction": "ASCENDING",
        "relative_orbit": None,
        "rows": int(height),
        "cols": int(width),
        "n_dates": 2,
        "dates": ["2025-01-05", "2025-01-17"],
        "sampling_days": [0, 12],
        "wavelength_m": WAVELENGTH_M,
        "incidence_deg": INCIDENCE_DEG,
        "range_spacing_m": 2.329562,
        "azimuth_spacing_m": 13.9,
        "slant_range_m": 705000.0,
        "bperp_m": [0.0, 0.0],
        "start_times": ["2025-01-05T00:00:00", "2025-01-17T00:00:00"],
        "corner_lon": lon0,
        "corner_lat": lat0,
        "lon_step": lon_step,
        "lat_step": lat_step,
        "crs": "EPSG:4326",
        "ref_index": 0,
        "already_coregistered": True,
        "geo_referenced": True,
        "notes": (
            "Derived from the CC-BY-4.0 GAMMA unwrapped product on Zenodo "
            "14836721. The coseismic lobe was cropped, no-data pixels were "
            "nearest-filled, a Gaussian sigma=10 smoothing and 3x block mean "
            "were applied, and the phase was divided by 10 to keep the built-in "
            "unwrapper stable. Multiply the workbench displacement by 10. "
            "This is a replay/derived stack, not raw Sentinel-1 SLC."
        ),
        "replay_phase_scale": 1.0 / PHASE_SCALE,
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_md5": SOURCE_MD5,
        "source_license": SOURCE_LICENSE,
        "crop_row": r0,
        "crop_col": c0,
        "crop_size": size,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "downsample_factor": DOWNSAMPLE,
        "reference_row": REFERENCE_ROW,
        "reference_col": REFERENCE_COL,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "schema": "insar-studio/example-replay/v1",
        "case": "01_dingri_2025_dinsar",
        "created_at": date.today().isoformat(),
        "source": {
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
            "citation": SOURCE_CITATION,
            "file": SOURCE_NAME,
            "md5": SOURCE_MD5,
            "size_bytes": source.stat().st_size,
        },
        "build": {
            "crop_row": r0,
            "crop_col": c0,
            "crop_size": size,
            "no_data_fill": "nearest-neighbour (scipy.ndimage.distance_transform_edt)",
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "downsample_factor": DOWNSAMPLE,
            "phase_scale": PHASE_SCALE,
            "phase_scale_note": "workbench displacement × 10 = published LOS",
            "wavelength_m": WAVELENGTH_M,
            "reference_pixel": [REFERENCE_ROW, REFERENCE_COL],
        },
        "outputs": {
            "slc.npy": {"sha256": _sha256(out_dir / "slc.npy"), "bytes": (out_dir / "slc.npy").stat().st_size},
            "dem.npy": {"sha256": _sha256(out_dir / "dem.npy"), "bytes": (out_dir / "dem.npy").stat().st_size},
            "mask.npy": {"sha256": _sha256(out_dir / "mask.npy"), "bytes": (out_dir / "mask.npy").stat().st_size},
            "reference_los_mm.npy": {
                "sha256": _sha256(out_dir / "reference_los_mm.npy"),
                "bytes": (out_dir / "reference_los_mm.npy").stat().st_size,
            },
        },
        "reference_stats": {
            "valid_fraction": float(valid.mean()),
            "los_min_mm": float(np.nanmin(los_mm)),
            "los_max_mm": float(np.nanmax(los_mm)),
            "los_p05_mm": float(np.nanpercentile(los_mm, 5)),
            "los_p95_mm": float(np.nanpercentile(los_mm, 95)),
            "shape": [int(height), int(width)],
        },
    }
    (out_dir / "replay_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="重建定日 2025 D-InSAR 回放栈")
    parser.add_argument("--source", type=Path, help="已下载的 .tif 路径")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="输出 data 目录")
    parser.add_argument("--download", action="store_true", help="源文件不存在时自动下载")
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = SOURCE_DIR / SOURCE_NAME
    if not source.exists():
        if not args.download:
            raise SystemExit(
                f"找不到源文件: {source}\n"
                f"请先下载 {SOURCE_URL}，或加 --download 自动下载。"
            )
        _download(SOURCE_URL, source)
    config = build(source, args.out)
    print(json.dumps(config["reference_stats"], ensure_ascii=False, indent=2))
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
