#!/usr/bin/env python
"""Rebuild the Baige landslide SBAS replay stack.

The script downloads the public CC-BY-4.0 displacement time series from Zenodo
record 10605366 (``disp_data.rar``), reads the MATLAB v7.3 ``Dt.mat`` cube,
selects every fifth epoch plus the final one, and writes a compact complex SLC
stack that InSAR Studio can process with its SBAS route.

The source package does not document acquisition dates.  The 41 selected
epochs are therefore mapped onto a *reconstructed* 6-day axis starting
2018-12-14.  This axis is only used to build the SBAS design matrix; absolute
velocities from this example must not be treated as a new geological result.

Example::

    python examples/02_baige_landslide_sbas/scripts/build_case.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
DATA_DIR = CASE_DIR / "data"
SOURCE_DIR = CASE_DIR / "source"

SOURCE_URL = "https://zenodo.org/api/records/10605366/files/disp_data.rar/content"
SOURCE_NAME = "disp_data.rar"
SOURCE_MD5 = "0540704e40a1924c4a726aea6482855b"
SOURCE_DOI = "10.5281/zenodo.10605366"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_CITATION = "Displacement time series of the Baige landslide (Zenodo record 10605366)"

EPOCH_STEP = 5
RECONSTRUCTED_START = date(2018, 12, 14)
RECONSTRUCTED_INTERVAL_DAYS = 6
WAVELENGTH_M = 0.055465763
INCIDENCE_DEG = 38.0
REFERENCE_ROW = 9
REFERENCE_COL = 30

# From disp_data/disp_par2 (GAMMA DIFF&GEO DEM/MAP parameter file).
CORNER_LON = 98.69603
CORNER_LAT = 31.08736
LON_STEP = 1.8518519e-04
LAT_STEP = -1.8518519e-04


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


def _extract_rar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # Windows 10+ ships bsdtar as ``tar`` and libarchive reads RAR.
    result = subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and (destination / "disp_data" / "Dt.mat").exists():
        return
    try:
        import rarfile  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "无法自动解压 RAR。请安装 7-Zip 或 bsdtar 后手动解压，"
            "再用 --source-dir 指向包含 Dt.mat 和 mask.mat 的 disp_data 目录。\n"
            f"tar 输出: {result.stderr.strip()}"
        ) from exc
    with rarfile.RarFile(archive) as handle:
        handle.extractall(destination)


def _read_matlab(path: Path, key: str) -> np.ndarray:
    from scipy.io import loadmat

    try:
        return np.asarray(loadmat(path)[key])
    except NotImplementedError:
        pass
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"{path.name} 是 MATLAB v7.3 文件，需要 h5py 才能读取：\n"
            "    python -m pip install h5py"
        ) from exc
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key])


def build(disp_dir: Path, out_dir: Path, source_size: int | None = None) -> dict:
    dt_path = disp_dir / "Dt.mat"
    mask_path = disp_dir / "mask.mat"
    if not dt_path.exists() or not mask_path.exists():
        raise SystemExit(f"目录中缺少 Dt.mat 或 mask.mat: {disp_dir}")

    cube = _read_matlab(dt_path, "Dt").astype(np.float64)
    mask = _read_matlab(mask_path, "mask").astype(bool)
    if cube.ndim != 3:
        raise SystemExit(f"Dt.mat 维度异常: {cube.shape}")
    # The published cube is stored as (time, cols, rows) relative to mask.mat;
    # transpose the two spatial axes so everything is (time, rows, cols).
    cube = np.transpose(cube, (0, 2, 1))
    n_epochs, rows, cols = cube.shape
    if mask.shape != (rows, cols):
        raise SystemExit(f"mask 尺寸 {mask.shape} 与 Dt.mat 空间尺寸 {(rows, cols)} 不一致")

    disp = cube.astype(np.float32)
    indices = list(range(0, n_epochs, EPOCH_STEP))
    if indices[-1] != n_epochs - 1:
        indices.append(n_epochs - 1)
    selected = disp[indices].copy()

    dates = [
        (RECONSTRUCTED_START + timedelta(days=RECONSTRUCTED_INTERVAL_DAYS * i)).isoformat()
        for i in indices
    ]
    sampling_days = [RECONSTRUCTED_INTERVAL_DAYS * i for i in indices]

    wavelength = WAVELENGTH_M
    k = 4.0 * np.pi / wavelength
    phase = (k * selected / 1000.0).astype(np.float32)

    yy, xx = np.mgrid[0:rows, 0:cols]
    amplitude = (
        0.86
        + 0.10 * np.cos(2.0 * np.pi * xx / 23.0)
        + 0.05 * np.cos(2.0 * np.pi * yy / 17.0)
    ).astype(np.float32)
    amplitude = np.clip(amplitude, 0.55, 1.25)

    slc = np.empty((len(indices), rows, cols), dtype=np.complex64)
    for epoch in range(len(indices)):
        slc[epoch] = (amplitude * np.exp(1j * phase[epoch])).astype(np.complex64)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    np.save(out_dir / "slc.npy", slc, allow_pickle=False)
    np.save(out_dir / "dem.npy", np.zeros((rows, cols), dtype=np.float32), allow_pickle=False)
    np.save(out_dir / "mask.npy", mask, allow_pickle=False)
    np.save(out_dir / "truth_disp_mm.npy", selected.astype(np.float32), allow_pickle=False)

    meta = {
        "schema": "insar-studio/stack/v1",
        "name": "Baige landslide — derived SBAS replay stack (41 epochs)",
        "source": "public-displacement-timeseries-replay",
        "demo": False,
        "platform": "Unknown (C-band assumed)",
        "orbit_direction": "Unknown",
        "relative_orbit": None,
        "rows": int(rows),
        "cols": int(cols),
        "n_dates": int(len(indices)),
        "dates": dates,
        "dates_source": "reconstructed",
        "sampling_days": sampling_days,
        "wavelength_m": wavelength,
        "incidence_deg": INCIDENCE_DEG,
        "range_spacing_m": 2.329562,
        "azimuth_spacing_m": 13.9,
        "slant_range_m": 705000.0,
        "bperp_m": [0.0] * len(indices),
        "start_times": [f"{day}T00:00:00" for day in dates],
        "corner_lon": CORNER_LON,
        "corner_lat": CORNER_LAT,
        "lon_step": LON_STEP,
        "lat_step": LAT_STEP,
        "crs": "EPSG:4326",
        "ref_index": 0,
        "already_coregistered": True,
        "geo_referenced": True,
        "notes": (
            "Derived from the CC-BY-4.0 Baige displacement time series on Zenodo "
            "10605366. Every fifth epoch plus the final epoch was selected and "
            "converted to phase at the C-band wavelength. The source package has "
            "no acquisition dates and does not document platform or orbit, so the "
            "41 epochs use a reconstructed 6-day axis starting 2018-12-14. This is "
            "a replay/derived stack, not raw SLC."
        ),
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_md5": SOURCE_MD5,
        "source_license": SOURCE_LICENSE,
        "reconstructed_time_axis": True,
        "reconstructed_start": dates[0],
        "reconstructed_interval_days": RECONSTRUCTED_INTERVAL_DAYS,
        "source_epoch_indices": indices,
        "reference_row": REFERENCE_ROW,
        "reference_col": REFERENCE_COL,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    masked = selected[:, mask]
    config = {
        "schema": "insar-studio/example-replay/v1",
        "case": "02_baige_landslide_sbas",
        "created_at": date.today().isoformat(),
        "source": {
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
            "citation": SOURCE_CITATION,
            "file": SOURCE_NAME,
            "md5": SOURCE_MD5,
            "size_bytes": source_size,
            "dt_mat_md5": _md5(dt_path),
            "mask_mat_md5": _md5(mask_path),
        },
        "build": {
            "source_shape": [int(n_epochs), int(rows), int(cols)],
            "selected_epoch_indices": indices,
            "epoch_step": EPOCH_STEP,
            "time_axis": "reconstructed",
            "time_axis_note": (
                "Source package has no acquisition dates. A 6-day axis starting "
                "2018-12-14 is used only for the SBAS design matrix; absolute "
                "velocities are not a new geological conclusion."
            ),
            "unit_assumption": "source values interpreted as millimetres",
            "wavelength_m": wavelength,
            "reference_pixel": [REFERENCE_ROW, REFERENCE_COL],
        },
        "outputs": {
            name: {
                "sha256": _sha256(out_dir / name),
                "bytes": (out_dir / name).stat().st_size,
            }
            for name in ("slc.npy", "dem.npy", "mask.npy", "truth_disp_mm.npy")
        },
        "reference_stats": {
            "shape": [int(rows), int(cols)],
            "mask_pixels": int(mask.sum()),
            "mask_fraction": float(mask.mean()),
            "disp_min_mm": float(np.nanmin(masked)),
            "disp_max_mm": float(np.nanmax(masked)),
            "disp_last_max_mm": float(np.nanmax(selected[-1][mask])),
            "first_epoch_all_zero": bool(np.all(selected[0] == 0)),
        },
    }
    (out_dir / "replay_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="重建白格滑坡 SBAS 回放栈")
    parser.add_argument("--source", type=Path, help="已下载的 disp_data.rar 路径")
    parser.add_argument("--source-dir", type=Path, help="已解压的 disp_data 目录")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="输出 data 目录")
    parser.add_argument("--download", action="store_true", help="源文件不存在时自动下载")
    args = parser.parse_args()

    if args.source_dir is not None:
        disp_dir = args.source_dir
        source_size = None
    else:
        archive = args.source or (SOURCE_DIR / SOURCE_NAME)
        if not archive.exists():
            if not args.download:
                raise SystemExit(
                    f"找不到源文件: {archive}\n"
                    f"请先下载 {SOURCE_URL}，或加 --download 自动下载。"
                )
            _download(SOURCE_URL, archive)
        digest = _md5(archive)
        if digest != SOURCE_MD5:
            raise SystemExit(
                f"源文件 MD5 不匹配：期望 {SOURCE_MD5}，实际 {digest}。"
                "请重新下载，不要使用被截断的文件。"
            )
        source_size = archive.stat().st_size
        extracted = SOURCE_DIR / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        _extract_rar(archive, extracted)
        disp_dir = extracted / "disp_data"

    config = build(disp_dir, args.out, source_size=source_size)
    print(json.dumps(config["reference_stats"], ensure_ascii=False, indent=2))
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
