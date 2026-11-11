#!/usr/bin/env python
"""Run the Baige landslide SBAS example end to end.

The script imports the derived 41-epoch replay stack in ``data/`` into a
throw-away InSAR Studio workspace, runs the SBAS route with the documented
parameters, compares the recovered time series with the published displacement
cube, and writes compact results to ``results/``.

The source package has no acquisition dates.  The example uses a reconstructed
6-day axis only to build the SBAS design matrix; the absolute velocities are
not a new geological conclusion.

Example::

    python examples/02_baige_landslide_sbas/scripts/run_case.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from insar_studio.core.geotiff import write_geotiff  # noqa: E402
from insar_studio.importer import import_stack  # noqa: E402
from insar_studio.pipeline import run_pipeline  # noqa: E402
from insar_studio.project import ProjectStore  # noqa: E402

DATA_DIR = CASE_DIR / "data"
RESULTS_DIR = CASE_DIR / "results"
PARAMS = {
    "remove_ramp": False,
    "multilook": 1,
    "filter": "none",
    "coherence_window": 5,
    "min_coherence": 0.0,
    "unwrap_coherence_min": 0.0,
    "unwrap_method": "quality",
    "max_temporal_baseline": 60,
    "max_pairs": 300,
    "estimate_dem_error": False,
    "aps_method": "off",
    "aps_iterations": 0,
    "robust_iterations": 1,
    "regularization": 0.0,
    "reference_mode": "manual",
    "reference_row": 9,
    "reference_col": 30,
}


def _json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figures(
    velocity: np.ndarray,
    disp: np.ndarray,
    truth: np.ndarray,
    dates: list[str],
    mask: np.ndarray,
    reference: tuple[int, int],
    bounds: list[float],
    out_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    row, col = reference
    peak_row, peak_col = np.unravel_index(np.nanargmax(disp[-1]), disp.shape[1:])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    image = ax.imshow(velocity, cmap="RdBu_r", extent=extent)
    ax.set_title("SBAS LOS velocity - Baige landslide (reconstructed time axis)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.colorbar(image, ax=ax, shrink=0.9, label="Velocity (mm/yr)")
    path = out_dir / "baige_velocity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    dates_x = np.arange(len(dates))
    truth_ref = truth - truth[:, row, col][:, None, None]
    disp_ref = disp - disp[:, row, col][:, None, None]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    axes[0].plot(dates_x, truth_ref[:, peak_row, peak_col], "k--", linewidth=1.8, label="Published")
    axes[0].plot(dates_x, disp_ref[:, peak_row, peak_col], color="#c0392b", linewidth=1.8, label="Workbench")
    tick_idx = list(range(0, len(dates), 5))
    if tick_idx[-1] != len(dates) - 1:
        tick_idx.append(len(dates) - 1)
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels([dates[i][:7] for i in tick_idx], rotation=35, ha="right")
    axes[0].set_xlabel("Reconstructed date")
    axes[0].set_ylabel("LOS displacement (mm)")
    axes[0].set_title(f"Peak pixel (row {peak_row}, col {peak_col})")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    valid = np.isfinite(disp_ref) & mask[None, :, :]
    axes[1].scatter(truth_ref[valid][::17], disp_ref[valid][::17], s=4, alpha=0.3, linewidths=0)
    low = float(np.nanmin(truth_ref[valid]))
    high = float(np.nanmax(truth_ref[valid]))
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_xlabel("Published LOS (mm)")
    axes[1].set_ylabel("Workbench LOS (mm)")
    axes[1].set_title("All valid pixels and epochs")
    axes[1].grid(alpha=0.25)
    path2 = out_dir / "baige_timeseries.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    return [path.relative_to(CASE_DIR).as_posix(), path2.relative_to(CASE_DIR).as_posix()]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行白格滑坡 SBAS 示例")
    parser.add_argument("--work", type=Path, default=CASE_DIR / "workspace", help="临时工作区")
    parser.add_argument("--keep", action="store_true", help="保留临时工作区")
    parser.add_argument("--skip-figures", action="store_true", help="不生成 PNG 图件")
    args = parser.parse_args()

    if not (DATA_DIR / "meta.json").exists():
        raise SystemExit(f"缺少示例数据: {DATA_DIR}\n请先运行 scripts/build_case.py")

    meta = _json_read(DATA_DIR / "meta.json")
    truth = np.load(DATA_DIR / "truth_disp_mm.npy", allow_pickle=False)
    mask = np.load(DATA_DIR / "mask.npy", allow_pickle=False).astype(bool)
    dates = list(meta["dates"])

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    started = time.time()
    store = ProjectStore(work)
    project = store.create(
        "Baige landslide SBAS example",
        "sbas",
        params={"remove_ramp": False, "max_temporal_baseline": 60, "max_pairs": 300},
        source="import",
        notes="Derived replay stack from Zenodo 10605366; reconstructed 6-day time axis",
    )
    root = store.path(project["id"])
    import_stack(root, DATA_DIR, technique="sbas", meta_overrides={})
    run_pipeline(root, technique="sbas", params=PARAMS, steps=["network", "interferometry", "inversion"])
    elapsed = time.time() - started

    inversion = _json_read(root / "products" / "inversion.json")
    arrays = root / "products" / "arrays"
    disp = np.load(arrays / "disp_ts_mm.npy", allow_pickle=False)
    velocity = np.load(arrays / "velocity_mm_yr.npy", allow_pickle=False)
    valid = np.load(arrays / "valid.npy", allow_pickle=False).astype(bool)
    reference_row, reference_col = (int(v) for v in inversion["reference_pixel"])

    truth_ref = truth - truth[:, reference_row, reference_col][:, None, None]
    disp_ref = disp - disp[:, reference_row, reference_col][:, None, None]
    sample = np.isfinite(disp_ref) & valid[None, :, :] & mask[None, :, :]
    residual = disp_ref[sample].astype(np.float64) - truth_ref[sample].astype(np.float64)

    peak_flat = np.nanargmax(np.where(mask, disp[-1], np.nan))
    peak_row, peak_col = np.unravel_index(peak_flat, disp.shape[1:])
    peak_workbench = float(disp_ref[-1, peak_row, peak_col])
    peak_published = float(truth_ref[-1, peak_row, peak_col])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "baige_disp_ts_mm.npy", disp_ref, allow_pickle=False)
    np.save(RESULTS_DIR / "baige_velocity_mm_yr.npy", velocity, allow_pickle=False)
    write_geotiff(
        RESULTS_DIR / "baige_velocity_mm_yr.tif",
        velocity,
        corner_lon=float(inversion["grid"]["corner_lon"]),
        corner_lat=float(inversion["grid"]["corner_lat"]),
        lon_step=float(inversion["grid"]["lon_step"]),
        lat_step=float(inversion["grid"]["lat_step"]),
    )
    write_geotiff(
        RESULTS_DIR / "baige_disp_last_mm.tif",
        disp_ref[-1],
        corner_lon=float(inversion["grid"]["corner_lon"]),
        corner_lat=float(inversion["grid"]["corner_lat"]),
        lon_step=float(inversion["grid"]["lon_step"]),
        lat_step=float(inversion["grid"]["lat_step"]),
    )

    figures: list[str] = []
    if not args.skip_figures:
        figures = _save_figures(
            velocity,
            disp,
            truth,
            dates,
            mask,
            (reference_row, reference_col),
            inversion["bounds"],
            RESULTS_DIR / "figures",
        )

    inside = velocity[np.isfinite(velocity) & mask]
    summary = {
        "case": "02_baige_landslide_sbas",
        "title": "Baige landslide - SBAS replay on a reconstructed time axis",
        "technique": "sbas",
        "elapsed_s": round(elapsed, 1),
        "stack": {
            "rows": int(meta["rows"]),
            "cols": int(meta["cols"]),
            "n_dates": int(meta["n_dates"]),
            "date_start": dates[0],
            "date_end": dates[-1],
            "time_axis": "reconstructed 6-day axis; source package has no acquisition dates",
            "source": meta.get("source"),
            "source_doi": meta.get("source_doi"),
        },
        "parameters": PARAMS,
        "reference_pixel": [reference_row, reference_col],
        "velocity_mm_yr": {
            "min": float(np.nanmin(inside)),
            "median": float(np.nanmedian(inside)),
            "p05": float(np.nanpercentile(inside, 5)),
            "p95": float(np.nanpercentile(inside, 95)),
            "max": float(np.nanmax(inside)),
        },
        "peak_pixel": {
            "row": int(peak_row),
            "col": int(peak_col),
            "workbench_last_mm": peak_workbench,
            "published_last_mm": peak_published,
            "difference_mm": peak_workbench - peak_published,
        },
        "validation_against_published_timeseries": {
            "rmse_mm": float(np.sqrt(np.mean(residual**2))),
            "p95_abs_residual_mm": float(np.percentile(np.abs(residual), 95)),
            "correlation": float(np.corrcoef(disp_ref[sample], truth_ref[sample])[0, 1]),
            "valid_samples": int(sample.sum()),
            "note": (
                "Both series are referenced to the same stable pixel. The reconstructed "
                "time axis does not affect the displacement fit; it only scales velocity."
            ),
        },
        "caveats": [
            "Source package has no acquisition dates; the 6-day axis is reconstructed.",
            "Absolute velocities are not a new geological conclusion.",
            "The replay stack is derived from a published displacement product, not raw SLC.",
        ],
        "figures": figures,
        "outputs": [
            "results/baige_disp_ts_mm.npy",
            "results/baige_velocity_mm_yr.npy",
            "results/baige_velocity_mm_yr.tif",
            "results/baige_disp_last_mm.tif",
            "results/summary.json",
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary["validation_against_published_timeseries"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["peak_pixel"], ensure_ascii=False, indent=2))
    print(f"结果目录: {RESULTS_DIR}")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
        print("已删除临时工作区（--keep 可保留）")
    else:
        print(f"临时工作区保留在: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
