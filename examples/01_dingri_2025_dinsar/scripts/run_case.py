#!/usr/bin/env python
"""Run the Dingri 2025 D-InSAR example end to end.

The script imports the derived replay stack in ``data/`` into a throw-away
InSAR Studio workspace, runs the D-InSAR route with the parameters documented
in the tutorial, multiplies the phase-compressed output by the replay scale,
validates it against the published LOS product, and writes compact results to
``results/``.  The workspace is removed afterwards unless ``--keep`` is given.

Example::

    python examples/01_dingri_2025_dinsar/scripts/run_case.py
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
    "reference_mode": "manual",
    "reference_row": 24,
    "reference_col": 63,
}


def _json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figures(los_mm: np.ndarray, published_mm: np.ndarray, bounds: list[float], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(los_mm) & np.isfinite(published_mm)
    limit = float(np.nanpercentile(np.abs(los_mm[valid]), 99))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(los_mm, cmap="RdBu_r", norm=norm, extent=extent)
    ax.set_title("D-InSAR LOS displacement - Dingri 2025-01-07 (workbench, x10)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.colorbar(image, ax=ax, shrink=0.85, label="LOS displacement (mm)")
    path = out_dir / "dingri_los_mm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    residual = los_mm - published_mm
    residual = residual - np.nanmedian(residual[valid])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    sample = valid & (np.abs(residual) < np.nanpercentile(np.abs(residual[valid]), 99.5))
    axes[0].scatter(published_mm[sample][::7], los_mm[sample][::7], s=4, alpha=0.35, linewidths=0)
    low = float(np.nanmin(published_mm[sample]))
    high = float(np.nanmax(published_mm[sample]))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Published LOS (mm)")
    axes[0].set_ylabel("Workbench LOS (mm)")
    axes[0].set_title("Agreement with the published product")
    axes[0].grid(alpha=0.25)
    image = axes[1].imshow(residual, cmap="PuOr_r", extent=extent)
    axes[1].set_title("Residual after median offset (mm)")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    path2 = out_dir / "dingri_validation.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    return [path.relative_to(CASE_DIR).as_posix(), path2.relative_to(CASE_DIR).as_posix()]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行定日 2025 D-InSAR 示例")
    parser.add_argument("--work", type=Path, default=CASE_DIR / "workspace", help="临时工作区")
    parser.add_argument("--keep", action="store_true", help="保留临时工作区")
    parser.add_argument("--skip-figures", action="store_true", help="不生成 PNG 图件")
    args = parser.parse_args()

    if not (DATA_DIR / "meta.json").exists():
        raise SystemExit(f"缺少示例数据: {DATA_DIR}\n请先运行 scripts/build_case.py")

    meta = _json_read(DATA_DIR / "meta.json")
    scale = float(meta.get("replay_phase_scale", 1.0))
    published = np.load(DATA_DIR / "reference_los_mm.npy", allow_pickle=False)

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    started = time.time()
    store = ProjectStore(work)
    project = store.create(
        "Dingri 2025 D-InSAR example",
        "dinsar",
        params={"remove_ramp": False},
        source="import",
        notes="Derived replay stack from Zenodo 14836721",
    )
    root = store.path(project["id"])
    import_stack(root, DATA_DIR, technique="dinsar", meta_overrides={})
    run_pipeline(root, technique="dinsar", params=PARAMS, steps=["network", "interferometry", "inversion"])
    elapsed = time.time() - started

    inversion = _json_read(root / "products" / "inversion.json")
    raw = np.load(root / "products" / "arrays" / "disp_pair_mm.npy", allow_pickle=False)
    los_mm = (raw.astype(np.float64) * scale).astype(np.float32)
    valid = np.isfinite(los_mm) & np.isfinite(published)
    diff = los_mm[valid].astype(np.float64) - published[valid].astype(np.float64)
    offset = float(np.median(diff))
    centred = diff - offset
    reference_row = int(inversion["reference_pixel"][0])
    reference_col = int(inversion["reference_pixel"][1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "dingri_los_mm.npy", los_mm, allow_pickle=False)
    write_geotiff(
        RESULTS_DIR / "dingri_los_mm.tif",
        los_mm,
        corner_lon=float(inversion["grid"]["corner_lon"]),
        corner_lat=float(inversion["grid"]["corner_lat"]),
        lon_step=float(inversion["grid"]["lon_step"]),
        lat_step=float(inversion["grid"]["lat_step"]),
    )

    figures: list[str] = []
    if not args.skip_figures:
        figures = _save_figures(los_mm, published, inversion["bounds"], RESULTS_DIR / "figures")

    summary = {
        "case": "01_dingri_2025_dinsar",
        "title": "2025-01-07 Dingri Mw 7.0 earthquake - D-InSAR replay",
        "technique": "dinsar",
        "elapsed_s": round(elapsed, 1),
        "stack": {
            "rows": int(meta["rows"]),
            "cols": int(meta["cols"]),
            "dates": meta["dates"],
            "orbit_direction": meta.get("orbit_direction"),
            "platform": meta.get("platform"),
            "source": meta.get("source"),
            "source_doi": meta.get("source_doi"),
            "replay_phase_scale": scale,
        },
        "parameters": PARAMS,
        "reference_pixel": [reference_row, reference_col],
        "statistics_mm": {
            "min": float(np.nanmin(los_mm[valid])),
            "max": float(np.nanmax(los_mm[valid])),
            "p05": float(np.nanpercentile(los_mm[valid], 5)),
            "p50": float(np.nanpercentile(los_mm[valid], 50)),
            "p95": float(np.nanpercentile(los_mm[valid], 95)),
        },
        "validation_against_published_product": {
            "median_offset_mm": offset,
            "rmse_mm": float(np.sqrt(np.mean(centred**2))),
            "p95_abs_residual_mm": float(np.percentile(np.abs(centred), 95)),
            "correlation": float(np.corrcoef(los_mm[valid], published[valid])[0, 1]),
            "valid_fraction": float(valid.mean()),
            "note": (
                "Workbench output multiplied by the replay scale 10. The median offset is a "
                "constant gauge difference; the centred residual reflects smoothing and 3x downsampling."
            ),
        },
        "figures": figures,
        "outputs": [
            "results/dingri_los_mm.npy",
            "results/dingri_los_mm.tif",
            "results/summary.json",
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary["validation_against_published_product"], ensure_ascii=False, indent=2))
    print(f"结果目录: {RESULTS_DIR}")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
        print("已删除临时工作区（--keep 可保留）")
    else:
        print(f"临时工作区保留在: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
