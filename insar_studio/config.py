"""Application paths and global defaults.

Everything is resolved inside the repository / workspace so the application
never writes outside the project directory tree.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATE_DIR = PACKAGE_DIR / "templates"
WORKSPACE_DIR = Path(os.environ.get("INSAR_STUDIO_WORKSPACE", APP_ROOT / "workspace"))
PROJECTS_DIR = WORKSPACE_DIR / "projects"
CACHE_DIR = WORKSPACE_DIR / "cache"
LOG_DIR = WORKSPACE_DIR / "logs"

#: Sentinel-1 C-band carrier wavelength (m)
S1_WAVELENGTH = 0.05546576
#: Default incidence angle for the demo (degrees)
DEFAULT_INCIDENCE = 38.5
DEFAULT_RANGE_SPACING = 2.329562
DEFAULT_AZIMUTH_SPACING = 13.968

DEMO_ROWS = 220
DEMO_COLS = 260
DEMO_N_DATES = 24
DEMO_INTERVAL_DAYS = 12
DEMO_START = "2023-01-03"
DEMO_MULTILOOK = 2

#: Default processing parameters.  These are intentionally conservative and
#: suitable for a first run by a non-expert; expert mode exposes every value.
DEFAULT_PROCESSING = {
    "multilook": 2,
    "coherence_window": 5,
    "filter": "boxcar",
    "boxcar_size": 3,
    "goldstein_alpha": 0.65,
    "goldstein_patch": 32,
    "unwrap_method": "quality",
    "unwrap_coherence_min": 0.18,
    "max_temporal_baseline": 72,
    "max_perpendicular_baseline": 180.0,
    "max_pairs": 80,
    "ps_star_all": False,
    "min_coherence": 0.25,
    "estimate_dem_error": False,
    "remove_ramp": True,
    "ramp_order": 1,
    "robust_iterations": 3,
    "regularization": 0.30,
    "aps_method": "spatiotemporal",
    "aps_sigma_px": 10.0,
    "aps_iterations": 1,
    "reference_mode": "auto",
    "ps_adi_max": 0.28,
    "ps_coherence_min": 0.62,
    "ds_patch": 9,
    "dinsar_pair": None,
}

STEP_ORDER = [
    "project",
    "data",
    "network",
    "interferometry",
    "atmosphere",
    "inversion",
    "results",
]

STEP_LABELS = {
    "project": "项目",
    "data": "数据",
    "network": "网络",
    "interferometry": "干涉",
    "atmosphere": "大气",
    "inversion": "反演",
    "results": "成果",
}

TECHNIQUES = {
    "dinsar": {
        "name": "D-InSAR",
        "cn": "差分干涉",
        "tagline": "单对 / 多对差分，快速提取同震、滑坡、沉降形变",
        "min_dates": 2,
        "default_pairs": 1,
        "badge": "快速",
    },
    "ps": {
        "name": "PS-InSAR",
        "cn": "永久散射体",
        "tagline": "相干点时序，适合城市建筑、岩石裸露区毫米级监测",
        "min_dates": 8,
        "badge": "毫米级",
    },
    "sbas": {
        "name": "SBAS-InSAR",
        "cn": "小基线集",
        "tagline": "多主影像小基线网络，兼顾精度与空间覆盖",
        "min_dates": 8,
        "badge": "推荐",
    },
    "ds": {
        "name": "DS-InSAR",
        "cn": "分布式散射体",
        "tagline": "相位链接 + 自适应多视，适合植被、裸土等低相干区",
        "min_dates": 10,
        "badge": "前沿",
    },
}

FORMATS = {
    "stack": ["目录 + stack.json", "ENVI 栈 (.hdr/.dat)", "NumPy 栈 (.npy/.npz)"],
    "slc": ["Sentinel-1 SAFE（实验）", "复数 GeoTIFF", "SNAP/ISCE 导出"],
    "dem": [".tif/.tiff", ".npy", "ENVI"],
    "gacos": [
        "官方 tar.gz（申请 / 邮件导入）",
        "邮件页面下载链接",
        ".ztd + .rsc",
        ".tif",
        ".npy",
    ],
}


def ensure_runtime_dirs() -> None:
    for path in (WORKSPACE_DIR, PROJECTS_DIR, CACHE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def new_project_id() -> str:
    import secrets
    import time

    return f"prj_{int(time.time()):x}_{secrets.token_hex(3)}"
