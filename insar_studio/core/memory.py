"""Memory budgeting for large SLC stacks.

The engine keeps the SLC stack memory-mapped, but coregistration, pair
formation and inversion still need full-resolution or pair-wide work arrays.
This module centralises the estimates and a preflight guard so an oversized
job fails with an actionable message instead of being killed by the operating
system.

All estimates are deliberately approximate.  They are used to decide whether a
job is clearly beyond the machine's capacity, not to reserve memory.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
GIB = 1024 * MIB

# Keep this much headroom for the Python process, the browser, the operating
# system and memory-mapped page cache.  The guard never tries to use all of the
# reported free memory.
MEMORY_HEADROOM_RATIO = 0.10
MEMORY_HEADROOM_MIN_BYTES = 256 * MIB


class MemoryBudgetError(RuntimeError):
    """Raised before a stage whose estimated peak clearly exceeds free RAM."""


def _format_bytes(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.1f} GiB"
    if value >= MIB:
        return f"{value / MIB:.0f} MiB"
    return f"{value / 1024:.0f} KiB"


def _parse_budget(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?", text, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    scale = {"": 1, "k": 1024, "m": MIB, "g": GIB, "t": 1024 * GIB}[unit]
    result = int(number * scale)
    return result if result > 0 else None


def parse_size(value: str) -> int | None:
    """Parse ``123``, ``4M`` or ``2GiB`` into bytes; return None if invalid."""
    return _parse_budget(value)


def _windows_available_bytes() -> int | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        return None


def _linux_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _posix_available_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except Exception:
        return None
    return None


def available_memory_bytes() -> int | None:
    """Best-effort available physical memory in bytes.

    ``INSAR_MEMORY_BUDGET_BYTES`` overrides detection.  The value may be a
    plain byte count or use ``K/M/G/T`` suffixes (for example ``16G``).
    """
    override = _parse_budget(os.environ.get("INSAR_MEMORY_BUDGET_BYTES", ""))
    if override is not None:
        return override
    if sys.platform == "win32":
        value = _windows_available_bytes()
        if value:
            return value
    elif sys.platform.startswith("linux"):
        value = _linux_available_bytes()
        if value:
            return value
    elif sys.platform == "darwin":
        value = _posix_available_bytes()
        if value:
            return value
    return None


def usable_memory_bytes(available: int | None = None) -> int | None:
    """Return the portion of free memory that a stage may use."""
    value = available if available is not None else available_memory_bytes()
    if not value or value <= 0:
        return None
    headroom = max(int(value * MEMORY_HEADROOM_RATIO), MEMORY_HEADROOM_MIN_BYTES)
    return max(0, value - headroom)


def choose_block_rows(
    rows: int,
    cols: int,
    *,
    budget_bytes: int = 256 * MIB,
    arrays: int = 3,
    itemsize: int = 8,
    multiple: int = 1,
) -> int:
    """Choose a row-block height that keeps a working set under a budget."""
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    multiple = max(1, int(multiple))
    bytes_per_row = max(1, cols * max(1, int(arrays)) * max(1, int(itemsize)))
    block = max(multiple, int(max(1, budget_bytes) // bytes_per_row))
    block = max(multiple, (block // multiple) * multiple)
    return max(1, min(rows, block))


def estimate_coregistration_peak(rows: int, cols: int) -> int:
    """Peak RAM for resampling one acquisition while writing a memmapped output.

    ``warp`` needs contiguous real and imaginary inputs, two float32 outputs,
    and the final complex image.  The reference amplitude and a small margin
    are included; the output stack itself is memory-mapped and not counted.
    """
    image = max(1, int(rows)) * max(1, int(cols)) * 8
    return int(image * 3.5)


def estimate_interferometry_peak(rows: int, cols: int, multilook: int = 2) -> int:
    """Peak RAM for one pair after block-wise multilook."""
    m = max(1, int(multilook))
    block_rows = choose_block_rows(
        max(1, int(rows)),
        max(1, int(cols)),
        budget_bytes=256 * MIB,
        arrays=3,
        itemsize=8,
        multiple=m,
    )
    block = block_rows * max(1, int(cols)) * 3 * 8
    out_pixels = max(1, int(rows) // m) * max(1, int(cols) // m)
    # ifg / ref / sec (complex64), filtered (complex64), wrapped (float32),
    # coherence / unwrapped / displacement / mask (float32 / bool).
    multilooked = out_pixels * (8 * 3 + 8 + 4 * 4 + 1)
    return int(block * 2.0 + multilooked + rows * cols * 16.0)


def inversion_block_pixels(n_dates: int) -> int:
    """Pixel-block height for batched weighted normal equations.

    ``GtWG`` is ``n_dates x n_dates`` per pixel, so the block is reduced on
    long time series to keep the einsum/solve workspace bounded.  On a 48-date
    stack this picks 3 640 pixels, close to the historical 4 096 block.
    """
    n = max(1, int(n_dates))
    per_pixel = 4 * n * n * 8  # GtWG plus einsum and solve temporaries
    budget = 256 * MIB
    return int(max(512, min(4096, budget // max(1, per_pixel))))


def estimate_inversion_peak(n_pairs: int, n_pixels: int, n_dates: int, *, aps: bool = True) -> int:
    """Peak RAM for the pair cube, batched solve and residual reduction.

    The 2026-11 PS large-stack run (93 pairs, 490 000 pixels, 48 dates)
    measured about 1.65 GB after the inversion rewrite, down from 5.16 GB with
    the old float64/intermediate-heavy solver.  The breakdown below keeps a
    15 % margin over that measured working set.
    """
    p = max(1, int(n_pairs))
    npix = max(1, int(n_pixels))
    n = max(1, int(n_dates))
    # phase cube + corrected copy + coherence + bool mask + float64 Y
    pair_arrays = p * npix * (4 + 4 + 4 + 1 + 8)
    # X, displacement, APS screens and velocity helpers
    date_arrays = n * npix * (4 + 4 + 4 + 4)
    # Batched normal equations: GtWG and its einsum/solve temporaries.
    block = inversion_block_pixels(n)
    workspace = 4 * n * n * block * 8
    return int((pair_arrays + date_arrays + workspace) * 1.15 + 64 * MIB)


def ensure_memory_budget(required_bytes: int, stage: str, *, available: int | None = None) -> None:
    """Raise :class:`MemoryBudgetError` when a stage clearly cannot fit."""
    required = int(required_bytes or 0)
    if required <= 0:
        return
    free = available if available is not None else available_memory_bytes()
    usable = usable_memory_bytes(free)
    if usable is None or required <= usable:
        return
    free_text = _format_bytes(free) if free else "未知"
    raise MemoryBudgetError(
        f"{stage}预计峰值内存约 {_format_bytes(required)}，当前可用物理内存约 {free_text}。"
        "为避免被系统强制终止，任务已提前停止。请增大 multilook、减少日期或干涉对数、"
        "关闭 APS 或 DEM 误差估计，或在内存更大的机器上运行。"
        "如需覆盖自动检测值，可设置环境变量 INSAR_MEMORY_BUDGET_BYTES（字节，或带 K/M/G/T 后缀）。"
    )


def memory_report() -> dict[str, Any]:
    """Small diagnostic payload used by tests and doctor-like callers."""
    free = available_memory_bytes()
    return {
        "available_bytes": free,
        "available_human": _format_bytes(free) if free else None,
        "usable_bytes": usable_memory_bytes(free),
        "usable_human": _format_bytes(usable_memory_bytes(free)) if free else None,
    }
