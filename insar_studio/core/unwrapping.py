"""2-D phase unwrapping.

The default is a coherence-guided region-growing unwrapper that respects
invalid masks and disconnected low-coherence patches.  SNAPHU and
scikit-image's least-squares unwrapper are available as alternatives.
"""

from __future__ import annotations

import heapq
import itertools
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .utils import TWO_PI, fill_nan_nearest, wrap_phase


def _align_components(unwrapped: np.ndarray, valid: np.ndarray, labels: np.ndarray, main_label: int) -> np.ndarray:
    out = unwrapped.copy()
    main = labels == main_label
    if not main.any():
        return out
    dist, (ri, ci) = ndimage.distance_transform_edt(~main, return_indices=True, return_distances=True)
    main_values = np.where(main, out, np.nan)
    nearest_main = main_values[ri, ci]
    for lab in range(1, int(labels.max()) + 1):
        if lab == main_label:
            continue
        comp = labels == lab
        boundary = comp & (dist <= 1.5) & np.isfinite(nearest_main)
        if boundary.any():
            delta = np.nanmedian(nearest_main[boundary] - out[boundary])
            if np.isfinite(delta):
                out[comp] += delta
    return out


def unwrap_quality_guided(
    wrapped: np.ndarray,
    quality: np.ndarray,
    valid: np.ndarray,
    *,
    coherence_min: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Region-growing unwrapping with a max-quality priority queue."""
    wrapped = np.asarray(wrapped, dtype=np.float64)
    quality = np.nan_to_num(np.asarray(quality, dtype=np.float64), nan=0.0)
    mask = np.asarray(valid, dtype=bool) & (quality >= coherence_min) & np.isfinite(wrapped)
    rows, cols = wrapped.shape
    if mask.sum() < 20:
        raise ValueError("有效像元不足，无法解缠（请降低相干性阈值或检查数据）")

    flat_q = quality.ravel()
    flat_w = wrapped.ravel()
    flat_m = mask.ravel()
    order = np.argsort(np.where(flat_m, flat_q, -np.inf))[::-1]
    order = order[np.isfinite(flat_q[order])]
    visited = np.zeros(rows * cols, dtype=bool)
    out = np.full(rows * cols, np.nan, dtype=np.float64)
    heap: list[tuple[float, int, int, int]] = []
    counter = itertools.count()
    pointer = 0

    def push_neighbours(idx: int) -> None:
        r, c = divmod(idx, cols)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < rows and 0 <= cc < cols:
                j = rr * cols + cc
                if flat_m[j] and not visited[j]:
                    edge_q = min(flat_q[idx], flat_q[j])
                    heapq.heappush(heap, (-edge_q, next(counter), j, idx))

    seed = int(order[pointer])
    components = 0
    while True:
        if not visited[seed]:
            out[seed] = flat_w[seed]
            visited[seed] = True
            push_neighbours(seed)
            components += 1
        while heap:
            _, _, idx, parent = heapq.heappop(heap)
            if visited[idx]:
                continue
            # choose the best already-unwrapped neighbour (robust to stale parents)
            r, c = divmod(idx, cols)
            best_parent = parent
            best_q = -1.0
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    if visited[j] and flat_q[j] > best_q:
                        best_q = flat_q[j]
                        best_parent = j
            k = round((out[best_parent] - flat_w[idx]) / TWO_PI)
            out[idx] = flat_w[idx] + TWO_PI * k
            visited[idx] = True
            push_neighbours(idx)
        # next disconnected component
        while pointer < len(order) and visited[int(order[pointer])]:
            pointer += 1
        if pointer >= len(order):
            break
        seed = int(order[pointer])

    unwrapped = out.reshape(rows, cols)
    unwrapped[~mask] = np.nan

    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3)))
    if n_labels > 1:
        counts = ndimage.sum(mask, labels, range(1, n_labels + 1))
        main_label = int(np.argmax(counts)) + 1
        unwrapped = _align_components(unwrapped, mask, labels, main_label)
    else:
        main_label = 1

    info = {
        "method": "quality_guided",
        "valid_fraction": float(mask.mean()),
        "components": int(n_labels),
        "main_component_fraction": float((labels == main_label).sum() / max(mask.sum(), 1)),
        "coherence_min": float(coherence_min),
    }
    return unwrapped.astype(np.float32), info


def unwrap_skimage(wrapped: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from skimage.restoration import unwrap_phase

    masked = np.where(valid, wrapped, np.nan)
    filled = fill_nan_nearest(masked)
    unwrapped = unwrap_phase(filled.astype(np.float32), wrap_around=False)
    unwrapped = unwrapped.astype(np.float32)
    unwrapped[~valid] = np.nan
    return unwrapped, {
        "method": "skimage_least_squares",
        "valid_fraction": float(valid.mean()),
        "components": 1,
        "main_component_fraction": 1.0,
    }


def unwrap_snaphu(
    wrapped: np.ndarray,
    coherence: np.ndarray,
    valid: np.ndarray,
    *,
    executable: str = "snaphu",
    workdir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    exe = shutil.which(executable) or (executable if Path(executable).exists() else None)
    if not exe:
        raise RuntimeError("未找到 SNAPHU 可执行文件；请安装 SNAPHU 或在专家模式选择内置解缠器")
    rows, cols = wrapped.shape
    workdir = Path(workdir or tempfile.mkdtemp(prefix="insar_snaphu_"))
    workdir.mkdir(parents=True, exist_ok=True)
    infile = workdir / "phase.in"
    corrfile = workdir / "coh.in"
    outfile = workdir / "phase.out"
    cfg = workdir / "snaphu.cfg"
    phase = np.where(valid, wrapped, 0.0).astype(np.float32)
    phase.tofile(infile)
    np.where(valid, coherence, 0.0).astype(np.float32).tofile(corrfile)
    cfg.write_text(
        "\n".join(
            [
                "INFILE phase.in",
                "OUTFILE phase.out",
                "CORRFILE coh.in",
                "CORRFILEFORMAT FLOAT_DATA",
                "LINELENGTH %d" % cols,
                "ORIGIN 0 0",
            ]
        ),
        encoding="ascii",
    )
    proc = subprocess.run(
        [exe, "-f", str(cfg.name)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0 or not outfile.exists():
        raise RuntimeError(f"SNAPHU 运行失败: {proc.stderr[-500:]}")
    unwrapped = np.fromfile(outfile, dtype=np.float32).reshape(rows, cols)
    unwrapped[~valid] = np.nan
    return unwrapped, {"method": "snaphu", "valid_fraction": float(valid.mean()), "components": 1, "main_component_fraction": 1.0}


def unwrap(
    wrapped: np.ndarray,
    coherence: np.ndarray,
    valid: np.ndarray,
    *,
    method: str = "quality",
    coherence_min: float = 0.0,
    snaphu_exe: str = "snaphu",
) -> tuple[np.ndarray, dict[str, Any]]:
    method = (method or "quality").lower()
    mask = valid
    if method == "snaphu":
        try:
            return unwrap_snaphu(wrapped, coherence, valid, executable=snaphu_exe)
        except Exception as exc:
            result, info = unwrap_quality_guided(wrapped, coherence, valid, coherence_min=coherence_min)
            info["fallback"] = f"SNAPHU 不可用，已回退内置算法: {exc}"
            return result, info
    if method in ("skimage", "least_squares", "ls"):
        return unwrap_skimage(wrapped, mask & (coherence >= coherence_min))
    return unwrap_quality_guided(wrapped, coherence, mask, coherence_min=coherence_min)
