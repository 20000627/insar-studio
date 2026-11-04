"""Interferometric network construction and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_PROCESSING
from .plotting import save_network
from .utils import atomic_write_json, parse_date


def _pair(i: int, j: int, dates: list[str], bperp: list[float]) -> dict[str, Any]:
    i, j = sorted((int(i), int(j)))
    di = parse_date(dates[i])
    dj = parse_date(dates[j])
    return {
        "id": f"{i:02d}_{j:02d}",
        "ref": i,
        "sec": j,
        "ref_date": dates[i],
        "sec_date": dates[j],
        "dt_days": int((dj - di).days),
        "bperp_ref": float(bperp[i]),
        "bperp_sec": float(bperp[j]),
        "bperp_diff": float(bperp[j] - bperp[i]),
    }


def _components(n: int, pairs: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        union(p["ref"], p["sec"])
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_network(
    dates: list[str],
    bperp: list[float],
    technique: str = "sbas",
    params: dict[str, Any] | None = None,
    *,
    ref_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p = {**DEFAULT_PROCESSING, **(params or {})}
    n = len(dates)
    technique = technique.lower()
    pairs: list[dict[str, Any]] = []

    if technique == "dinsar":
        chosen = p.get("dinsar_pair")
        if chosen and len(chosen) == 2:
            pairs.append(_pair(chosen[0], chosen[1], dates, bperp))
        else:
            i = max(0, n // 3)
            pairs.append(_pair(i, n - 1, dates, bperp))
    elif technique == "ps":
        # single-reference star + adjacent pairs for loop closure
        for j in range(n):
            if j != ref_index:
                if p.get("ps_star_all", True):
                    pairs.append(_pair(ref_index, j, dates, bperp))
                else:
                    dt = abs((parse_date(dates[j]) - parse_date(dates[ref_index])).days)
                    if dt <= p["max_temporal_baseline"]:
                        pairs.append(_pair(ref_index, j, dates, bperp))
        for j in range(n - 1):
            pairs.append(_pair(j, j + 1, dates, bperp))
    else:
        max_dt = int(p.get("max_temporal_baseline", 72))
        max_bp = float(p.get("max_perpendicular_baseline", 180.0))
        for i in range(n - 1):
            for j in range(i + 1, n):
                dt = abs((parse_date(dates[j]) - parse_date(dates[i])).days)
                db = abs(float(bperp[j]) - float(bperp[i]))
                if dt <= max_dt and db <= max_bp:
                    pairs.append(_pair(i, j, dates, bperp))

    # de-duplicate and sort
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in pairs:
        unique[(pair["ref"], pair["sec"])] = pair
    pairs = [unique[k] for k in sorted(unique)]

    # ensure every acquisition is connected; add nearest temporal bridge
    groups = _components(n, pairs)
    if len(groups) > 1 and technique != "dinsar":
        for gi in range(len(groups) - 1):
            a, b = groups[gi], groups[gi + 1]
            best = min(
                ((abs((parse_date(dates[x]) - parse_date(dates[y])).days), x, y) for x in a for y in b),
                key=lambda t: (t[0], abs(bperp[t[1]] - bperp[t[2]])),
            )
            pairs.append(_pair(best[1], best[2], dates, bperp))
        unique = {(p["ref"], p["sec"]): p for p in pairs}
        pairs = [unique[k] for k in sorted(unique)]
        groups = _components(n, pairs)

    # cap very large networks while preserving connectivity / time coverage
    max_pairs = int(p.get("max_pairs", 80))
    if len(pairs) > max_pairs and technique != "dinsar":
        # keep short-baseline pairs first, then ensure every date appears
        ranked = sorted(pairs, key=lambda q: (q["dt_days"], abs(q["bperp_diff"])))
        keep: dict[tuple[int, int], dict[str, Any]] = {}
        for date_idx in range(n):
            for q in ranked:
                if q["ref"] == date_idx or q["sec"] == date_idx:
                    keep[(q["ref"], q["sec"])] = q
                    break
        for q in ranked:
            if len(keep) >= max_pairs:
                break
            keep[(q["ref"], q["sec"])] = q
        trial = [keep[k] for k in sorted(keep)]
        # only accept the cap when the date graph stays connected
        if len(_components(n, trial)) == len(groups):
            pairs = trial

    groups = _components(n, pairs)
    redundancy = len(pairs) / max(n - 1, 1)
    dts = np.asarray([p["dt_days"] for p in pairs], dtype=float)
    bps = np.asarray([abs(p["bperp_diff"]) for p in pairs], dtype=float)
    stats = {
        "n_pairs": len(pairs),
        "n_dates": n,
        "redundancy": float(redundancy),
        "connected_components": len(groups),
        "max_temporal_days": float(dts.max()) if len(dts) else 0.0,
        "mean_temporal_days": float(dts.mean()) if len(dts) else 0.0,
        "max_perpendicular_m": float(bps.max()) if len(bps) else 0.0,
        "technique": technique,
        "ref_index": int(ref_index),
    }
    return pairs, stats


def network_closure_candidates(pairs: list[dict[str, Any]]) -> int:
    have = {(p["ref"], p["sec"]) for p in pairs}
    count = 0
    for a, b in have:
        for c in range(int(b) + 1, max(y for _, y in have) + 1):
            if (b, c) in have and (a, c) in have:
                count += 1
    return count


def persist_network(project_root: Path, pairs: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root)
    payload = {"pairs": pairs, "stats": stats}
    atomic_write_json(root / "data" / "network.json", payload)
    try:
        from .io import load_stack

        stack = load_stack(root)
        save_network(
            pairs,
            stack.dates,
            stack.meta.get("bperp_m", [0.0] * stack.n_dates),
            root / "products" / "figures" / "network.png",
            ref_index=int(stats.get("ref_index", 0)),
        )
    except Exception:
        pass
    return payload
