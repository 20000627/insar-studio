"""Validation, normalisation and fingerprinting of processing parameters.

The web UI only exposes a curated subset of the engine parameters.  Direct API
clients and imported projects can nevertheless contain arbitrary values, so
every entry point funnels parameters through :func:`validate_params` before a
job is submitted or a project is saved.

The helpers here are deliberately dependency-free (standard library only) and
keep the existing permissive contract: unknown keys are preserved, while known
keys are type-checked and range-checked with a readable Chinese error message.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from .config import DEFAULT_PROCESSING


class ParamError(ValueError):
    """Raised when a processing parameter cannot be used safely."""


SECRET_HINTS = ("password", "secret", "token", "authorization", "credential")

BOOL_KEYS = {
    "estimate_dem_error",
    "remove_ramp",
    "ps_star_all",
}

INT_RANGES: dict[str, tuple[int, int]] = {
    "multilook": (1, 8),
    "coherence_window": (3, 31),
    "boxcar_size": (1, 31),
    "goldstein_patch": (8, 128),
    "max_temporal_baseline": (6, 730),
    "max_pairs": (1, 1000),
    "robust_iterations": (1, 12),
    "aps_iterations": (0, 6),
    "ds_patch": (3, 41),
    "ref_index": (0, 1_000_000),
    "reference_row": (0, 1_000_000),
    "reference_col": (0, 1_000_000),
    "tile": (16, 256),
    "upsample": (1, 100),
}

FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "max_perpendicular_baseline": (5.0, 3000.0),
    "min_coherence": (0.0, 0.99),
    "unwrap_coherence_min": (0.0, 0.99),
    "regularization": (0.0, 2.0),
    "goldstein_alpha": (0.0, 1.5),
    "gaussian_sigma": (0.3, 12.0),
    "aps_sigma_px": (1.0, 120.0),
    "ps_adi_max": (0.01, 0.8),
    "ps_coherence_min": (0.01, 0.99),
    "ds_overlap": (0.0, 0.95),
}

ENUMS: dict[str, set[str]] = {
    "filter": {"boxcar", "goldstein", "gaussian", "gauss", "nlmeans", "nl-means", "nlm", "none"},
    "unwrap_method": {"quality", "snaphu", "skimage", "least-squares", "least_squares"},
    "aps_method": {"spatiotemporal", "off"},
    "ds_phase_linking": {"emi", "eigenvalue"},
    "reference_mode": {"auto", "manual"},
    "atmosphere_method": {"gacos", "empirical", "off", "demo"},
    "gacos_source": {"demo", "local", "online"},
    "model": {"affine", "translation", "similarity"},
    "orbit_direction": {"ascending", "descending", "unknown"},
}

ODD_KEYS = {"coherence_window", "boxcar_size", "ds_patch"}

#: Parameters that influence each processing stage.  The hashes are used to
#: warn when results were produced with a different configuration.
STAGE_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "coregistration": ("ref_index", "model", "tile", "upsample"),
    "network": (
        "max_temporal_baseline",
        "max_perpendicular_baseline",
        "max_pairs",
        "ref_index",
    ),
    "interferometry": (
        "multilook",
        "coherence_window",
        "filter",
        "boxcar_size",
        "goldstein_alpha",
        "goldstein_patch",
        "gaussian_sigma",
        "min_coherence",
        "unwrap_method",
        "unwrap_coherence_min",
        "remove_ramp",
        "snaphu_exe",
    ),
    "atmosphere": (
        "atmosphere_method",
        "gacos_source",
        "gacos_email",
        "aps_method",
        "aps_sigma_px",
        "aps_iterations",
    ),
    "inversion": (
        "multilook",
        "estimate_dem_error",
        "remove_ramp",
        "robust_iterations",
        "regularization",
        "aps_method",
        "aps_sigma_px",
        "aps_iterations",
        "ps_adi_max",
        "ps_coherence_min",
        "ps_star_all",
        "ds_patch",
        "ds_overlap",
        "ds_phase_linking",
        "dinsar_pair",
        "ref_index",
        "reference_mode",
        "reference_row",
        "reference_col",
    ),
}


def _label(key: str) -> str:
    return key.replace("_", " ")


def _as_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or value is None or value == "":
        raise ParamError(f"参数 {_label(key)} 必须是整数")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParamError(f"参数 {_label(key)} 必须是整数") from exc
    if not math.isfinite(number) or abs(number - round(number)) > 1e-9:
        raise ParamError(f"参数 {_label(key)} 必须是整数")
    return int(round(number))


def _as_float(key: str, value: Any) -> float:
    if isinstance(value, bool) or value is None or value == "":
        raise ParamError(f"参数 {_label(key)} 必须是数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParamError(f"参数 {_label(key)} 必须是数字") from exc
    if not math.isfinite(number):
        raise ParamError(f"参数 {_label(key)} 必须是有限数字")
    return number


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on", "是"):
            return True
        if text in ("0", "false", "no", "off", "否", ""):
            return False
    raise ParamError(f"参数 {_label(key)} 必须是布尔值")


def is_secret_key(key: str) -> bool:
    lower = str(key).lower()
    return any(hint in lower for hint in SECRET_HINTS) or lower in {
        "gacos_email",
        "username",
        "user",
        "email",
    }


def public_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy safe to persist in project metadata and reports."""
    return {k: v for k, v in (params or {}).items() if not is_secret_key(k)}


def validate_params(
    params: dict[str, Any] | None,
    *,
    technique: str | None = None,
    n_dates: int | None = None,
) -> dict[str, Any]:
    """Validate and normalise a parameter dictionary.

    Unknown keys are retained untouched so that forward-compatible clients do
    not lose information.  Values that are known to the engine are converted
    to canonical Python types.
    """
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ParamError("参数必须是 JSON 对象")
    out: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key:
            raise ParamError("参数名必须是非空字符串")
        if value is None:
            # Null is only meaningful for optional selection parameters.
            if key in ("dinsar_pair", "reference_row", "reference_col"):
                out[key] = None
            continue
        if key in BOOL_KEYS:
            out[key] = _as_bool(key, value)
            continue
        if key in INT_RANGES:
            number = _as_int(key, value)
            low, high = INT_RANGES[key]
            if not low <= number <= high:
                raise ParamError(f"参数 {_label(key)} 必须在 {low}–{high} 之间")
            # ``ds_patch`` is only used by the DS route.  Legacy projects
            # may still carry the old even default, and rejecting an
            # irrelevant hidden value would block SBAS/PS/D-InSAR runs.
            relevant = key != "ds_patch" or technique in (None, "ds")
            if key in ODD_KEYS and number % 2 == 0 and relevant:
                raise ParamError(f"参数 {_label(key)} 必须是奇数（当前 {number}）")
            out[key] = number
            continue
        if key in FLOAT_RANGES:
            number = _as_float(key, value)
            low, high = FLOAT_RANGES[key]
            if not low <= number <= high:
                raise ParamError(f"参数 {_label(key)} 必须在 {low}–{high} 之间")
            out[key] = number
            continue
        if key in ENUMS:
            text = str(value).strip().lower()
            if text not in ENUMS[key]:
                allowed = " / ".join(sorted(ENUMS[key]))
                raise ParamError(f"参数 {_label(key)} 只能是 {allowed}")
            out[key] = text
            continue
        if key == "dinsar_pair":
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    out[key] = None
                elif len(text.split("_")) == 2 and all(part.isdigit() for part in text.split("_")):
                    out[key] = text
                else:
                    raise ParamError("dinsar_pair 必须是干涉对编号（如 00_01）或两个日期索引")
            elif isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise ParamError("dinsar_pair 必须包含两个日期索引")
                pair = sorted((_as_int("dinsar_pair", value[0]), _as_int("dinsar_pair", value[1])))
                if pair[0] < 0 or pair[0] == pair[1]:
                    raise ParamError("dinsar_pair 的两个日期索引必须不同且非负")
                out[key] = pair
            else:
                raise ParamError("dinsar_pair 格式不正确")
            continue
        if key == "snaphu_exe":
            text = str(value).strip()
            out[key] = text[:260]
            continue
        # Unknown / free-form values are retained, but must be JSON serialisable.
        try:
            json.dumps(value)
        except TypeError as exc:
            raise ParamError(f"参数 {_label(key)} 不能序列化") from exc
        out[key] = value

    if technique:
        technique = technique.lower()
        if technique not in ("sbas", "ps", "ds", "dinsar"):
            raise ParamError("未知技术路线")
    if n_dates is not None:
        for key in ("ref_index", "reference_row", "reference_col"):
            if key in out and out[key] is not None and int(out[key]) >= int(n_dates) and key == "ref_index":
                raise ParamError(f"参考影像索引必须在 0–{max(0, int(n_dates) - 1)} 之间")
        ref = out.get("dinsar_pair")
        if isinstance(ref, (list, tuple)) and n_dates and max(ref) >= int(n_dates):
            raise ParamError("D-InSAR 干涉对索引超出影像数量")
    return out


def merge_params(project_params: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Merge project and request parameters without persisting secrets."""
    merged = {**DEFAULT_PROCESSING, **public_params(project_params)}
    for key, value in (incoming or {}).items():
        merged[key] = value
    return merged


def with_atmosphere_defaults(params: dict[str, Any], *, demo: bool) -> dict[str, Any]:
    """Fill the implicit atmosphere method/source used by the engine.

    The defaults are data-source dependent (demo projects use simulated GACOS,
    imported projects use the empirical correction).  Recording them keeps
    stage fingerprints comparable even when the UI never had to save them.
    """
    out = dict(params)
    if not out.get("atmosphere_method"):
        out["atmosphere_method"] = "gacos" if demo else "empirical"
    if not out.get("gacos_source"):
        out["gacos_source"] = "demo" if demo else "local"
    return out


def fingerprint(
    params: dict[str, Any] | None,
    *,
    stage: str | None = None,
    technique: str | None = None,
) -> str:
    """Stable short hash of the parameters that affect a processing stage."""
    source = public_params(params)
    if stage and stage in STAGE_PARAM_KEYS:
        picked = {key: source.get(key) for key in STAGE_PARAM_KEYS[stage] if key in source}
    else:
        picked = source
    payload = {
        "technique": (technique or "").lower() or None,
        "stage": stage,
        "params": picked,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def changed_keys(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    stage: str | None = None,
) -> list[str]:
    """Return the public keys whose values differ for a given stage."""
    a = public_params(old)
    b = public_params(new)
    keys: Iterable[str] = STAGE_PARAM_KEYS.get(stage or "", ()) or set(a) | set(b)
    return sorted(key for key in keys if a.get(key) != b.get(key))
