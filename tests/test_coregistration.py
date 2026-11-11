"""Coregistration transform-model regression tests."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from insar_studio.core.coregistration import estimate_transform


def _texture(rows: int = 160, cols: int = 200, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.random((rows, cols)).astype(np.float32), 3)
    base = (base - base.min()) / (base.max() - base.min()) + 0.2
    return base.astype(np.float32)


def test_affine_model_performs_a_real_affine_fit():
    ref = _texture()
    mov = ndimage.shift(ref, shift=(3, -4), order=3, mode="reflect")

    _, info = estimate_transform(ref, mov, model="affine", tile=48, upsample=20)

    assert info["model"] == "affine"
    assert "matrix" in info
    assert info["affine_rms_px"] < 0.75
    matrix = np.asarray(info["matrix"], dtype=float)
    # A pure translation must not be mistaken for the old fallback path.
    assert abs(matrix[0, 1]) < 0.05
    assert abs(matrix[1, 0]) < 0.05


def test_similarity_model_returns_conformal_transform():
    ref = _texture()
    mov = ndimage.shift(ref, shift=(3.2, -2.4), order=3, mode="reflect")

    _, info = estimate_transform(ref, mov, model="similarity", tile=48, upsample=20)

    assert info["model"] == "similarity"
    assert "similarity_rms_px" in info
    assert info["similarity_rms_px"] < 0.75
    assert abs(info["scale"] - 1.0) < 0.01
    assert abs(info["rotation_deg"]) < 0.2
    matrix = np.asarray(info["matrix"], dtype=float)
    # Similarity matrices are conformal: a = d and b = -c.
    assert np.isclose(matrix[0, 0], matrix[1, 1], atol=1e-9)
    assert np.isclose(matrix[0, 1], -matrix[1, 0], atol=1e-9)


def test_translation_model_skips_matrix_fit():
    ref = _texture(rows=96, cols=120)

    _, info = estimate_transform(ref, ref, model="translation", tile=32, upsample=10)

    assert info["model"] == "translation"
    assert "matrix" not in info
    assert info["shift_row"] == 0.0
    assert info["shift_col"] == 0.0
