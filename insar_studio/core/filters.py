"""Interferometric phase filters."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def goldstein_filter(ifg: np.ndarray, alpha: float = 0.65, patch: int = 32) -> np.ndarray:
    """Goldstein adaptive spectral filter.

    Strong spectral peaks (dominant scatterers) are preserved while the
    low-amplitude speckle floor is suppressed.
    """
    z = np.fft.fftshift(np.fft.fft2(ifg))
    amp = np.abs(z)
    smooth = ndimage.uniform_filter(amp, size=max(3, int(patch)), mode="nearest")
    ratio = smooth / (amp + 1e-8)
    gain = np.clip(ratio**float(alpha), 0.05, 8.0)
    # do not amplify the DC term
    cr, cc = np.array(z.shape) // 2
    gain[cr, cc] = 1.0
    out = np.fft.ifft2(np.fft.ifftshift(z * gain))
    return out.astype(np.complex64)


def boxcar_filter(ifg: np.ndarray, size: int = 5) -> np.ndarray:
    size = max(1, int(size))
    return (
        ndimage.uniform_filter(ifg.real, size=size, mode="nearest")
        + 1j * ndimage.uniform_filter(ifg.imag, size=size, mode="nearest")
    ).astype(np.complex64)


def gaussian_filter_complex(ifg: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return (
        ndimage.gaussian_filter(ifg.real, sigma=sigma, mode="nearest")
        + 1j * ndimage.gaussian_filter(ifg.imag, sigma=sigma, mode="nearest")
    ).astype(np.complex64)


def nlmeans_filter(ifg: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Non-local means on the real/imaginary parts (experimental).

    This suppresses speckle while retaining discontinuities better than a
    boxcar, at the cost of runtime.  Falls back to Goldstein-like behaviour if
    scikit-image is unavailable.
    """
    try:
        from skimage.restoration import denoise_nl_means

        sigma = max(0.02, float(np.std(ifg.real)) * 0.25)
        h = max(0.05, strength * sigma)
        real = denoise_nl_means(
            ifg.real.astype(np.float32),
            h=h,
            sigma=sigma,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
            preserve_range=True,
        )
        imag = denoise_nl_means(
            ifg.imag.astype(np.float32),
            h=h,
            sigma=sigma,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
            preserve_range=True,
        )
        return (real + 1j * imag).astype(np.complex64)
    except Exception:
        return goldstein_filter(ifg, alpha=0.5, patch=24)


def filter_complex(ifg: np.ndarray, method: str = "goldstein", **kwargs) -> np.ndarray:
    method = (method or "goldstein").lower()
    if method == "none":
        return ifg.astype(np.complex64)
    if method == "boxcar":
        return boxcar_filter(ifg, size=int(kwargs.get("size", 5)))
    if method in ("gaussian", "gauss"):
        return gaussian_filter_complex(ifg, sigma=float(kwargs.get("sigma", 2.0)))
    if method in ("nlmeans", "nl-means", "nlm"):
        return nlmeans_filter(ifg, strength=float(kwargs.get("strength", 0.5)))
    return goldstein_filter(
        ifg,
        alpha=float(kwargs.get("alpha", 0.65)),
        patch=int(kwargs.get("patch", 32)),
    )


FILTERS = {
    "goldstein": "Goldstein 自适应谱滤波",
    "boxcar": "Boxcar 多视平滑",
    "gaussian": "高斯低通",
    "nlmeans": "NL-Means 非局部滤波",
    "none": "不滤波",
}
