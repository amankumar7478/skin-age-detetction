"""Depth -> curvature utilities (shared by the demo script, training and the Streamlit app)."""
import cv2
import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def mean_curvature(depth: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Approximate mean curvature of the surface z = depth(x, y).

    depth : 2-D array (MiDaS relative inverse depth is fine; it is normalised to [0, 1] first).
    sigma : Gaussian smoothing before differentiating (MiDaS maps are noisy, second
            derivatives amplify noise, so some smoothing is essential).
    Returns a float32 map; sign = convex/concave, magnitude = how strongly curved.
    """
    z = normalize01(depth)
    if sigma > 0:
        z = cv2.GaussianBlur(z, (0, 0), sigma)
    zx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    zy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    zxx = cv2.Sobel(z, cv2.CV_32F, 2, 0, ksize=3) / 8.0
    zyy = cv2.Sobel(z, cv2.CV_32F, 0, 2, ksize=3) / 8.0
    zxy = cv2.Sobel(z, cv2.CV_32F, 1, 1, ksize=3) / 4.0
    num = (1 + zx ** 2) * zyy - 2 * zx * zy * zxy + (1 + zy ** 2) * zxx
    den = 2 * (1 + zx ** 2 + zy ** 2) ** 1.5
    return (num / den).astype(np.float32)


def curvature_for_display(curv: np.ndarray, clip_pct: float = 99.0) -> np.ndarray:
    """Symmetric clip + rescale to [0, 1] so 0.5 = flat (good for a diverging colormap)."""
    lim = np.percentile(np.abs(curv), clip_pct) + 1e-8
    return np.clip(curv, -lim, lim) / (2 * lim) + 0.5


def silhouette_mask(depth: np.ndarray, sigma: float = 2.0, frac: float = 0.25, dilate: int = 9) -> np.ndarray:
    """Boolean mask of pixels sitting on (or next to) a depth discontinuity such as the head outline.

    These edges produce huge curvature spikes that drown out the real facial surface.
    A pixel is flagged when its depth gradient exceeds `frac` x the largest gradient in the image
    (adaptive, so it works whether the head is big or small), then the flag is dilated by `dilate` px.
    """
    z = normalize01(depth)
    if sigma > 0:
        z = cv2.GaussianBlur(z, (0, 0), sigma)
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx ** 2 + gy ** 2)
    mask = (g > frac * g.max()).astype(np.uint8)
    if dilate > 1:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return mask.astype(bool)


def suppress_edges(curv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero the curvature on silhouette pixels (those pixels are then treated as 'flat')."""
    out = curv.copy()
    out[mask] = 0.0
    return out


def curvature_for_display_masked(curv: np.ndarray, mask: np.ndarray, clip_pct: float = 98.0) -> np.ndarray:
    """Like curvature_for_display, but the colour scale is set only by the unmasked pixels."""
    valid = np.abs(curv[~mask])
    lim = (np.percentile(valid, clip_pct) if valid.size else 1.0) + 1e-8
    return np.clip(curv, -lim, lim) / (2 * lim) + 0.5
