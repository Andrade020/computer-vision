"""Vectorized Kuwahara edge-preserving smoothing filter.

The original `kuwahara_filter` was a manual per-pixel Python loop --
O(H*W*window_size^2) -- which is why the GUI hardcoded a 400px resize on
load and warned above window=9 ("may cause slow processing"). This version
computes the four quadrant means/variances for *every* pixel of the image at
once via 2D summed-area tables (integral images: `S = cumsum(cumsum(pad(a)))`
lets any axis-aligned rectangle's sum be read off in O(1) with a handful of
array lookups), so the whole image is a fixed number of numpy operations
independent of a per-pixel loop -- no cap needed regardless of window size.

It also fixes a real bug found while porting: the original computed `idx`,
the index of the quadrant with the lowest brightness variance, but then
re-sliced `image[tl_y:br_y+1, tl_x:br_x+1]` -- the union of *all four*
quadrants, not `quadrants[idx]` -- to compute the output color. `idx` was
therefore computed and silently discarded, and the filter always behaved
like a plain box blur regardless of local structure, never a true
edge-preserving Kuwahara filter. This version actually uses the winning
quadrant's mean color, which is what makes Kuwahara smooth flat regions
while keeping edges sharp.
"""
import cv2
import numpy as np


def _integral(a):
    """2D summed-area table, zero-padded on top/left, so that the sum over
    the inclusive rectangle rows [r0, r1], cols [c0, c1] is:
        S[r1+1, c1+1] - S[r0, c1+1] - S[r1+1, c0] + S[r0, c0]
    """
    return np.pad(a.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _box_sum(integral, r0, r1, c0, c1):
    """Sum over the inclusive rectangle [r0,r1] x [c0,c1] for every pixel of
    the image at once -- r0/r1/c0/c1 are (H,W) arrays of per-pixel bounds,
    so this is a vectorized summed-area-table lookup (numpy fancy indexing),
    never a per-pixel Python loop."""
    return (integral[r1 + 1, c1 + 1] - integral[r0, c1 + 1]
            - integral[r1 + 1, c0] + integral[r0, c0])


def kuwahara_filter(image, window_size):
    """Apply the Kuwahara edge-preserving filter to a BGR image.

    For every pixel, four overlapping quadrant windows anchored at that
    pixel (clamped at the image border -- same convention as the original
    manual-loop version) are compared by the variance of their HSV
    brightness (V channel). The quadrant with the lowest variance -- the
    flattest one, least likely to straddle an edge -- has its mean BGR
    color written to the output pixel.

    Parameters:
        image: BGR uint8 image.
        window_size: window size; even values are bumped up by one (the
            original GUI's slider could land on an even value).

    Returns:
        Filtered BGR uint8 image, same shape as the input.
    """
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].astype(np.float64)
    bgr = image.astype(np.float64)

    integral_b = _integral(brightness)
    integral_b2 = _integral(brightness ** 2)
    integral_c = [_integral(bgr[:, :, ch]) for ch in range(3)]

    y, x = np.mgrid[0:h, 0:w]
    y0 = np.clip(y - half, 0, h - 1)
    y2 = np.clip(y + half, 0, h - 1)
    x0 = np.clip(x - half, 0, w - 1)
    x2 = np.clip(x + half, 0, w - 1)

    # (row_start, row_end, col_start, col_end), inclusive -- one tuple per
    # quadrant, in the same order as the original: top-left, top-right,
    # bottom-left, bottom-right, all sharing the corner at (y, x).
    quadrants = [
        (y0, y, x0, x),
        (y0, y, x, x2),
        (y, y2, x0, x),
        (y, y2, x, x2),
    ]

    variances = []
    for r0, r1, c0, c1 in quadrants:
        count = (r1 - r0 + 1) * (c1 - c0 + 1)
        s = _box_sum(integral_b, r0, r1, c0, c1)
        s2 = _box_sum(integral_b2, r0, r1, c0, c1)
        mean = s / count
        var = np.maximum(s2 / count - mean ** 2, 0.0)
        variances.append(var)
    best = np.argmin(np.stack(variances, axis=-1), axis=-1)  # (h, w) in 0..3

    result = np.empty_like(bgr)
    for ch in range(3):
        chan_means = []
        for r0, r1, c0, c1 in quadrants:
            count = (r1 - r0 + 1) * (c1 - c0 + 1)
            chan_means.append(_box_sum(integral_c[ch], r0, r1, c0, c1) / count)
        chan_means = np.stack(chan_means, axis=-1)  # (h, w, 4)
        result[:, :, ch] = np.take_along_axis(chan_means, best[..., None], axis=-1)[..., 0]

    return np.clip(result, 0, 255).astype(np.uint8)
