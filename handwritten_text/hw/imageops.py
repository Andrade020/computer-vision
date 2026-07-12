"""
Lightweight image ops shared by the renderer and the math engine (numpy+scipy
only -- no matplotlib, so importing this from the text path stays cheap).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, grey_dilation, grey_erosion


def disk(r):
    r = int(max(1, round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def stroke_px(alpha, thr=0.45):
    """Stroke width in px ~= 2 x median distance-to-edge over the ink."""
    ink = alpha > thr
    if ink.sum() < 4:
        return 0.0
    d = distance_transform_edt(ink)
    return float(2.0 * np.median(d[ink]))


def normalize_stroke(mask, target_px, strength=1.0, max_grow=7.0, max_shrink=5.0):
    """Bring a glyph's stroke width toward target_px so all glyphs share a
    consistent weight on the page (thin strokes dilate, thick ones erode).
    strength in [0,1] blends between the original and fully-normalized weight."""
    if strength <= 0 or mask.max() <= 0:
        return mask
    cur = stroke_px(mask)
    if cur <= 0:
        return mask
    delta = (target_px - cur) * strength
    if delta > 0.5:
        return grey_dilation(mask, footprint=disk(min(delta / 2.0, max_grow)))
    if delta < -0.5:
        return grey_erosion(mask, footprint=disk(min(-delta / 2.0, max_shrink)))
    return mask
