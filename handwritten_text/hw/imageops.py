"""
Lightweight image ops shared by the renderer and the math engine (numpy+scipy
only -- no matplotlib, so importing this from the text path stays cheap).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           grey_dilation, grey_erosion, map_coordinates)


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


def elastic(alpha, sigma, amp, rng):
    """Smooth random displacement field -> organic hand tremor / wobble."""
    h, w = alpha.shape
    if h < 3 or w < 3 or amp <= 0:
        return alpha
    dx = gaussian_filter(rng.randn(h, w), sigma) * amp
    dy = gaussian_filter(rng.randn(h, w), sigma) * amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = map_coordinates(alpha, [(yy + dy).ravel(), (xx + dx).ravel()],
                             order=1, mode="constant").reshape(h, w)
    return warped.astype(np.float32)


def ink_texture(mask, strength=1.0, rng=None):
    """Ballpoint-pen ink density variation within a stroke.

    The OCR glyph bank stores hard binary masks (thresholded from the scan),
    so the real ink's density gradation was thrown away. This reintroduces a
    plausible version: the stroke stays essentially fully inked (real ballpoint
    ink is mostly solid), with gentle low-frequency "dry patch" / pressure
    variation dipping the opacity down in places -- same idea as this
    project's original simulate_ink, ported to numpy + scipy, but tuned much
    more conservatively (the naive version washed out thin strokes almost
    everywhere, since a distance-transform core is rare on a 3-6px-wide line).
    Returns a modulated alpha array, same shape as mask."""
    if strength <= 0 or mask.max() <= 0:
        return mask
    rng = rng or np.random.RandomState(0)
    if (mask > 0.35).sum() < 4:
        return mask

    h, w = mask.shape
    cell = max(2, min(h, w) // 4)
    small = rng.uniform(-1, 1, (h // cell + 2, w // cell + 2)).astype(np.float32)
    noise = np.asarray(Image.fromarray(((small + 1) * 127.5).astype(np.uint8))
                       .resize((w, h), Image.BILINEAR), np.float32) / 127.5 - 1
    noise = gaussian_filter(noise, sigma=max(1.0, cell * 0.5))
    noise /= max(1e-6, np.abs(noise).max())              # normalize to [-1, 1]

    density = np.clip(1.0 + 0.35 * noise, 0.55, 1.0)     # mostly ~0.9-1.0, rare dry dips
    blended = mask * (1 - strength + strength * density)
    return np.clip(blended, 0, 1)
