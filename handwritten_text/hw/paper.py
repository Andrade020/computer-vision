"""
Paper-scan post-processing: gentle page warp, fold/crease shadows, grain, and
lighting drift applied to a finished page image, so it reads like a scanned
or photographed sheet of paper instead of a flat digital render.

Ported from this project's original cv2-based effect (simulate_paper_folds /
simulate_ink in writting_colos.py, see the "fake your handwriting" article)
to numpy + scipy + PIL only, since cv2 isn't installed in this environment.
Same five ingredients: global sinusoidal warp, random crease shadows, a
low-frequency noise field, a vertical brightness gradient, and fine grain.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates


def _low_freq_field(h, w, cell, rng):
    """Smooth random field: a coarse random grid, upsampled and blurred."""
    small = rng.uniform(-1, 1, (h // cell + 2, w // cell + 2)).astype(np.float32)
    big = np.asarray(Image.fromarray(((small + 1) * 127.5).astype(np.uint8))
                     .resize((w, h), Image.BILINEAR), np.float32) / 127.5 - 1
    return gaussian_filter(big, sigma=cell * 0.6)


def _global_warp(arr, amplitude, period, rng):
    """Gentle horizontal sine warp, as if the page isn't lying perfectly flat."""
    h, w = arr.shape[:2]
    phase = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = np.clip(xx + amplitude * np.sin(2 * np.pi * yy / period + phase),
                    0, w - 1)
    out = np.empty_like(arr)
    coords = [yy.ravel(), src_x.ravel()]
    for c in range(arr.shape[2]):
        out[..., c] = map_coordinates(arr[..., c], coords, order=1,
                                      mode="nearest").reshape(h, w)
    return out


def _crease_mask(h, w, count, max_intensity, shadow_width, rng):
    """Random fold-shadow lines, blurred into soft bands."""
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for _ in range(count):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        theta = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.2 * min(w, h), 1.2 * max(w, h))
        x1, y1 = cx - np.cos(theta) * length / 2, cy - np.sin(theta) * length / 2
        x2, y2 = cx + np.cos(theta) * length / 2, cy + np.sin(theta) * length / 2
        val = int(np.clip(rng.uniform(0, max_intensity), 0, 1) * 255)
        width = max(1, int(rng.uniform(0.5, 1.5) * shadow_width))
        d.line([(x1, y1), (x2, y2)], fill=val, width=width)
    return gaussian_filter(np.asarray(mask, np.float32) / 255.0, sigma=15)


def scan_effect(page, strength=1.0, seed=None):
    """Apply a paper-scan look to a finished page. strength scales every
    effect together (0 = no-op, 1 = default, >1 = heavier). Returns a new
    PIL RGB image; the input is left untouched."""
    if strength <= 0:
        return page
    rng = np.random.RandomState(seed)
    arr = np.asarray(page.convert("RGB"), np.float32) / 255.0
    h, w = arr.shape[:2]

    arr = _global_warp(arr, amplitude=4.0 * strength, period=260, rng=rng)

    crease = _crease_mask(h, w, count=max(1, round(3 * strength)),
                          max_intensity=0.10 * strength,
                          shadow_width=26, rng=rng)[..., None]
    arr = arr * (1 - crease)

    noise = _low_freq_field(h, w, cell=24, rng=rng)[..., None]
    arr = np.clip(arr + 0.05 * strength * noise, 0, 1)

    bv = 0.06 * strength
    gradient = np.linspace(1 - bv, 1 + bv, h).astype(np.float32)
    arr = np.clip(arr * gradient[:, None, None], 0, 1)

    grain = rng.normal(0, 0.035 * strength, (h, w, 3)).astype(np.float32)
    arr = np.clip(arr + grain, 0, 1)

    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")
