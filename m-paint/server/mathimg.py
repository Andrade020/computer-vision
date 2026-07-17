"""
Rasterize LaTeX math to a PIL image, offline.

Adapted from handwritten_text/hw/mathimg.py (same repo) -- vendored here to
keep m-paint self-contained instead of importing across sibling projects.

Backend: matplotlib mathtext (no external TeX needed, handles most
inline/display math). Returns an RGBA image with the given ink color,
transparent background, scaled so its visual height matches `height_px`.
"""
import io

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mathtext_png(expr, ink, dpi=220):
    fig = plt.figure(figsize=(0.01, 0.01))
    color = tuple(c / 255 for c in ink)
    # wrap in $...$ for mathtext; caller passes bare expression
    t = fig.text(0, 0, f"${expr}$", fontsize=24, color=color)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w, h = int(np.ceil(bbox.width)), int(np.ceil(bbox.height))
    fig.set_size_inches((w + 8) / fig.dpi, (h + 8) / fig.dpi)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=fig.dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def render_math(expr, height_px, ink=(20, 24, 60)):
    """Return (RGBA image, width_px). height_px sets the target visual height."""
    try:
        img = _mathtext_png(expr, ink)
    except Exception:
        # fall back to rendering as literal text if mathtext can't parse
        img = _mathtext_png(r"\mathrm{" + _escape(expr) + "}", ink)
    scale = height_px / img.height
    w = max(1, int(round(img.width * scale)))
    img = img.resize((w, height_px), Image.LANCZOS)
    return img, w


def _escape(s):
    for a, b in [("\\", ""), ("{", ""), ("}", ""), ("_", " "), ("^", " ")]:
        s = s.replace(a, b)
    return s or "?"
