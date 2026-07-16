"""Edge detection: Canny, Laplacian/LoG, and Sobel gradient magnitude.

Three different answers to "where are the edges in this image?", each with
a different definition of "edge" worth knowing:

- **Gradient magnitude** (``gradient_magnitude``) says an edge is wherever
  brightness changes quickly in *some* direction -- it reuses this project's
  own Sobel presets from ``convolution.py`` (one kernel per direction) and
  combines them as ``sqrt(gx^2 + gy^2)``, the magnitude of the brightness
  gradient vector. Cheap, direct, but produces thick, fuzzy edge bands and
  is sensitive to noise (a single noisy pixel has a "high gradient" too).
- **Laplacian / LoG** (``laplacian_edges``) looks at the *second* derivative
  instead of the first: an edge is where brightness curvature crosses zero
  (a peak/valley in the first derivative). This responds to edges in every
  direction at once with a single kernel (no separate horizontal/vertical
  pass), but is even more noise-sensitive than the gradient -- which is why
  it's almost always paired with a Gaussian blur first (the "LoG" in the
  name: Laplacian of Gaussian) to smooth out the noise the second derivative
  would otherwise amplify.
- **Canny** (``canny_edges``) is the one actually meant to produce a clean,
  thin, usable edge *map* rather than just a "how much edge is here"
  brightness image. It runs a fixed pipeline: blur to suppress noise,
  compute the gradient (magnitude and direction), thin it down to
  single-pixel-wide lines by keeping only local maxima along the gradient
  direction ("non-maximum suppression"), then a two-threshold ("hysteresis")
  decision -- pixels above the high threshold are definite edges, pixels
  above the low threshold only count if they connect to a definite edge.
  ``canny_stages`` exposes each of these intermediate images instead of
  just the final result, since seeing *why* Canny picked an edge is far
  more instructive than only seeing the answer.
"""

import cv2
import numpy as np

from .convolution import KERNELS


def _to_gray_float(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float64)


def gradient_magnitude(image, keep_color=False):
    """Sobel gradient magnitude: sqrt(gx^2 + gy^2), normalized to uint8.

    Reuses this project's own ``sobel-h``/``sobel-v`` kernel presets (see
    ``convolution.py``) instead of reintroducing new ones -- the two Sobel
    convolutions ARE the gradient's x/y components; this just combines them
    into a single magnitude instead of looking at each direction alone.
    """
    def _one_channel(chan):
        chan = chan.astype(np.float64)
        gx = cv2.filter2D(chan, cv2.CV_64F, KERNELS["sobel-h"], borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(chan, cv2.CV_64F, KERNELS["sobel-v"], borderType=cv2.BORDER_REFLECT)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        peak = mag.max()
        if peak > 0:
            mag = mag / peak * 255.0
        return mag.astype(np.uint8)

    if keep_color and image.ndim == 3:
        channels = [_one_channel(image[:, :, c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)

    gray_mag = _one_channel(_to_gray_float(image))
    return np.stack([gray_mag] * 3, axis=2) if image.ndim == 3 else gray_mag


def laplacian_edges(image, ksize=3, blur_sigma=1.0):
    """Laplacian of Gaussian: blur with a Gaussian of the given sigma (skip
    if ``blur_sigma <= 0``, giving a plain Laplacian instead of LoG), then
    take the discrete Laplacian, then take the absolute value and normalize
    to uint8 for display (the raw Laplacian is signed -- zero-crossings, not
    magnitude, are technically "the edge", but the absolute value is what's
    actually useful to look at)."""
    gray = _to_gray_float(image)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    mag = np.abs(lap)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak * 255.0
    out = mag.astype(np.uint8)
    return np.stack([out] * 3, axis=2) if image.ndim == 3 else out


def canny_edges(image, low_threshold=50, high_threshold=150, blur_sigma=1.0):
    """Canny edge map (0/255 binary-ish image, thin single-pixel lines).
    Thin wrapper over cv2.Canny (a well-established, carefully-tuned
    algorithm not worth reimplementing) -- the pre-blur is applied here
    explicitly so its sigma is a visible, tunable parameter rather than
    buried inside cv2's own internal default."""
    gray = _to_gray_float(image)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    edges = cv2.Canny(gray.astype(np.uint8), threshold1=low_threshold, threshold2=high_threshold)
    return np.stack([edges] * 3, axis=2) if image.ndim == 3 else edges


def canny_stages(image, low_threshold=50, high_threshold=150, blur_sigma=1.0):
    """Same pipeline as ``canny_edges``, but returns a dict of every
    intermediate stage instead of just the final edge map -- the point
    being to make Canny's internal reasoning visible rather than a black
    box that spits out lines. Stages, in order:

        blurred     -- after the Gaussian pre-blur (noise suppression)
        gradient    -- Sobel gradient magnitude of the blurred image
        direction   -- gradient direction, visualized as a hue wheel (HSV
                       with direction -> hue, magnitude -> value) so you can
                       see *which way* each edge is oriented, not just where
        edges       -- the final Canny result (after non-max suppression +
                       hysteresis thresholding, both done internally by
                       cv2.Canny -- the two steps that turn a fuzzy gradient
                       into thin, connected lines)

    All returned as uint8 grayscale/BGR images ready to display or save.
    """
    gray_f = _to_gray_float(image)
    blurred = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=blur_sigma) if blur_sigma > 0 else gray_f

    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_norm = (mag / mag.max() * 255.0).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)

    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0..1
    hsv = np.zeros((*gray_f.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 179).astype(np.uint8)   # hue = direction
    hsv[..., 1] = 255                              # full saturation
    hsv[..., 2] = mag_norm                         # value = edge strength
    direction_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    edges = cv2.Canny(blurred.astype(np.uint8), threshold1=low_threshold, threshold2=high_threshold)

    return {
        "blurred": blurred.astype(np.uint8),
        "gradient": mag_norm,
        "direction": direction_img,
        "edges": edges,
    }
