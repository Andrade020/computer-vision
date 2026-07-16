"""Frequency-domain filtering for images -- the 2D sibling of the
sibling `audio_processor` project's `stft.py`.

Why this exists (worth reading once): every filter in ``convolution.py``
works in the **spatial domain** -- it looks at a small neighborhood of
pixels around each output pixel and combines them with a kernel. That's
intuitive, but some ideas are much easier to state in the **frequency
domain** instead: "keep only the smooth, low-frequency content" (blur),
"keep only the sharp edges/fine texture" (high-pass/sharpen-adjacent), or
"this image has a repeating scan-line/moire pattern at a specific spatial
frequency -- remove just that" (notch filtering) -- none of which have an
obvious small convolution kernel, but are a one-line mask in the frequency
domain.

The 2D Fourier transform of an image decomposes it into sinusoidal
gratings of every possible orientation and frequency; low frequencies
(near the center of the shifted spectrum) correspond to slow, smooth
brightness changes, and high frequencies (toward the edges) correspond to
sharp edges, fine texture, and noise. Multiplying the transform by a mask
that zeroes out some region, then inverse-transforming, is mathematically
equivalent to convolving with some (possibly huge, impractical-to-write-
down) spatial kernel -- the frequency domain just makes some filters
trivial to express and see, instead of needing a hand-crafted kernel.

A concrete, slightly counter-intuitive teaching example lives in
``LOW_PASS_KIND``/``ideal`` vs ``gaussian`` below: an "ideal" low-pass
filter (a hard brick-wall cutoff) sounds like it should be the *best*
version of the idea, but its sharp edge in the frequency domain corresponds
to a `sinc` ripple in the spatial domain, which shows up as visible ringing
artifacts (ghostly echoes) near edges in the filtered image -- the same
"Gibbs phenomenon" the sibling audio project's naive Fourier `compress_audio`
exhibits as audible ringing. A ``gaussian`` mask (smooth falloff, no hard
edge) avoids that ringing entirely, at the cost of a less crisp cutoff
frequency. Comparing the two on the same image is a genuinely useful,
visual way to understand the trade-off.
"""

import cv2
import numpy as np


def to_grayscale(image):
    """BGR -> grayscale, or a no-op if already single-channel."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def fft2_channel(channel):
    """2D FFT of a single-channel image, DC (zero frequency) shifted to the
    center via ``fftshift``. Shifting is purely a display/reasoning
    convenience -- it puts low frequencies in the middle and high
    frequencies toward the corners, matching how frequency masks are drawn
    and described below (``_distance_grid`` measures distance from the
    center for exactly this reason)."""
    return np.fft.fftshift(np.fft.fft2(channel.astype(np.float64)))


def ifft2_channel(F):
    """Inverse of ``fft2_channel``: un-shift, inverse FFT, keep the real
    part (any imaginary component left is floating-point noise, since a
    correctly-symmetric mask applied to a real-valued image's transform
    keeps the result real up to that noise)."""
    return np.fft.ifft2(np.fft.ifftshift(F)).real


def magnitude_spectrum_image(image, log_scale=True):
    """A displayable grayscale uint8 "photo" of an image's frequency
    content -- brightness at each point shows how strong that spatial
    frequency is, with the center = low frequencies (smooth areas) and the
    corners/edges = high frequencies (sharp detail, texture, noise).

    Raw FFT magnitudes span an enormous range (the DC/average-brightness
    term totally dominates), so ``log_scale`` compresses it with
    ``log(1 + x)`` -- the standard way this is displayed -- before
    normalizing to 0-255 for viewing.
    """
    gray = to_grayscale(image)
    F = fft2_channel(gray)
    mag = np.abs(F)
    if log_scale:
        mag = np.log1p(mag)
    mag = mag - mag.min()
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    return mag.astype(np.uint8)


def _distance_grid(shape, center_offset=(0.0, 0.0)):
    """Per-pixel Euclidean distance from the (shifted-spectrum) center,
    optionally re-centered by ``center_offset`` (dy, dx) in pixels -- used
    by ``notch_mask`` to place a hole away from the true center."""
    h, w = shape
    dy, dx = center_offset
    cy, cx = h / 2.0 + dy, w / 2.0 + dx
    y, x = np.ogrid[:h, :w]
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def low_pass_mask(shape, cutoff, kind="gaussian"):
    """A mask that keeps low frequencies (near the center) and suppresses
    high ones -- the frequency-domain equivalent of blurring.

    ``kind="ideal"`` is a hard 1/0 cutoff at radius ``cutoff`` -- simple to
    reason about, but its sharp edge causes ringing artifacts (see module
    docstring). ``kind="gaussian"`` tapers smoothly and avoids that at the
    cost of a fuzzier cutoff.
    """
    d = _distance_grid(shape)
    if kind == "ideal":
        return (d <= cutoff).astype(np.float64)
    if kind == "gaussian":
        return np.exp(-(d ** 2) / (2.0 * cutoff ** 2))
    raise ValueError(f"unknown mask kind: {kind!r} (expected 'ideal' or 'gaussian')")


def high_pass_mask(shape, cutoff, kind="gaussian"):
    """The complement of ``low_pass_mask``: suppresses low frequencies
    (smooth shading) and keeps high ones (edges, fine texture) -- the
    frequency-domain equivalent of edge/detail extraction."""
    return 1.0 - low_pass_mask(shape, cutoff, kind=kind)


def band_pass_mask(shape, low_cutoff, high_cutoff, kind="gaussian"):
    """Keeps a ring of frequencies between the two cutoffs, suppressing
    both very smooth content (below ``low_cutoff``) and very fine
    detail/noise (above ``high_cutoff``). Requires ``high_cutoff >
    low_cutoff``."""
    if high_cutoff <= low_cutoff:
        raise ValueError("high_cutoff must be greater than low_cutoff")
    return low_pass_mask(shape, high_cutoff, kind=kind) - low_pass_mask(shape, low_cutoff, kind=kind)


def notch_mask(shape, points, radius, kind="ideal"):
    """Suppresses one or more specific frequency *points* (not a radius
    from the center) -- useful for removing a periodic pattern (scan
    lines, moire, mains hum in a scanned/photographed page, halftone
    dots) that shows up as a sharp, isolated peak away from the center
    in the magnitude spectrum, without touching the rest of the image.

    ``points`` is a list of ``(du, dv)`` pixel offsets from the spectrum's
    center. Each point is suppressed **together with its mirror image**
    ``(-du, -dv)`` automatically: a real-valued image's FFT is always
    conjugate-symmetric around the center, so silencing only one side of a
    symmetric pair would make the inverse transform come back complex
    (i.e. produce an invalid image) instead of a clean real-valued result.
    """
    mask = np.ones(shape, dtype=np.float64)
    for du, dv in points:
        for offset in {(du, dv), (-du, -dv)}:
            d = _distance_grid(shape, center_offset=(offset[1], offset[0]))
            if kind == "ideal":
                mask[d <= radius] = 0.0
            else:
                mask = mask * (1.0 - np.exp(-(d ** 2) / (2.0 * radius ** 2)))
    return mask


FILTER_TYPES = ("low-pass", "high-pass", "band-pass")


def build_mask(shape, filter_type, cutoff, cutoff2=None, kind="gaussian"):
    """Dispatch helper mirroring ``convolution.KERNELS``'s "pick a preset by
    name" convenience -- builds one of the three standard filter shapes
    above from a short string, for callers (CLI/GUI) that just want "give
    me a low-pass mask of this size" without importing each function."""
    if filter_type == "low-pass":
        return low_pass_mask(shape, cutoff, kind=kind)
    if filter_type == "high-pass":
        return high_pass_mask(shape, cutoff, kind=kind)
    if filter_type == "band-pass":
        if cutoff2 is None:
            raise ValueError("band-pass requires cutoff2 (the upper cutoff)")
        return band_pass_mask(shape, cutoff, cutoff2, kind=kind)
    raise ValueError(f"unknown filter_type: {filter_type!r} (expected one of {FILTER_TYPES})")


def apply_frequency_filter(image, mask, keep_color=False):
    """Apply a precomputed frequency-domain ``mask`` (from ``build_mask``/
    ``low_pass_mask``/etc., or a hand-built one) to ``image`` via
    FFT -> multiply -> inverse FFT.

    Mirrors ``convolution_filter``'s ``keep_color`` convention: by default
    (False) the image is converted to grayscale first (frequency-domain
    filtering is most easily reasoned about on a single brightness channel);
    if True, the same mask is applied to each B/G/R channel independently.
    """
    if keep_color and image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            F = fft2_channel(image[:, :, c])
            channels.append(ifft2_channel(F * mask))
        result = np.stack(channels, axis=2)
    else:
        gray = to_grayscale(image)
        F = fft2_channel(gray)
        filtered = ifft2_channel(F * mask)
        result = np.stack([filtered] * 3, axis=2) if image.ndim == 3 else filtered

    return np.clip(result, 0, 255).astype(np.uint8)
