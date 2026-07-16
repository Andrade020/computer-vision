"""Per-pixel / whole-image operations that were already vectorized in the
original app -- ported as-is (same math, same defaults), just with clean
docstrings instead of the original's vowel-stripped pseudo-English comments.
"""
import cv2
import numpy as np


def adjust_brightness_contrast(image, beta, k):
    """Adjust brightness (additive `beta`) and contrast (multiplicative `k`).

    output = clip(k * image + beta, 0, 255)

    Parameters:
        image: BGR (or grayscale) uint8 numpy array.
        beta: additive brightness offset (e.g. -100..100).
        k: contrast multiplier (e.g. 0.0..3.0; 1.0 = unchanged).

    Returns:
        uint8 array, same shape as input.
    """
    image_float = image.astype(np.float32)
    adjusted = np.clip(k * image_float + beta, 0, 255).astype(np.uint8)
    return adjusted


def add_gaussian_noise(image, std_dev):
    """Add zero-mean Gaussian noise with standard deviation `std_dev` to an
    image and clip back to the valid uint8 range.

    Parameters:
        image: BGR (or grayscale) uint8 numpy array.
        std_dev: standard deviation of the noise (0 = no-op).

    Returns:
        uint8 array, same shape as input.
    """
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def resize_image(image, max_dim):
    """Downscale an image so its largest dimension is at most `max_dim`
    pixels, preserving aspect ratio. If the image is already within bounds
    it is returned unchanged (never upscaled).

    Note: with `convolution_filter` and `kuwahara_filter` now vectorized,
    callers no longer *need* to force a hard cap here for performance --
    this is kept purely as an optional convenience (e.g. for faster previews
    on very large images), not a workaround for slow processing.
    """
    height, width = image.shape[:2]
    scaling = max_dim / float(max(height, width))
    if scaling < 1.0:
        new_size = (int(width * scaling), int(height * scaling))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image
