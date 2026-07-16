"""Image loading/saving helpers around cv2, with real error handling.

The original app never checked `cv2.imread`'s return value: on a bad path or
unsupported/corrupt file, `cv2.imread` silently returns `None`, and the crash
would only surface later (and confusingly) inside `resize_image` or the first
filter call. This module raises a clear, specific exception right at load
time instead.
"""
import os

import cv2


def load_image(path):
    """Load an image from disk as a BGR uint8 numpy array.

    Parameters:
        path: path to an image file (png/jpg/bmp/...).

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file exists but cv2 could not decode it (unsupported
            format, truncated/corrupt file, not actually an image, etc).

    Returns:
        BGR uint8 numpy array of shape (H, W, 3).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not decode image (unsupported or corrupt file): {path}")
    return image


def save_image(image, path):
    """Save a BGR uint8 numpy array to disk, creating parent directories as
    needed. Returns the path on success, raises ValueError on failure (e.g.
    an unsupported extension or an unwritable location)."""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise ValueError(f"Could not save image to: {path}")
    return path
