"""Morphological operations: erosion, dilation, and the four combinations
built from them (opening, closing, tophat, blackhat).

Where convolution treats every pixel as a weighted average of its
neighbors, morphology treats a neighborhood (the "structuring element") as
a shape probe and asks a min/max question instead of a weighted-sum one:

- **Erosion** replaces each pixel with the MINIMUM brightness in its
  neighborhood -- bright regions shrink, dark regions grow. A thin bright
  line or a small bright speck can vanish entirely if the structuring
  element is bigger than it.
- **Dilation** is the mirror image: each pixel becomes the MAXIMUM in its
  neighborhood -- bright regions grow, dark regions shrink/fill in.

Chaining the two in a fixed order gives two more useful operations that
only look brutal in isolation:

- **Opening** (erode, then dilate) shrinks bright specks away entirely in
  the erosion step, and the following dilation only grows back what
  *survived* -- so an opening removes small bright noise/protrusions while
  leaving the overall shape of anything bigger essentially unchanged. Good
  for cleaning up salt-like bright noise without blurring real edges.
- **Closing** (dilate, then erode) is opening's mirror: it fills in small
  dark holes/gaps (e.g. a scanned line with tiny gaps, or pepper noise)
  without changing the overall shape of anything bigger.

Tophat and blackhat isolate exactly what those two operations threw away:

- **Tophat** = original - opening -- what's left is the small bright
  details opening erased, i.e. "the small bright stuff", isolated on its
  own against a near-black background.
- **Blackhat** = closing - original -- the small dark details closing
  filled in, isolated the same way.
"""

import cv2
import numpy as np


def _structuring_element(size, shape="ellipse"):
    size = max(1, int(size))
    cv2_shape = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE,
                "cross": cv2.MORPH_CROSS}.get(shape)
    if cv2_shape is None:
        raise ValueError(f"unknown structuring element shape: {shape!r}")
    return cv2.getStructuringElement(cv2_shape, (size, size))


def _apply(image, op, kernel_size, shape, keep_color, iterations):
    kernel = _structuring_element(kernel_size, shape)
    if keep_color and image.ndim == 3:
        channels = [op(image[:, :, c], kernel, iterations=iterations)
                   for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    result = op(gray, kernel, iterations=iterations)
    return np.stack([result] * 3, axis=2) if image.ndim == 3 else result


def erode(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, cv2.erode, kernel_size, shape, keep_color, iterations)


def dilate(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, cv2.dilate, kernel_size, shape, keep_color, iterations)


def _morph_ex(op_code):
    def op(channel, kernel, iterations=1):
        return cv2.morphologyEx(channel, op_code, kernel, iterations=iterations)
    return op


def opening(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, _morph_ex(cv2.MORPH_OPEN), kernel_size, shape, keep_color, iterations)


def closing(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, _morph_ex(cv2.MORPH_CLOSE), kernel_size, shape, keep_color, iterations)


def tophat(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, _morph_ex(cv2.MORPH_TOPHAT), kernel_size, shape, keep_color, iterations)


def blackhat(image, kernel_size=3, shape="ellipse", keep_color=False, iterations=1):
    return _apply(image, _morph_ex(cv2.MORPH_BLACKHAT), kernel_size, shape, keep_color, iterations)


OPERATIONS = {
    "erode": erode, "dilate": dilate, "opening": opening,
    "closing": closing, "tophat": tophat, "blackhat": blackhat,
}


def apply_morphology(image, op_name, kernel_size=3, shape="ellipse", keep_color=False,
                     iterations=1):
    """Dispatch helper mirroring convolution.KERNELS / frequency.build_mask's
    "pick by name" convenience."""
    if op_name not in OPERATIONS:
        raise ValueError(f"unknown morphology op: {op_name!r} (expected one of {sorted(OPERATIONS)})")
    return OPERATIONS[op_name](image, kernel_size=kernel_size, shape=shape,
                               keep_color=keep_color, iterations=iterations)
