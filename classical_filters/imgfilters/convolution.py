"""2D convolution filters via cv2.filter2D.

The original `convolution_filter` was a manual nested Python for-loop over
every output pixel (O(H*W*k^2)), which is why the GUI hardcoded a 400px
resize on load. It also had two correctness issues, both fixed here:

  1. It always converted the input to grayscale first, even for color
     images -- there was no way to get a color convolution result.
  2. It left an unprocessed, literally black border of `kernel_size // 2`
     pixels around the whole image (the loop's range started at `pad_y`/
     `pad_x` and stopped `pad_y`/`pad_x` early, and `result` started as
     zeros that those border pixels never got overwritten).

`cv2.filter2D` is a single vectorized call that handles borders properly
(via `borderType=cv2.BORDER_REFLECT`, so edges are mirrored instead of
zero/black) and applies to color images per-channel natively, so both
problems disappear and there is no need for a manual loop or a size cap.
"""
import cv2
import numpy as np

# Same 5 presets the original GUI's combobox offered, keyed by a short slug
# so both the CLI and the GUI can share this single dict.
KERNELS = {
    "blur3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "horizontal-derivative": np.array([[1, 0, -1]], dtype=np.float32),
    "vertical-derivative": np.array([[1], [0], [-1]], dtype=np.float32),
    "sobel-h": np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32),
    "sobel-v": np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32),
}


def convolution_filter(image, kernel, keep_color=False):
    """Apply a 2D convolution kernel to an image.

    Parameters:
        image: BGR (H,W,3) or grayscale (H,W) uint8 image.
        kernel: 2D float32 filter matrix.
        keep_color: if False (default, matches the original app's only
            behavior), the image is converted to grayscale first and the
            single-channel result is expanded back to 3-channel BGR for
            display. If True, the kernel is applied independently to each
            of the B/G/R channels and the result stays in color.

    Returns:
        uint8 array, same H/W as the input, with borders handled by
        reflection instead of being left black.
    """
    kernel = kernel.astype(np.float32)

    if image.ndim == 3 and not keep_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        work = gray.astype(np.float32) / 255.0
        filtered = cv2.filter2D(work, -1, kernel, borderType=cv2.BORDER_REFLECT)
        result = np.clip(filtered * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    work = image.astype(np.float32) / 255.0
    filtered = cv2.filter2D(work, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(filtered * 255.0, 0, 255).astype(np.uint8)
