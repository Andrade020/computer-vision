"""Thresholding and segmentation: turning "a picture" into "which pixels
belong together" -- five different answers, each solving a different
version of that problem.

- **Otsu thresholding** (`otsu_threshold`) picks a single global brightness
  cutoff automatically: it tries every possible threshold and keeps the one
  that best splits the image's brightness histogram into two separated
  clusters (minimizing the spread *within* each cluster). Works well when
  the whole image has roughly even lighting and a genuinely bimodal
  histogram (a clear "dark stuff" and "light stuff").
- **Adaptive thresholding** (`adaptive_threshold`) uses a *different*
  threshold for each neighborhood (comparing each pixel to the local mean
  of a small window around it) instead of one global number -- this is
  what Otsu can't do: if the lighting is uneven across the image (a
  gradient, a shadow), a single global cutoff misclassifies whole regions,
  while an adaptive one keeps adjusting to the local baseline.
- **Connected components** (`connected_components`) takes an already-binary
  mask and groups touching foreground pixels into separate labeled blobs --
  "how many distinct objects are there, and where/how big is each one."
- **Watershed** (`watershed_segments`) solves the harder case connected
  components can't: two objects that *touch* (their pixels are literally
  connected) but should still count as separate objects. It treats
  brightness like a topographic map and "floods" it from confident interior
  points outward, drawing a dividing ridge wherever two floods would
  otherwise merge -- the peaks used as flood starting points come from a
  distance transform (how far is each foreground pixel from the nearest
  background pixel; local maxima of that are "probably the center of one
  object").
- **K-means color segmentation** (`kmeans_color_segments`) ignores shape
  entirely and groups pixels by color similarity alone -- "posterize this
  image down to k representative colors," useful when regions are better
  distinguished by color than by brightness/connectivity.
"""

import cv2
import numpy as np


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def otsu_threshold(image, invert=False):
    """Global automatic threshold (Otsu's method). Returns (mask, thresh)
    where mask is a 0/255 uint8 image and thresh is the brightness cutoff
    Otsu actually picked -- worth surfacing, since seeing the chosen number
    demystifies "automatic" thresholding."""
    gray = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thresh_val, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return mask, thresh_val


def adaptive_threshold(image, method="gaussian", block_size=11, C=2, invert=False):
    """Per-neighborhood threshold: each pixel is compared to the (weighted)
    mean brightness of a ``block_size``x``block_size`` window around it,
    minus ``C`` -- so the effective cutoff drifts with local lighting
    instead of using one fixed number for the whole image. ``method``
    picks how the local mean is weighted: "mean" (flat average) or
    "gaussian" (center-weighted, usually smoother results)."""
    gray = to_gray(image)
    if block_size % 2 == 0:
        block_size += 1  # cv2 requires an odd block size
    adaptive_method = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C,
                       "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}.get(method)
    if adaptive_method is None:
        raise ValueError(f"unknown method: {method!r} (expected 'mean' or 'gaussian')")
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, adaptive_method, flag, block_size, C)


def connected_components(binary_mask, connectivity=8):
    """Labels each blob of touching foreground pixels (mask > 0) with its
    own integer ID (0 = background). Returns (num_labels, labels, stats,
    centroids) straight from cv2 -- ``num_labels`` includes the background,
    so ``num_labels - 1`` is the actual object count; ``stats`` has one row
    per label with [x, y, width, height, area] of its bounding box."""
    mask01 = (binary_mask > 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(mask01, connectivity=connectivity)


def colorize_labels(labels, seed=0):
    """Turn an integer label map (from connected_components or watershed)
    into a BGR uint8 image with one random-but-consistent color per label,
    background (label <= 0) forced to black -- for visualizing "which
    pixels the algorithm grouped together" rather than reading raw
    integers."""
    rng = np.random.RandomState(seed)
    max_label = int(labels.max())
    colors = rng.randint(50, 256, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = 0
    out = colors[np.clip(labels, 0, max_label)]
    out[labels <= 0] = 0
    return out


def find_contours(binary_mask, mode="external"):
    """Traces the boundary of each blob in a binary mask. ``mode``:
    "external" (cv2.RETR_EXTERNAL, outermost boundaries only -- holes
    inside a blob are ignored) or "all" (cv2.RETR_LIST, every boundary
    including holes). Returns the list of contours as cv2 gives them
    (each an (N,1,2) array of points)."""
    retr = {"external": cv2.RETR_EXTERNAL, "all": cv2.RETR_LIST}.get(mode)
    if retr is None:
        raise ValueError(f"unknown mode: {mode!r} (expected 'external' or 'all')")
    mask01 = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask01, retr, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(image, contours, color=(0, 0, 255), thickness=2):
    """Draws contours on top of a copy of ``image`` (BGR) -- default
    bright red so they stand out regardless of the underlying image."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def watershed_segments(image, fg_ratio=0.5):
    """Separates TOUCHING objects that a plain threshold + connected-
    components would merge into one blob -- the classic watershed demo is
    exactly this: two overlapping circles.

    Pipeline: Otsu-threshold to a rough foreground/background split ->
    distance transform (how far is each foreground pixel from the nearest
    background pixel) -> threshold that distance map at ``fg_ratio`` of its
    own peak to get "sure foreground" seed blobs (one seed per real object,
    since the distance transform peaks near each object's center) -> dilate
    the rough mask for "sure background" -> whatever's neither sure-fg nor
    sure-bg is "unknown", marked 0 so cv2.watershed knows to resolve it ->
    flood-fill outward from each seed; wherever two floods would collide,
    draw a boundary there instead of merging.

    Returns (markers, boundary_overlay): ``markers`` is an int32 label map
    (watershed's convention: -1 marks the drawn boundaries, 1 is
    background, 2+ are individual objects); ``boundary_overlay`` is the
    original image with the boundaries painted on top in red, for a quick
    visual check.
    """
    gray = to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # so background becomes 1, not 0
    markers[unknown == 255] = 0    # 0 = "unknown, let watershed decide"

    color_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_image = color_image.copy()
    markers = cv2.watershed(color_image, markers)

    overlay = color_image.copy()
    overlay[markers == -1] = (0, 0, 255)
    return markers, overlay


def kmeans_color_segments(image, k=4, attempts=3, seed=0):
    """Groups pixels by color similarity alone (ignoring position/shape)
    into ``k`` clusters via cv2's k-means, then paints every pixel with
    its cluster's mean color -- a "posterize to k representative colors"
    view of the image. Returns (segmented_image, labels) where labels is
    an (H, W) int array of which cluster each pixel belongs to."""
    h, w = image.shape[:2]
    data = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts,
                                    cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented, labels.reshape(h, w)
