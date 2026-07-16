"""Command-line interface for the vectorized classical filters in imgfilters/.

Chains any combination of operations in a fixed, sensible order -- only the
ones you actually pass flags for run -- and writes the result to disk:

    python imgfilter.py in.png -o out.png --brightness 20 --contrast 1.2 \\
        --conv sobel-h --keep-color --noise 15 --kuwahara 7

Order of operations (fixed, regardless of flag order on the command line):
    1. resize        (--resize)
    2. brightness/contrast (--brightness / --contrast)
    3. convolution    (--conv [--keep-color])
    4. gaussian noise (--noise)
    5. kuwahara       (--kuwahara)
"""
import argparse
import sys

from imgfilters.io import load_image, save_image
from imgfilters.pointops import adjust_brightness_contrast, add_gaussian_noise, resize_image
from imgfilters.convolution import convolution_filter, KERNELS
from imgfilters.kuwahara import kuwahara_filter


def build_parser():
    p = argparse.ArgumentParser(
        description="Apply classical image filters (vectorized) and save the result.")
    p.add_argument("input", help="Path to the input image.")
    p.add_argument("-o", "--output", required=True, help="Path to write the output image.")

    p.add_argument("--resize", type=int, metavar="MAX_DIM",
                    help="Downscale so the largest dimension is at most MAX_DIM px "
                         "(aspect-preserving; never upscales).")
    p.add_argument("--brightness", type=float, default=None, metavar="BETA",
                    help="Additive brightness offset (used together with --contrast; "
                         "defaults to 0 if only --contrast is given).")
    p.add_argument("--contrast", type=float, default=None, metavar="K",
                    help="Contrast multiplier (used together with --brightness; "
                         "defaults to 1.0 if only --brightness is given).")
    p.add_argument("--conv", choices=sorted(KERNELS), metavar="KERNEL",
                    help=f"Convolution kernel preset: {', '.join(sorted(KERNELS))}.")
    p.add_argument("--keep-color", action="store_true",
                    help="With --conv, apply the kernel per-channel instead of "
                         "converting to grayscale first.")
    p.add_argument("--noise", type=float, metavar="STD_DEV",
                    help="Add Gaussian noise with this standard deviation.")
    p.add_argument("--kuwahara", type=int, metavar="WINDOW_SIZE",
                    help="Apply the Kuwahara edge-preserving filter with this window size.")
    return p


def run(args):
    image = load_image(args.input)

    if args.resize:
        image = resize_image(image, args.resize)

    if args.brightness is not None or args.contrast is not None:
        beta = args.brightness if args.brightness is not None else 0.0
        k = args.contrast if args.contrast is not None else 1.0
        image = adjust_brightness_contrast(image, beta, k)

    if args.conv:
        image = convolution_filter(image, KERNELS[args.conv], keep_color=args.keep_color)

    if args.noise is not None:
        image = add_gaussian_noise(image, args.noise)

    if args.kuwahara:
        image = kuwahara_filter(image, args.kuwahara)

    save_image(image, args.output)
    return image


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
