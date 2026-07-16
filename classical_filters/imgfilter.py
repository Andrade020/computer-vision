"""Command-line interface for the vectorized classical filters in imgfilters/.

Chains any combination of operations in a fixed, sensible order -- only the
ones you actually pass flags for run -- and writes the result to disk:

    python imgfilter.py in.png -o out.png --brightness 20 --contrast 1.2 \\
        --conv sobel-h --keep-color --noise 15 --kuwahara 7

Order of operations (fixed, regardless of flag order on the command line):
    1. resize        (--resize)
    2. brightness/contrast (--brightness / --contrast)
    3. convolution    (--conv [--keep-color])
    4. frequency-domain filter (--freq-filter [--freq-cutoff/--freq-cutoff2/--freq-kind/--freq-keep-color])
    5. gaussian noise (--noise)
    6. kuwahara       (--kuwahara)
"""
import argparse
import sys

from imgfilters.io import load_image, save_image
from imgfilters.pointops import adjust_brightness_contrast, add_gaussian_noise, resize_image
from imgfilters.convolution import convolution_filter, KERNELS
from imgfilters.frequency import build_mask, apply_frequency_filter, magnitude_spectrum_image, FILTER_TYPES
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
    p.add_argument("--freq-filter", choices=FILTER_TYPES, metavar="TYPE",
                    help=f"Frequency-domain filter: {', '.join(FILTER_TYPES)}. Operates via "
                         "2D FFT instead of a spatial convolution kernel -- see "
                         "imgfilters/frequency.py for why this is a different (and for some "
                         "effects, easier-to-express) way to filter an image.")
    p.add_argument("--freq-cutoff", type=float, default=30.0, metavar="RADIUS",
                    help="Frequency cutoff radius in pixels (the low cutoff, for band-pass). "
                         "Default: 30.")
    p.add_argument("--freq-cutoff2", type=float, default=None, metavar="RADIUS",
                    help="Upper cutoff radius in pixels, required for --freq-filter band-pass.")
    p.add_argument("--freq-kind", choices=("ideal", "gaussian"), default="gaussian", metavar="KIND",
                    help="ideal = hard cutoff (simple, but rings/ghosts near edges); "
                         "gaussian = smooth falloff (default, no ringing). See --explain.")
    p.add_argument("--freq-keep-color", action="store_true",
                    help="With --freq-filter, apply the mask per-channel instead of "
                         "converting to grayscale first.")
    p.add_argument("--spectrum-out", default=None, metavar="PNG_PATH",
                    help="Also save a visualization of the final image's magnitude "
                         "spectrum (the 'FFT photo' of the image) to this PNG file.")
    p.add_argument("--explain", action="store_true",
                    help="Print a plain-language explanation of the frequency-domain "
                         "filter settings (and the ideal-vs-gaussian ringing trade-off) "
                         "before processing.")
    p.add_argument("--noise", type=float, metavar="STD_DEV",
                    help="Add Gaussian noise with this standard deviation.")
    p.add_argument("--kuwahara", type=int, metavar="WINDOW_SIZE",
                    help="Apply the Kuwahara edge-preserving filter with this window size.")
    return p


def explain_frequency_filter(filter_type, cutoff, cutoff2, kind):
    lines = [
        f"Filtro de frequencia: {filter_type}, cutoff={cutoff}px"
        + (f", cutoff2={cutoff2}px" if cutoff2 is not None else "") + f", kind={kind}",
        "  - Baixa frequencia = variacoes suaves de brilho (perto do centro do espectro).",
        "  - Alta frequencia = bordas nitidas, textura fina, ruido (perto das bordas do espectro).",
    ]
    if kind == "ideal":
        lines.append(
            "  - kind=ideal usa um corte duro (tudo ou nada) -- simples de entender, mas "
            "sua borda abrupta no dominio da frequencia produz 'ringing' (ecos fantasmas) "
            "perto de bordas nitidas na imagem, o mesmo fenomeno de Gibbs que aparece como "
            "artefato audivel no --compress do audio_processor irmao deste projeto.")
    else:
        lines.append(
            "  - kind=gaussian usa uma transicao suave -- evita o ringing do corte duro, "
            "ao custo de um corte de frequencia menos preciso.")
    return "\n".join(lines)


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

    if args.freq_filter:
        mask = build_mask(image.shape[:2], args.freq_filter, cutoff=args.freq_cutoff,
                          cutoff2=args.freq_cutoff2, kind=args.freq_kind)
        image = apply_frequency_filter(image, mask, keep_color=args.freq_keep_color)

    if args.noise is not None:
        image = add_gaussian_noise(image, args.noise)

    if args.kuwahara:
        image = kuwahara_filter(image, args.kuwahara)

    save_image(image, args.output)

    if args.spectrum_out:
        spectrum_img = magnitude_spectrum_image(image)
        save_image(spectrum_img, args.spectrum_out)

    return image


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.explain and args.freq_filter:
        print(explain_frequency_filter(args.freq_filter, args.freq_cutoff,
                                       args.freq_cutoff2, args.freq_kind))
        print()

    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved: {args.output}")
    if args.spectrum_out:
        print(f"Spectrum saved: {args.spectrum_out}")


if __name__ == "__main__":
    main()
