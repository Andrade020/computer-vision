"""audioprocess.py -- command-line audio DSP pipeline.

Loads a WAV/FLAC/etc. file, applies zero or more effects in a fixed,
sensible order (trim -> compress -> echo -> reverb), writes the result, and
optionally dumps a spectrum plot of the final signal.

Example:
    python audioprocess.py in.wav -o out.wav --trim 10 --compress 0.5 \\
        --echo-delay 0.5 --echo-gain 0.6 --reverb-delays 10 \\
        --reverb-time 0.05 --spectrum spec.png
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audiodsp import io as audio_io
from audiodsp import effects
from audiodsp import spectrum as spectrum_mod


def build_parser():
    p = argparse.ArgumentParser(
        description="Apply DSP effects to an audio file from the command line.")
    p.add_argument("input", help="path to the input audio file")
    p.add_argument("-o", "--output", default="processed_audio.wav",
                    help="path to write the processed audio (default: processed_audio.wav)")

    p.add_argument("--trim", type=float, default=None, metavar="SECONDS",
                    help="keep only the first SECONDS seconds of audio")
    p.add_argument("--compress", type=float, default=None, metavar="FRACTION",
                    help="Fourier compression: fraction (0-1) of low/high FFT bins kept")
    p.add_argument("--echo-delay", type=float, default=None, metavar="SECONDS",
                    help="enable echo with this delay in seconds (default gain 0.6 "
                         "unless --echo-gain is also given)")
    p.add_argument("--echo-gain", type=float, default=None, metavar="GAIN",
                    help="enable echo with this gain (default delay 0.5s "
                         "unless --echo-delay is also given)")
    p.add_argument("--reverb-delays", type=int, default=None, metavar="N",
                    help="enable reverb with N delayed copies (default delay-time 0.05s)")
    p.add_argument("--reverb-time", type=float, default=None, metavar="SECONDS",
                    help="enable reverb with this delay time (default 10 copies)")
    p.add_argument("--spectrum", default=None, metavar="PNG_PATH",
                    help="also plot the magnitude spectrum of the final audio to this PNG file")
    return p


def process(audio, sr, args):
    """Apply the requested effects, in a fixed order, and return the result."""
    if args.trim is not None:
        audio = effects.trim_audio(audio, sr, duration=args.trim)

    if args.compress is not None:
        audio = effects.compress_audio(audio, args.compress)

    if args.echo_delay is not None or args.echo_gain is not None:
        delay = args.echo_delay if args.echo_delay is not None else 0.5
        gain = args.echo_gain if args.echo_gain is not None else 0.6
        audio = effects.add_echo(audio, sr, delay=delay, echo_gain=gain)

    if args.reverb_delays is not None or args.reverb_time is not None:
        num_delays = args.reverb_delays if args.reverb_delays is not None else 10
        delay_time = args.reverb_time if args.reverb_time is not None else 0.05
        audio = effects.add_reverb(audio, sr, num_delays=num_delays, delay_time=delay_time)

    return audio


def dump_spectrum(audio, sr, png_path):
    freqs, mags = spectrum_mod.magnitude_spectrum(audio, sr)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(freqs, mags)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Magnitude Spectrum")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        audio, sr = audio_io.load_audio(args.input)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    audio = process(audio, sr, args)
    audio_io.save_audio(audio, sr, args.output)
    print(f"Processed audio written to {args.output}")

    if args.spectrum:
        dump_spectrum(audio, sr, args.spectrum)
        print(f"Spectrum plot written to {args.spectrum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
