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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audiodsp import io as audio_io
from audiodsp import effects
from audiodsp import spectrum as spectrum_mod
from audiodsp import stft as stft_mod
from audiodsp import filters as filters_mod


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
    p.add_argument("--spectrogram", default=None, metavar="PNG_PATH",
                    help="also plot a time-frequency spectrogram (STFT magnitude, in dB) "
                         "of the final audio to this PNG file")
    p.add_argument("--n-fft", type=int, default=1024, metavar="N",
                    help="STFT window length in samples for --spectrogram (default: 1024)")
    p.add_argument("--hop", type=int, default=256, metavar="N",
                    help="STFT hop length in samples for --spectrogram (default: 256)")
    p.add_argument("--window", default="hann", metavar="NAME",
                    help="STFT window function for --spectrogram, any name scipy.signal.get_window "
                         "accepts, e.g. hann/hamming/blackman (default: hann)")
    p.add_argument("--explain", action="store_true",
                    help="print a plain-language explanation of what the chosen "
                         "--n-fft/--hop/--window mean in real-world terms (ms, Hz, trade-offs) "
                         "before processing")
    p.add_argument("--spectral-region", action="append", default=None,
                    metavar="T0,T1,F0,F1,GAIN",
                    help="paint a rectangular time/frequency region of the STFT with a "
                         "constant gain -- 0.0 erases it, 1.0 leaves it unchanged, >1.0 "
                         "boosts it. Repeatable (each use paints one region). Runs after "
                         "the time-domain effects, using the --n-fft/--hop/--window STFT "
                         "settings. Example: --spectral-region 1.0,2.0,1100,1300,0.0 "
                         "silences 1100-1300Hz between the 1s and 2s marks. The scriptable "
                         "CLI equivalent of the GUI's spectral paint tool.")
    p.add_argument("--eq", action="append", default=None, metavar="TYPE,FREQ,Q,GAIN",
                    help="apply a real (biquad/IIR) EQ band, streaming/causal like a real "
                         f"EQ, not an STFT round-trip. TYPE is one of "
                         f"{', '.join(sorted(filters_mod.FILTER_BUILDERS))}; FREQ in Hz; Q "
                         "controls bandwidth/resonance (e.g. 0.707); GAIN in dB (ignored "
                         "except for peak/lowshelf/highshelf). Repeatable -- chains bands in "
                         "order like a graphic EQ. Runs after echo/reverb, before "
                         "--spectral-region. Example: --eq peak,1000,1.0,6 boosts 1000Hz by "
                         "6dB; --eq notch,60,10,0 removes a narrow 60Hz hum.")
    p.add_argument("--response-out", default=None, metavar="PNG_PATH",
                    help="with --eq, plot the combined analytic frequency response of all "
                         "EQ bands (no audio needed) to this PNG file.")
    return p


def parse_eq_band(spec):
    parts = spec.split(",")
    if len(parts) != 4:
        raise ValueError(f"--eq expects TYPE,FREQ,Q,GAIN (4 comma-separated values), got {spec!r}")
    type_, freq, q, gain = parts
    if type_ not in filters_mod.FILTER_BUILDERS:
        raise ValueError(f"--eq: unknown filter type {type_!r} "
                         f"(expected one of {sorted(filters_mod.FILTER_BUILDERS)})")
    return {"type": type_, "freq": float(freq), "q": float(q), "gain_db": float(gain)}


def dump_response(bands, sr, png_path, n_points=1024):
    """Combined analytic frequency response of a cascade of EQ bands --
    cascaded filters multiply in the linear domain, i.e. add in dB, so the
    total response is just the sum of each band's own response."""
    total_db = None
    w = None
    for band in bands:
        b, a = filters_mod.build_biquad(band["type"], band["freq"], sr,
                                        q=band["q"], gain_db=band["gain_db"])
        w, mag_db = filters_mod.frequency_response(b, a, sr, n_points=n_points)
        total_db = mag_db if total_db is None else total_db + mag_db
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.semilogx(w, total_db)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("Resposta em frequencia do EQ")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def parse_spectral_region(spec):
    parts = spec.split(",")
    if len(parts) != 5:
        raise ValueError(
            f"--spectral-region expects T0,T1,F0,F1,GAIN (5 comma-separated numbers), got {spec!r}")
    t0, t1, f0, f1, gain = (float(x) for x in parts)
    return (t0, t1, f0, f1, gain)


def apply_spectral_regions(audio, sr, region_specs, n_fft, hop, window):
    """Round-trips through the STFT once, paints every requested region into
    a shared mask, and inverse-transforms back -- the same primitive
    (stft -> paint_region -> istft) the GUI's interactive paint tool uses,
    just driven by flags instead of a mouse."""
    freqs, times, S = stft_mod.stft(audio, sr, n_fft=n_fft, hop=hop, window=window)
    mask = np.ones(S.shape, dtype=float)
    for spec in region_specs:
        t0, t1, f0, f1, gain = parse_spectral_region(spec)
        stft_mod.paint_region(mask, freqs, times, (f0, f1), (t0, t1), gain)
    return stft_mod.istft(S * mask, sr, hop=hop, window=window, length=len(audio))


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

    if args.eq:
        bands = [parse_eq_band(spec) for spec in args.eq]
        audio = filters_mod.apply_eq_bands(audio, sr, bands)

    if args.spectral_region:
        audio = apply_spectral_regions(audio, sr, args.spectral_region,
                                       n_fft=args.n_fft, hop=args.hop, window=args.window)

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


def dump_spectrogram(audio, sr, png_path, n_fft=1024, hop=256, window="hann"):
    times, freqs, db = stft_mod.spectrogram_db(audio, sr, n_fft=n_fft, hop=hop, window=window)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
    mesh = ax.pcolormesh(times, freqs, db, shading="gouraud", cmap="magma",
                         vmin=-80, vmax=0)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequencia (Hz)")
    ax.set_title(f"Espectrograma (n_fft={n_fft}, hop={hop}, {window})")
    fig.colorbar(mesh, ax=ax, label="dB (relativo ao pico)")
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

    if args.explain:
        print(stft_mod.describe_params(sr, n_fft=args.n_fft, hop=args.hop, window=args.window))
        print()

    audio = process(audio, sr, args)
    if args.eq:
        print(f"Applied {len(args.eq)} EQ band(s): {args.eq}")
        if args.response_out:
            bands = [parse_eq_band(spec) for spec in args.eq]
            dump_response(bands, sr, args.response_out)
            print(f"Frequency response plot written to {args.response_out}")
    if args.spectral_region:
        print(f"Applied {len(args.spectral_region)} spectral region(s): {args.spectral_region}")
    audio_io.save_audio(audio, sr, args.output)
    print(f"Processed audio written to {args.output}")

    if args.spectrum:
        dump_spectrum(audio, sr, args.spectrum)
        print(f"Spectrum plot written to {args.spectrum}")

    if args.spectrogram:
        dump_spectrogram(audio, sr, args.spectrogram, n_fft=args.n_fft, hop=args.hop,
                         window=args.window)
        print(f"Spectrogram plot written to {args.spectrogram}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
