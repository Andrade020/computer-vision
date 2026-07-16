"""Real EQ/filtering via biquads -- the audio-engineering counterpart to the
STFT-based spectral painter in ``stft.py``.

Why this exists alongside the spectral editor (worth reading once): A2's
``paint_region`` can silence "this frequency band, but only during this
1-second window" -- something only possible because it works on the full
time-frequency plane (an STFT) computed from the *entire* recording ahead
of time. A biquad filter answers a different, narrower question: "always
attenuate/boost this frequency band, everywhere, streaming, sample by
sample, forever" -- it has no notion of "only between 1.0s and 2.0s", but
unlike the STFT it needs no lookahead at all, which is exactly why every
real analog and digital EQ, every synthesizer filter, and every
real-time audio effect is built from biquads, not FFTs: a biquad can run
one sample at a time as audio streams in, with no buffering.

A "biquad" (biquadratic filter) is the smallest IIR (infinite impulse
response) building block that can shape frequency content in genuinely
useful ways: it looks at the last 2 input samples and the last 2 output
samples (hence "bi-quad" -- two poles, two zeros) to produce each new
output sample::

    y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0

Different (b0,b1,b2,a0,a1,a2) coefficient formulas turn that same recurrence
into a low-pass, high-pass, band-pass, notch, shelf, or parametric-peak
filter -- the formulas below are the well-known "Audio EQ Cookbook" (Robert
Bristow-Johnson) derivations, the same math behind most real-world digital
EQ plugins and mixing-console EQ sections.
"""

import numpy as np
from scipy.signal import lfilter, freqz


def _normalize(b0, b1, b2, a0, a1, a2):
    """Cookbook formulas produce un-normalized coefficients (a0 often != 1);
    scipy.signal.lfilter expects the a0=1 convention, so divide through."""
    return (np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))


def biquad_lowpass(freq, sr, q=0.707):
    """Passes frequencies below ``freq`` (Hz), attenuates above it at
    -12dB/octave past the cutoff (a single biquad is a 2nd-order filter).
    ``q`` controls the sharpness/resonance at the cutoff: 0.707 (the
    default) is "maximally flat" (Butterworth-like, no bump); higher Q
    adds a resonant peak right at the cutoff frequency before it rolls off."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_highpass(freq, sr, q=0.707):
    """Mirror of ``biquad_lowpass``: passes above ``freq``, attenuates below."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_bandpass(freq, sr, q=1.0):
    """Passes only a band centered on ``freq``; ``q`` sets how narrow the
    band is (higher Q = narrower). Constant 0dB peak gain convention."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_notch(freq, sr, q=1.0):
    """The opposite of band-pass: removes a narrow band centered on
    ``freq``, passes everything else -- e.g. for pulling out a 60Hz mains
    hum or a single whistling tone, without touching the rest of the
    spectrum (the streaming/real-time counterpart to painting that same
    band out on the spectrogram in A2 -- same intent, different mechanism:
    a notch filter applies everywhere/forever, a spectral paint can target
    one moment in time)."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0
    b1 = -2 * cos_w0
    b2 = 1.0
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_peak(freq, sr, gain_db, q=1.0):
    """Parametric EQ "bell" band: boosts (gain_db > 0) or cuts (gain_db < 0)
    a band centered on ``freq``, width set by ``q``, leaving frequencies
    far from ``freq`` essentially untouched -- the classic "parametric EQ"
    knob (center frequency, gain, and bandwidth all independently tunable)."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_lowshelf(freq, sr, gain_db, q=0.707):
    """Boosts/cuts everything BELOW ``freq`` by ``gain_db``, leaving
    frequencies well above it unchanged -- a "bass boost/cut" knob."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_a * alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def biquad_highshelf(freq, sr, gain_db, q=0.707):
    """Mirror of ``biquad_lowshelf``: boosts/cuts everything ABOVE ``freq``
    -- a "treble boost/cut" knob."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_a * alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


FILTER_BUILDERS = {
    "lowpass": biquad_lowpass,
    "highpass": biquad_highpass,
    "bandpass": biquad_bandpass,
    "notch": biquad_notch,
    "peak": biquad_peak,
    "lowshelf": biquad_lowshelf,
    "highshelf": biquad_highshelf,
}

# which of the builders above take a gain_db argument (peak/shelf types)
_GAIN_TYPES = {"peak", "lowshelf", "highshelf"}


def build_biquad(filter_type, freq, sr, q=0.707, gain_db=0.0):
    """Dispatch helper mirroring this repo's other build_mask/apply_morphology
    "pick by name" conveniences."""
    if filter_type not in FILTER_BUILDERS:
        raise ValueError(f"unknown filter_type: {filter_type!r} "
                         f"(expected one of {sorted(FILTER_BUILDERS)})")
    builder = FILTER_BUILDERS[filter_type]
    if filter_type in _GAIN_TYPES:
        return builder(freq, sr, gain_db, q=q)
    return builder(freq, sr, q=q)


def apply_filter(audio, b, a):
    """Run the biquad recurrence over the whole signal via
    ``scipy.signal.lfilter`` -- a real (causal) IIR filter, same as a
    hardware/software EQ would apply: it only ever looks at past samples,
    so (unlike ``stft.istft`` or a zero-phase ``filtfilt``) it has a real
    phase response -- transients shift slightly in time near the cutoff,
    exactly like an analog EQ's knobs would do to the signal."""
    return lfilter(b, a, audio)


def apply_eq_bands(audio, sr, bands):
    """Cascade several biquads in series -- a simple graphic/parametric EQ.

    ``bands``: list of dicts, each ``{"type": ..., "freq": ..., "q": ...,
    "gain_db": ...}`` (gain_db only meaningful for peak/shelf types).
    Applied in the given order; each band's output feeds the next band's
    input, the same way a multi-band hardware EQ chains its sections.
    """
    result = audio
    for band in bands:
        b, a = build_biquad(band["type"], band["freq"], sr,
                            q=band.get("q", 0.707), gain_db=band.get("gain_db", 0.0))
        result = apply_filter(result, b, a)
    return result


def frequency_response(b, a, sr, n_points=512):
    """The filter's magnitude response in dB across 0..sr/2 Hz -- "if you
    fed this filter every possible frequency, how much would each one be
    attenuated or boosted?" -- computed analytically from the coefficients
    (no audio needed), which is what makes it useful as a live preview
    while turning an EQ knob, before touching any actual signal."""
    w, h = freqz(b, a, worN=n_points, fs=sr)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, mag_db


def describe_filter(filter_type, freq, sr, q=0.707, gain_db=0.0):
    """Plain-language explanation of what a given filter setting does --
    the --explain text for the CLI, and the same numbers the GUI could
    show next to its sliders."""
    nyquist = sr / 2.0
    lines = [f"Filtro: {filter_type}, freq={freq:.0f}Hz, Q={q:.2f}"
            + (f", ganho={gain_db:+.1f}dB" if filter_type in _GAIN_TYPES else "")]
    if freq >= nyquist:
        lines.append(f"  - AVISO: {freq:.0f}Hz esta acima (ou na) frequencia de "
                     f"Nyquist ({nyquist:.0f}Hz) para uma taxa de {sr}Hz -- o "
                     f"filtro nao vai se comportar como esperado.")
    if filter_type == "lowpass":
        lines.append(f"  - Deixa passar abaixo de {freq:.0f}Hz, atenua acima "
                     "(-12dB/oitava a partir do corte).")
    elif filter_type == "highpass":
        lines.append(f"  - Deixa passar acima de {freq:.0f}Hz, atenua abaixo "
                     "(-12dB/oitava a partir do corte).")
    elif filter_type == "bandpass":
        lines.append(f"  - So deixa passar uma faixa em torno de {freq:.0f}Hz; "
                     f"Q={q:.2f} maior = faixa mais estreita.")
    elif filter_type == "notch":
        lines.append(f"  - Remove so uma faixa estreita em torno de {freq:.0f}Hz, "
                     "deixa o resto intacto -- equivalente em tempo real ao "
                     "editor espectral (A2), mas aplicado sempre/em todo o "
                     "audio, nao so num trecho de tempo especifico.")
    elif filter_type == "peak":
        lines.append(f"  - Realca ou corta uma faixa em torno de {freq:.0f}Hz "
                     f"em {gain_db:+.1f}dB, sem afetar frequencias distantes "
                     "(o botao classico de EQ parametrico).")
    elif filter_type == "lowshelf":
        lines.append(f"  - Realca ou corta tudo ABAIXO de {freq:.0f}Hz em "
                     f"{gain_db:+.1f}dB (grave).")
    elif filter_type == "highshelf":
        lines.append(f"  - Realca ou corta tudo ACIMA de {freq:.0f}Hz em "
                     f"{gain_db:+.1f}dB (agudo).")
    lines.append("  - E um filtro IIR (biquad) causal, como um EQ de verdade -- "
                "roda amostra a amostra, sem precisar olhar o audio inteiro "
                "de antemao (diferente da STFT usada no espectrograma/editor "
                "espectral).")
    return "\n".join(lines)
