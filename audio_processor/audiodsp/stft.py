"""Short-Time Fourier Transform (STFT) / inverse STFT (ISTFT) -- the
time-frequency backbone the rest of the "Spectral Studio" work builds on.

Why this exists (the didactic bit, worth reading once):

``spectrum.py``'s ``magnitude_spectrum`` answers "which frequencies are
present in this clip, overall?" -- a single FFT of the whole signal. That
throws away *when* each frequency happened: a flute note at the start and
the same note at the end look identical in a whole-signal FFT.

The STFT answers "which frequencies are present, and when?" by sliding a
short analysis window along the signal and taking an FFT of each window
in turn. Stack those FFTs side by side (frequency on one axis, time on the
other) and you get a **spectrogram** -- literally a picture of the sound.

That slicing is also *the* classic trade-off of this whole family of
techniques, sometimes called the time-frequency uncertainty principle (a
cousin of the same idea from physics):

- A **long window** (big ``n_fft``) sees many cycles of a low frequency, so
  it can tell two nearby pitches apart precisely -- good frequency
  resolution. But it smears together anything that happens *within* that
  window, so two clicks 5ms apart become one blur -- poor time resolution.
- A **short window** (small ``n_fft``) does the opposite: it nails *when*
  something happened, but can't distinguish nearby frequencies well.

There is no free lunch here, only a dial: ``n_fft`` in samples, ``hop`` in
samples (how far the window slides between frames -- smaller hop = more
overlap = smoother look in time, at the cost of more frames to compute).

Everything below is plain numpy/scipy, headless (no plotting), so it can be
unit tested with a synthetic signal and reused identically by the CLI and
the GUI -- same layering discipline as ``spectrum.py``.
"""

import numpy as np
from scipy.signal import get_window


def stft(audio, sr, n_fft=1024, hop=256, window="hann"):
    """Slide a window of length ``n_fft`` across ``audio`` in steps of
    ``hop`` samples, taking the FFT of each windowed slice.

    The signal is reflect-padded by ``n_fft // 2`` samples on each side
    first, so that frame 0 is centered on sample 0 of the original audio
    (rather than starting there) -- this "centered" convention is what
    makes ``istft`` line back up with the original sample positions, and
    is the same convention most STFT implementations (e.g. librosa) use.

    Returns:
        freqs: 1-D array, length ``n_fft // 2 + 1``, in Hz (0 to sr/2 --
            same positive-half convention as ``spectrum.magnitude_spectrum``).
        times: 1-D array, one timestamp (in seconds) per frame, taken at
            the *center* of each analysis window.
        S: complex array of shape ``(len(freqs), len(times))`` -- the STFT
            matrix. ``np.abs(S)`` is magnitude, ``np.angle(S)`` is phase;
            both matter for ``istft`` to reconstruct audio exactly, but
            only the magnitude is needed for a spectrogram *picture*.
    """
    if n_fft <= 0 or hop <= 0:
        raise ValueError("n_fft and hop must be positive")
    if hop > n_fft:
        raise ValueError("hop must not exceed n_fft (frames would miss samples)")

    win = get_window(window, n_fft, fftbins=True)
    pad = n_fft // 2
    padded = np.pad(np.asarray(audio, dtype=np.float64), (pad, pad), mode="reflect")

    # The frame grid (starts at 0, hop, 2*hop, ...) only covers the padded
    # signal up to the last start position where a full n_fft-sample frame
    # still fits -- generally a few dozen samples short of the true end.
    # Left uncorrected, that tail is simply never analyzed, and istft() has
    # nothing to reconstruct it from (it comes back as silence). Padding
    # with a few more zeros so the tail lands exactly on the hop grid keeps
    # every input sample inside at least one frame.
    remainder = (len(padded) - n_fft) % hop
    if remainder:
        padded = np.pad(padded, (0, hop - remainder), mode="constant")

    n_frames = 1 + (len(padded) - n_fft) // hop
    # A "sliding window view" via stride tricks: instead of a Python loop
    # copying each n_fft-sample slice, we describe the same data with a new
    # shape/stride so overlapping frames share the underlying memory. This
    # is the vectorized equivalent of the double for-loop a naive STFT would
    # use, in the same spirit as filter2D/summed-area-table replacing manual
    # pixel loops in the sibling classical_filters project.
    frame_stride = padded.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(n_fft, n_frames), strides=(frame_stride, frame_stride * hop))

    windowed = frames * win[:, np.newaxis]
    S = np.fft.rfft(windowed, axis=0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(n_frames) * hop / sr
    return freqs, times, S


def istft(S, sr, hop, window="hann", length=None):
    """Invert ``stft``: turn a complex STFT matrix back into a waveform.

    Each frame is taken back to the time domain (inverse FFT), re-windowed,
    and overlap-added into an output buffer -- the standard "weighted
    overlap-add" reconstruction. Overlapping windows don't sum to a flat 1.0
    on their own, so each output sample is divided by the sum of squared
    window values that touched it (this is what makes ``istft(stft(x))``
    reconstruct ``x`` instead of a volume-modulated version of it).

    ``length``, if given, trims/pads the result to an exact sample count --
    useful because the padding ``stft`` added back on each side would
    otherwise leave a few extra samples on the output.
    """
    n_freq, n_frames = S.shape
    n_fft = (n_freq - 1) * 2
    win = get_window(window, n_fft, fftbins=True)

    frames = np.fft.irfft(S, n=n_fft, axis=0)
    windowed = frames * win[:, np.newaxis]

    pad = n_fft // 2
    out_len = (n_frames - 1) * hop + n_fft
    output = np.zeros(out_len, dtype=np.float64)
    win_sq_sum = np.zeros(out_len, dtype=np.float64)

    # Overlap-add: each frame's contribution lands at a shifted position.
    # A plain Python loop over frames (not samples) -- readable, and cheap
    # since n_frames is small relative to the sample count.
    for i in range(n_frames):
        start = i * hop
        output[start:start + n_fft] += windowed[:, i]
        win_sq_sum[start:start + n_fft] += win ** 2

    win_sq_sum[win_sq_sum < 1e-8] = 1e-8
    output /= win_sq_sum

    # Undo the centering pad stft() added.
    output = output[pad:out_len - pad]

    if length is not None:
        if len(output) < length:
            output = np.pad(output, (0, length - len(output)))
        else:
            output = output[:length]
    return output


def spectrogram_db(audio, sr, n_fft=1024, hop=256, window="hann", db_floor=-80.0):
    """Convenience wrapper for plotting: STFT -> magnitude -> decibels.

    Decibels because raw magnitude spectrograms are almost all near-zero
    with a few tall spikes -- the ear (and a useful plot) cares about
    relative *ratios* of loudness, not absolute linear amplitude. ``db_floor``
    clips very quiet content so silence doesn't dominate the color range
    with numerical noise.

    Returns (times, freqs, db) ready to hand to e.g. matplotlib's
    ``pcolormesh``/``imshow``.
    """
    freqs, times, S = stft(audio, sr, n_fft=n_fft, hop=hop, window=window)
    mag = np.abs(S)
    peak = mag.max()
    ref = peak if peak > 0 else 1.0
    db = 20 * np.log10(np.maximum(mag, 1e-10) / ref)
    db = np.maximum(db, db_floor)
    return times, freqs, db


def describe_params(sr, n_fft=1024, hop=256, window="hann"):
    """Human-readable explanation of what a given (n_fft, hop, window)
    choice means in real-world terms -- the ``--explain`` text for the CLI
    and the same numbers the GUI could show next to its sliders.
    """
    window_ms = n_fft / sr * 1000
    hop_ms = hop / sr * 1000
    overlap_pct = (1 - hop / n_fft) * 100
    freq_res_hz = sr / n_fft
    return (
        f"STFT settings: n_fft={n_fft} samples, hop={hop} samples, window={window!r}\n"
        f"  - Cada janela de analise cobre {window_ms:.1f} ms de audio.\n"
        f"  - O passo entre janelas e {hop_ms:.1f} ms ({overlap_pct:.0f}% de sobreposicao).\n"
        f"  - Resolucao de frequencia: ~{freq_res_hz:.1f} Hz por bin\n"
        f"    (duas frequencias mais proximas que isso tendem a se misturar).\n"
        f"  - Resolucao temporal: ~{window_ms:.1f} ms\n"
        f"    (dois eventos mais proximos que isso tendem a se misturar no tempo).\n"
        f"  - Trade-off: n_fft maior = mais preciso em frequencia, mais borrado no "
        f"tempo; n_fft menor = o oposto. Nao ha almoco gratis aqui, so um dial."
    )
