"""Audio effects: trim, Fourier compression, echo, reverb.

Ported from the original ``audiointerface.py`` prototype. The math is
unchanged (it was already correct) -- only the comments/docstrings were
rewritten (the originals were vowel-stripped pseudo-Portuguese, e.g. "corta
o audo para a durco") and the functions were made GUI-free so they can be
imported and tested without Tkinter.
"""

import numpy as np
from scipy.fft import fft, ifft


def trim_audio(audio, sr, duration=10):
    """Slice the first ``duration`` seconds off the start of ``audio``.

    If the clip is shorter than ``duration``, the whole clip is returned
    unchanged (numpy slicing clamps the stop index automatically).
    """
    num_samples = int(duration * sr)
    return audio[:num_samples]


def compress_audio(audio, p):
    """Fourier "compression": keep only the lowest and highest ``p`` fraction
    of FFT bins (by index) and zero out everything in between, then invert.

    Because real-signal FFTs are conjugate-symmetric, the low indices near 0
    and the high indices near ``len(ft)`` both correspond to low frequencies,
    so this keeps the low-frequency content and discards the mid/high
    frequencies -- a crude low-pass filter. ``p`` is the fraction of bins
    (from each end) retained, e.g. ``p=0.5`` keeps everything (the two halves
    already cover the whole spectrum), smaller ``p`` filters more aggressively.
    """
    ft = fft(audio)
    limit_index = int(len(ft) * p)
    ft_filtered = np.zeros_like(ft)
    ft_filtered[:limit_index] = ft[:limit_index]
    ft_filtered[-limit_index:] = ft[-limit_index:]
    return ifft(ft_filtered).real


def add_echo(audio, sr, delay=0.5, echo_gain=0.6):
    """Add a single delayed, attenuated copy of the signal on top of itself.

    ``delay`` is in seconds, ``echo_gain`` scales the delayed copy's
    amplitude before mixing it back in.
    """
    delay_samples = int(sr * delay)
    echo_signal = np.zeros_like(audio)
    if delay_samples > 0:
        echo_signal[delay_samples:] = audio[:-delay_samples]
    else:
        echo_signal[:] = audio
    return audio + echo_gain * echo_signal


def add_reverb(audio, sr, num_delays=10, delay_time=0.05):
    """Simulate reverb as a sum of decaying, zero-padded delayed copies.

    Each of the ``num_delays`` taps is shifted by ``i * delay_time`` seconds
    and scaled by ``1 / (i + 1)``, and copies are zero-padded (not wrapped
    around) so the tail doesn't bleed back into the start of the clip.
    """
    delay_samples = int(sr * delay_time)
    reverb_audio = audio.copy()
    for i in range(1, num_delays + 1):
        shift = i * delay_samples
        if shift <= 0 or shift >= len(audio):
            continue
        padded = np.concatenate((np.zeros(shift), audio[:-shift]))
        reverb_audio += padded / (i + 1)
    return reverb_audio
