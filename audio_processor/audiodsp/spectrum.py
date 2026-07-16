"""Magnitude spectrum computation -- headless, no matplotlib import here.

Kept separate from any plotting code so it can be imported and unit tested
in environments without a display, and reused identically by both the CLI
(which plots with matplotlib) and the GUI (which embeds a
FigureCanvasTkAgg). Plotting lives in the CLI/GUI layer, not here.
"""

import numpy as np


def magnitude_spectrum(audio, sr):
    """Compute the FFT magnitude spectrum, positive-frequency half only.

    The original prototype plotted frequencies from 0 up to ``sr`` using
    ``np.linspace(0, sr, len(magnitude_spectrum))``, which includes the
    redundant, mirrored upper half of a real signal's FFT (frequencies above
    the Nyquist rate, ``sr / 2``, carry no extra information for real-valued
    audio). This returns only the informative half, from 0 to ``sr / 2``.

    Returns:
        (freqs, mags): both 1-D arrays of the same length, ``freqs`` in Hz
        ranging from 0 to ``sr / 2``.
    """
    n = len(audio)
    ft = np.fft.rfft(audio)
    mags = np.abs(ft)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, mags
