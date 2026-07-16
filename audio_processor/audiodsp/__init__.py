"""audiodsp -- small, headless (no GUI, no plotting) audio DSP toolkit.

Modules:
    io       -- load/save audio files (soundfile), mono-mixing, error handling
    effects  -- trim, Fourier compression, echo, reverb
    spectrum -- magnitude spectrum (positive-frequency half only)
    stft     -- STFT/ISTFT + spectrogram-in-dB (time-frequency analysis)
    filters  -- real EQ/filtering via biquads (low/high/band-pass, notch,
                parametric peak, shelves) -- the streaming/real-time
                counterpart to the STFT-based spectral painter
    playback -- cross-platform play/stop via sounddevice

Everything here operates on plain numpy arrays + a sample rate, so it can be
unit tested and reused from both the CLI (audioprocess.py) and the GUI
(audio_gui.py) without pulling in Tkinter or matplotlib.
"""

from . import io, effects, spectrum, stft, filters, playback

__all__ = ["io", "effects", "spectrum", "stft", "filters", "playback"]
