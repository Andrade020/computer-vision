"""Audio file I/O: load/save, with mono-mixing and clean error handling.

The original prototype called ``sf.read`` directly with no error handling,
so a missing or corrupt file crashed the whole GUI with a raw traceback.
``load_audio`` here catches that and re-raises a plain ``RuntimeError`` with
a readable message, which callers (CLI or GUI) can show to the user instead
of a stack trace.
"""

import numpy as np
import soundfile as sf


def load_audio(path):
    """Load an audio file and mix it down to mono.

    Returns:
        (audio, sr): 1-D float array and the integer sample rate.

    Raises:
        RuntimeError: if the file does not exist, is not a readable audio
            file, or soundfile otherwise fails to decode it.
    """
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file '{path}': {exc}") from exc

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def save_audio(audio, sr, path):
    """Write a numpy audio buffer to disk as a WAV (or whatever soundfile
    infers from the extension)."""
    sf.write(path, audio, sr)
    return path
