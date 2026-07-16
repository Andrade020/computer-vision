"""Cross-platform audio playback via sounddevice.

The original prototype played audio through the OS's default media player
using ``os.startfile(...)``, which is Windows-only and appeared in two
places: ``play_audio_file()`` and directly inline inside
``AudioProcessingApp.play_original_audio()``. Both call sites are replaced
by this module: ``sounddevice.play`` plays a numpy array straight through
the default output device (no temp file, no external player, works on
Windows/macOS/Linux alike).

``Player`` extends that into a small stateful transport (play/pause/resume/
seek/position) without needing a raw ``sounddevice.OutputStream`` callback:
it just remembers, in wall-clock time, when the current playback segment
started and how many samples had already been consumed before that, and
combines the two to know "where" playback currently is. Pausing stops the
stream and freezes that position; resuming (or seeking while playing)
restarts ``sd.play()`` from the frozen/new position with a fresh start
timestamp.
"""

import time

import sounddevice as sd


class Player:
    """Stateful play/pause/seek transport built on sd.play()/sd.stop().

    ``time_fn`` is injectable (defaults to ``time.time``) so tests can drive
    a fake clock instead of sleeping in real wall-clock time.
    """

    def __init__(self, time_fn=time.time):
        self._time_fn = time_fn
        self._audio = None
        self._sr = None
        self._pos_samples = 0
        self._playing = False
        self._started_at = None  # time_fn() value when current segment began

    def load(self, audio, sr):
        """Load a new buffer, stopping any current playback and resetting
        position to the start."""
        self.stop()
        self._audio = audio
        self._sr = sr
        self._pos_samples = 0

    def play(self):
        """(Re)starts playback from the current position."""
        if self._audio is None:
            return
        sd.stop()
        sd.play(self._audio[self._pos_samples:], self._sr)
        self._started_at = self._time_fn()
        self._playing = True

    def pause(self):
        if not self._playing:
            return
        sd.stop()
        self._pos_samples = min(
            self._pos_samples + int((self._time_fn() - self._started_at) * self._sr),
            len(self._audio))
        self._playing = False

    def resume(self):
        self.play()

    def stop(self):
        sd.stop()
        self._playing = False
        self._pos_samples = 0

    def seek(self, seconds):
        """Jump to an absolute position (in seconds), whether playing or
        paused. If playback was in progress, it keeps playing from the new
        spot; if paused, it stays paused there."""
        if self._audio is None or self._sr is None:
            return
        self._pos_samples = max(0, min(int(seconds * self._sr), len(self._audio)))
        if self._playing:
            self.play()

    def position_seconds(self):
        if self._audio is None or self._sr is None:
            return 0.0
        pos = self._pos_samples
        if self._playing:
            pos += int((self._time_fn() - self._started_at) * self._sr)
        return min(pos, len(self._audio)) / self._sr

    def duration_seconds(self):
        if self._audio is None or self._sr is None:
            return 0.0
        return len(self._audio) / self._sr

    def is_playing(self):
        return self._playing

    def finished(self):
        """True once a playing buffer has reached its end. The GUI's poll
        loop uses this to notice natural end-of-playback and reset its
        Play/Pause button rather than showing "Pause" forever at a stuck
        position."""
        return self._playing and self.position_seconds() >= self.duration_seconds()


# Module-level shared instance + thin functional wrappers, kept for
# backward compatibility with any existing call sites that just want
# fire-and-forget playback without transport controls.
_shared_player = Player()


def play(audio, sr, blocking=False):
    """Play a numpy audio array through the default output device.

    Any playback already in progress is stopped first so overlapping calls
    don't talk over each other.
    """
    sd.stop()
    sd.play(audio, sr)
    if blocking:
        sd.wait()


def stop():
    """Stop any playback currently in progress."""
    sd.stop()
