"""Add a metronome click on every beat.

Without a metronome the output is trimmed to start at the first sung line.
With one, the intro is kept, so the clicks give a count-in before the vocal
comes in, and keep everyone in time with the soloist the whole way through.
"""
import numpy as np

CLICK_HZ = 1500  # high enough to cut through the vocals without clashing
CLICK_LENGTH_S = 0.03
CLICK_VOLUME = 0.35  # relative to the vocals, which are normalized to ~1


def click_sound(sr):
    """A short beep that dies away quickly, like a woodblock or metronome tick."""
    t = np.arange(int(CLICK_LENGTH_S * sr)) / sr
    return np.sin(2 * np.pi * CLICK_HZ * t) * np.exp(-t / 0.005)


def add_clicks(samples, sr, beat_times):
    """Mix a click into samples at each time in beat_times (seconds)."""
    click = click_sound(sr) * CLICK_VOLUME
    out = samples.copy()
    for beat in beat_times:
        start = int(beat * sr)
        end = min(start + len(click), len(out))
        if start >= len(out):
            break
        out[start:end] += click[: end - start, None]  # same click in every channel
    return out
