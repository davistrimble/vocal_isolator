"""Add a metronome click on every beat.

Without a metronome the output is trimmed to start at the first sung line.
With one, the intro is kept, so the clicks give a count-in before the vocal
comes in, and keep everyone in time with the soloist the whole way through.
"""
import numpy as np
from scipy.signal import savgol_filter

# 2.5 kHz is a range the ear is very sensitive to, and it's above where most
# of a voice's energy is, so the click stands out instead of blending in
CLICK_HZ = 2500
CLICK_LENGTH_S = 0.08
CLICK_DECAY_S = 0.02  # how fast the click dies away

# the first version had the click at about a third of the vocals' volume and
# you couldn't hear it while anyone was singing. now the vocals get turned
# down to half and the click goes near the max, which puts the click about
# 14 dB above the vocals on a typical beat (9 dB on the loudest vocal parts)
VOCAL_LEVEL = 0.5
CLICK_LEVEL = 0.9

# the beat tracker only places beats on a grid of ~12 ms steps, and gets the
# odd beat wrong, so raw clicks wobble audibly. each click time gets replaced
# with the average of the beats around it (a moving average of the beat
# times). wider = steadier, but slower to follow the band if they speed up or
# slow down. 9 beats was the best balance on songs with tempo changes
SMOOTHING_BEATS = 9


def click_sound(sr):
    """A short tick: a tone plus its octave, dying away quickly like a woodblock."""
    t = np.arange(int(CLICK_LENGTH_S * sr)) / sr
    tone = (np.sin(2 * np.pi * CLICK_HZ * t) + 0.5 * np.sin(2 * np.pi * 2 * CLICK_HZ * t)) / 1.5
    return tone * np.exp(-t / CLICK_DECAY_S)


def steady_beats(beat_times, song_length_s):
    """Smooth out the beat tracker's wobble, and extend the beats to cover the
    whole song so there's a count-in even if the music starts late.
    """
    if len(beat_times) >= SMOOTHING_BEATS:
        # a Savitzky-Golay filter with polyorder=1 is a moving average that
        # also handles the first and last few beats sensibly
        beat_times = savgol_filter(beat_times, SMOOTHING_BEATS, polyorder=1)
    if len(beat_times) < 2:
        return beat_times

    # keep going backwards to the start at the opening tempo, and forwards to
    # the end at the closing tempo
    start_gap = np.median(np.diff(beat_times[:SMOOTHING_BEATS]))
    end_gap = np.median(np.diff(beat_times[-SMOOTHING_BEATS:]))
    before = np.arange(beat_times[0] - start_gap, 0, -start_gap)[::-1]
    after = np.arange(beat_times[-1] + end_gap, song_length_s, end_gap)
    return np.concatenate([before, beat_times, after])


def add_clicks(vocals, sr, beat_times):
    """Mix a click into the (already normalized) vocals at each time in beat_times."""
    click = click_sound(sr) * CLICK_LEVEL
    out = vocals * VOCAL_LEVEL
    for beat in beat_times:
        start = int(beat * sr)
        if start >= len(out):
            break
        end = min(start + len(click), len(out))
        out[start:end] += click[: end - start, None]  # same click in every channel
    return out
