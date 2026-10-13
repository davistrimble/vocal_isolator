"""Change the key and tempo of the separated vocals, for rehearsing a song in
a different key or at a slower speed than the recording.

Both use a phase vocoder (via librosa). Naively speeding up audio also raises
the pitch (like a chipmunk record), and naively changing pitch by resampling
also changes the speed. A phase vocoder separates the two: it cuts the audio
into tiny overlapping slices, and spaces the slices further apart or closer
together while adjusting each one's phase so they still line up smoothly.
That changes the length without changing the pitch. Pitch shifting is then
"stretch it, then resample it back to the original length".
"""
import librosa
import numpy as np

from vocal_isolator.analysis import NOTE_NAMES


def semitones_between(from_key, to_key):
    """How many semitones to shift to go from one key to another, e.g. C -> D
    is +2. Always picks the shorter way round (C -> A is -3, not +9), since
    big shifts sound more unnatural.
    """
    shift = (NOTE_NAMES.index(to_key) - NOTE_NAMES.index(from_key)) % 12
    if shift > 6:
        shift -= 12
    return shift


def _per_channel(func, samples):
    """librosa works on one channel at a time, so run func on each and stack them."""
    return np.stack([func(samples[:, ch]) for ch in range(samples.shape[1])], axis=1)


def change_key(samples, sr, semitones):
    if semitones == 0:
        return samples
    return _per_channel(lambda ch: librosa.effects.pitch_shift(ch, sr=sr, n_steps=semitones), samples)


def change_tempo(samples, original_bpm, new_bpm):
    """rate > 1 speeds it up, rate < 1 slows it down."""
    rate = new_bpm / original_bpm
    if rate == 1:
        return samples
    return _per_channel(lambda ch: librosa.effects.time_stretch(ch, rate=rate), samples)
