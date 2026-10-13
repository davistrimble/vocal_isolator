"""Change the key and tempo of the separated vocals, for rehearsing a song in
a different key or at a slower speed than the recording.

Naively speeding up audio also raises the pitch (like a chipmunk record), and
naively changing pitch by resampling also changes the speed. A phase vocoder
separates the two: it cuts the audio into tiny overlapping slices, and spaces
the slices further apart or closer together while adjusting each one's phase
so they still line up smoothly. That changes the length without changing the
pitch. Pitch shifting is then "stretch it, then resample it back".

I started with librosa's phase vocoder, but it smeared the vocals (that
"phasey", underwater sound). A good test: feed it two nearly identical inputs
(3% different) and the outputs came out ~50% different, meaning it was adding
a lot of its own artifacts. Rubber Band, the library behind the time-stretch
in Audacity and other audio editors, kept them within ~9% using its newer
"fine" engine, so that's what this uses now.

Rubber Band also has formant preservation. Formants are the resonances of the
singer's throat and mouth that make a voice sound like *them*. Plain pitch
shifting moves the formants too, so shifting up sounds chipmunky and shifting
down sounds like a giant. Preserving them keeps the soloist sounding like the
soloist in the new key.

Needs the rubberband command line tool: `brew install rubberband` on a Mac.
"""
import numpy as np
import pyrubberband

from vocal_isolator.analysis import NOTE_NAMES

# "--fine" picks Rubber Band's newer, higher quality engine
ENGINE = {"--fine": ""}


def semitones_between(from_key, to_key):
    """How many semitones to shift to go from one key to another, e.g. C -> D
    is +2. Always picks the shorter way round (C -> A is -3, not +9), since
    big shifts sound more unnatural.
    """
    shift = (NOTE_NAMES.index(to_key) - NOTE_NAMES.index(from_key)) % 12
    if shift > 6:
        shift -= 12
    return shift


def change_key(samples, sr, semitones):
    if semitones == 0:
        return samples
    args = {**ENGINE, "--formant": ""}
    return pyrubberband.pitch_shift(samples, sr, semitones, rbargs=args).astype(np.float32)


def change_tempo(samples, sr, original_bpm, new_bpm):
    """rate > 1 speeds it up, rate < 1 slows it down."""
    rate = new_bpm / original_bpm
    if rate == 1:
        return samples
    return pyrubberband.time_stretch(samples, sr, rate, rbargs=ENGINE).astype(np.float32)
