"""Figure out a song's tempo, beat positions, and key. All classical signal
processing, no ML.

Tempo/beats: librosa's beat tracker (Ellis, 2007). It finds "onsets" (moments
where a new sound starts, like a drum hit or a strummed chord), then looks for
the steady pulse that lines up with the most onsets.

Key: the Krumhansl-Schmuckler algorithm. Music psychologists measured how
important each of the 12 notes sounds in a major key and in a minor key (the
"key profiles" below). We measure how much each note is actually used in the
song, then compare against the profile for every possible key. The key whose
profile matches best wins.
"""
import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl & Kessler (1982). index 0 = the key's root note, 1 = a half step
# above it, etc. e.g. in a major key the root (6.35) and the 5th (5.19, index 7)
# sound the most "at home"
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# librosa's analysis runs at this rate. it's plenty for finding beats and
# notes, and twice as fast as working at 44.1 kHz
ANALYSIS_SR = 22050


def _mono(samples, sr):
    return librosa.resample(samples.mean(axis=1), orig_sr=sr, target_sr=ANALYSIS_SR)


def detect_beats(samples, sr):
    """Returns (tempo_bpm, beat_times_in_seconds)."""
    tempo, beat_times = librosa.beat.beat_track(y=_mono(samples, sr), sr=ANALYSIS_SR, units="time")
    return float(np.atleast_1d(tempo)[0]), beat_times


def detect_key(samples, sr):
    """Returns (root_note, mode), e.g. ("D", "major")."""
    # chroma = how much energy is in each of the 12 notes, ignoring octave,
    # for every moment. averaging over the song tells us which notes it uses
    chroma = librosa.feature.chroma_cqt(y=_mono(samples, sr), sr=ANALYSIS_SR)
    note_usage = chroma.mean(axis=1)

    best_score, best_key = -np.inf, None
    for root in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            # roll the profile so its root lines up with this candidate root
            score = np.corrcoef(note_usage, np.roll(profile, root))[0, 1]
            if score > best_score:
                best_score, best_key = score, (NOTE_NAMES[root], mode)
    return best_key
