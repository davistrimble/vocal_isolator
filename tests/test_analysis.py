import numpy as np
import pytest

from conftest import SR, note, stereo
from vocal_isolator.analysis import NOTE_NAMES, detect_beats, detect_key


def chord_progression(root, mode):
    """I - IV - V - I, played twice. The V chord uses a major third in both
    modes, like most real minor-key songs do."""
    third = 4 if mode == "major" else 3
    chords = [[0, third, 7], [5, 5 + third, 12], [7, 11, 14], [0, third, 7]]
    a4 = 440
    sound = np.concatenate([
        sum(note(a4 * 2 ** ((root + n - 9) / 12), 1.0) for n in chord) for chord in chords * 2
    ])
    return stereo(sound * 0.1)


@pytest.mark.parametrize("root", ["C", "F#", "A"])
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_detects_key(root, mode):
    assert detect_key(chord_progression(NOTE_NAMES.index(root), mode), SR) == (root, mode)


def click_track(bpm, seconds):
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    hit = note(1000, 0.05)
    for beat in np.arange(0, seconds - 0.1, 60 / bpm):
        start = int(beat * SR)
        out[start:start + len(hit)] += hit
    return stereo(out)


@pytest.mark.parametrize("bpm", [90, 120])
def test_detects_tempo_and_beats(bpm):
    tempo, beats = detect_beats(click_track(bpm, 20), SR)
    assert tempo == pytest.approx(bpm, rel=0.03)
    # the beats should land on the clicks, every 60/bpm seconds
    gaps = np.diff(beats)
    assert np.median(gaps) == pytest.approx(60 / bpm, rel=0.03)
