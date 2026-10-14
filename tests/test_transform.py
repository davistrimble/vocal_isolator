import pytest

from conftest import SR, peak_frequency, stereo, tone
from vocal_isolator.transform import change_key, change_tempo, semitones_between


@pytest.mark.parametrize(
    "start, end, expected",
    [("C", "D", 2), ("D", "C", -2), ("C", "A", -3), ("A", "C", 3), ("B", "C", 1), ("C", "F#", 6), ("G", "G", 0)],
)
def test_semitones_takes_the_short_way(start, end, expected):
    assert semitones_between(start, end) == expected


def test_change_key_moves_pitch_but_not_length():
    shifted = change_key(stereo(tone(440, 2.0)), SR, 2)
    assert peak_frequency(shifted[:, 0]) == pytest.approx(493.9, abs=3)  # A -> B
    assert len(shifted) == pytest.approx(2 * SR, rel=0.01)


def test_change_tempo_moves_length_but_not_pitch():
    slower = change_tempo(stereo(tone(440, 2.0)), SR, original_bpm=120, new_bpm=90)
    assert len(slower) / SR == pytest.approx(2 * 120 / 90, rel=0.01)
    assert peak_frequency(slower[:, 0]) == pytest.approx(440, abs=3)


def test_no_change_returns_input():
    audio = stereo(tone(440, 0.5))
    assert change_key(audio, SR, 0) is audio
    assert change_tempo(audio, SR, 100, 100) is audio
