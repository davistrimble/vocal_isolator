import numpy as np
import pytest

from conftest import SR, stereo, tone
from vocal_isolator.metronome import CLICK_LEVEL, VOCAL_LEVEL, add_clicks, click_sound, steady_beats


def test_smoothing_removes_jitter():
    rng = np.random.default_rng(0)
    perfect = np.arange(2, 60, 0.5)
    wobbly = perfect + rng.uniform(-0.015, 0.015, len(perfect))  # +-15 ms, like the beat tracker
    smoothed = steady_beats(wobbly, 60)
    smoothed = smoothed[(smoothed > 4) & (smoothed < 56)]  # ignore the ends
    assert np.std(np.diff(smoothed)) < np.std(np.diff(wobbly)) / 3


def test_beats_extended_back_to_the_start():
    beats = steady_beats(np.arange(3.0, 20.0, 0.5), 25)
    assert beats[0] == pytest.approx(0)  # count-in from the very start
    assert beats[-1] > 24  # and all the way to the end
    assert np.allclose(np.diff(beats), 0.5)


def test_clicks_land_on_the_beats():
    silence = np.zeros((SR * 3, 2), dtype=np.float32)
    out = add_clicks(silence, SR, [0.5, 1.5, 2.5])
    for beat in (0.5, 1.5, 2.5):
        i = int(beat * SR)
        assert np.abs(out[i - 100:i, 0]).max() == 0  # silent right before the beat
        assert np.abs(out[i:i + 100, 0]).max() > 0.5  # loud right after it


def test_click_is_louder_than_the_vocals():
    vocals = stereo(tone(300, 2.0)) * 0.95  # vocals at full volume
    out = add_clicks(vocals, SR, [1.0])
    assert np.abs(out[:, 0]).max() > VOCAL_LEVEL
    assert np.abs(click_sound(SR)).max() * CLICK_LEVEL > 1.5 * VOCAL_LEVEL
