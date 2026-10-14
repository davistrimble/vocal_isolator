"""Shared helpers for the tests. The tests make their own audio instead of
using real songs, so nothing copyrighted needs to be in the repo.
"""
import numpy as np
import pytest
import soundfile as sf

SR = 44100


def tone(freq, seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def note(freq, seconds, sr=SR):
    """A pitched note with a few harmonics that fades out, like a piano."""
    t = np.arange(int(seconds * sr)) / sr
    sound = sum(np.sin(2 * np.pi * k * freq * t) / k for k in range(1, 5))
    return (sound * np.exp(-t * 3)).astype(np.float32)


def stereo(mono):
    return np.stack([mono, mono], axis=1)


def peak_frequency(mono, sr=SR):
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    return freqs[np.argmax(np.abs(np.fft.rfft(mono)))]


@pytest.fixture
def mp3_file(tmp_path):
    """A real 2 second mp3 of a quiet tone."""
    path = tmp_path / "song.mp3"
    sf.write(path, stereo(tone(440, 2.0)) * 0.5, SR)
    return str(path)
