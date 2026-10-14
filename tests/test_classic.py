"""The classic (no ML) pipeline, on a made-up mix where we know what the
vocals should sound like."""
import numpy as np

from scipy.signal import stft

from conftest import SR, note
from vocal_isolator.center_extract import extract_center
from vocal_isolator.repet import find_period


def si_sdr(reference, estimate):
    """How close estimate is to reference, in dB, ignoring overall volume. Higher is better."""
    target = (estimate @ reference) / (reference @ reference) * reference
    return 10 * np.log10((target @ target) / ((estimate - target) @ (estimate - target)))


def make_mix(seconds=12):
    rng = np.random.default_rng(1)
    n = int(seconds * SR)
    # "vocal": a melody that keeps changing, panned center
    melody = rng.choice([261.6, 293.7, 329.6, 349.2, 392.0, 440.0], size=int(seconds / 0.4))
    vocal = np.concatenate([note(f, 0.4) for f in melody])[:n] * 0.3
    # backing loop that repeats every 2s, also center
    loop = np.concatenate([note(f, 0.5) for f in (130.8, 164.8, 196.0, 164.8)])
    backing = np.tile(loop, seconds // 2 + 1)[:n] * 0.3
    # guitar panned hard left, noise panned hard right
    guitar = note(330, seconds)[:n] * 0.2
    hiss = rng.normal(0, 0.05, n).astype(np.float32)
    left = vocal + backing + guitar
    right = vocal + backing + hiss
    return np.stack([left, right], axis=1).astype(np.float32), vocal


def test_finds_repeating_period():
    """The backing loop repeats every 2 seconds. This used to come back as 4
    (a multiple of the real period) before I loosened the check in find_period."""
    mix, _ = make_mix()
    _, _, spec = stft(mix.mean(axis=1), SR, nperseg=2048, noverlap=1536)
    period = find_period(np.abs(spec), frames_per_second=SR / 512)
    assert abs(period / (SR / 512) - 2.0) < 0.05


def test_classic_pipeline_cleans_up_the_mix():
    mix, vocal = make_mix()
    before = si_sdr(vocal, mix.mean(axis=1))
    after = si_sdr(vocal, extract_center(mix, SR)[:, 0])
    assert after > before + 5  # at least 5 dB cleaner
