"""Demucs. Skipped by default since it downloads a ~90 MB model and is slow
without a GPU. Run with:  RUN_SLOW_TESTS=1 python -m pytest tests/test_ml.py
"""
import os

import numpy as np
import pytest

from conftest import SR, stereo, tone

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="slow, set RUN_SLOW_TESTS=1")


def test_demucs_returns_vocals_same_shape():
    from vocal_isolator.ml_separate import extract_vocals_ml

    audio = stereo(tone(440, 3.0)) * 0.3
    progress = []
    vocals = extract_vocals_ml(audio, SR, on_progress=progress.append)
    assert vocals.shape == audio.shape
    assert vocals.dtype == np.float32
    assert progress  # reported progress at least once


def test_demucs_handles_mono():
    from vocal_isolator.ml_separate import extract_vocals_ml

    mono = tone(440, 2.0)[:, None] * 0.3
    assert extract_vocals_ml(mono, SR).shape == (len(mono), 2)
