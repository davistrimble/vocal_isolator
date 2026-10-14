import numpy as np
import pytest
import soundfile as sf

from conftest import SR, stereo, tone
from vocal_isolator.audio_io import NotAnMp3Error, check_is_mp3, load_audio, normalize, trim_silence


def test_loads_mp3(mp3_file):
    samples, sr = load_audio(mp3_file)
    assert sr == SR
    assert samples.shape[1] == 2
    assert samples.dtype == np.float32


def test_rejects_wav(tmp_path):
    path = tmp_path / "song.wav"
    sf.write(path, stereo(tone(440, 1.0)), SR)
    with pytest.raises(NotAnMp3Error, match="isn't an .mp3"):
        check_is_mp3(str(path))


def test_rejects_wav_renamed_to_mp3(tmp_path):
    path = tmp_path / "sneaky.mp3"
    sf.write(path, stereo(tone(440, 1.0)), SR, format="WAV")
    with pytest.raises(NotAnMp3Error, match="actually a WAV"):
        check_is_mp3(str(path))


def test_rejects_non_audio(tmp_path):
    path = tmp_path / "notes.mp3"
    path.write_text("not a song")
    with pytest.raises(NotAnMp3Error, match="couldn't be read"):
        check_is_mp3(str(path))


def test_normalize_sets_peak():
    quiet = stereo(tone(440, 1.0)) * 0.1
    assert np.abs(normalize(quiet)).max() == pytest.approx(0.95, abs=1e-4)


def test_normalize_leaves_silence_alone():
    silence = np.zeros((1000, 2), dtype=np.float32)
    assert np.all(normalize(silence) == 0)


def test_trim_silence_cuts_both_ends():
    # 3s silence, 2s tone, 4s silence
    mono = np.concatenate([np.zeros(3 * SR), tone(440, 2.0), np.zeros(4 * SR)]).astype(np.float32)
    trimmed, cut = trim_silence(stereo(mono), SR)
    assert cut == pytest.approx(3 - 0.25, abs=0.03)  # keeps a quarter second before the sound
    assert len(trimmed) / SR == pytest.approx(2 + 0.5, abs=0.05)
