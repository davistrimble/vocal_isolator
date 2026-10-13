import numpy as np
import soundfile as sf


def load_audio(path):
    """Load an audio file. Returns (samples, sample_rate).

    samples has shape (num_samples, num_channels) and is float32 in [-1, 1].
    """
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    return samples, sr


def save_audio(path, samples, sr):
    # clip so we don't get nasty distortion if something went above 1.0
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(path, samples, sr)
