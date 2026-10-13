import numpy as np
import soundfile as sf


def load_audio(path):
    """Load an audio file. Returns (samples, sample_rate).

    samples has shape (num_samples, num_channels) and is float32 in [-1, 1].
    """
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    return samples, sr


def normalize(samples, peak=0.95):
    """Scale so the loudest sample hits `peak` (just under the max of 1.0).

    Separation throws away a lot of the song's energy, so the vocals come out
    much quieter than the original. This turns them back up.
    """
    loudest = np.abs(samples).max()
    if loudest == 0:
        return samples  # silent file, nothing to scale
    return samples * (peak / loudest)


def trim_silence(samples, sr, threshold_db=-50, keep_s=0.25):
    """Cut off silence at the start and end so playback starts right at the
    first vocal line instead of making you wait through the intro.

    "Silent" means quieter than threshold_db relative to the loudest part.
    keep_s seconds are left on each side so the first word isn't clipped.
    Returns (trimmed_samples, seconds_cut_from_start).
    """
    # loudness of each 20ms chunk
    chunk = int(0.02 * sr)
    num_chunks = len(samples) // chunk
    if num_chunks == 0:
        return samples, 0.0
    chunks = samples[: num_chunks * chunk].reshape(num_chunks, chunk, -1)
    loudness = np.sqrt((chunks ** 2).mean(axis=(1, 2)))

    threshold = loudness.max() * 10 ** (threshold_db / 20)
    loud_chunks = np.nonzero(loudness > threshold)[0]
    if len(loud_chunks) == 0:
        return samples, 0.0  # whole thing is silent, leave it alone

    start = max(0, loud_chunks[0] * chunk - int(keep_s * sr))
    end = min(len(samples), (loud_chunks[-1] + 1) * chunk + int(keep_s * sr))
    return samples[start:end], start / sr


def save_audio(path, samples, sr):
    # clip so we don't get nasty distortion if something went above 1.0
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(path, samples, sr)
