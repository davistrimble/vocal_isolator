"""Simple, non-ML vocal extraction.

Most songs are mixed with the lead vocal panned dead center, meaning it's
identical in the left and right channels. Instruments are usually spread out
across the stereo field. So:

    mid  = (L + R) / 2   -> everything in the center (vocals + some other stuff)
    side = (L - R) / 2   -> only stuff that's different between L and R (no vocals)

If we look at the spectrum (STFT) of each, we can subtract the side magnitude
from the mid magnitude, and what's left is mostly the centered vocal.
"""
import numpy as np
from scipy.signal import stft, istft

N_FFT = 2048


def extract_center(samples, sr):
    if samples.shape[1] < 2:
        raise ValueError("center extraction needs a stereo file")

    left = samples[:, 0]
    right = samples[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2

    _, _, mid_spec = stft(mid, sr, nperseg=N_FFT)
    _, _, side_spec = stft(side, sr, nperseg=N_FFT)

    # spectral subtraction: remove the side energy from the mid, never go below 0
    magnitude = np.maximum(np.abs(mid_spec) - np.abs(side_spec), 0)
    phase = np.angle(mid_spec)
    vocals_spec = magnitude * np.exp(1j * phase)

    _, vocals = istft(vocals_spec, sr, nperseg=N_FFT)
    vocals = vocals[: len(mid)]

    # back to stereo so it plays normally
    return np.stack([vocals, vocals], axis=1).astype(np.float32)
