"""A "presence boost" - a standard mixing trick for making vocals easier to
understand.

The consonants that make words intelligible (t, k, s, ch...) and the "edge"
of a voice live around 2-5 kHz. Turning that range up a few dB makes the
words cut through whatever background is left, without removing anything.

How: take the FFT of the whole song (every frequency in it), multiply each
frequency by a gain from a bell-shaped curve centered at 3 kHz, then inverse
FFT back to audio.
"""
import numpy as np

CENTER_HZ = 3000
BOOST_DB = 4.0
# width of the bell in octaves (1 octave = doubling of frequency). at 1.5,
# the boost is about half strength at 1.8 kHz and 5 kHz
WIDTH_OCTAVES = 1.5


def boost_curve(freqs):
    """Gain (as a multiplier, not dB) for each frequency."""
    # distance from the center in octaves, so the bell is symmetric on a
    # musical (log) scale instead of in raw Hz
    octaves_away = np.log2(np.maximum(freqs, 1) / CENTER_HZ)
    boost_db = BOOST_DB * np.exp(-0.5 * (octaves_away / (WIDTH_OCTAVES / 2.355)) ** 2)
    return 10 ** (boost_db / 20)  # dB -> multiplier


def presence_boost(samples, sr):
    """samples is (num_samples, num_channels). Returns the boosted audio."""
    spectrum = np.fft.rfft(samples, axis=0)
    freqs = np.fft.rfftfreq(len(samples), d=1 / sr)
    spectrum *= boost_curve(freqs)[:, None]
    return np.fft.irfft(spectrum, n=len(samples), axis=0).astype(np.float32)
