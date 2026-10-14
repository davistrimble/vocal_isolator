"""REPET: REpeating Pattern Extraction Technique (Rafii & Pardo, 2012).

The idea: the backing music in a song repeats (same chords/beat every bar),
but the vocal melody keeps changing. So:

  1. Find how long the repeating pattern is (the "period"), by checking how
     similar the spectrogram is to a time-shifted copy of itself.
  2. Chop the spectrogram into period-length pieces and take the median of
     all the pieces. Stuff that's in every piece (the backing music) survives
     the median, stuff that only shows up once (the vocal) gets thrown out.
     That gives us a model of what one repetition of the backing sounds like.
  3. Anywhere the actual spectrogram matches that model, it's background.
     Anything left over is the vocal.
"""
import numpy as np

# only look for repeating periods in this range (seconds). a bar of music is
# usually somewhere around 1-4 seconds
MIN_PERIOD_S = 1.0
MAX_PERIOD_S = 8.0

# songs change tempo / switch sections, so instead of finding one period for
# the whole song we run REPET separately on windows of this many seconds
WINDOW_S = 20.0


def beat_spectrum(magnitude):
    """How similar the spectrogram is to itself shifted by each lag (in frames).

    This is the autocorrelation of each frequency row, averaged together.
    A peak at lag L means the music tends to repeat every L frames.
    """
    power = magnitude ** 2
    n_frames = power.shape[1]

    # autocorrelation via FFT (way faster than doing it directly). pad to 2x
    # length so it doesn't wrap around
    spectrum = np.fft.rfft(power, n=2 * n_frames, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :n_frames]

    # normalize each lag by how many frames overlapped at that lag
    autocorr = autocorr / np.arange(n_frames, 0, -1)
    return autocorr.mean(axis=0)


def find_period(magnitude, frames_per_second):
    beats = beat_spectrum(magnitude)
    lo = int(MIN_PERIOD_S * frames_per_second)
    hi = min(int(MAX_PERIOD_S * frames_per_second), len(beats) // 3)
    if hi <= lo:
        return None  # clip too short to see any repetition
    best = lo + int(np.argmax(beats[lo:hi]))

    # if something repeats every 2s it also repeats every 4s, 6s, etc, so the
    # best peak is sometimes a multiple of the real period. check if half,
    # a third, or a quarter of it fits almost as well, and prefer the shortest
    for divisor in (4, 3, 2):
        candidate = best // divisor
        if candidate < lo:
            continue
        # look a couple frames either side since it won't divide perfectly
        nearby = beats[candidate - 2:candidate + 3]
        if nearby.max() >= 0.8 * beats[best]:
            return candidate - 2 + int(np.argmax(nearby))
    return best


def background_mask(magnitude, frames_per_second):
    """Returns a soft mask (values 0 to 1) of how much of each point is
    repeating background. Same shape as magnitude.
    """
    window = int(WINDOW_S * frames_per_second)
    mask = np.zeros_like(magnitude)
    for start in range(0, magnitude.shape[1], window):
        chunk = magnitude[:, start:start + window]
        mask[:, start:start + window] = _mask_for_chunk(chunk, frames_per_second)
    return mask


def _mask_for_chunk(magnitude, frames_per_second):
    period = find_period(magnitude, frames_per_second)
    if period is None:
        return np.zeros_like(magnitude)

    n_freqs, n_frames = magnitude.shape
    n_segments = int(np.ceil(n_frames / period))

    # pad with nan so the last partial segment doesn't drag the median down
    padded = np.full((n_freqs, n_segments * period), np.nan, dtype=magnitude.dtype)
    padded[:, :n_frames] = magnitude
    segments = padded.reshape(n_freqs, n_segments, period)

    # what one repetition of the background looks like
    repeating_segment = np.nanmedian(segments, axis=1)

    # tile it back out to the full length of the song
    model = np.tile(repeating_segment, n_segments)[:, :n_frames]

    # the background can't be louder than what's actually there
    model = np.minimum(model, magnitude)
    return model / (magnitude + 1e-10)
