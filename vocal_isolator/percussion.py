"""Remove drums using harmonic/percussive separation (Fitzgerald, 2010).

In a spectrogram (frequency on the y axis, time on the x axis):
  - sustained, pitched sounds like a voice show up as HORIZONTAL lines
    (one frequency held for a while)
  - drum hits show up as VERTICAL lines
    (a short burst of energy across lots of frequencies at once)

A median filter replaces each point with the median of its neighbors, which
wipes out anything that doesn't match the direction you filter in:
  - filtering along time keeps horizontal lines   -> "harmonic" estimate
  - filtering along frequency keeps vertical lines -> "percussive" estimate

Then each point in the spectrogram is kept in proportion to how harmonic it
looks compared to how percussive it looks.
"""
import numpy as np
from scipy.ndimage import median_filter

# how many neighbors the median filters look at. bigger = stronger separation
# but smears things more. time is in STFT frames, freq is in frequency bins
TIME_KERNEL = 33
FREQ_KERNEL = 17


def harmonic_mask(magnitude):
    """Returns an array of values from 0 to 1, same shape as magnitude (freq x time).

    Close to 1 where the spectrogram is harmonic (voice-like), close to 0 where
    it's percussive (drum-like), and in between when it's a mix.

    This used to be a hard 0/1 cutoff, but flipping bins fully on and off
    made the output sound warbly/metallic. A soft mask (this is called a
    Wiener-style mask) fades between them instead.
    """
    harmonic = median_filter(magnitude, size=(1, TIME_KERNEL))
    percussive = median_filter(magnitude, size=(FREQ_KERNEL, 1))
    return harmonic ** 2 / (harmonic ** 2 + percussive ** 2 + 1e-10)
