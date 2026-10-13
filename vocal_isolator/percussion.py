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

Then each point in the spectrogram is kept if it looks more harmonic than
percussive.
"""
import numpy as np
from scipy.ndimage import median_filter

# how many neighbors the median filters look at. bigger = stronger separation
# but smears things more
KERNEL_SIZE = 17


def harmonic_mask(magnitude):
    """Returns a 0/1 array the same shape as magnitude (freq x time).

    1 where the spectrogram is more harmonic (voice-like), 0 where it's
    more percussive (drum-like).
    """
    harmonic = median_filter(magnitude, size=(1, KERNEL_SIZE))
    percussive = median_filter(magnitude, size=(KERNEL_SIZE, 1))
    return (harmonic >= percussive).astype(magnitude.dtype)
