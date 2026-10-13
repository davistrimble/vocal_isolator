# vocal_isolator

Take a song in, get just the vocals out.

Work in progress.

## Usage

```
python -m vocal_isolator song.mp3
```

Writes `<song>_vocals.wav` next to the input. Takes a few seconds for a typical song.

## How it works

No machine learning — just classical signal processing on the song's spectrogram
(a picture of how loud each frequency is at each moment). Each step removes a different
kind of non-vocal sound:

1. **Center channel extraction** — lead vocals are almost always mixed dead center, so
   they're identical in the left and right channels. Subtracting the side channel (L - R)
   from the mid channel (L + R) removes instruments panned to the sides.
2. **Band limiting** — cut everything below 100 Hz and above 8 kHz, outside the vocal range.
   Gets rid of bass and kick drum, which are also usually panned center.
3. **Harmonic/percussive separation** — drums look like vertical lines in a spectrogram, singing
   looks like horizontal lines. Median filtering in each direction tells them apart, and the
   percussive parts get dropped.
4. **REPET** — the backing music repeats every bar while the vocal melody doesn't. Find the
   repeat length, take the median across all repetitions to model the background, and subtract it.
   Runs on 20 second windows since songs change tempo.

## Limitations

- Needs a stereo file (step 1 doesn't work on mono).
- Anything else panned center, pitched, and non-repeating (a lead guitar solo, a sax) gets kept.
- Long held vocal notes can get partly removed by REPET since they look like they "repeat".
- Reverb on the vocal usually gets removed along with the background, so the result sounds dry.

## References

- D. Fitzgerald, "Harmonic/Percussive Separation using Median Filtering," DAFx 2010.
- Z. Rafii and B. Pardo, "REpeating Pattern Extraction Technique (REPET): A Simple Method for
  Music/Voice Separation," IEEE TASLP 2013.
