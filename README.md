# vocal_isolator

Take a song in, get just the vocals out.

Work in progress.

## Usage

```
python -m vocal_isolator song.mp3
```

Writes `<song>_vocals.wav` next to the input.

Optionally, give it timestamped lyrics so it can mute the parts of the song where nobody's singing:

```
python -m vocal_isolator song.mp3 --lyrics song.lrc
python -m vocal_isolator song.mp3 --artist "Dexys Midnight Runners" --title "Come On Eileen"
```

The second one looks the lyrics up on [LRCLIB](https://lrclib.net). It only uses them if it finds a
version of the song the same length as your file, since a radio edit and an album version have
different timings. Takes a few seconds for a typical song.

## How it works

No machine learning — just classical signal processing on the song's spectrogram
(a picture of how loud each frequency is at each moment). Each step removes a different
kind of non-vocal sound:

1. **Center channel extraction** — lead vocals are almost always mixed dead center, so
   they're identical in the left and right channels. Subtracting the side channel (L - R)
   from the mid channel (L + R) removes instruments panned to the sides.
2. **Band limiting** — cut everything below 80 Hz and above 16 kHz. Gets rid of bass and kick
   drum, which are also usually panned center. (This was originally 100 Hz - 8 kHz, but cutting
   the highs made vocals sound like they were coming out of a phone speaker.)
3. **Harmonic/percussive separation** — drums look like vertical lines in a spectrogram, singing
   looks like horizontal lines. Median filtering in each direction tells them apart, and the
   percussive parts get faded out with a soft mask.
4. **REPET** — the backing music repeats every bar while the vocal melody doesn't. Find the
   repeat length, take the median across all repetitions to model the background, and subtract it.
   Runs on 20 second windows since songs change tempo.
5. **Presence boost** — turn up 2-5 kHz by a few dB, where consonants live. Doesn't remove anything,
   but makes the words easier to follow over whatever background is left. Skip with `--no-boost`.
6. **Lyric gating** (optional) — the steps above never remove everything, and leftovers are most
   obvious when no one is singing over them. LRC lyrics have a timestamp for each line, so anything
   outside a sung line gets faded out.

## Limitations

- Needs a stereo file (step 1 doesn't work on mono).
- Anything else panned center, pitched, and non-repeating (a lead guitar solo, a sax) gets kept.
- Long held vocal notes can get partly removed by REPET since they look like they "repeat".
- Lyric gating only knows when each *line* starts, not each word, so gaps inside a line aren't muted.
- Reverb on the vocal usually gets removed along with the background, so the result sounds dry.

## Things I tried that didn't work

- **Melody tracking / harmonic comb filter** — find the note being sung in each frame and keep only
  its harmonics. Pitch tracking was accurate on test signals, but on real songs the guitars and keys
  play in the same key as the singer, so their harmonics land in the same places. It only removed
  ~0.5 dB and locked onto instruments almost as often as the voice, so I took it out.
- **ADRess** (Barry et al., 2004) instead of mid/side subtraction — estimates where each frequency
  bin is panned and keeps only the center. Should've helped with instruments panned slightly off
  center, but when a bin has both the voice and an off-center instrument in it, its pan position
  lands in between and the whole bin (voice included) gets thrown out. Scored worse than plain
  subtraction on every test mix at every width I tried (best: 9.1 dB vs 9.4 dB).

## References

- D. Fitzgerald, "Harmonic/Percussive Separation using Median Filtering," DAFx 2010.
- Z. Rafii and B. Pardo, "REpeating Pattern Extraction Technique (REPET): A Simple Method for
  Music/Voice Separation," IEEE TASLP 2013.
