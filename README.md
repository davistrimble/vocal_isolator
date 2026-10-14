# vocal_isolator

[![tests](https://github.com/davistrimble/vocal_isolator/actions/workflows/tests.yml/badge.svg)](https://github.com/davistrimble/vocal_isolator/actions/workflows/tests.yml)

Take a song in, get just the vocals out.

## Why I built this

I sing in an a cappella group, and every so often the soloist for a song can't make it to
practice. Playing the original recording doesn't really work as a stand-in, since the instruments
cover up the parts the rest of us are supposed to be singing. What we actually want is just the
lead vocal, so we can sing our parts around it like the soloist is in the room.

There are free vocal removers online, but none of the ones I found let you change the tempo or the
key, and for rehearsal I needed both. Our arrangements aren't always in the same key as the
original recording, and when we're learning a song it helps to slow it down. So on top of
isolating the vocals, this tool can transpose them to a new key, change the tempo without changing
the pitch, and add a metronome. It also trims the output to start right at the first sung line so
it's ready to play at rehearsal.

## Results

I built this twice. First with classical signal processing and no machine learning, to see how far
that could get. It worked, but not well enough for rehearsal. Instruments playing in the same key
as the singer kept bleeding through. So the second version uses a neural network.

Scored on 50 songs from the MUSDB18 benchmark, where the real isolated vocals are available to
compare against (SDR = signal-to-distortion ratio, higher is better):

| Method | Vocal SDR |
|---|---|
| Do nothing (full mix) | -4.6 dB |
| Attempt 1: classical DSP | 1.2 dB |
| Attempt 2: Demucs neural network | **8.8 dB** |

Every 3 dB is roughly a halving of the leftover background. Reproduce with `python benchmark.py`.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests generate their own audio (tones, chord progressions, made-up mixes) instead of using real
songs. They run on every push with GitHub Actions. The Demucs tests are skipped by default since
they download the model; run them with `RUN_SLOW_TESTS=1`.

## Usage

```
brew install rubberband        # used for key/tempo changes (Linux: apt install rubberband-cli)
pip install -r requirements.txt
python -m vocal_isolator song.mp3
```

Writes `<song>_vocals.wav` next to the input. The first run downloads the model (~90 MB). A
4-minute song takes about a minute and a half on an Apple Silicon Mac.

### Website

```
python -m vocal_isolator.web
```

Then open http://localhost:8000. Pick an MP3 and it detects the tempo and key right away and fills
them in; check "Change the key" or "Change the tempo" to adjust them, or "Add a metronome" for a
click track with a count-in. The page checks on progress every 5 seconds while the song processes,
then lets you play or download the result.

It's a small [Flask](https://flask.palletsprojects.com/) app (`vocal_isolator/web.py`) with a plain
HTML/JavaScript page. Processing runs in a background thread so the browser isn't stuck waiting on
one long request. The page polls a `/status` endpoint instead. All the audio work goes through
the same `pipeline.process()` function as the command line.

### Rehearsal options

```
python -m vocal_isolator song.mp3 --new-key A          # transpose into A
python -m vocal_isolator song.mp3 --new-tempo 90       # slow it down to 90 BPM, same pitch
python -m vocal_isolator song.mp3 --metronome          # click on every beat, keep the intro as a count-in
```

The song's original key and tempo are detected automatically. If the detection is off (tempo
detection sometimes picks double or half the tempo you'd count), pass `--original-key` /
`--original-tempo` to override it.

- **Tempo and beats** are found with librosa's beat tracker, which looks for the steady pulse that
  lines up with the most note onsets (drum hits, strummed chords) in the original mix.
- **Key** uses the Krumhansl-Schmuckler algorithm: measure how much each of the 12 notes is used in
  the song, and compare that against profiles of how much each note is used in every major and
  minor key.
- **Key and tempo changes** use a phase vocoder, which can stretch audio in time without changing
  its pitch (and vice versa), unlike just speeding up or slowing down the recording. I started with
  librosa's, but it made the vocals sound smeared and "phasey", so it now uses
  [Rubber Band](https://breakfastquay.com/rubberband/) with formant preservation, which keeps the
  singer sounding like themselves in the new key instead of chipmunky or giant-like.
- **Metronome.** My first version was hard to use: the clicks wobbled, and they were quieter than
  the vocals so you lost them whenever someone was singing. Two fixes. The beat tracker now runs
  with a high "tightness" setting so it holds one steady tempo instead of slipping out of step at
  section changes, and each click time is averaged with the 8 beats around it to smooth out the
  tracker's ~12 ms timing steps. The vocals are also turned down under a louder 2.5 kHz click, so
  the click sits about 14 dB above the singing. One tradeoff: since the clicks hold a steady tempo,
  they won't follow a song that deliberately speeds up or slows down partway through.

### Classical method

To use the classical pipeline instead:

```
python -m vocal_isolator song.mp3 --method classic
```

The classical pipeline can also use timestamped lyrics to mute the parts of the song where nobody's
singing (this isn't needed with Demucs, whose output is already silent there):

```
python -m vocal_isolator song.mp3 --method classic --lyrics song.lrc
python -m vocal_isolator song.mp3 --method classic --artist "Blues Traveler" --title "Run-Around"
```

The second one looks the lyrics up on [LRCLIB](https://lrclib.net). It only uses them if it finds a
version of the song the same length as your file, since a radio edit and an album version have
different timings.

## Attempt 1: classical signal processing (`--method classic`)

This version works on the song's spectrogram, a picture of how loud each frequency is at each
moment. Each step removes a different kind of non-vocal sound:

1. **Center channel extraction.** Lead vocals are almost always mixed dead center, so
   they're identical in the left and right channels. Subtracting the side channel (L - R)
   from the mid channel (L + R) removes instruments panned to the sides.
2. **Band limiting.** Cut everything below 80 Hz and above 16 kHz. Gets rid of bass and kick
   drum, which are also usually panned center. (This was originally 100 Hz - 8 kHz, but cutting
   the highs made vocals sound like they were coming out of a phone speaker.)
3. **Harmonic/percussive separation.** Drums look like vertical lines in a spectrogram, singing
   looks like horizontal lines. Median filtering in each direction tells them apart, and the
   percussive parts get faded out with a soft mask.
4. **REPET.** The backing music repeats every bar while the vocal melody doesn't. Find the
   repeat length, take the median across all repetitions to model the background, and subtract it.
   Runs on 20 second windows since songs change tempo.
5. **Presence boost.** Turn up 2-5 kHz by a few dB, where consonants live. This makes the words
   easier to follow over whatever background is left. Skip with `--no-boost`.
6. **Lyric gating** (optional). The steps above never remove everything, and leftovers are most
   obvious when no one is singing over them. LRC lyrics have a timestamp for each line, so anything
   outside a sung line gets faded out.

### Limitations

- Needs a stereo file (step 1 doesn't work on mono).
- Anything else panned center, pitched, and non-repeating (a lead guitar solo, a sax) gets kept.
- Long held vocal notes can get partly removed by REPET since they look like they "repeat".
- Lyric gating only knows when each *line* starts, not each word, so gaps inside a line aren't muted.
- Reverb on the vocal usually gets removed along with the background, so the result sounds dry.

## Attempt 2: Demucs (`--method ml`, the default)

[Demucs](https://github.com/facebookresearch/demucs) is a neural network from Meta AI Research,
trained on songs where every instrument and the vocals were recorded separately. That lets it
learn the mapping from "full mix" to "just the vocals" directly. I use the pretrained `htdemucs`
model (Hybrid Transformer Demucs), which looks at both the raw waveform and the spectrogram, and
splits a song into drums, bass, other, and vocals. This tool keeps the vocals.

After separation, both methods share the same finishing steps: the volume is normalized (separated
vocals come out much quieter than the original song) and silence is trimmed off the start and end.

## References

- D. Fitzgerald, "Harmonic/Percussive Separation using Median Filtering," DAFx 2010.
- Z. Rafii and B. Pardo, "REpeating Pattern Extraction Technique (REPET): A Simple Method for
  Music/Voice Separation," IEEE TASLP 2013.
- S. Rouard, F. Massa, and A. Défossez, "Hybrid Transformers for Music Source Separation,"
  ICASSP 2023.
- Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, and R. Bittner, "The MUSDB18 corpus for
  music separation," 2017.
