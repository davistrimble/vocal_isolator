# vocal_isolator

Take a song in, get just the vocals out.

Work in progress.

## Usage

```
python -m vocal_isolator song.mp3
```

Currently uses a simple center-channel extraction (no ML): vocals are usually panned to the
center of a stereo mix, so we take the mid channel and spectrally subtract the side channel.
Frequencies outside the vocal range (~100 Hz - 8 kHz) are cut, which gets rid of most of the
bass and kick. Only works on stereo files, and anything else panned center in the vocal range
(snare, some synths) still leaks through.
