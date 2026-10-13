# vocal_isolator

Take a song in, get just the vocals out.

Work in progress.

## Usage

```
python -m vocal_isolator song.mp3
```

Currently uses a simple center-channel extraction (no ML): vocals are usually panned to the
center of a stereo mix, so we take the mid channel and spectrally subtract the side channel.
Only works on stereo files, and anything else panned center (bass, kick, snare) leaks through.
