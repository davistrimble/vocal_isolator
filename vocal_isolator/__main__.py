import argparse
import os

from vocal_isolator.analysis import NOTE_NAMES
from vocal_isolator.audio_io import load_audio, save_audio
from vocal_isolator.lrclib import find_synced_lyrics
from vocal_isolator.pipeline import process


def main():
    parser = argparse.ArgumentParser(description="Isolate the vocals from a song.")
    parser.add_argument("input", help="path to the song (wav, mp3, flac)")
    parser.add_argument("-o", "--output", help="where to save the vocals (default: <name>_vocals.wav)")
    parser.add_argument(
        "--method",
        choices=["ml", "classic"],
        default="ml",
        help="ml = Demucs neural network (default, much cleaner). classic = my original signal processing pipeline",
    )
    parser.add_argument("--lyrics", help="timestamped lyrics (.lrc file), used to mute parts with no singing")
    parser.add_argument("--artist", help="with --title, look up timestamped lyrics online instead of using --lyrics")
    parser.add_argument("--title", help="song title, for looking up lyrics")
    parser.add_argument("--no-boost", action="store_true", help="skip the 2-5 kHz presence boost (classic method only)")
    parser.add_argument("--new-key", choices=NOTE_NAMES, help="shift the vocals into this key")
    parser.add_argument("--original-key", choices=NOTE_NAMES, help="the song's key (detected automatically if left out)")
    parser.add_argument("--new-tempo", type=float, help="change the tempo to this many BPM, without changing the pitch")
    parser.add_argument("--original-tempo", type=float, help="the song's tempo in BPM (detected automatically if left out)")
    parser.add_argument(
        "--metronome",
        action="store_true",
        help="add a click on every beat and keep the intro, so there's a count-in before the vocals",
    )
    args = parser.parse_args()

    if bool(args.artist) != bool(args.title):
        parser.error("--artist and --title need to be used together")

    output = args.output
    if output is None:
        name = os.path.splitext(os.path.basename(args.input))[0]
        output = name + "_vocals.wav"

    samples, sr = load_audio(args.input)
    print(f"loaded {args.input}: {samples.shape[0] / sr:.1f}s, {sr} Hz, {samples.shape[1]} channel(s)")

    lrc_text = None
    if args.lyrics:
        with open(args.lyrics, encoding="utf-8") as f:
            lrc_text = f.read()
    elif args.artist:
        try:
            lrc_text = find_synced_lyrics(args.artist, args.title, len(samples) / sr)
        except LookupError as e:
            print(f"couldn't use online lyrics: {e}")
            print("continuing without lyrics")

    vocals = process(
        samples,
        sr,
        method=args.method,
        new_key=args.new_key,
        original_key=args.original_key,
        new_tempo=args.new_tempo,
        original_tempo=args.original_tempo,
        metronome=args.metronome,
        lrc_text=lrc_text,
        boost=not args.no_boost,
    )

    save_audio(output, vocals, sr)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
