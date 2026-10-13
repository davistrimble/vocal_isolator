import argparse
import os

from vocal_isolator.analysis import NOTE_NAMES, detect_beats, detect_key
from vocal_isolator.audio_io import load_audio, normalize, save_audio, trim_silence
from vocal_isolator.center_extract import extract_center
from vocal_isolator.eq import presence_boost
from vocal_isolator.ml_separate import extract_vocals_ml
from vocal_isolator.lrclib import find_synced_lyrics
from vocal_isolator.lyrics import parse_lrc, sung_sections, vocal_gate
from vocal_isolator.metronome import add_clicks
from vocal_isolator.transform import change_key, change_tempo, semitones_between


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

    if args.method == "ml":
        print("separating with Demucs (takes a minute or two)...")
        vocals = extract_vocals_ml(samples, sr)
    else:
        vocals = extract_center(samples, sr)
        if not args.no_boost:
            vocals = presence_boost(vocals, sr)

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

    if lrc_text:
        lines = parse_lrc(lrc_text)
        sections = sung_sections(lines, len(vocals) / sr)
        print(f"found {len(sections)} sung lines in the lyrics")
        gate = vocal_gate(sections, len(vocals), sr)
        vocals = vocals * gate[:, None]

    # tempo and beats come from the original mix, not the separated vocals,
    # since the drums and bass are what make the beat easy to find
    beat_times = None
    original_tempo = args.original_tempo
    if args.metronome or (args.new_tempo and not original_tempo):
        detected_tempo, beat_times = detect_beats(samples, sr)
        print(f"detected tempo: {detected_tempo:.0f} BPM")
        original_tempo = original_tempo or detected_tempo

    if args.new_key:
        original_key = args.original_key
        if not original_key:
            original_key, mode = detect_key(samples, sr)
            print(f"detected key: {original_key} {mode}")
        shift = semitones_between(original_key, args.new_key)
        print(f"shifting {shift:+d} semitones ({original_key} -> {args.new_key})")
        vocals = change_key(vocals, sr, shift)

    if args.new_tempo:
        print(f"changing tempo: {original_tempo:.0f} -> {args.new_tempo:.0f} BPM")
        vocals = change_tempo(vocals, original_tempo, args.new_tempo)
        if beat_times is not None:
            # everything happens sooner/later by the same ratio
            beat_times = beat_times * original_tempo / args.new_tempo

    vocals = normalize(vocals)
    if args.metronome:
        vocals = add_clicks(vocals, sr, beat_times)
    else:
        vocals, cut = trim_silence(vocals, sr)
        if cut > 0:
            print(f"skipped {cut:.1f}s of silence at the start")

    save_audio(output, normalize(vocals), sr)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
