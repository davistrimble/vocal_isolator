import argparse
import os

from vocal_isolator.audio_io import load_audio, save_audio
from vocal_isolator.center_extract import extract_center
from vocal_isolator.lyrics import parse_lrc, sung_sections, vocal_gate


def main():
    parser = argparse.ArgumentParser(description="Isolate the vocals from a song.")
    parser.add_argument("input", help="path to the song (wav, mp3, flac)")
    parser.add_argument("-o", "--output", help="where to save the vocals (default: <name>_vocals.wav)")
    parser.add_argument("--lyrics", help="timestamped lyrics (.lrc file), used to mute parts with no singing")
    args = parser.parse_args()

    output = args.output
    if output is None:
        name = os.path.splitext(os.path.basename(args.input))[0]
        output = name + "_vocals.wav"

    samples, sr = load_audio(args.input)
    print(f"loaded {args.input}: {samples.shape[0] / sr:.1f}s, {sr} Hz, {samples.shape[1]} channel(s)")

    vocals = extract_center(samples, sr)

    if args.lyrics:
        with open(args.lyrics, encoding="utf-8") as f:
            lines = parse_lrc(f.read())
        sections = sung_sections(lines, len(vocals) / sr)
        print(f"found {len(sections)} sung lines in the lyrics")
        gate = vocal_gate(sections, len(vocals), sr)
        vocals = vocals * gate[:, None]

    save_audio(output, vocals, sr)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
