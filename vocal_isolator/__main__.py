import argparse
import os

from vocal_isolator.audio_io import load_audio, save_audio


def main():
    parser = argparse.ArgumentParser(description="Isolate the vocals from a song.")
    parser.add_argument("input", help="path to the song (wav, mp3, flac)")
    parser.add_argument("-o", "--output", help="where to save the vocals (default: <name>_vocals.wav)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        name = os.path.splitext(os.path.basename(args.input))[0]
        output = name + "_vocals.wav"

    samples, sr = load_audio(args.input)
    print(f"loaded {args.input}: {samples.shape[0] / sr:.1f}s, {sr} Hz, {samples.shape[1]} channel(s)")

    # TODO: actually separate the vocals. just passing audio through for now
    save_audio(output, samples, sr)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
