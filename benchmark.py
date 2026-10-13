"""Score both separation methods against real isolated vocals.

Uses the MUSDB18 sample dataset: 7 second clips from 50 songs where every
instrument and the vocals were recorded separately, so we know exactly what
the "right answer" sounds like. Score is SDR (signal-to-distortion ratio, in
dB) from museval, the standard tool for this benchmark - higher is better.

    pip install -r requirements-dev.txt
    python benchmark.py

The dataset (~150 MB) downloads into data/ the first time.
"""
import numpy as np
import musdb
import museval

from vocal_isolator.center_extract import extract_center
from vocal_isolator.ml_separate import extract_vocals_ml


def sdr(reference, estimate, sr):
    """Median SDR over 1 second windows, which is how MUSDB results are reported."""
    scores, _, _, _ = museval.evaluate(reference[None], estimate[None], win=sr, hop=sr)
    return np.nanmedian(scores)


def main():
    db = musdb.DB(root="data/musdb18", download=True, subsets="test")
    methods = {
        "do nothing (full mix)": lambda mix, sr: mix,
        "classic (my DSP pipeline)": extract_center,
        "ml (Demucs)": extract_vocals_ml,
    }
    results = {name: [] for name in methods}

    for i, track in enumerate(db):
        mix = track.audio.astype(np.float32)
        truth = track.targets["vocals"].audio
        if np.abs(truth).max() < 1e-3:
            continue  # no singing in this clip, nothing to score
        for name, separate in methods.items():
            results[name].append(sdr(truth, separate(mix, track.rate), track.rate))
        print(f"[{i + 1}/{len(db)}] {track.name}")

    print(f"\nvocal SDR on {len(results['ml (Demucs)'])} MUSDB18 test clips (higher is better):")
    for name, scores in results.items():
        print(f"  {name:28s} median {np.nanmedian(scores):5.1f} dB")


if __name__ == "__main__":
    main()
