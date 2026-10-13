"""Vocal separation with Demucs, a neural network from Meta AI Research.

The classical pipeline (center_extract.py) hit a wall: an instrument that's
panned center and playing in the same key as the singer looks just like the
voice to every rule-based method I tried. Telling them apart means knowing
what a voice actually sounds like vs a guitar, which is what a neural
network learns from examples.

Demucs was trained on songs where every instrument and the vocals were
recorded separately, so it could learn to map "full mix" -> "just the vocals".
We use the pretrained model "htdemucs" (Hybrid Transformer Demucs), which
looks at both the raw waveform and the spectrogram.

The model splits a song into 4 stems: drums, bass, other, vocals. We keep
the vocals.
"""
import numpy as np
import torch
from demucs.api import Separator

MODEL_NAME = "htdemucs"

_separator = None  # loading the model takes a few seconds, so only do it once


def _device():
    # use the Mac's GPU (Apple "Metal Performance Shaders") if there is one,
    # it's about 3x faster than the CPU
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_separator():
    global _separator
    if _separator is None:
        _separator = Separator(model=MODEL_NAME, device=_device())
    return _separator


def extract_vocals_ml(samples, sr, on_progress=None):
    """samples is (num_samples, num_channels). Returns vocals in the same shape.

    on_progress, if given, gets called with the percent done (0-100) as the
    model works through the song.
    """
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)  # the model expects stereo

    separator = get_separator()

    # demucs processes the song in overlapping chunks and can call a function
    # each time a chunk finishes, telling us how far through the song it is
    def callback(info):
        if on_progress and info["state"] == "end":
            on_progress(100 * info["segment_offset"] / info["audio_length"])

    separator.update_parameter(callback=callback)
    # torch wants (channels, samples), we store (samples, channels)
    wav = torch.from_numpy(np.ascontiguousarray(samples.T))
    _, stems = separator.separate_tensor(wav, sr)

    vocals = stems["vocals"].cpu().numpy().T
    if separator.samplerate != sr:
        raise ValueError(f"model returned {separator.samplerate} Hz audio, expected {sr} Hz")
    return vocals.astype(np.float32)
