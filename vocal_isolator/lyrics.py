"""Use timestamped lyrics to silence the parts of the song where nobody sings.

Lyrics come in LRC format, where each line starts with the time it's sung:

    [00:22.24]Poor old Johnnie Ray
    [00:25.44]
    [00:28.57]Sounded sad upon the radio

An empty line (just a timestamp) means the singing stops there, like before an
instrumental break.

The separation never gets rid of everything, and leftover instruments are most
noticeable when there's no voice covering them up. If we know when the singing
happens, we can just mute everything else.
"""
import re

import numpy as np
from scipy.ndimage import uniform_filter1d

# matches a timestamp like [01:23.45]
TIMESTAMP = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")

# lines don't say when they end, so we assume a line lasts until the next one
# starts, but cap it at this many seconds in case there's a long gap
MAX_LINE_S = 10.0

# start the vocals a little before each line and end a little after, since
# LRC timings aren't exact and we don't want to cut off the start of words
PADDING_S = 0.4

# fade in/out over this long instead of cutting instantly (which clicks)
FADE_S = 0.15


def parse_lrc(text):
    """Returns a list of (time_in_seconds, words), sorted by time."""
    lines = []
    for raw_line in text.splitlines():
        stamps = TIMESTAMP.findall(raw_line)
        if not stamps:
            continue  # metadata like [ar:Artist] or a blank line
        words = TIMESTAMP.sub("", raw_line).strip()
        # a line can have more than one timestamp if it's repeated (like a chorus)
        for minutes, seconds in stamps:
            lines.append((int(minutes) * 60 + float(seconds), words))
    return sorted(lines)


def sung_sections(lines, song_length_s):
    """Turns parsed lyrics into a list of (start, end) times where someone is singing."""
    sections = []
    for i, (start, words) in enumerate(lines):
        if not words:
            continue  # empty line = singing stops here
        if i + 1 < len(lines):
            end = lines[i + 1][0]
        else:
            end = song_length_s
        end = min(end, start + MAX_LINE_S)
        sections.append((start, end))
    return sections


def vocal_gate(sections, num_samples, sr):
    """A volume curve for the whole song: 1 while singing, 0 otherwise,
    with short fades in between. Multiply the audio by this to apply it.
    """
    gate = np.zeros(num_samples, dtype=np.float32)
    for start, end in sections:
        first = max(0, int((start - PADDING_S) * sr))
        last = min(num_samples, int((end + PADDING_S) * sr))
        gate[first:last] = 1.0

    # a moving average turns each instant on/off jump into a smooth ramp
    return uniform_filter1d(gate, size=int(FADE_S * sr))
