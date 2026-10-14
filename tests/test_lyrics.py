import numpy as np
import pytest

from vocal_isolator import lrclib
from vocal_isolator.lyrics import parse_lrc, sung_sections, vocal_gate

LRC = """[ar:Some Artist]
[ti:Some Song]
[00:05.00]first line
[00:08.50]second line
[00:12.00]
[00:30.00][01:00.00]chorus line
"""


def test_parse_lrc():
    lines = parse_lrc(LRC)
    assert lines == [
        (5.0, "first line"),
        (8.5, "second line"),
        (12.0, ""),  # empty line = singing stops
        (30.0, "chorus line"),
        (60.0, "chorus line"),  # one line with two timestamps
    ]


def test_sung_sections():
    sections = sung_sections(parse_lrc(LRC), song_length_s=65)
    # a line lasts until the next one starts, capped at 10 seconds
    assert sections == [(5.0, 8.5), (8.5, 12.0), (30.0, 40.0), (60.0, 65)]


def test_vocal_gate_mutes_outside_sung_lines():
    sr = 1000
    gate = vocal_gate([(2.0, 4.0)], num_samples=6 * sr, sr=sr)
    assert gate[int(1.0 * sr)] == 0  # before
    assert gate[int(3.0 * sr)] == 1  # during
    assert gate[int(5.0 * sr)] == 0  # after
    assert 0 < gate[int(1.6 * sr)] < 1  # fading in (0.4s padding, 0.15s fade)


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


@pytest.fixture
def fake_lrclib(monkeypatch):
    """Pretend lrclib has three versions of the song, so the tests don't need the internet."""
    results = [
        {"albumName": "Album", "duration": 240.0, "syncedLyrics": "[00:10.00]album version"},
        {"albumName": "Single", "duration": 206.0, "syncedLyrics": "[00:05.00]radio edit"},
        {"albumName": "Live", "duration": 300.0, "syncedLyrics": None},
    ]
    monkeypatch.setattr(lrclib.requests, "get", lambda *args, **kwargs: FakeResponse(results))


def test_picks_version_matching_song_length(fake_lrclib):
    assert lrclib.find_synced_lyrics("Artist", "Song", 207.3) == "[00:05.00]radio edit"
    assert lrclib.find_synced_lyrics("Artist", "Song", 239.0) == "[00:10.00]album version"


def test_refuses_when_no_version_matches(fake_lrclib):
    with pytest.raises(LookupError, match="none for a version this length"):
        lrclib.find_synced_lyrics("Artist", "Song", 280.0)
