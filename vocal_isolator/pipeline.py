"""The full chain of processing steps, from song to finished vocal track.

Both the command line (__main__.py) and the website (web.py) call process(),
so they always do exactly the same thing.
"""
from vocal_isolator.analysis import detect_beats, detect_key
from vocal_isolator.audio_io import normalize, trim_silence
from vocal_isolator.center_extract import extract_center
from vocal_isolator.eq import presence_boost
from vocal_isolator.lyrics import parse_lrc, sung_sections, vocal_gate
from vocal_isolator.metronome import add_clicks, steady_beats
from vocal_isolator.ml_separate import extract_vocals_ml
from vocal_isolator.transform import change_key, change_tempo, semitones_between


def process(
    samples,
    sr,
    method="ml",
    new_key=None,
    original_key=None,
    new_tempo=None,
    original_tempo=None,
    metronome=False,
    lrc_text=None,
    boost=True,
    report=print,
):
    """Returns the finished vocals as a (num_samples, num_channels) array.

    report is called with a short status message as each step starts, so the
    caller can show progress (the command line prints it, the website shows
    it on the page).
    """
    # 1. separate the vocals from everything else
    if method == "ml":
        report("separating vocals with Demucs")
        vocals = extract_vocals_ml(samples, sr, on_progress=lambda pct: report(f"separating vocals: {pct:.0f}%"))
    else:
        report("separating vocals (classic method)")
        vocals = extract_center(samples, sr)
        if boost:
            vocals = presence_boost(vocals, sr)

    # 2. optionally mute everything between sung lines
    if lrc_text:
        sections = sung_sections(parse_lrc(lrc_text), len(vocals) / sr)
        report(f"muting between {len(sections)} sung lines")
        vocals = vocals * vocal_gate(sections, len(vocals), sr)[:, None]

    # 3. find the beat. this uses the original mix, not the separated vocals,
    # since the drums and bass are what make the beat easy to find
    beat_times = None
    if metronome or (new_tempo and not original_tempo):
        report("finding the beat")
        detected_tempo, beat_times = detect_beats(samples, sr)
        report(f"detected tempo: {detected_tempo:.0f} BPM")
        original_tempo = original_tempo or detected_tempo
        beat_times = steady_beats(beat_times, len(samples) / sr)

    # 4. change key
    if new_key:
        if not original_key:
            original_key, mode = detect_key(samples, sr)
            report(f"detected key: {original_key} {mode}")
        shift = semitones_between(original_key, new_key)
        report(f"changing key: {original_key} -> {new_key} ({shift:+d} semitones)")
        vocals = change_key(vocals, sr, shift)

    # 5. change tempo
    if new_tempo:
        report(f"changing tempo: {original_tempo:.0f} -> {new_tempo:.0f} BPM")
        vocals = change_tempo(vocals, sr, original_tempo, new_tempo)
        if beat_times is not None:
            # everything happens sooner/later by the same ratio
            beat_times = beat_times * original_tempo / new_tempo

    # 6. finishing: fix the volume, then either add the metronome (keeping the
    # intro as a count-in) or cut the silence before the first vocal
    vocals = normalize(vocals)
    if metronome:
        report("adding metronome")
        vocals = add_clicks(vocals, sr, beat_times)
    else:
        vocals, cut = trim_silence(vocals, sr)
        if cut > 0:
            report(f"skipped {cut:.1f}s of silence at the start")

    return normalize(vocals)
