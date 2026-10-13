"""Look up timestamped lyrics on LRCLIB (https://lrclib.net), a free public
database of synced lyrics. No account or API key needed.

This is just a normal web request: we ask the API for a song by artist and
title, it sends back JSON with every version it has, and we pick the version
whose length matches our audio file.
"""
import requests

SEARCH_URL = "https://lrclib.net/api/search"

# lrclib asks apps to say who they are
USER_AGENT = "vocal_isolator (https://github.com/davistrimble/vocal_isolator)"

# the same song often exists in several versions (single edit, album version,
# extended mix) with different timings. if the closest one is further off than
# this, the timestamps won't line up with our file, so don't use it
MAX_DURATION_DIFF_S = 4.0


def search(artist, title):
    """Returns the list of results from lrclib (each one is a dict)."""
    response = requests.get(
        SEARCH_URL,
        params={"artist_name": artist, "track_name": title},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()  # error out if the server returned 404/500/etc
    return response.json()


def find_synced_lyrics(artist, title, song_length_s):
    """Returns the LRC text for the best matching version of the song.

    Raises LookupError with a readable message if nothing usable is found.
    """
    results = [r for r in search(artist, title) if r.get("syncedLyrics")]
    if not results:
        raise LookupError(f"no synced lyrics found for '{title}' by {artist}")

    best = min(results, key=lambda r: abs(r["duration"] - song_length_s))
    diff = abs(best["duration"] - song_length_s)
    if diff > MAX_DURATION_DIFF_S:
        lengths = sorted({int(r["duration"]) for r in results})
        raise LookupError(
            f"found lyrics, but none for a version this length ({song_length_s:.0f}s). "
            f"available versions are {lengths} seconds long, so the timing wouldn't line up. "
            f"you can pass your own .lrc file with --lyrics instead"
        )

    print(f"using lyrics from '{best['albumName']}' ({best['duration']:.0f}s)")
    return best["syncedLyrics"]
