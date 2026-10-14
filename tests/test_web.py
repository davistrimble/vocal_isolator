"""The website, using Flask's test client (fake requests, no real server needed)."""
import io
import time

import pytest
import soundfile as sf

from conftest import SR, stereo, tone
from vocal_isolator import web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WORK_DIR", str(tmp_path))  # keep test uploads out of the real folder
    web.jobs.clear()
    return web.app.test_client()


def upload(client, path, name="song.mp3"):
    with open(path, "rb") as f:
        return client.post("/upload", data={"song": (f, name)}, content_type="multipart/form-data")


def test_home_page(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"Vocal Isolator" in page.data


def test_upload_detects_tempo_and_key(client, mp3_file):
    response = upload(client, mp3_file)
    assert response.status_code == 200
    data = response.get_json()
    assert set(data) == {"job_id", "tempo", "key", "mode", "duration"}
    assert data["duration"] == pytest.approx(2.0, abs=0.1)


def test_upload_rejects_wav(client, tmp_path):
    path = tmp_path / "song.wav"
    sf.write(path, stereo(tone(440, 1.0)), SR)
    response = upload(client, path, "song.wav")
    assert response.status_code == 400
    assert "Only .mp3" in response.get_json()["error"]


def test_upload_rejects_fake_mp3(client):
    fake = (io.BytesIO(b"definitely not audio"), "song.mp3")
    response = client.post("/upload", data={"song": fake}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "isn't a valid" in response.get_json()["error"]


@pytest.mark.parametrize(
    "form, message",
    [
        ({"change_tempo": "on", "original_tempo": "abc", "new_tempo": "100"}, "as numbers"),
        ({"change_tempo": "on", "original_tempo": "120", "new_tempo": "900"}, "between 20 and 300"),
        ({"change_key": "on", "original_key": "H", "new_key": "C"}, "Pick both"),
    ],
)
def test_bad_options_are_rejected(client, mp3_file, form, message):
    job_id = upload(client, mp3_file).get_json()["job_id"]
    response = client.post(f"/separate/{job_id}", data=form)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_unknown_job(client):
    assert client.get("/status/nope").status_code == 404
    assert client.get("/download/nope").status_code == 404


def test_full_job_with_classic_method(client, mp3_file):
    """Upload, start processing, poll /status until done, then download."""
    job_id = upload(client, mp3_file).get_json()["job_id"]
    start = client.post(f"/separate/{job_id}", data={"method": "classic", "metronome": "on"})
    assert start.status_code == 200

    for _ in range(100):
        status = client.get(f"/status/{job_id}").get_json()
        if status["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert status["state"] == "done", status["message"]

    download = client.get(f"/download/{job_id}?save=1")
    assert download.status_code == 200
    assert download.mimetype == "audio/wav"
    assert "song_vocals.wav" in download.headers["Content-Disposition"]
