"""The website. Run it with:

    python -m vocal_isolator.web

then open http://localhost:8000 in a browser.

How a request flows through the site:

  1. GET  /                  -> the page itself (templates/index.html)
  2. POST /upload            -> the browser sends the mp3. We save it, detect
                                its tempo and key, and send those back so the
                                page can pre-fill the tempo/key boxes.
  3. POST /separate/<job_id> -> the browser sends the chosen options. We start
                                processing in a background thread and reply
                                right away, instead of making the browser wait
                                a minute or two on one request.
  4. GET  /status/<job_id>   -> the page asks this every 15 seconds to show
                                progress ("separating vocals: 40%").
  5. GET  /download/<job_id> -> once it's done, the finished vocals.

All the actual audio work happens in pipeline.process(), the same function
the command line uses. This file just collects the user's choices and passes
them along.
"""
import os
import shutil
import tempfile
import threading
import time
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from vocal_isolator.analysis import NOTE_NAMES, detect_beats, detect_key
from vocal_isolator.audio_io import NotAnMp3Error, load_audio, save_audio
from vocal_isolator.pipeline import process

PORT = 8000  # Flask's usual 5000 is taken by AirPlay on Macs
MAX_UPLOAD_MB = 50
JOB_MAX_AGE_S = 60 * 60  # delete uploads and results after an hour

app = Flask(__name__)
# Flask rejects anything bigger than this before it even reaches our code
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# uploads and results go in the system temp folder, not the project
WORK_DIR = os.path.join(tempfile.gettempdir(), "vocal_isolator")

# everything we know about each song that's been uploaded, by job id.
# e.g. {"state": "working", "message": "separating vocals: 40%", ...}
# states: uploaded -> queued -> working -> done (or error)
jobs = {}

# only process one song at a time. the model uses a lot of memory, and two at
# once would just make both slower
processing_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html", note_names=NOTE_NAMES, max_upload_mb=MAX_UPLOAD_MB)


@app.post("/upload")
def upload():
    file = request.files.get("song")
    if file is None or file.filename == "":
        return jsonify(error="No file was uploaded."), 400

    # secure_filename strips anything dangerous, like "../../" that could
    # make us write outside our folder
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".mp3"):
        return jsonify(error="Only .mp3 files are supported."), 400

    clean_up_old_jobs()
    job_id = uuid.uuid4().hex  # a random id nobody could guess
    folder = os.path.join(WORK_DIR, job_id)
    os.makedirs(folder)
    song_path = os.path.join(folder, "song.mp3")
    file.save(song_path)

    try:
        samples, sr = load_audio(song_path)  # also checks it's a real mp3
    except NotAnMp3Error:
        shutil.rmtree(folder)
        return jsonify(error="That file isn't a valid .mp3."), 400

    tempo, _ = detect_beats(samples, sr)
    key, mode = detect_key(samples, sr)

    jobs[job_id] = {
        "state": "uploaded",
        "message": "",
        "folder": folder,
        "song_name": filename[: -len(".mp3")],
        "created": time.time(),
    }
    return jsonify(job_id=job_id, tempo=round(tempo), key=key, mode=mode, duration=len(samples) / sr)


@app.post("/separate/<job_id>")
def separate(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404)
    if job["state"] in ("queued", "working"):
        return jsonify(error="That song is already being processed."), 409

    try:
        options = read_options(request.form)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    job.update(state="queued", message="waiting to start")
    # daemon=True means this thread won't stop the server from shutting down
    threading.Thread(target=run_job, args=(job_id, options), daemon=True).start()
    return jsonify(ok=True)


def read_options(form):
    """Turn the submitted form into keyword arguments for pipeline.process().

    Checkboxes only show up in the form when they're checked.
    """
    options = {
        "method": "classic" if form.get("method") == "classic" else "ml",
        "metronome": "metronome" in form,
    }

    if "change_key" in form:
        original_key, new_key = form.get("original_key"), form.get("new_key")
        if original_key not in NOTE_NAMES or new_key not in NOTE_NAMES:
            raise ValueError("Pick both an original key and a new key.")
        options.update(original_key=original_key, new_key=new_key)

    if "change_tempo" in form:
        try:
            original_tempo = float(form["original_tempo"])
            new_tempo = float(form["new_tempo"])
        except (KeyError, ValueError):
            raise ValueError("Enter both tempos as numbers.")
        if not (20 <= original_tempo <= 300 and 20 <= new_tempo <= 300):
            raise ValueError("Tempos should be between 20 and 300 BPM.")
        options.update(original_tempo=original_tempo, new_tempo=new_tempo)

    return options


def run_job(job_id, options):
    """Runs in a background thread. Updates the job's message as it goes, which
    is what /status reports back to the page.
    """
    job = jobs[job_id]
    with processing_lock:
        job.update(state="working", message="loading song")
        try:
            samples, sr = load_audio(os.path.join(job["folder"], "song.mp3"))
            vocals = process(samples, sr, report=lambda message: job.update(message=message), **options)
            save_audio(os.path.join(job["folder"], "vocals.wav"), vocals, sr)
            job.update(state="done", message="done")
        except Exception as e:  # show the error on the page instead of failing silently
            job.update(state="error", message=f"Something went wrong: {e}")


@app.get("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(state=job["state"], message=job["message"])


@app.get("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if job is None or job["state"] != "done":
        abort(404)
    return send_file(
        os.path.join(job["folder"], "vocals.wav"),
        mimetype="audio/wav",
        # ?save=1 makes the browser download it, otherwise it can play it in the page
        as_attachment=request.args.get("save") == "1",
        download_name=job["song_name"] + "_vocals.wav",
    )


@app.errorhandler(413)
def too_big(_error):
    return jsonify(error=f"That file is too big (max {MAX_UPLOAD_MB} MB)."), 413


def clean_up_old_jobs():
    now = time.time()
    for job_id, job in list(jobs.items()):
        if now - job["created"] > JOB_MAX_AGE_S and job["state"] not in ("queued", "working"):
            shutil.rmtree(job["folder"], ignore_errors=True)
            del jobs[job_id]


if __name__ == "__main__":
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"open http://localhost:{PORT} in your browser")
    app.run(port=PORT)
