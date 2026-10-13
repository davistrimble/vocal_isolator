// Everything interactive on the page. Plain JavaScript, no libraries.
//
// fetch() sends a request to our Flask server without reloading the page, and
// gives back the server's response. Our server replies in JSON, which
// response.json() turns into a normal JavaScript object.

const POLL_SECONDS = 15; // how often to ask the server how it's going

const form = document.getElementById("options-form");
const songInput = document.getElementById("song");
const goButton = document.getElementById("go");

let jobId = null; // the server's id for the uploaded song

// --- checkboxes that show/hide extra options ---
// each one has data-reveals="<id of the box to show>"
for (const checkbox of document.querySelectorAll("[data-reveals]")) {
  checkbox.addEventListener("change", () => {
    document.getElementById(checkbox.dataset.reveals).hidden = !checkbox.checked;
  });
}

// --- step 1: as soon as a song is picked, upload it and get its tempo/key ---
songInput.addEventListener("change", async () => {
  const file = songInput.files[0];
  hideError();
  goButton.disabled = true;
  jobId = null;
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".mp3")) {
    showError("Only .mp3 files are supported.");
    return;
  }

  setText("analysis", "Uploading and detecting tempo and key...");
  const upload = new FormData();
  upload.append("song", file);

  const response = await fetch("/upload", { method: "POST", body: upload });
  const data = await response.json();
  if (!response.ok) {
    setText("analysis", "");
    showError(data.error);
    return;
  }

  jobId = data.job_id;
  setText("analysis", `Detected: ${data.tempo} BPM, ${data.key} ${data.mode}. ` +
                      "You can change these below if they're off.");

  // pre-fill the "original" boxes, and start the "new" ones at the same values
  document.getElementById("original_key").value = data.key;
  document.getElementById("new_key").value = data.key;
  document.getElementById("original_tempo").value = data.tempo;
  document.getElementById("new_tempo").value = data.tempo;
  goButton.disabled = false;
});

// --- step 2: send the options and start processing ---
form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stop the browser's normal form submit (a full page reload)
  hideError();

  // FormData collects every field in the form. we don't need to re-send the
  // song itself, the server already has it under jobId
  const options = new FormData(form);
  options.delete("song");

  const response = await fetch(`/separate/${jobId}`, { method: "POST", body: options });
  const data = await response.json();
  if (!response.ok) {
    showError(data.error);
    return;
  }

  goButton.disabled = true;
  document.getElementById("result").hidden = true;
  document.getElementById("progress").hidden = false;
  setText("status-message", "Starting...");
  startElapsedTimer();
  checkStatus();
});

// --- step 3: ask the server how it's going, every POLL_SECONDS ---
async function checkStatus() {
  const response = await fetch(`/status/${jobId}`);
  const data = await response.json();

  if (data.state === "done") {
    stopElapsedTimer();
    showResult();
  } else if (data.state === "error") {
    stopElapsedTimer();
    document.getElementById("progress").hidden = true;
    showError(data.message);
    goButton.disabled = false;
  } else {
    setText("status-message", capitalize(data.message) + "...");
    setTimeout(checkStatus, POLL_SECONDS * 1000);
  }
}

function showResult() {
  document.getElementById("progress").hidden = true;
  document.getElementById("result").hidden = false;
  // ?t= makes the URL different each time, so the browser doesn't play an
  // old cached version if you process the same song twice
  document.getElementById("player").src = `/download/${jobId}?t=${Date.now()}`;
  document.getElementById("download-link").href = `/download/${jobId}?save=1`;
  goButton.disabled = false;
}

// --- a seconds counter, so the page doesn't look frozen between status checks ---
let timer = null;

function startElapsedTimer() {
  const start = Date.now();
  stopElapsedTimer();
  timer = setInterval(() => {
    const seconds = Math.round((Date.now() - start) / 1000);
    setText("elapsed", `${seconds}s elapsed (progress updates every ${POLL_SECONDS}s)`);
  }, 1000);
}

function stopElapsedTimer() {
  clearInterval(timer);
}

// --- small helpers ---
function setText(id, text) {
  document.getElementById(id).textContent = text;
}

function showError(message) {
  const box = document.getElementById("error");
  box.textContent = message;
  box.hidden = false;
}

function hideError() {
  document.getElementById("error").hidden = true;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
