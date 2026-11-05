// Wires the page together. All text from the server is set with textContent, so a
// strange filename can only ever appear as text.

import {
  fetchHealth,
  fetchModes,
  fetchRecordings,
  rescan,
  saveSidecar,
  type ModesResponse,
  type Recording,
} from "./api";
import { AudioEngine, type EngineStatus } from "./audio";
import { runBoot, type BootFacts } from "./boot";
import { ChunkStore } from "./chunks";
import { CloseUp } from "./closeup";
import { centerAt, newId, type Marker } from "./markers";
import { notesFor } from "./notes";
import { RawPairs } from "./rawpairs";
import { applySettings, changeSettings, settings, type Settings } from "./settings";
import { SignalPath } from "./signalpath";
import { colors } from "./theme";
import { SPEEDS, Transport, type Speed } from "./transport";
import { Uploader, type Job } from "./upload";
import { IqViews } from "./views";
import { Waterfall, type Cursor } from "./waterfall";
import "./style.css";

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  fileList: byId<HTMLUListElement>("file-list"),
  fileInfo: byId("file-info"),
  question: byId("question"),
  banner: byId("banner"),
  archive: byId("archive"),
  titleFrequency: byId("title-frequency"),
  titleMode: byId("title-mode"),
  modes: byId("modes"),
  tune: byId<HTMLInputElement>("tune"),
  bw: byId<HTMLInputElement>("bw"),
  bwLabel: byId("bw-label"),
  play: byId<HTMLButtonElement>("play"),
  scrub: byId<HTMLInputElement>("scrub"),
  time: byId("time"),
  status: byId("status"),
  floor: byId<HTMLInputElement>("floor"),
  ceiling: byId<HTMLInputElement>("ceiling"),
  panels: byId("panels"),
  speeds: byId("speeds"),
  muted: byId("muted"),
  stepBack: byId<HTMLButtonElement>("step-back"),
  stepForward: byId<HTMLButtonElement>("step-forward"),
  tuning: document.querySelector<HTMLElement>(".tuning")!,
  tabs: byId("tabs"),
  waterfallCanvas: byId("waterfall"),
  path: byId("path"),
  notes: byId("notes"),
  markers: byId("markers"),
  markerList: byId<HTMLUListElement>("marker-list"),
  markFrequency: byId<HTMLInputElement>("mark-frequency"),
  showFiles: byId("show-files"),
  showExamples: byId("show-examples"),
  showUpload: byId("show-upload"),
  fileTools: byId("file-tools"),
  rescanStatus: byId("rescan-status"),
  arriving: byId<HTMLUListElement>("arriving"),
  upload: byId("upload"),
  uploadJobs: byId<HTMLUListElement>("upload-jobs"),
  uploadUnfinished: byId<HTMLUListElement>("upload-unfinished"),
  uploadInput: byId<HTMLInputElement>("upload-input"),
  archiveNote: byId("archive-note"),
  analysisToggle: byId("analysis-toggle"),
  analysisBar: byId("analysis-bar"),
  analysisControls: byId("analysis-controls"),
  analysisNumbers: byId("analysis-numbers"),
  cursorArm: byId("cursor-arm"),
  cursorReadout: byId("cursor-readout"),
  constellationSource: byId("constellation-source"),
  constellationTuned: byId("constellation-tuned"),
  constellationRaw: byId("constellation-raw"),
  settings: byId<HTMLDialogElement>("settings"),
};

applySettings();

const engine = new AudioEngine();
const transport = new Transport(engine);
const store = new ChunkStore();
const views = new IqViews(
  (index) => (engine.isPlaying ? (engine.chunkFor(index) ?? store.peek(index)) : store.get(index)),
  byId<HTMLCanvasElement>("constellation"),
  byId<HTMLCanvasElement>("phasor"),
  byId<HTMLCanvasElement>("traces"),
  byId<HTMLCanvasElement>("waveform"),
  byId("pair-sample"),
);
const closeUp = new CloseUp(byId<HTMLCanvasElement>("spectrum"));
const signalPath = new SignalPath(
  {
    raw: byId<HTMLCanvasElement>("path-raw"),
    tuned: byId<HTMLCanvasElement>("path-tuned"),
    filtered: byId<HTMLCanvasElement>("path-filtered"),
    demod: byId<HTMLCanvasElement>("path-demod"),
    sound: byId<HTMLCanvasElement>("path-sound"),
  },
  byId("path-formula"),
);
const waterfall = new Waterfall(byId<HTMLCanvasElement>("waterfall"));
const rawPairs = new RawPairs();
const uploader = new Uploader();

let modes: ModesResponse = { ssb_low_hz: 300, modes: {} };
let recordings: Recording[] = [];
let current: Recording | null = null;
let offsetHz = 0;
let mode = "am";
let bwHz = 6000;
let scrubbing = false;
let following = true;
let tab: "panels" | "path" | "notes" | "markers" = "panels";
let listing: "files" | "examples" | "upload" = "files";
let arriving: { name: string; size_bytes: number }[] = [];
let shownCenter: number | null = null;
let shownMarkers = "";
let lastSegment: number | null = null;
let markerSaves: Promise<void> = Promise.resolve();
let markerSavesPending = 0;
let rawConstellation = false;
let analysisTextAt = 0;

const MODE_ORDER = ["am", "usb", "lsb", "cw", "nfm", "wfm"];

// --- Formatting -----------------------------------------------------------------------

function formatTime(seconds: number, digits = 3): string {
  const total = Math.max(seconds, 0);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = (total % 60).toFixed(digits).padStart(3 + digits, "0");
  const mm = String(minutes).padStart(2, "0");
  return hours ? `${hours}:${mm}:${rest}` : `${mm}:${rest}`;
}

function formatDuration(seconds: number): string {
  if (seconds >= 3600) return `${Math.floor(seconds / 3600)}h${String(Math.floor((seconds % 3600) / 60)).padStart(2, "0")}m`;
  if (seconds >= 60) return `${Math.floor(seconds / 60)}m${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  return `${seconds.toFixed(1)}s`;
}

function formatHz(hz: number): string {
  return `${(hz / 1e6).toFixed(6).replace(/(\.\d{3})(\d{3})$/, "$1 $2")} MHz`;
}

function kindLabel(recording: Recording): string {
  return { iq: "IQ", audio: "AUDIO", unknown: "?", unreadable: "ERR" }[recording.kind];
}

// --- File list ------------------------------------------------------------------------

function formatBytes(bytes: number): string {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(2)} GiB`;
  if (bytes >= 2 ** 20) return `${(bytes / 2 ** 20).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(0)} KiB`;
}

function renderFileList(): void {
  ui.fileList.replaceChildren();
  let folder: string | null = null;
  ui.showFiles.setAttribute("aria-pressed", String(listing === "files"));
  ui.showExamples.setAttribute("aria-pressed", String(listing === "examples"));
  ui.showUpload.setAttribute("aria-pressed", String(listing === "upload"));
  ui.fileList.hidden = listing === "upload";
  ui.upload.hidden = listing !== "upload";
  ui.fileTools.hidden = listing !== "files";
  ui.arriving.hidden = listing !== "files" || arriving.length === 0;
  ui.arriving.replaceChildren(
    ...arriving.map((file) => {
      const item = document.createElement("li");
      item.textContent = `COPYING... ${file.name} (${formatBytes(file.size_bytes)})`;
      return item;
    }),
  );
  if (listing === "upload") return;
  const shown = recordings.filter((r) => Boolean(r.example) === (listing === "examples"));
  if (listing === "examples") shown.sort((a, b) => a.example!.order - b.example!.order);
  if (shown.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = listing === "examples" ? "GENERATING EXAMPLES..." : "NO RECORDINGS FOUND";
    ui.fileList.append(empty);
  }
  for (const recording of shown) {
    if (recording.folder !== folder) {
      folder = recording.folder;
      if (folder) {
        const heading = document.createElement("li");
        heading.className = "folder";
        heading.textContent = `${folder}/`;
        ui.fileList.append(heading);
      }
    }
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "file";
    button.dataset.testid = `file-${recording.name}`;
    button.setAttribute("data-testid", `file-${recording.name}`);
    if (recording.id === current?.id) button.classList.add("selected");
    if (!recording.available) button.classList.add("unavailable");

    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = recording.example?.title ?? recording.name;

    const meta = document.createElement("span");
    meta.className = "file-meta";
    const parts = [kindLabel(recording)];
    if (recording.duration_s !== undefined) parts.push(formatDuration(recording.duration_s));
    if (recording.frequency_hz) parts.push(`${(recording.frequency_hz / 1e6).toFixed(3)} MHz`);
    if (recording.ingest.total > recording.ingest.done) {
      parts.push(`INDEXING ${Math.floor((100 * recording.ingest.done) / recording.ingest.total)}%`);
    }
    meta.textContent = parts.join(" . ");
    button.append(name, meta);
    if (recording.example) {
      const about = document.createElement("span");
      about.className = "example-note";
      about.textContent = recording.example.description;
      button.append(about);
    }
    button.addEventListener("click", () => selectRecording(recording.id));
    item.append(button);
    ui.fileList.append(item);
  }
}

// --- Selecting a recording --------------------------------------------------------------

function describe(recording: Recording): string {
  if (recording.kind === "unreadable") return `${recording.name}: CAN'T READ THIS FILE. ${recording.kind_reason}`;
  const parts = [
    recording.name,
    `${recording.kind.toUpperCase()} (${recording.kind_reason})`,
    `${((recording.sample_rate ?? 0) / 1e6).toFixed(3)} MS/s`,
    recording.sample_type ?? "",
    recording.container ?? "",
  ];
  if (recording.recorded_with) parts.push(recording.recorded_with);
  if (recording.started_at) parts.push(`START ${recording.started_at}`);
  return parts.filter(Boolean).join(" | ");
}

function showBanner(recording: Recording): void {
  const messages: string[] = [];
  if (recording.kind === "audio") {
    messages.push(
      "AUDIO RECORDING: NO IQ DATA. Already demodulated; the radio signal was not saved. " +
        "To capture IQ, set the SDR++ Recorder module to Baseband.",
    );
    if (recording.channels === 2 && recording.lr_match) messages.push("Left and right are the same: playing one channel.");
  }
  const hasFrequencyMarks = (recording.markers ?? []).some((m) => m.kind === "frequency");
  if (recording.frequency_note && !hasFrequencyMarks) {
    messages.push(`${recording.frequency_note} If the radio was retuned while recording, mark where in [ MARKERS ].`);
  }
  messages.push(...recording.warnings.map((warning) => `WARNING: ${warning}`));
  ui.banner.textContent = messages.join("  ");
  ui.banner.hidden = messages.length === 0;
}

function showQuestion(recording: Recording): void {
  ui.question.replaceChildren();
  ui.question.hidden = true;
  if (recording.kind === "unknown") {
    const text = document.createElement("span");
    text.textContent = `WHAT IS THIS FILE? ${recording.kind_reason}.`;
    ui.question.append(text);
    for (const answer of ["iq", "audio"] as const) {
      const button = document.createElement("button");
      button.textContent = answer === "iq" ? "IT'S IQ" : "IT'S AUDIO";
      button.addEventListener("click", () => void answerSidecar(recording.id, { kind: answer }));
      ui.question.append(button);
    }
    ui.question.hidden = false;
  } else if (recording.kind === "iq") {
    const text = document.createElement("span");
    text.textContent = recording.frequency_hz
      ? `CENTER ${formatHz(recording.frequency_hz)} (${recording.frequency_source}).`
      : "CENTER FREQUENCY UNKNOWN: showing offsets from center.";
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.placeholder = "center kHz";
    input.setAttribute("aria-label", "Center frequency in kHz");
    input.setAttribute("data-testid", "center-input");
    const set = document.createElement("button");
    set.textContent = "SET";
    set.addEventListener("click", () => {
      const khz = parseFloat(input.value);
      if (Number.isFinite(khz) && khz > 0) void answerSidecar(recording.id, { frequency_hz: Math.round(khz * 1000) });
    });
    ui.question.append(text, input, set);
    if (recording.frequency_source === "set by the user") {
      const clear = document.createElement("button");
      clear.textContent = "CLEAR";
      clear.addEventListener("click", () => void answerSidecar(recording.id, { frequency_hz: null }));
      ui.question.append(clear);
    }
    ui.question.hidden = false;
  }
}

async function answerSidecar(
  id: string,
  values: { kind?: "iq" | "audio"; frequency_hz?: number | null; markers?: Marker[] },
): Promise<void> {
  try {
    const updated = await saveSidecar(id, values);
    recordings = recordings.map((r) => (r.id === id ? updated : r));
    if (current?.id === id) applyRecording(updated, Object.keys(values).includes("kind"));
    renderFileList();
  } catch (error) {
    ui.status.textContent = (error as Error).message;
  }
}

function applyRecording(recording: Recording, reload: boolean): void {
  current = recording;
  ui.fileInfo.textContent = describe(recording);
  showBanner(recording);
  showQuestion(recording);
  waterfall.setCenterHz(recording.frequency_hz ?? null);
  if (reload) {
    engine.unload();
    offsetHz = 0;
    waterfall.load(recording.kind === "iq" || recording.kind === "audio" ? recording : null);
    store.setRecording(recording.kind === "iq" ? recording.id : null);
    views.reset();
    closeUp.setRecording(recording.kind === "iq" ? recording.id : null, recording.sample_rate ?? 48_000, recording.duration_s ?? 0);
    store.setTuning({ offsetHz, mode, bwHz });
    engine.retune({ offsetHz, mode, bwHz });
    transport.load(recording.duration_s ?? 0);
    renderSpeeds();
    signalPath.setRecording(recording.kind === "iq" ? recording.id : null);
    rawPairs.setRecording(recording.kind === "iq" ? recording.id : null);
    lastSegment = null;
    shownCenter = null;
    if (recording.kind === "iq" || recording.kind === "audio") {
      store.setRecording(recording.id);
      engine.load(recording.id, recording.duration_s ?? 0);
    }
    if (recording.kind !== "iq" && tab === "path") tab = "panels";
    renderSpeeds();
  }
  const isIq = recording.kind === "iq";
  const playable = (isIq || recording.kind === "audio") && recording.available;
  // Sound already going can still be paused, even if the archive has just dropped out.
  ui.play.disabled = !playable && !transport.isRunning;
  ui.scrub.disabled = !playable;
  ui.stepBack.disabled = !isIq || !recording.available;
  ui.stepForward.disabled = !isIq || !recording.available;
  ui.panels.classList.toggle("audio-only", recording.kind === "audio");
  ui.tuning.hidden = !isIq;
  ui.speeds.hidden = !isIq;
  ui.stepBack.hidden = !isIq;
  ui.stepForward.hidden = !isIq;
  applyTab();
  applyAnalysis();
  renderMarkers();
  ui.scrub.max = String(recording.duration_s ?? 1);
  syncTuningControls();
  updateTuning(false);
}

/** The open file is gone from the archive for good, unlike one that's only offline. */
function closeRemovedRecording(gone: Recording): void {
  engine.unload();
  waterfall.load(null);
  store.setRecording(null);
  closeUp.setRecording(null, 48_000, 0);
  signalPath.setRecording(null);
  rawPairs.setRecording(null);
  views.reset();
  transport.load(0);
  current = null;
  ui.fileInfo.textContent = "SELECT A RECORDING";
  ui.question.hidden = true;
  ui.banner.textContent = `${gone.name} IS NO LONGER IN THE ARCHIVE. It was moved, renamed or deleted.`;
  ui.banner.hidden = false;
  ui.play.disabled = true;
  ui.scrub.disabled = true;
  ui.stepBack.disabled = true;
  ui.stepForward.disabled = true;
  ui.tuning.hidden = true;
  ui.status.textContent = "";
  waterfall.setOverlay(null);
  ui.titleFrequency.textContent = "--";
  ui.titleMode.textContent = "--";
  applyTab();
  applyAnalysis();
}

function selectRecording(id: string): void {
  const recording = recordings.find((r) => r.id === id);
  if (!recording) return;
  following = true;
  applyRecording(recording, true);
  ui.status.textContent = "";
  // An example opens tuned and ready, so it can be heard straight away.
  if (recording.example) {
    setMode(recording.example.mode);
    tuneTo(recording.example.offset_hz);
  }
  renderFileList();
}

ui.showFiles.addEventListener("click", () => {
  listing = "files";
  renderFileList();
});
ui.showExamples.addEventListener("click", () => {
  listing = "examples";
  renderFileList();
});
ui.showUpload.addEventListener("click", () => {
  listing = "upload";
  renderFileList();
  void uploader.refresh();
});

byId("rescan").addEventListener("click", async () => {
  ui.rescanStatus.textContent = "LOOKING...";
  try {
    await rescan();
    await refreshRecordings();
    ui.rescanStatus.textContent = "";
  } catch (error) {
    ui.rescanStatus.textContent = (error as Error).message;
  }
});

// --- Uploading --------------------------------------------------------------------------

const JOB_LABELS: Record<Job["state"], string> = {
  waiting: "WAITING",
  checking: "CHECKING",
  sending: "SENDING",
  retrying: "CONNECTION TROUBLE",
  done: "DONE",
  failed: "FAILED",
  cancelled: "CANCELLED",
};

function renderUploads(): void {
  ui.uploadJobs.replaceChildren(
    ...uploader.jobs
      .filter((job) => job.state !== "cancelled")
      .map((job) => {
        const item = document.createElement("li");
        item.className = `upload-job ${job.state}`;
        item.setAttribute("data-testid", "upload-job");
        const name = document.createElement("div");
        name.className = "file-name";
        name.textContent = job.file.name;
        const percent = job.file.size ? (100 * job.sent) / job.file.size : 0;
        const bar = document.createElement("progress");
        bar.max = 100;
        bar.value = job.state === "done" ? 100 : percent;
        const line = document.createElement("div");
        line.className = "file-meta";
        const speed = job.state === "sending" && job.bytesPerSecond ? `  ${formatBytes(job.bytesPerSecond)}/s` : "";
        line.textContent =
          `${JOB_LABELS[job.state]}${job.resumed && job.state !== "done" ? " (CONTINUED)" : ""}` +
          `  ${formatBytes(job.sent)} OF ${formatBytes(job.file.size)}${speed}`;
        item.append(name, bar, line);
        if (job.message) {
          const message = document.createElement("div");
          message.className = job.state === "failed" || job.state === "retrying" ? "status alert" : "file-meta";
          message.textContent = job.message;
          item.append(message);
        }
        if (uploader.busy && job.state !== "done" && job.state !== "failed") {
          const cancel = document.createElement("button");
          cancel.textContent = "CANCEL";
          cancel.setAttribute("data-testid", "upload-cancel");
          cancel.addEventListener("click", () => uploader.cancel(job));
          item.append(cancel);
        }
        return item;
      }),
  );
  const active = new Set(uploader.jobs.filter((j) => uploader.busy && j.state !== "done").map((j) => j.uploadId));
  ui.uploadUnfinished.replaceChildren(
    ...uploader.unfinished
      .filter((info) => !active.has(info.id))
      .map((info) => {
        const item = document.createElement("li");
        item.className = "upload-job";
        item.setAttribute("data-testid", "upload-unfinished-item");
        const name = document.createElement("div");
        name.className = "file-name";
        name.textContent = info.original_name;
        const line = document.createElement("div");
        line.className = "file-meta";
        line.textContent = `UNFINISHED: ${formatBytes(info.received)} OF ${formatBytes(info.size)}. CHOOSE THE SAME FILE TO CONTINUE.`;
        const discard = document.createElement("button");
        discard.textContent = "DISCARD";
        discard.setAttribute("data-testid", "upload-discard");
        discard.addEventListener("click", () => void uploader.discard(info.id));
        item.append(name, line, discard);
        return item;
      }),
  );
}
uploader.onChange = renderUploads;
uploader.onFinished = () => void refreshRecordings();

ui.uploadInput.addEventListener("change", () => {
  uploader.add(ui.uploadInput.files ?? []);
  ui.uploadInput.value = ""; // so the same file can be chosen again
});
const filesColumn = document.querySelector<HTMLElement>(".files")!;
filesColumn.addEventListener("dragover", (event) => {
  event.preventDefault();
  filesColumn.classList.add("dropping");
});
filesColumn.addEventListener("dragleave", () => filesColumn.classList.remove("dropping"));
filesColumn.addEventListener("drop", (event) => {
  event.preventDefault();
  filesColumn.classList.remove("dropping");
  if (!event.dataTransfer?.files.length) return;
  listing = "upload";
  renderFileList();
  uploader.add(event.dataTransfer.files);
});
// Leaving mid-upload loses nothing that's arrived, but it does stop the rest.
window.addEventListener("beforeunload", (event) => {
  if (uploader.busy) event.preventDefault();
});

// --- Lower tabs -----------------------------------------------------------------------

function applyTab(): void {
  const kind = current?.kind;
  const hasPlayer = kind === "iq" || kind === "audio";
  ui.tabs.hidden = !hasPlayer;
  for (const button of ui.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
    button.setAttribute("aria-pressed", String(button.dataset.tab === tab));
    if (button.dataset.tab === "path") button.hidden = kind !== "iq";
  }
  ui.panels.hidden = !hasPlayer || tab !== "panels";
  ui.path.hidden = !hasPlayer || tab !== "path";
  ui.notes.hidden = !hasPlayer || tab !== "notes";
  ui.markers.hidden = !hasPlayer || tab !== "markers";
  if (tab === "notes") renderNotes();
}

for (const button of ui.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
  button.addEventListener("click", () => {
    tab = button.dataset.tab as typeof tab;
    applyTab();
  });
}

function renderNotes(): void {
  const { main, others } = notesFor(current?.kind === "audio" ? "audio" : mode);
  const parts: HTMLElement[] = [];
  const add = (tag: string, text: string, className?: string) => {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    parts.push(element);
  };
  add("h3", main.heading);
  add("p", main.body);
  if (others.length) add("p", "THINGS THAT CATCH PEOPLE OUT", "notes-rule");
  for (const note of others) {
    add("h3", note.heading);
    add("p", note.body);
  }
  ui.notes.replaceChildren(...parts);
}

// --- Markers --------------------------------------------------------------------------

function markerList(): Marker[] {
  return current?.markers ?? [];
}

function saveMarkers(markers: Marker[]): void {
  if (!current) return;
  const sorted = [...markers].sort((a, b) => a.t - b.t);
  current.markers = sorted;
  renderMarkers(true);

  // Saves go strictly one after another, and until the last has finished, this page's
  // list is the true one. A slower reply or a background refresh can't undo a newer edit.
  const id = current.id;
  markerSavesPending++;
  markerSaves = markerSaves.then(async () => {
    try {
      const updated = await saveSidecar(id, { markers: sorted });
      if (markerSavesPending === 1) {
        recordings = recordings.map((r) => (r.id === id ? updated : r));
        if (current?.id === id) {
          current = updated;
          showBanner(updated);
          renderMarkers();
        }
      }
    } catch (error) {
      ui.status.textContent = `Couldn't save markers: ${(error as Error).message}`;
      ui.status.classList.add("alert");
    } finally {
      markerSavesPending--;
    }
  });
}

function addNoteMarker(): void {
  if (!current || !isPlayable()) return;
  const marker: Marker = { id: newId(), t: transport.seconds, label: "", kind: "note" };
  tab = "markers";
  applyTab();
  saveMarkers([...markerList(), marker]);
  // Ready to type a label straight away; Enter or clicking away keeps it.
  ui.markerList.querySelector<HTMLInputElement>(`[data-marker="${marker.id}"]`)?.focus();
}

function addFrequencyMark(): void {
  if (!current || current.kind !== "iq") return;
  const khz = parseFloat(ui.markFrequency.value);
  if (!Number.isFinite(khz) || khz <= 0) return;
  const hz = Math.round(khz * 1000);
  ui.markFrequency.value = "";
  saveMarkers([
    ...markerList(),
    { id: newId(), t: transport.seconds, label: `${(hz / 1e6).toFixed(3)} MHz`, kind: "frequency", frequency_hz: hz },
  ]);
}

function renderMarkers(ownEdit = false): void {
  const list = markerList();
  const signature = JSON.stringify(list);
  waterfall.setMarkers(list);
  if (signature === shownMarkers) return;
  // A background refresh waits, so it can't rebuild the list under someone typing a label.
  // The user's own edits always show, keeping the cursor in whichever label it was in.
  const focused = document.activeElement as HTMLInputElement | null;
  const typingIn = focused && ui.markerList.contains(focused) ? focused.getAttribute("data-marker") : null;
  if (!ownEdit && typingIn) return;
  const caret = typingIn ? focused!.selectionStart : null;
  shownMarkers = signature;
  ui.markerList.replaceChildren();
  if (list.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "NO MARKERS YET";
    ui.markerList.append(empty);
  }
  for (const marker of list) {
    const row = document.createElement("li");
    row.className = "marker-row";
    const go = document.createElement("button");
    go.textContent = formatTime(marker.t);
    go.title = "Jump to this moment";
    go.setAttribute("data-testid", "marker-go");
    go.addEventListener("click", () => {
      transport.seek(marker.t);
      following = true;
    });
    row.append(go);
    if (marker.kind === "frequency") {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "RETUNED";
      row.append(tag);
    }
    const label = document.createElement("input");
    label.type = "text";
    label.value = marker.label;
    label.maxLength = 200;
    label.placeholder = "label";
    label.setAttribute("aria-label", "Marker label");
    label.setAttribute("data-marker", marker.id);
    label.setAttribute("data-testid", "marker-label");
    label.addEventListener("keydown", (event) => {
      if (event.key === "Enter") label.blur();
    });
    label.addEventListener("change", () => {
      saveMarkers(markerList().map((m) => (m.id === marker.id ? { ...m, label: label.value } : m)));
    });
    const remove = document.createElement("button");
    remove.textContent = "DEL";
    remove.setAttribute("data-testid", "marker-delete");
    remove.addEventListener("click", () => saveMarkers(markerList().filter((m) => m.id !== marker.id)));
    row.append(label, remove);
    ui.markerList.append(row);
  }
  if (typingIn) {
    const again = ui.markerList.querySelector<HTMLInputElement>(`[data-marker="${typingIn}"]`);
    again?.focus();
    if (again && caret !== null) again.setSelectionRange(caret, caret);
  }
}

byId("add-marker").addEventListener("click", addNoteMarker);
byId("add-frequency").addEventListener("click", addFrequencyMark);
ui.markFrequency.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    addFrequencyMark();
    ui.markFrequency.blur();
  }
});

function centerNow(): number | null {
  return centerAt(current?.frequency_hz, markerList(), transport.seconds);
}

function isPlayable(): boolean {
  return current?.kind === "iq" || current?.kind === "audio";
}

// --- Tuning ---------------------------------------------------------------------------

function renderModes(): void {
  ui.modes.replaceChildren();
  for (const name of MODE_ORDER) {
    const button = document.createElement("button");
    button.textContent = name.toUpperCase();
    button.setAttribute("data-testid", `mode-${name}`);
    button.setAttribute("aria-pressed", String(name === mode));
    button.addEventListener("click", () => setMode(name));
    ui.modes.append(button);
  }
}

function setMode(name: string): void {
  mode = name;
  bwHz = modes.modes[name]?.default_bw_hz ?? bwHz;
  renderModes();
  syncTuningControls();
  updateTuning(true);
  if (tab === "notes") renderNotes();
}

function syncTuningControls(): void {
  const isIq = current?.kind === "iq";
  ui.tune.disabled = !isIq;
  ui.bw.disabled = !isIq;
  ui.bw.value = String(bwHz);
  ui.bwLabel.textContent = mode === "usb" || mode === "lsb" ? "TOP Hz" : "BW Hz";
  ui.tune.value = displayFrequencyKhz();
  ui.titleMode.textContent = isIq ? mode.toUpperCase() : "--";
}

function displayFrequencyKhz(): string {
  if (!current || current.kind !== "iq") return "";
  return (((centerNow() ?? 0) + offsetHz) / 1000).toFixed(3);
}

function passband(): { lowHz: number; highHz: number } {
  const low = modes.ssb_low_hz;
  if (mode === "usb") return { lowHz: low, highHz: bwHz };
  if (mode === "lsb") return { lowHz: -bwHz, highHz: -low };
  return { lowHz: -bwHz / 2, highHz: bwHz / 2 };
}

function updateTuning(applyToEngine: boolean): void {
  if (!current || current.kind !== "iq") {
    waterfall.setOverlay(null);
    ui.titleFrequency.textContent = "--";
    return;
  }
  const center = centerNow();
  ui.titleFrequency.textContent =
    center != null ? formatHz(center + offsetHz) : `${offsetHz >= 0 ? "+" : ""}${offsetHz} Hz`;
  ui.tune.value = displayFrequencyKhz();
  waterfall.setOverlay({ offsetHz, ...passband() });
  if (applyToEngine) {
    engine.retune({ offsetHz, mode, bwHz });
    store.setTuning({ offsetHz, mode, bwHz });
    ui.status.textContent = "";
  }
}

function tuneTo(newOffsetHz: number): void {
  if (!current?.sample_rate) return;
  const limit = current.sample_rate / 2;
  offsetHz = Math.round(Math.min(Math.max(newOffsetHz, -limit), limit));
  updateTuning(true);
  waterfall.reveal(offsetHz);
}

waterfall.onTune = tuneTo;
closeUp.onTune = tuneTo;
closeUp.onSliceChange = (lowHz, highHz) => waterfall.setSlice(lowHz, highHz);
byId("slice-narrow").addEventListener("click", () => closeUp.changeSpan(-1));
byId("slice-wide").addEventListener("click", () => closeUp.changeSpan(1));
byId("slice-center").addEventListener("click", () => closeUp.recenter());
waterfall.onSeek = (seconds) => {
  transport.seek(seconds);
  following = true;
};
waterfall.onPanByUser = () => {
  following = false;
};
waterfall.onContrastAuto = (floor, ceiling) => {
  ui.floor.value = String(floor);
  ui.ceiling.value = String(ceiling);
};

ui.tune.addEventListener("change", () => {
  const khz = parseFloat(ui.tune.value);
  if (!Number.isFinite(khz)) {
    ui.tune.value = displayFrequencyKhz();
    return;
  }
  tuneTo(Math.round(khz * 1000) - (centerNow() ?? 0));
});

ui.bw.addEventListener("change", () => {
  const value = parseFloat(ui.bw.value);
  if (Number.isFinite(value) && value > 0) {
    bwHz = value;
    updateTuning(true);
  } else {
    ui.bw.value = String(bwHz);
  }
});

// Enter finishes typing, so the arrow keys and spacebar go back to controlling playback.
for (const input of [ui.tune, ui.bw]) {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") input.blur();
  });
}

for (const button of document.querySelectorAll<HTMLButtonElement>("[data-step]")) {
  button.addEventListener("click", () => tuneTo(offsetHz + Number(button.dataset.step)));
}

// --- Analysis Mode ----------------------------------------------------------------------
// It only adds things to look at. Nothing here changes the tuning or what the sound is made of.

const withCommas = (n: number): string => Math.round(n).toLocaleString("en-US");

function signedHz(hz: number): string {
  const rounded = Math.round(hz);
  return `${rounded >= 0 ? "+" : "-"}${withCommas(Math.abs(rounded))} Hz`;
}

function applyAnalysis(): void {
  const on = settings.analysis;
  const hasPlayer = isPlayable();
  ui.analysisToggle.textContent = `[ ANALYSIS: ${on ? "ON" : "OFF"} ]`;
  ui.analysisToggle.setAttribute("aria-pressed", String(on));
  ui.analysisBar.hidden = !on || !hasPlayer;
  ui.analysisControls.hidden = !on || !hasPlayer;
  ui.constellationSource.hidden = current?.kind !== "iq";
  ui.constellationTuned.setAttribute("aria-pressed", String(!rawConstellation));
  ui.constellationRaw.setAttribute("aria-pressed", String(rawConstellation));
  views.rawSource = on && rawConstellation && current?.kind === "iq" ? () => rawPairs.burst : null;
  ui.cursorArm.setAttribute("aria-pressed", String(waterfall.placingCursors));
  if (!on) ui.waterfallCanvas.classList.remove("placing");
  else ui.waterfallCanvas.classList.toggle("placing", waterfall.placingCursors);
  renderCursors(waterfall.cursors);
}

function cursorText(cursor: Cursor): { text: string; absolute: number } {
  const time = formatTime(cursor.seconds);
  if (current?.kind === "audio") return { text: `${(cursor.hz / 1000).toFixed(3)} kHz  ${time}`, absolute: cursor.hz };
  // On a retuned recording, the same place on the screen can mean different real frequencies.
  const center = centerAt(current?.frequency_hz, markerList(), cursor.seconds);
  if (center === null) return { text: `${signedHz(cursor.hz)}  ${time}`, absolute: cursor.hz };
  return { text: `${formatHz(center + cursor.hz)} ${time}`, absolute: center + cursor.hz };
}

function renderCursors(cursors: Cursor[]): void {
  waterfall.cursorNote = "";
  if (cursors.length === 0) {
    ui.cursorReadout.textContent = "CURSORS: SHIFT+CLICK THE WATERFALL";
    return;
  }
  const a = cursorText(cursors[0]);
  if (cursors.length === 1) {
    ui.cursorReadout.textContent = `A ${a.text}\nB: SHIFT+CLICK AGAIN`;
    return;
  }
  const b = cursorText(cursors[1]);
  const seconds = cursors[1].seconds - cursors[0].seconds;
  const dF = `dF ${signedHz(b.absolute - a.absolute)}`;
  const dT = `dT ${seconds >= 0 ? "+" : "-"}${Math.abs(seconds).toFixed(3)} s`;
  ui.cursorReadout.textContent = `A ${a.text}\nB ${b.text}\n${dF}\n${dT}`;
  waterfall.cursorNote = `${dF}  ${dT}`;
}

/** The average level over this stretch, in decibels relative to full scale. */
function levelDb(...parts: Float32Array[]): number | null {
  let total = 0;
  let count = 0;
  for (let n = 0; n < parts[0].length; n++) {
    if (Number.isNaN(parts[0][n])) continue;
    for (const part of parts) total += part[n] * part[n];
    count++;
  }
  return count && total > 0 ? 10 * Math.log10(total / count) : null;
}

const dbText = (db: number | null): string => (db === null ? "--" : `${db.toFixed(1)} dBFS`);

function updateAnalysis(seconds: number): void {
  if (!settings.analysis || !current || !isPlayable()) return;
  const rate = current.sample_rate ?? 48_000;
  const lastSample = Math.max((current.frames ?? 1) - 1, 0);
  const sourceSample = Math.min(Math.floor(seconds * rate), lastSample);
  if (views.rawSource && tab === "panels") rawPairs.want(sourceSample);

  // At full speed these change too fast to read, so refresh them less often.
  const now = performance.now();
  if (transport.speed === 1 && !transport.stepped && now - analysisTextAt < 100) return;
  analysisTextAt = now;

  const lines: string[] = [];
  if (current.kind === "iq") {
    const center = centerNow();
    if (center !== null) lines.push(`TUNED  ${formatHz(center + offsetHz)}`);
    lines.push(`OFFSET ${signedHz(offsetHz)}`);
    lines.push(`RAW    #${withCommas(sourceSample)}`);
    const iqRate = currentIqRate();
    if (views.lastPair !== null) lines.push(`PAIR   #${withCommas(views.lastPair)}`);
    if (views.lastValues) lines.push(`AUDIO  #${withCommas(views.lastValues.sample)}`);
    const start = Math.floor(seconds * iqRate) - 1200;
    lines.push(`FILTER ${dbText(levelDb(views.read("i", iqRate, start, 2400), views.read("q", iqRate, start, 2400)))}`);
    const burst = rawPairs.burst;
    if (views.rawSource && burst) lines.push(`BAND   ${dbText(levelDb(burst.i, burst.q))}`);
  } else {
    lines.push(`SAMPLE #${withCommas(sourceSample)}`);
    lines.push(`RATE   ${withCommas(rate)} S/s`);
    const start = Math.floor(seconds * 48_000) - 1200;
    lines.push(`LEVEL  ${dbText(levelDb(views.read("audio", 48_000, start, 2400)))}`);
  }
  const text = lines.join("\n");
  if (ui.analysisNumbers.textContent !== text) ui.analysisNumbers.textContent = text;
}

ui.analysisToggle.addEventListener("click", () => {
  changeSettings({ analysis: !settings.analysis });
  applyAnalysis();
});
waterfall.onCursors = renderCursors;
ui.cursorArm.addEventListener("click", () => {
  waterfall.placingCursors = !waterfall.placingCursors;
  applyAnalysis();
});
byId("cursor-clear").addEventListener("click", () => waterfall.clearCursors());
ui.constellationTuned.addEventListener("click", () => {
  rawConstellation = false;
  applyAnalysis();
});
ui.constellationRaw.addEventListener("click", () => {
  rawConstellation = true;
  applyAnalysis();
});

// --- Settings ---------------------------------------------------------------------------

function renderSettings(): void {
  for (const button of ui.settings.querySelectorAll<HTMLButtonElement>("[data-setting]")) {
    const key = button.dataset.setting as keyof Settings;
    button.setAttribute("aria-pressed", String(String(settings[key]) === button.dataset.value));
  }
  byId("motion-note").hidden = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

for (const button of ui.settings.querySelectorAll<HTMLButtonElement>("[data-setting]")) {
  button.addEventListener("click", () => {
    const key = button.dataset.setting as "theme" | "effects" | "boot";
    const value = button.dataset.value!;
    changeSettings(key === "theme" ? { theme: value as Settings["theme"] } : { [key]: value === "true" });
    waterfall.repaint();
    renderSettings();
  });
}
byId("open-settings").addEventListener("click", () => {
  renderSettings();
  ui.settings.showModal();
});
byId("close-settings").addEventListener("click", () => ui.settings.close());
// A click on the darkened page around the panel closes it too.
ui.settings.addEventListener("click", (event) => {
  if (event.target === ui.settings) ui.settings.close();
});

// --- Transport ------------------------------------------------------------------------

function togglePlay(): void {
  if (!isPlayable()) return;
  following = true;
  transport.toggle();
}

function currentIqRate(): number {
  return modes.modes[mode]?.iq_rate ?? 48_000;
}

function step(pairs: number): void {
  if (!current || current.kind !== "iq") return;
  following = true;
  transport.step(pairs, currentIqRate());
}

const speedLabels: Record<Speed, string> = { 1: "1x", 0.1: "1/10", 0.001: "1/1000" };

function renderSpeeds(): void {
  ui.speeds.replaceChildren();
  for (const speed of SPEEDS) {
    const button = document.createElement("button");
    button.textContent = speedLabels[speed];
    button.setAttribute("data-testid", `speed-${speed}`);
    button.setAttribute("aria-pressed", String(speed === transport.speed));
    button.addEventListener("click", () => {
      if (!current || current.kind !== "iq") return;
      transport.setSpeed(speed);
      renderSpeeds();
    });
    ui.speeds.append(button);
  }
}

ui.play.addEventListener("click", togglePlay);
ui.stepBack.addEventListener("click", () => step(-1));
ui.stepForward.addEventListener("click", () => step(1));
byId("back").addEventListener("click", () => transport.seek(transport.seconds - 5));
byId("forward").addEventListener("click", () => transport.seek(transport.seconds + 5));

ui.scrub.addEventListener("pointerdown", () => {
  scrubbing = true;
  transport.beginScrub();
});
ui.scrub.addEventListener("input", () => {
  if (!scrubbing) {
    scrubbing = true;
    transport.beginScrub();
  }
  following = true;
  transport.scrubTo(Number(ui.scrub.value));
});
ui.scrub.addEventListener("change", () => {
  scrubbing = false;
  transport.endScrub(Number(ui.scrub.value));
});
// A press and release with no movement fires neither input nor change.
ui.scrub.addEventListener("pointerup", () => {
  if (scrubbing) {
    scrubbing = false;
    transport.endScrub(Number(ui.scrub.value));
  }
});

document.addEventListener("keydown", (event) => {
  if (ui.settings.open) return; // keys belong to the settings panel while it's up
  const target = event.target as HTMLElement;
  if (target.tagName === "INPUT" && target !== ui.scrub && (target as HTMLInputElement).type !== "range") return;
  if (event.code === "Space") {
    event.preventDefault();
    togglePlay();
  } else if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
    if (target === volumeSlider) return; // arrows on the volume slider change the volume
    event.preventDefault();
    const pairs = event.shiftKey ? 100 : 1;
    step(event.code === "ArrowLeft" ? -pairs : pairs);
  } else if (event.code === "KeyM" && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault();
    addNoteMarker();
  }
});

engine.onStatus = (status: EngineStatus, message?: string) => {
  const labels: Record<EngineStatus, string> = {
    stopped: "",
    playing: "",
    paused: "PAUSED",
    buffering: "BUFFERING",
    ended: "END OF RECORDING",
  };
  ui.status.textContent = message ?? labels[status];
  ui.status.classList.toggle("alert", Boolean(message));
};

// --- View controls ----------------------------------------------------------------------

byId("freq-in").addEventListener("click", () => waterfall.zoomFrequency(0.5));
byId("freq-out").addEventListener("click", () => waterfall.zoomFrequency(2));
byId("time-in").addEventListener("click", () => waterfall.zoomTime(0.5));
byId("time-out").addEventListener("click", () => waterfall.zoomTime(2));
byId("fit").addEventListener("click", () => waterfall.fit());
const contrast = () => waterfall.setContrast(Number(ui.floor.value), Number(ui.ceiling.value));
ui.floor.addEventListener("input", contrast);
ui.ceiling.addEventListener("input", contrast);
byId("contrast-reset").addEventListener("click", () => waterfall.resetContrast());

// The slider follows how loudness is heard: halfway sounds about half as loud.
const VOLUME_KEY = "iqscope.volume";
const volumeSlider = byId<HTMLInputElement>("volume");
function applyVolume(): void {
  engine.setVolume((Number(volumeSlider.value) / 100) ** 2);
}
volumeSlider.addEventListener("input", () => {
  applyVolume();
  try {
    localStorage.setItem(VOLUME_KEY, volumeSlider.value);
  } catch {
    // Without storage the slider still works; it just won't be remembered.
  }
});
try {
  const saved = localStorage.getItem(VOLUME_KEY);
  if (saved !== null && Number.isFinite(Number(saved))) volumeSlider.value = saved;
} catch {
  // As above.
}
applyVolume();

// --- Polling and the animation loop -----------------------------------------------------

async function refreshRecordings(): Promise<void> {
  try {
    const data = await fetchRecordings();
    recordings = data.recordings;
    arriving = data.arriving ?? [];
    // Just after a restart, or with the share down, an empty list proves nothing, and a
    // file that's merely being looked at again isn't gone either.
    if (
      current &&
      data.complete &&
      !recordings.some((r) => r.id === current!.id) &&
      !arriving.some((a) => a.name === current!.name)
    ) {
      closeRemovedRecording(current);
    }
    if (current) {
      const fresh = recordings.find((r) => r.id === current!.id);
      if (fresh && markerSavesPending > 0) {
        fresh.markers = current.markers; // this page's edits are newer than that snapshot
      }
      if (fresh) {
        const wasAvailable = current.available;
        current = fresh;
        if (wasAvailable !== fresh.available) applyRecording(fresh, false);
        else renderMarkers();
      }
    }
    renderFileList();
  } catch {
    // The health poll below reports a server that isn't answering.
  }
}

async function refreshHealth(): Promise<string> {
  try {
    const health = await fetchHealth();
    ui.archive.textContent = health.share.readable ? "ARCHIVE ONLINE" : "ARCHIVE OFFLINE";
    ui.archive.title = health.share.detail;
    ui.archive.classList.toggle("alert", !health.share.readable);
    ui.archiveNote.textContent = health.share.readable ? "" : `ARCHIVE OFFLINE: ${health.share.detail}`;
    ui.archiveNote.hidden = health.share.readable;
    return health.share.readable ? "ONLINE" : "OFFLINE";
  } catch {
    ui.archive.textContent = "SERVER NOT RESPONDING";
    ui.archive.title = "";
    ui.archive.classList.add("alert");
    ui.archiveNote.textContent = "SERVER NOT RESPONDING: this page can't reach the app's server.";
    ui.archiveNote.hidden = false;
    return "NOT RESPONDING";
  }
}

function frame(): void {
  // Whatever goes wrong in one panel, the next frame still happens, so an error can't
  // freeze the whole screen. The browser's console still gets to see it.
  requestAnimationFrame(frame);
  const seconds = transport.seconds;
  if (following && transport.isRunning) waterfall.follow(seconds);
  waterfall.draw(seconds);
  // One pair is about 20 microseconds, so slow motion and stepping need more digits.
  ui.time.textContent = formatTime(seconds, transport.muted ? 6 : 3);

  const isIq = current?.kind === "iq";
  if (isIq) {
    // A frequency mark can change what the band is called as the playhead crosses it.
    const center = centerNow();
    if (center !== shownCenter) {
      shownCenter = center;
      waterfall.setCenterHz(center);
      updateTuning(false);
    }
    // In the clip that changes modes, the mode follows along.
    const segments = current!.example?.segments;
    if (segments) {
      const at = segments.findIndex((s) => seconds >= s.start_s && seconds < s.end_s);
      if (at >= 0 && at !== lastSegment) {
        lastSegment = at;
        if (segments[at].mode !== mode) setMode(segments[at].mode);
      }
    }
    const band = passband();
    closeUp.setTuning(offsetHz, center, band.lowHz, band.highHz);
    if (tab === "panels") closeUp.draw(seconds, transport.speed === 1 && !transport.stepped);
    views.draw(seconds, currentIqRate(), transport.speed, transport.stepped);
    if (tab === "path") {
      const source = engine.isPlaying ? engine.chunkFor(Math.floor(seconds)) : store.peek(Math.floor(seconds));
      signalPath.draw(
        seconds,
        current!.sample_rate ?? 48_000,
        current!.duration_s ?? 0,
        { offsetHz, mode, bwHz },
        currentIqRate(),
        source,
        (field, rate, start, count) => views.read(field, rate, start, count),
        center,
      );
    }
    updateAnalysis(seconds);
    transport.tick(views.dataReady);
    // Keep the seconds on either side ready, so stepping across a boundary is instant.
    if (!engine.isPlaying && views.dataReady) {
      const index = Math.floor(seconds);
      const last = Math.ceil(current!.duration_s ?? 0) - 1;
      if (index < last) store.get(index + 1);
      if (index > 0) store.get(index - 1);
    }
    if (store.refusal && !engine.isPlaying) {
      ui.status.textContent = store.refusal;
      ui.status.classList.add("alert");
    }
  } else if (current?.kind === "audio") {
    views.drawAudioOnly(seconds);
    updateAnalysis(seconds);
  }
  const label = transport.isRunning ? "PAUSE" : "PLAY";
  if (ui.play.textContent !== label) ui.play.textContent = label;
  ui.muted.hidden = !isIq || !transport.muted;
  if (transport.ended && ui.status.textContent !== "END OF RECORDING") ui.status.textContent = "END OF RECORDING";
  if (!scrubbing && document.activeElement !== ui.scrub) ui.scrub.value = String(seconds);
  else if (!scrubbing) ui.scrub.value = String(seconds);
}

async function start(): Promise<void> {
  renderModes();
  renderSpeeds();
  syncTuningControls();
  applyAnalysis();
  try {
    modes = await fetchModes();
    bwHz = modes.modes[mode]?.default_bw_hz ?? bwHz;
    syncTuningControls();
  } catch {
    // Modes are retried by reloading; the defaults above are still valid.
  }
  // The boot screen reports the real result of these first checks, whenever they land.
  let reportBoot: (facts: BootFacts) => void = () => {};
  runBoot(new Promise<BootFacts>((resolve) => (reportBoot = resolve)));
  const [, archive] = await Promise.all([refreshRecordings(), refreshHealth()]);
  reportBoot({ archive, recordings: `${recordings.filter((r) => !r.example).length} FOUND` });
  setInterval(() => void refreshRecordings(), 5000);
  setInterval(() => void refreshHealth(), 5000);
  requestAnimationFrame(frame);
}

// Test hook for the automated sync check; it exposes the same values the UI uses.
(window as unknown as { iqScopeDebug: unknown }).iqScopeDebug = {
  engine,
  transport,
  views,
  closeUp,
  signalPath,
  store,
  waterfall,
  rawPairs,
  colors,
  uploader,
  state: () => ({ offsetHz, mode, bwHz, current: current?.id }),
};

void start();
