// Sending recordings to the archive, one piece at a time. Each piece is hashed here and
// checked at the far end, and a failed piece is simply tried again, so an unreliable
// connection only slows an upload down. Nothing here touches playback.

import {
  cancelUpload,
  finishUpload,
  listUploads,
  sendUploadPiece,
  startUpload,
  UploadFailure,
  uploadState,
  type Recording,
  type UploadInfo,
} from "./api";

export type JobState = "waiting" | "checking" | "sending" | "retrying" | "done" | "failed" | "cancelled";

export interface Job {
  file: File;
  state: JobState;
  sent: number; // bytes the archive has confirmed, plus those of the piece on its way
  bytesPerSecond: number;
  message: string;
  resumed: boolean;
  uploadId: string | null;
  abort: AbortController;
}

const MAX_RETRY_DELAY_MS = 30_000;
const MAX_DAMAGED_TRIES = 5;

async function sha256Hex(data: BufferSource): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });

export class Uploader {
  jobs: Job[] = [];
  /** Uploads the archive still holds part of, from this visit or an earlier one. */
  unfinished: UploadInfo[] = [];
  onChange: () => void = () => {};
  onFinished: (recording: Recording) => void = () => {};

  private running = false;

  get busy(): boolean {
    return this.jobs.some((j) => ["waiting", "checking", "sending", "retrying"].includes(j.state));
  }

  add(files: Iterable<File>): void {
    for (const file of files) {
      const job: Job = {
        file,
        state: "waiting",
        sent: 0,
        bytesPerSecond: 0,
        message: "",
        resumed: false,
        uploadId: null,
        abort: new AbortController(),
      };
      if (!file.name.toLowerCase().endsWith(".wav")) {
        job.state = "failed";
        job.message = "ONLY .WAV FILES CAN BE UPLOADED";
      }
      this.jobs.push(job);
    }
    this.onChange();
    void this.run();
  }

  /** Stop this upload and remove what the archive has of it. */
  cancel(job: Job): void {
    job.state = "cancelled";
    job.message = "";
    job.abort.abort();
    if (job.uploadId) void this.discard(job.uploadId);
    this.onChange();
  }

  async discard(uploadId: string): Promise<void> {
    try {
      await cancelUpload(uploadId);
    } catch {
      // It's cleared automatically after two weeks if this didn't get through.
    }
    await this.refresh();
  }

  async refresh(): Promise<void> {
    try {
      this.unfinished = await listUploads();
    } catch {
      // The server isn't answering; the page's own status line already says so.
    }
    this.onChange();
  }

  /** Files go one at a time, leaving the connection free for listening. */
  private async run(): Promise<void> {
    if (this.running) return;
    this.running = true;
    for (let job; (job = this.jobs.find((j) => j.state === "waiting")); ) {
      await this.send(job);
    }
    this.running = false;
    await this.refresh();
  }

  private set(job: Job, state: JobState, message = ""): void {
    if (job.state === "cancelled") return;
    job.state = state;
    job.message = message;
    this.onChange();
  }

  /** Keep trying through dropped connections and archive outages, until told to stop. */
  private async persist<T>(job: Job, step: () => Promise<T>): Promise<T | null> {
    let delay = 1000;
    let damaged = 0;
    while (!job.abort.signal.aborted) {
      try {
        return await step();
      } catch (error) {
        if (job.abort.signal.aborted) return null;
        const failure = error as UploadFailure;
        if (error instanceof UploadFailure && !failure.retry) {
          this.set(job, "failed", failure.message);
          return null;
        }
        if (error instanceof UploadFailure && failure.status === 422 && ++damaged >= MAX_DAMAGED_TRIES) {
          this.set(job, "failed", `${failure.message} Gave up after ${damaged} tries.`);
          return null;
        }
        // The archive has told us where it's really up to, so carry on from there.
        if (error instanceof UploadFailure && failure.received !== null) return Promise.reject(error);
        this.set(job, "retrying", `${(error as Error).message || "CONNECTION LOST"} TRYING AGAIN...`);
        await sleep(delay, job.abort.signal);
        delay = Math.min(delay * 2, MAX_RETRY_DELAY_MS);
      }
    }
    return null;
  }

  /** Is this file the one that upload began with? Every piece so far has to match. */
  private async matches(job: Job, info: UploadInfo): Promise<boolean> {
    this.set(job, "checking", "CHECKING AGAINST WHAT THE ARCHIVE ALREADY HAS");
    let offset = 0;
    for (const expected of info.part_hashes) {
      if (job.abort.signal.aborted) return false;
      const end = Math.min(offset + info.part_bytes, job.file.size);
      const data = await job.file.slice(offset, end).arrayBuffer();
      if ((await sha256Hex(data)) !== expected) return false;
      offset = end;
      job.sent = offset;
      this.onChange();
    }
    return true;
  }

  private async send(job: Job): Promise<void> {
    try {
      await this.refresh();
      let info: UploadInfo | null = null;
      const candidate = this.unfinished.find((u) => u.original_name === job.file.name && u.size === job.file.size);
      if (candidate && (await this.matches(job, candidate))) {
        info = candidate;
        job.resumed = true;
      }
      if (job.abort.signal.aborted) return;
      if (!info) {
        job.sent = 0;
        info = await this.persist(job, () => startUpload(job.file.name, job.file.size));
        if (!info) return;
      }
      job.uploadId = info.id;
      if (job.state === "cancelled") {
        void this.discard(info.id);
        return;
      }
      await this.refresh();

      const hashes = [...info.part_hashes];
      let offset = info.received;
      const pieceBytes = info.part_bytes;
      while (offset < job.file.size) {
        const end = Math.min(offset + pieceBytes, job.file.size);
        const data = await job.file.slice(offset, end).arrayBuffer();
        const hash = await sha256Hex(data);
        const started = performance.now();
        const at = offset;
        const uploadId = info.id;
        this.set(job, "sending");
        let received: number | null;
        try {
          received = await this.persist(job, () =>
            sendUploadPiece(uploadId, at, hash, data, job.abort.signal, (bytes) => {
              job.sent = at + bytes;
              this.onChange();
            }),
          );
        } catch (error) {
          // Out of step with the archive, such as a reply lost after a piece landed, or
          // an earlier try that the server hasn't let go of yet.
          await sleep(500, job.abort.signal);
          const state = await this.persist(job, () => uploadState(uploadId));
          if (!state) return;
          hashes.length = 0;
          hashes.push(...state.part_hashes);
          offset = state.received;
          job.sent = offset;
          if (!(error instanceof UploadFailure)) throw error;
          continue;
        }
        if (received === null) return;
        hashes.push(hash);
        offset = end;
        job.sent = offset;
        const seconds = (performance.now() - started) / 1000;
        if (seconds > 0) job.bytesPerSecond = (end - at) / seconds;
        this.onChange();
      }

      this.set(job, "sending", "CHECKING THE FINISHED FILE");
      const pieces = await sha256Hex(new TextEncoder().encode(hashes.join("")));
      const uploadId = info.id;
      const recording = await this.persist(job, () => finishUpload(uploadId, pieces));
      if (!recording) return;
      this.set(job, "done", `SAVED AS uploads/${recording.name}`);
      this.onFinished(recording);
    } catch (error) {
      // Such as the file changing or moving on this computer while it was being read.
      this.set(job, "failed", (error as Error).message || "COULDN'T READ THE FILE");
    }
  }
}
