// Playback. The audio thread's clock is the single source of truth for position.

import { fetchChunk, RefusedError, type Chunk, type Tuning } from "./api";

const RATE = 48_000;
const BUFFER_AHEAD_S = 3;
const SNIPPET_S = 0.14;
const SNIPPET_FADE_S = 0.01;
const LATENCY_SMOOTHING = 0.01;
const CLOCK_CORRECTION = 0.05;
const CLOCK_SNAP_S = 0.1;
const FIRST_FETCH_ESTIMATE_S = 0.5;
const VOLUME_SMOOTHING_S = 0.015; // eases each change, so moving the slider can't click

export type EngineStatus = "stopped" | "playing" | "paused" | "buffering" | "ended";

interface Clock {
  sample: number;
  time: number;
  starved: boolean;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export class AudioEngine {
  onStatus: (status: EngineStatus, message?: string) => void = () => {};

  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private output: GainNode | null = null;
  private starting: Promise<AudioContext> | null = null;
  private volume = 1;
  private recordingId: string | null = null;
  private chunkCount = 0;
  private duration = 0;
  private tuning: Tuning = { offsetHz: 0, mode: "am", bwHz: 6000 };

  private generation = 0;
  private playing = false;
  private position = 0; // seconds, used whenever the clock isn't running
  private clock: Clock | null = null;
  private latency: number | null = null;
  private display: { seconds: number; at: number } | null = null;
  private abort: AbortController | null = null;
  private queued = new Map<number, number>(); // chunk index -> tuning version it was made with
  private heard = new Map<number, Chunk>(); // the exact chunks given to the audio thread
  private tuningVersion = 0;
  private fetchEstimate = FIRST_FETCH_ESTIMATE_S;
  private wake: (() => void) | null = null;
  private status: EngineStatus = "stopped";

  private scrubbing = false;
  private resumeAfterScrub = false;
  private snippetSource: AudioBufferSourceNode | null = null;
  private snippetToken = 0;
  private snippetCache = new Map<string, Float32Array>();

  /** Starting audio takes a moment. Everyone waits on the same start, so nothing can
   *  use a half-built audio path, however quickly play and scrub follow each other. */
  private ensureContext(): Promise<AudioContext> {
    this.starting ??= this.startAudio();
    return this.starting;
  }

  private async startAudio(): Promise<AudioContext> {
    const context = new AudioContext({ sampleRate: RATE, latencyHint: "interactive" });
    this.output = context.createGain();
    this.output.gain.value = this.volume;
    this.output.connect(context.destination);
    await context.audioWorklet.addModule("/player-worklet.js");
    this.node = new AudioWorkletNode(context, "player", { outputChannelCount: [1] });
    this.node.connect(this.output);
    this.node.port.onmessage = (event) => this.onWorkletMessage(event.data);
    this.context = context;
    return context;
  }

  private onWorkletMessage(message: { type: string; generation: number } & Partial<Clock>): void {
    if (message.generation !== this.generation) return;
    if (message.type === "clock") {
      this.clock = { sample: message.sample!, time: message.time!, starved: message.starved! };
      if (this.playing) this.setStatus(message.starved ? "buffering" : "playing");
    } else if (message.type === "ended" && this.playing) {
      this.generation++; // stops the fetch loop
      this.position = this.duration;
      this.clock = null;
      this.playing = false;
      this.setStatus("ended");
    }
  }

  /** Set the playback volume. 0 is silent and 1 is the level the audio was made at. */
  setVolume(gain: number): void {
    this.volume = gain;
    if (this.output && this.context) {
      this.output.gain.setTargetAtTime(gain, this.context.currentTime, VOLUME_SMOOTHING_S);
    }
  }

  private setStatus(status: EngineStatus, message?: string): void {
    if (status === this.status && !message) return;
    this.status = status;
    this.onStatus(status, message);
  }

  /** The current position in seconds. While playing, this comes from the audio clock. */
  get seconds(): number {
    if (this.playing && this.clock && this.context) {
      const elapsed = this.clock.starved ? 0 : Math.max(this.context.currentTime - this.clock.time, 0);
      // Browsers report a latency that jumps around (24, 32, 40 ms) while the true delay
      // holds steady. Averaging it stops those jumps from shaking the display.
      const reported = this.context.outputLatency || 0;
      this.latency = this.latency === null ? reported : this.latency + (reported - this.latency) * LATENCY_SMOOTHING;
      const heard = this.clock.sample / RATE + elapsed - this.latency;
      return Math.min(Math.max(this.smooth(heard), this.position), this.duration);
    }
    return this.position;
  }

  /** The audio clock only updates in steps (21 ms is common), longer than a video frame.
   *  Between steps, run forward on the display's own timer, and keep gently correcting
   *  toward the audio clock, so motion is steady but can never drift away from the sound. */
  private smooth(heard: number): number {
    const now = performance.now() / 1000;
    if (this.display === null) {
      this.display = { seconds: heard, at: now };
      return heard;
    }
    if (this.clock?.starved) {
      // Waiting for audio: hold still, rather than stepping back to meet the clock.
      const held = Math.max(this.display.seconds, heard);
      this.display = { seconds: held, at: now };
      return held;
    }
    const predicted = this.display.seconds + (now - this.display.at);
    const error = heard - predicted;
    const seconds = Math.abs(error) > CLOCK_SNAP_S ? heard : predicted + error * CLOCK_CORRECTION;
    this.display = { seconds, at: now };
    return seconds;
  }

  /** Exposed for the sync test: position from the clock, before output latency. */
  get clockSeconds(): number | null {
    if (!this.clock || !this.context) return null;
    const elapsed = this.clock.starved ? 0 : Math.max(this.context.currentTime - this.clock.time, 0);
    return this.clock.sample / RATE + elapsed;
  }

  /** Exposed for the sync test: the output latency currently being allowed for. */
  get latencySeconds(): number {
    return this.latency ?? 0;
  }

  /** The chunk being heard for this second. The panels draw from this, so that during
   *  a retune they show what's really playing, not what's about to. */
  chunkFor(index: number): Chunk | null {
    return this.heard.get(index) ?? null;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  load(id: string, duration: number): void {
    this.halt();
    this.recordingId = id;
    this.duration = duration;
    this.chunkCount = Math.ceil(duration);
    this.position = 0;
    this.snippetCache.clear();
    this.setStatus("stopped");
  }

  unload(): void {
    this.halt();
    this.recordingId = null;
    this.duration = 0;
    this.position = 0;
    this.setStatus("stopped");
  }

  /** Stop everything in flight. The next start begins a new generation. */
  private halt(): void {
    this.generation++;
    this.playing = false;
    this.clock = null;
    this.abort?.abort();
    this.abort = null;
    void this.context?.suspend();
  }

  async play(): Promise<void> {
    if (!this.recordingId || this.playing) return;
    if (this.position >= this.duration) this.position = 0;
    const context = await this.ensureContext();
    await context.resume();
    this.startFrom(this.position);
  }

  pause(): void {
    if (!this.playing) return;
    const now = this.seconds;
    this.halt();
    this.position = now;
    this.setStatus("paused");
  }

  /** Move the playhead. Sound continues from there if it was already playing. */
  seek(seconds: number): void {
    const target = Math.min(Math.max(seconds, 0), this.duration);
    if (this.playing) this.startFrom(target);
    else this.position = target;
  }

  /** Change what's being demodulated, keeping the position.
   *
   *  Nothing already queued is thrown away. The fetch loop replaces queued seconds with
   *  newly tuned ones as fast as the server allows, so a slow server means hearing the
   *  old tuning for a moment longer, never silence. */
  retune(tuning: Tuning): void {
    this.tuning = { ...tuning };
    this.tuningVersion++;
    this.snippetCache.clear();
    this.wake?.();
  }

  private refuse(message: string): void {
    this.position = this.seconds;
    this.generation++;
    this.playing = false;
    this.clock = null;
    void this.context?.suspend();
    this.setStatus("stopped", message);
  }

  private startFrom(seconds: number): void {
    this.abort?.abort();
    this.abort = new AbortController();
    const generation = ++this.generation;
    this.queued.clear();
    this.heard.clear();
    this.position = seconds;
    this.clock = null;
    this.display = null;
    this.playing = true;
    this.node!.port.postMessage({ type: "reset", generation, sample: Math.round(seconds * RATE) });
    this.setStatus("buffering");
    void this.fetchLoop(generation, this.abort.signal);
  }

  /** Which second to fetch next, or null if everything is ready. One request runs at a
   *  time, so a burst of retunes can't pile up work on the server. */
  private pickChunk(): number | null {
    const now = this.clockSeconds ?? this.position;
    const current = Math.floor(now);
    for (const index of this.queued.keys()) if (index < current) this.queued.delete(index);
    // The panels look a moment into the past, so keep the second before this one.
    for (const index of this.heard.keys()) if (index < current - 1) this.heard.delete(index);
    const last = this.chunkCount - 1;

    // First, never run out. A request already running plus the one we'd start now can
    // take two fetch times, so everything within that reach must exist, in any tuning.
    // The slower the server is right now, the further ahead this looks.
    const secureUntil = Math.min(Math.floor(now + 2 * this.fetchEstimate + 0.5), last);
    for (let i = current; i <= secureUntil; i++) {
      if (!this.queued.has(i)) return i;
    }
    // Then, freshness: update the second the playhead will be in when the answer arrives,
    // and then the ones after it.
    const end = Math.min(Math.floor(now + Math.max(BUFFER_AHEAD_S, 2 * this.fetchEstimate + 1.5)), last);
    for (let i = Math.floor(now + this.fetchEstimate + 0.1); i <= end; i++) {
      const version = this.queued.get(i);
      if (version !== undefined && version !== this.tuningVersion) return i;
    }
    // Finally, top up the buffer.
    for (let i = current; i <= end; i++) {
      if (!this.queued.has(i)) return i;
    }
    return null;
  }

  private async idle(ms: number): Promise<void> {
    await new Promise<void>((resolve) => {
      const done = () => {
        clearTimeout(timer);
        this.wake = null;
        resolve();
      };
      const timer = setTimeout(done, ms);
      this.wake = done;
    });
  }

  private async fetchLoop(generation: number, signal: AbortSignal): Promise<void> {
    const id = this.recordingId!;
    while (generation === this.generation) {
      const index = this.pickChunk();
      if (index === null) {
        await this.idle(100);
        continue;
      }
      const version = this.tuningVersion;
      const started = performance.now();
      try {
        const chunk = await fetchChunk(id, index, this.tuning, signal);
        if (generation !== this.generation) return;
        this.fetchEstimate += ((performance.now() - started) / 1000 - this.fetchEstimate) * 0.3;
        const samples = chunk.arrays.audio.slice();
        this.node!.port.postMessage(
          { type: "chunk", generation, start: index * RATE, samples, last: index === this.chunkCount - 1 },
          [samples.buffer],
        );
        this.queued.set(index, version);
        this.heard.set(index, chunk);
      } catch (error) {
        if (generation !== this.generation || signal.aborted) return;
        if (error instanceof RefusedError) {
          this.refuse(error.message);
          return;
        }
        this.setStatus("buffering", `${(error as Error).message} Retrying...`);
        await sleep(2000);
      }
    }
  }

  // --- Scrubbing ---------------------------------------------------------------------

  beginScrub(): void {
    if (this.scrubbing) return;
    this.scrubbing = true;
    this.resumeAfterScrub = this.playing;
    const now = this.seconds;
    this.halt();
    this.position = now;
    void this.ensureContext().then((context) => context.resume());
  }

  /** Move while dragging: visuals follow at once, and a short snippet plays. */
  scrubTo(seconds: number): void {
    this.position = Math.min(Math.max(seconds, 0), this.duration);
    void this.playSnippet(this.position);
  }

  endScrub(seconds: number): void {
    if (!this.scrubbing) return;
    this.scrubbing = false;
    this.position = Math.min(Math.max(seconds, 0), this.duration);
    this.snippetToken++;
    this.snippetSource?.stop();
    this.snippetSource = null;
    if (this.resumeAfterScrub && this.position < this.duration) {
      this.startFrom(this.position);
    } else {
      void this.context?.suspend();
      this.setStatus(this.resumeAfterScrub ? "ended" : "paused");
    }
  }

  private async playSnippet(seconds: number): Promise<void> {
    const id = this.recordingId;
    if (!id) return;
    const token = ++this.snippetToken;
    const index = Math.min(Math.floor(seconds), this.chunkCount - 1);
    const key = `${id}:${index}`;
    let audio = this.snippetCache.get(key);
    if (!audio) {
      try {
        audio = (await fetchChunk(id, index, this.tuning)).arrays.audio;
      } catch {
        return; // the release will report anything that really is wrong
      }
      if (this.snippetCache.size > 8) this.snippetCache.clear();
      this.snippetCache.set(key, audio);
    }
    if (token !== this.snippetToken || !this.context) return;

    const start = Math.min(Math.round((seconds - index) * RATE), Math.max(audio.length - 1, 0));
    const count = Math.min(Math.round(SNIPPET_S * RATE), audio.length - start);
    if (count <= 0) return;
    const buffer = this.context.createBuffer(1, count, RATE);
    const data = buffer.getChannelData(0);
    data.set(audio.subarray(start, start + count));
    const fade = Math.min(Math.round(SNIPPET_FADE_S * RATE), Math.floor(count / 2));
    for (let i = 0; i < fade; i++) {
      data[i] *= i / fade;
      data[count - 1 - i] *= i / fade;
    }
    this.snippetSource?.stop();
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.output!);
    source.start();
    this.snippetSource = source;
  }
}
