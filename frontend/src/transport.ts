// Where we are in the recording, at any speed.
//
// At 1x the audio clock decides, through the engine. Slow motion and stepping can't use
// it, because audio is meaningless at those speeds, so a visual clock takes over and
// the sound is muted.

import type { AudioEngine } from "./audio";

export const SPEEDS = [1, 0.1, 0.001] as const;
export type Speed = (typeof SPEEDS)[number];

// Tolerates rounding when turning seconds back into a pair number, even hours in.
const PAIR_EPSILON = 1e-6;

export const pairAt = (seconds: number, iqRate: number) => Math.floor(seconds * iqRate + PAIR_EPSILON);

export class Transport {
  speed: Speed = 1;
  stepped = false; // the user is moving pair by pair, so the sound stays off
  ended = false;

  private duration = 0;
  private running = false;
  private anchorSeconds = 0;
  private anchorAt = 0;
  private scrubbing = false;
  private resumeAfterScrub = false;

  constructor(private engine: AudioEngine) {}

  private now(): number {
    return performance.now() / 1000;
  }

  get seconds(): number {
    if (this.speed === 1) return this.engine.seconds;
    if (!this.running) return this.anchorSeconds;
    return Math.min(this.anchorSeconds + (this.now() - this.anchorAt) * this.speed, this.duration);
  }

  get isRunning(): boolean {
    return this.speed === 1 ? this.engine.isPlaying : this.running;
  }

  /** The sound is off by design, not because something went wrong. */
  get muted(): boolean {
    return this.speed !== 1 || this.stepped;
  }

  load(duration: number): void {
    this.duration = duration;
    this.speed = 1;
    this.stepped = false;
    this.ended = false;
    this.running = false;
    this.anchorSeconds = 0;
  }

  private anchor(seconds: number): void {
    this.anchorSeconds = Math.min(Math.max(seconds, 0), this.duration);
    this.anchorAt = this.now();
  }

  play(): void {
    this.stepped = false;
    this.ended = false;
    if (this.speed === 1) {
      void this.engine.play();
      return;
    }
    this.anchor(this.anchorSeconds >= this.duration ? 0 : this.anchorSeconds);
    this.running = true;
  }

  pause(): void {
    if (this.speed === 1) {
      this.engine.pause();
      return;
    }
    this.anchor(this.seconds);
    this.running = false;
  }

  toggle(): void {
    if (this.isRunning) this.pause();
    else this.play();
  }

  setSpeed(speed: Speed): void {
    if (speed === this.speed) return;
    const seconds = this.seconds;
    const wasRunning = this.isRunning;
    if (this.speed === 1) {
      this.engine.pause();
      this.engine.seek(seconds);
      this.anchor(seconds);
      this.speed = speed;
      this.running = wasRunning;
      return;
    }
    this.running = false;
    this.speed = speed;
    if (speed === 1) {
      this.engine.seek(seconds);
      if (wasRunning) void this.engine.play();
    } else {
      this.anchor(seconds);
      this.running = wasRunning;
    }
  }

  seek(seconds: number): void {
    if (this.speed === 1) {
      this.engine.seek(seconds);
    } else {
      this.anchor(seconds);
    }
    this.ended = false;
  }

  /** Move by exactly this many tuned pairs, and stop there. */
  step(pairs: number, iqRate: number): void {
    if (this.isRunning) this.pause();
    this.stepped = true;
    this.ended = false;
    const lastPair = Math.max(Math.ceil(this.duration * iqRate) - 1, 0);
    const pair = Math.min(Math.max(pairAt(this.seconds, iqRate) + pairs, 0), lastPair);
    this.seek(pair / iqRate);
  }

  /** Called each video frame. If the panels are waiting for data, slow motion waits too,
   *  so it never runs ahead of what's on screen. */
  tick(dataReady: boolean): void {
    if (this.speed === 1 || !this.running) return;
    if (!dataReady) {
      this.anchor(this.seconds);
      return;
    }
    if (this.seconds >= this.duration) {
      this.anchor(this.duration);
      this.running = false;
      this.ended = true;
    }
  }

  beginScrub(): void {
    if (this.scrubbing) return;
    this.scrubbing = true;
    if (this.speed === 1) {
      this.engine.beginScrub();
      return;
    }
    this.resumeAfterScrub = this.running;
    this.pause();
  }

  scrubTo(seconds: number): void {
    if (this.speed === 1) this.engine.scrubTo(seconds);
    else this.anchor(seconds);
  }

  endScrub(seconds: number): void {
    if (!this.scrubbing) return;
    this.scrubbing = false;
    this.ended = false;
    if (this.speed === 1) {
      this.engine.endScrub(seconds);
      return;
    }
    this.anchor(seconds);
    this.running = this.resumeAfterScrub;
  }
}
