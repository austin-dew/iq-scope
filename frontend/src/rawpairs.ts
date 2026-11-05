// The newest raw pairs at the playhead, for Analysis Mode's constellation. Only one request
// is ever out, and none at all unless that view is switched on, so it can't crowd out sound.

import { fetchRaw, type RawBurst } from "./api";

const PAIRS = 2400;
const MIN_INTERVAL_MS = 66;
const RETRY_MS = 1000;

export class RawPairs {
  burst: RawBurst | null = null;
  requests = 0;

  private id: string | null = null;
  private fetching = false;
  private nextAt = 0;

  setRecording(id: string | null): void {
    this.id = id;
    this.burst = null;
  }

  /** Ask for the pairs ending at this one, unless they're already here or on the way. */
  want(pair: number): void {
    const end = this.burst ? this.burst.startPair + this.burst.i.length - 1 : -1;
    const id = this.id;
    if (!id || this.fetching || end === pair || performance.now() < this.nextAt) return;
    const start = Math.max(pair - PAIRS + 1, 0);
    this.fetching = true;
    this.requests++;
    this.nextAt = performance.now() + MIN_INTERVAL_MS;
    fetchRaw(id, start, pair - start + 1)
      .then((burst) => {
        if (id === this.id) this.burst = burst;
      })
      .catch(() => {
        this.nextAt = performance.now() + RETRY_MS;
      })
      .finally(() => {
        this.fetching = false;
      });
  }
}
