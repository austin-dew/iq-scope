// Chunks for the panels while the sound isn't playing: paused, stepping, or slow motion.
// Nearby seconds are kept in memory, so stepping backward is instant.

import { fetchChunk, RefusedError, type Chunk, type Tuning } from "./api";

const KEEP = 14;

export class ChunkStore {
  /** Why the current tuning can't be shown, when the server refused it. */
  refusal: string | null = null;

  private id: string | null = null;
  private tuning: Tuning | null = null;
  private cache = new Map<string, Chunk>();
  private wanted: number[] = [];
  private fetching = false;

  private key(index: number): string {
    const t = this.tuning!;
    return `${this.id}|${t.offsetHz}|${t.mode}|${t.bwHz}|${index}`;
  }

  setRecording(id: string | null): void {
    this.id = id;
    this.cache.clear();
    this.wanted = [];
    this.refusal = null;
  }

  setTuning(tuning: Tuning): void {
    this.tuning = { ...tuning };
    this.wanted = []; // only what the current tuning needs is worth fetching now
    this.refusal = null;
  }

  /** The chunk if it's here, without fetching anything. */
  peek(index: number): Chunk | null {
    if (!this.id || !this.tuning) return null;
    return this.cache.get(this.key(index)) ?? null;
  }

  /** The chunk if it's here; otherwise it's fetched, soonest for the latest request. */
  get(index: number): Chunk | null {
    const found = this.peek(index);
    if (found || !this.id || !this.tuning || this.refusal) return found;
    this.wanted = this.wanted.filter((i) => i !== index);
    this.wanted.push(index);
    void this.fetchNext();
    return null;
  }

  private async fetchNext(): Promise<void> {
    if (this.fetching) return;
    this.fetching = true;
    while (this.wanted.length > 0) {
      const index = this.wanted.pop()!;
      const id = this.id;
      const tuning = this.tuning;
      if (!id || !tuning) break;
      const key = this.key(index);
      if (this.cache.has(key)) continue;
      try {
        const chunk = await fetchChunk(id, index, tuning);
        this.cache.delete(key);
        this.cache.set(key, chunk);
        while (this.cache.size > KEEP) this.cache.delete(this.cache.keys().next().value!);
      } catch (error) {
        if (error instanceof RefusedError) {
          if (id === this.id && tuning === this.tuning) this.refusal = error.message;
          this.wanted = [];
        } else {
          // Probably a brief network or share problem. Asking again happens next frame.
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }
    }
    this.fetching = false;
  }
}
