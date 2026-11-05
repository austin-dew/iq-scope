// Spectrum slices from the server, kept in memory and fetched one at a time.

import { fetchSlice, type BandSlice } from "./api";

export class SliceCache {
  private id: string | null = null;
  private cache = new Map<string, BandSlice>();
  private wanted: { index: number; centerHz: number; spanHz: number }[] = [];
  private fetching = false;

  constructor(private keep: number) {}

  setRecording(id: string | null): void {
    this.id = id;
    this.cache.clear();
    this.wanted = [];
  }

  /** The slice if it's here; otherwise it's fetched, soonest for the latest request. */
  get(index: number, centerHz: number, spanHz: number): BandSlice | null {
    const found = this.cache.get(`${this.id}|${centerHz}|${spanHz}|${index}`);
    if (found || !this.id) return found ?? null;
    this.wanted = this.wanted.filter((w) => w.index !== index || w.centerHz !== centerHz || w.spanHz !== spanHz);
    this.wanted.push({ index, centerHz, spanHz });
    if (this.wanted.length > 4) this.wanted.shift(); // a tuner on the move leaves old requests behind
    void this.fetchNext();
    return null;
  }

  private async fetchNext(): Promise<void> {
    if (this.fetching) return;
    this.fetching = true;
    while (this.wanted.length > 0 && this.id) {
      const { index, centerHz, spanHz } = this.wanted.pop()!;
      const id = this.id;
      const key = `${id}|${centerHz}|${spanHz}|${index}`;
      if (this.cache.has(key)) continue;
      try {
        const slice = await fetchSlice(id, index, centerHz, spanHz);
        if (id !== this.id) continue;
        this.cache.set(key, slice);
        while (this.cache.size > this.keep) this.cache.delete(this.cache.keys().next().value!);
      } catch {
        // A brief network or share problem: leave the last plot up and try again shortly.
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
    this.fetching = false;
  }
}
