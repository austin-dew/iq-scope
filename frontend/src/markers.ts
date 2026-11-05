// Markers: named moments, and frequency marks for recordings that were retuned.

export interface Marker {
  id: string;
  t: number;
  label: string;
  kind: "note" | "frequency";
  frequency_hz?: number;
}

/** The recording's center frequency at this moment.
 *
 *  A frequency mark says "from here on, the radio was tuned to X". A single recording can
 *  hold several, as when the radio was retuned while it was recording. */
export function centerAt(baseHz: number | null | undefined, markers: Marker[], seconds: number): number | null {
  let center = baseHz ?? null;
  for (const marker of markers) {
    if (marker.t > seconds) break;
    if (marker.kind === "frequency" && marker.frequency_hz) center = marker.frequency_hz;
  }
  return center;
}

export function newId(): string {
  return Math.random().toString(36).slice(2, 12);
}
