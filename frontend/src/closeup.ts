// The close-up spectrum: a zoomed window locked onto one slice of the band.
//
// The band holds still and only the tuner moves. The plot comes from the raw recording,
// not the tuned audio, so moving the tuner within the slice needs no new data and the
// line can follow the cursor exactly. Frequencies here are hertz from the recording's center.

import type { BandSlice } from "./api";
import { SliceCache } from "./slices";
import { colors, display, plotFont } from "./theme";

const SPANS = [5_000, 10_000, 20_000, 50_000, 100_000, 250_000, 500_000];
const KEEP = 12;
const RANGE_DB = 70;
const SMOOTHING = 0.35;
const DRAG_SNAP_HZ = 10;
const FINE_SNAP_HZ = 1;
const FINE_FACTOR = 0.1;
const PLOT_TOP = 17;
const PLOT_BOTTOM_MARGIN = 15;


export interface CloseUpSummary {
  lockedCenterHz: number;
  spanHz: number;
  tunedHz: number;
  peakFromTuningHz: number | null; // strongest signal inside the filter, from the tuning
  peakDb: number | null;
  upperPower: number; // inside the filter, above and below the tuning
  lowerPower: number;
}

export class CloseUp {
  onTune: (offsetHz: number) => void = () => {};
  onSliceChange: (lowHz: number, highHz: number) => void = () => {};
  summary: CloseUpSummary | null = null;

  private duration = 0;
  private bandLimitHz = 1;
  private centerFrequencyHz: number | null = null;
  private tunedHz = 0;
  private filterLowHz = 0;
  private filterHighHz = 0;
  private spanHz = 10_000;
  private lockedCenterHz = 0;

  private slices = new SliceCache(KEEP);
  private smoothed: Float32Array | null = null;
  private smoothedFor = "";
  private topDb: number | null = null;
  private drag: { lastX: number; hz: number } | null = null;
  private hoverX: number | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", () => (this.drag = null));
    canvas.addEventListener("pointercancel", () => (this.drag = null));
    canvas.addEventListener("pointerleave", () => (this.hoverX = null));
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
  }

  setRecording(id: string | null, sampleRate: number, duration: number): void {
    this.duration = duration;
    this.bandLimitHz = sampleRate / 2;
    this.slices.setRecording(id);
    this.smoothed = null;
    this.summary = null;
    this.topDb = null;
    this.drag = null;
    this.spanHz = Math.min(10_000, sampleRate);
    this.lock(0);
  }

  private lock(centerHz: number): void {
    const half = this.spanHz / 2;
    const limit = Math.max(this.bandLimitHz - half, 0);
    this.lockedCenterHz = Math.round(Math.min(Math.max(centerHz, -limit), limit));
    this.smoothed = null;
    this.topDb = null;
    this.onSliceChange(this.lockedCenterHz - half, this.lockedCenterHz + half);
  }

  /** Move the slice to the tuner, for when you've finished in this area. */
  recenter(): void {
    this.lock(this.tunedHz);
  }

  /** Show more or less of the band around the tuner. */
  changeSpan(direction: 1 | -1): void {
    const allowed = SPANS.filter((s) => s <= this.bandLimitHz * 2);
    const index = allowed.indexOf(this.spanHz);
    const next = allowed[Math.min(Math.max(index + direction, 0), allowed.length - 1)];
    if (next === undefined || next === this.spanHz) return;
    this.spanHz = next;
    this.lock(this.tunedHz);
  }

  /** Follow the tuning. The slice stays locked unless the tuner has left it. */
  setTuning(tunedHz: number, centerFrequencyHz: number | null, filterLowHz: number, filterHighHz: number): void {
    this.tunedHz = tunedHz;
    this.centerFrequencyHz = centerFrequencyHz;
    this.filterLowHz = filterLowHz;
    this.filterHighHz = filterHighHz;

    // A signal wider than the window can't be seen, such as WFM, so widen it to fit.
    const needed = (filterHighHz - filterLowHz) * 1.25;
    if (needed > this.spanHz) {
      const allowed = SPANS.filter((s) => s <= this.bandLimitHz * 2);
      const fit = allowed.find((s) => s >= needed) ?? allowed[allowed.length - 1];
      if (fit !== undefined && fit !== this.spanHz) {
        this.spanHz = fit;
        this.lock(tunedHz);
        return;
      }
    }
    const half = this.spanHz / 2;
    if (tunedHz < this.lockedCenterHz - half || tunedHz > this.lockedCenterHz + half) this.lock(tunedHz);
  }

  // --- Data ------------------------------------------------------------------------------

  private get(index: number): BandSlice | null {
    return this.slices.get(index, this.lockedCenterHz, this.spanHz);
  }

  // --- Pointer ---------------------------------------------------------------------------

  private localX(e: PointerEvent): number {
    return e.clientX - this.canvas.getBoundingClientRect().left;
  }

  private hzAt(x: number): number {
    return this.lockedCenterHz + (x / this.canvas.clientWidth - 0.5) * this.spanHz;
  }

  /** Keep the tuner inside the slice, so a drag can never move the window under the cursor. */
  private clampToSlice(hz: number): number {
    const half = this.spanHz / 2;
    return Math.min(Math.max(hz, this.lockedCenterHz - half), this.lockedCenterHz + half);
  }

  private down(e: PointerEvent): void {
    this.canvas.setPointerCapture(e.pointerId);
    const x = this.localX(e);
    if (e.shiftKey) {
      // Fine tuning starts from where the tuner already is, with no jump to the cursor.
      this.drag = { lastX: x, hz: this.tunedHz };
      return;
    }
    const hz = this.clampToSlice(Math.round(this.hzAt(x) / DRAG_SNAP_HZ) * DRAG_SNAP_HZ);
    this.drag = { lastX: x, hz };
    this.onTune(hz);
  }

  private move(e: PointerEvent): void {
    const x = this.localX(e);
    this.hoverX = x;
    if (!this.drag) return;
    const dx = x - this.drag.lastX;
    if (dx === 0) return;
    this.drag.lastX = x;
    if (e.shiftKey) {
      const hzPerPixel = this.spanHz / this.canvas.clientWidth;
      this.drag.hz = this.clampToSlice(this.drag.hz + dx * hzPerPixel * FINE_FACTOR);
      this.onTune(Math.round(this.drag.hz / FINE_SNAP_HZ) * FINE_SNAP_HZ);
    } else {
      this.drag.hz = this.clampToSlice(this.hzAt(x));
      this.onTune(this.clampToSlice(Math.round(this.drag.hz / DRAG_SNAP_HZ) * DRAG_SNAP_HZ));
    }
  }

  private wheel(e: WheelEvent): void {
    e.preventDefault();
    const step = e.shiftKey ? FINE_SNAP_HZ : DRAG_SNAP_HZ;
    // Shift+scroll can arrive as sideways movement in some browsers.
    const movement = e.deltaY !== 0 ? e.deltaY : e.deltaX;
    if (movement !== 0) this.onTune(this.clampToSlice(this.tunedHz + (movement < 0 ? step : -step)));
  }

  private label(hz: number): string {
    if (this.centerFrequencyHz === null) return `${hz >= 0 ? "+" : ""}${Math.round(hz)} Hz`;
    return `${((this.centerFrequencyHz + hz) / 1000).toFixed(3)} kHz`;
  }

  // --- Drawing ---------------------------------------------------------------------------

  draw(seconds: number, smooth: boolean): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const g = this.canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = colors.plot;
    g.fillRect(0, 0, w, h);
    g.font = plotFont(11);
    g.textBaseline = "top";

    const index = Math.max(Math.min(Math.floor(seconds), Math.ceil(this.duration) - 1), 0);
    const slice = this.get(index);
    if (slice) {
      if (index + 1 < Math.ceil(this.duration)) this.get(index + 1);
      this.takeRow(slice, seconds - index, smooth);
    }

    const lowHz = this.lockedCenterHz - this.spanHz / 2;
    const xOf = (hz: number) => ((hz - lowHz) / this.spanHz) * w;
    const plotBottom = h - PLOT_BOTTOM_MARGIN;
    const top = this.topDb ?? 0;
    const yOf = (db: number) => PLOT_TOP + ((top - db) / RANGE_DB) * (plotBottom - PLOT_TOP);

    // Decibel lines.
    g.strokeStyle = colors.grid;
    g.lineWidth = 1;
    g.fillStyle = colors.dim;
    g.textAlign = "left";
    for (let db = top; db >= top - RANGE_DB; db -= 20) {
      const y = Math.round(yOf(db)) + 0.5;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(w, y);
      g.stroke();
      g.fillText(`${db} dBFS`, 3, y + 1);
    }

    // What the filter lets through, which travels with the tuner.
    const a = xOf(this.tunedHz + this.filterLowHz);
    const b = xOf(this.tunedHz + this.filterHighHz);
    g.fillStyle = colors.accentFill;
    g.fillRect(a, PLOT_TOP, b - a, plotBottom - PLOT_TOP);

    // The band itself.
    if (slice && this.smoothed) {
      g.beginPath();
      g.moveTo(xOf(slice.firstHz), plotBottom);
      for (let k = 0; k < slice.bins; k++) {
        g.lineTo(xOf(slice.firstHz + k * slice.binHz), Math.min(yOf(this.smoothed[k]), plotBottom));
      }
      g.lineTo(xOf(slice.firstHz + (slice.bins - 1) * slice.binHz), plotBottom);
      g.closePath();
      g.fillStyle = colors.fill;
      g.fill();
      g.strokeStyle = colors.fg;
      g.stroke();
    }

    // The tuner.
    const lineX = Math.round(xOf(this.tunedHz)) + 0.5;
    g.strokeStyle = colors.accent;
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(lineX, PLOT_TOP);
    g.lineTo(lineX, plotBottom);
    g.stroke();
    g.lineWidth = 1;

    // A fixed scale of real frequencies, as the band isn't moving.
    const step = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000].find((s) => s >= this.spanHz / 6) ?? 200000;
    const absolute = this.centerFrequencyHz ?? 0;
    g.fillStyle = colors.dim;
    g.textAlign = "center";
    for (let f = Math.ceil((lowHz + absolute) / step) * step; f <= lowHz + this.spanHz + absolute; f += step) {
      const text = this.centerFrequencyHz === null ? `${f >= 0 ? "+" : ""}${(f / 1000).toFixed(1)}k` : (f / 1000).toFixed(1);
      const half = g.measureText(text).width / 2 + 2;
      g.fillText(text, Math.min(Math.max(xOf(f - absolute), half), w - half), h - 13);
      if (display.analysis) {
        g.strokeStyle = colors.grid;
        g.beginPath();
        g.moveTo(Math.round(xOf(f - absolute)) + 0.5, PLOT_TOP);
        g.lineTo(Math.round(xOf(f - absolute)) + 0.5, plotBottom);
        g.stroke();
      }
    }

    this.updateSummary(slice);
    g.textAlign = "right";
    g.fillStyle = colors.bright;
    const peak = this.summary?.peakFromTuningHz;
    if (peak !== null && peak !== undefined) {
      const text = Math.abs(peak) >= 1000 ? `${(peak / 1000).toFixed(2)} kHz` : `${peak.toFixed(0)} Hz`;
      // In a narrow panel, drop words before anything can run under the buttons.
      const level = `${this.summary!.peakDb!.toFixed(1)} dBFS`;
      const room = w - 8 - ((this.canvas.previousElementSibling as HTMLElement | null)?.offsetWidth ?? 0);
      const full = `IN FILTER: PEAK ${text}  ${level}`;
      const shorter = `PEAK ${text}  ${level}`;
      g.fillText(g.measureText(full).width <= room ? full : g.measureText(shorter).width <= room ? shorter : level, w - 3, 2);
    }
    g.textAlign = "left";

    if (this.hoverX !== null) {
      const x = this.hoverX;
      g.strokeStyle = colors.dim;
      g.setLineDash([2, 3]);
      g.beginPath();
      g.moveTo(Math.round(x) + 0.5, PLOT_TOP);
      g.lineTo(Math.round(x) + 0.5, plotBottom);
      g.stroke();
      g.setLineDash([]);
      const hz = this.hzAt(x);
      const fromTuner = Math.round(hz - this.tunedHz);
      const text = display.analysis ? `${this.label(hz)}  ${fromTuner >= 0 ? "+" : ""}${fromTuner} Hz` : this.label(hz);
      const width = g.measureText(text).width + 8;
      const left = Math.min(x + 6, w - width);
      g.fillStyle = colors.scrim;
      g.fillRect(left, PLOT_TOP + 2, width, 15);
      g.fillStyle = colors.bright;
      g.fillText(text, left + 4, PLOT_TOP + 3);
    }
  }

  private takeRow(slice: BandSlice, fraction: number, smooth: boolean): void {
    const row = Math.min(Math.floor(fraction * slice.rowsPerSecond), slice.rows - 1);
    const data = slice.db.subarray(row * slice.bins, (row + 1) * slice.bins);
    const identity = `${this.lockedCenterHz}|${this.spanHz}|${slice.bins}`;
    if (!smooth || !this.smoothed || identity !== this.smoothedFor) {
      this.smoothed = data.slice();
      this.smoothedFor = identity;
    } else {
      for (let k = 0; k < slice.bins; k++) this.smoothed[k] += (data[k] - this.smoothed[k]) * SMOOTHING;
    }

    // A tidy top for the scale, which only moves when it needs to.
    let peak = -200;
    for (let k = 0; k < this.smoothed.length; k++) if (this.smoothed[k] > peak) peak = this.smoothed[k];
    const wanted = Math.min(Math.ceil(peak / 10) * 10, 0);
    if (this.topDb === null || wanted > this.topDb || wanted < this.topDb - 10) this.topDb = wanted;
  }

  private updateSummary(slice: BandSlice | null): void {
    let peakFromTuningHz: number | null = null;
    let peakDb: number | null = null;
    let upperPower = 0;
    let lowerPower = 0;
    if (slice && this.smoothed) {
      for (let k = 0; k < slice.bins; k++) {
        const fromTuning = slice.firstHz + k * slice.binHz - this.tunedHz;
        if (fromTuning < this.filterLowHz || fromTuning > this.filterHighHz) continue;
        const db = this.smoothed[k];
        if (peakDb === null || db > peakDb) {
          peakDb = db;
          peakFromTuningHz = fromTuning;
        }
        if (fromTuning > 0) upperPower += 10 ** (db / 10);
        else if (fromTuning < 0) lowerPower += 10 ** (db / 10);
      }
    }
    this.summary = {
      lockedCenterHz: this.lockedCenterHz,
      spanHz: this.spanHz,
      tunedHz: this.tunedHz,
      peakFromTuningHz,
      peakDb,
      upperPower,
      lowerPower,
    };
  }
}
