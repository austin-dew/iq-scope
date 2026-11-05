// The panels: constellation, phasor, I(t)/Q(t) traces and the audio waveform.
//
// They draw from whichever chunks are actually being heard, so what's on screen is what
// made the sound, even in the middle of a retune.

import type { Chunk, RawBurst } from "./api";
import { explain, pairValues, type PairValues } from "./pairsample";
import { pairAt } from "./transport";
import { colors, display, plotFont } from "./theme";

const TRAIL_BUCKETS = 8;

export type ChunkSource = (index: number) => Chunk | null;

interface Style {
  trailPairs: number;
  windowSeconds: number;
  markEachPair: boolean;
}

/** How much to draw at each speed: many pairs as a trail at 1x, individual ones when slow. */
function styleFor(speed: number, stepped: boolean): Style {
  if (stepped || speed <= 0.001) return { trailPairs: 96, windowSeconds: 0.0015, markEachPair: true };
  if (speed < 1) return { trailPairs: 480, windowSeconds: 0.01, markEachPair: false };
  return { trailPairs: 2400, windowSeconds: 0.02, markEachPair: false };
}

/** A value for an axis label, short and with its sign. */
const axisValue = (v: number): string => `${v > 0 ? "+" : ""}${v === 0 ? "0" : Number(v.toPrecision(2))}`;

/** Decibels relative to full scale, or a dash for silence. */
const dbfs = (magnitude: number): string =>
  magnitude > 0 ? `${(20 * Math.log10(magnitude)).toFixed(1)} dBFS` : "-- dBFS";

function prepare(canvas: HTMLCanvasElement): { g: CanvasRenderingContext2D; w: number; h: number } {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = colors.plot;
  g.fillRect(0, 0, w, h);
  g.font = plotFont(11);
  g.textBaseline = "top";
  return { g, w, h };
}

/** A plot range in tidy steps that only changes when it needs to, so plots don't jitter. */
function tidyRange(peak: number): number {
  const power = 10 ** Math.floor(Math.log10(Math.max(peak, 1e-7)));
  for (const m of [1, 2, 5, 10]) if (m * power >= peak) return m * power;
  return 10 * power;
}

export class IqViews {
  /** The pair most recently drawn, for the sync test. */
  lastPair: number | null = null;
  lastValues: PairValues | null = null;
  lastText = "";
  /** False when the data for the current position hasn't arrived yet. */
  dataReady = false;

  private range = 0.1;
  private peaks = new WeakMap<Chunk, number>();
  private lastTextAt = 0;
  private rawRange = 0.1;

  /** When set, the constellation shows the raw pairs from here instead of the tuned ones. */
  rawSource: (() => RawBurst | null) | null = null;

  constructor(
    private source: ChunkSource,
    private constellation: HTMLCanvasElement,
    private phasor: HTMLCanvasElement,
    private traces: HTMLCanvasElement,
    private waveform: HTMLCanvasElement,
    private panel: HTMLElement,
  ) {}

  /** Forget what was drawn, so one recording's values can't appear on another. */
  reset(): void {
    this.lastValues = null;
    this.lastPair = null;
  }

  private peakOf(chunk: Chunk): number {
    let peak = this.peaks.get(chunk);
    if (peak === undefined) {
      const { i, q } = chunk.arrays;
      let best = 0;
      for (let k = 0; k < i.length; k++) {
        const power = i[k] * i[k] + q[k] * q[k];
        if (power > best) best = power;
      }
      peak = Math.sqrt(best);
      this.peaks.set(chunk, peak);
    }
    return peak;
  }

  private updateRange(chunk: Chunk): void {
    const peak = this.peakOf(chunk);
    // Steps are at most 2.5x apart, so below 0.38 of the range a smaller one always fits.
    if (peak > this.range || peak < this.range * 0.38) this.range = tidyRange(peak * 1.05);
  }

  /** Read part of one array across chunk boundaries. Anything missing comes back as NaN. */
  read(field: string, rate: number, start: number, count: number): Float32Array {
    const out = new Float32Array(count).fill(NaN);
    let filled = 0;
    while (filled < count) {
      const position = start + filled;
      if (position < 0) {
        filled = Math.min(-start, count);
        continue;
      }
      const index = Math.floor(position / rate);
      const inChunk = position - index * rate;
      const data = this.source(index)?.arrays[field];
      const take = Math.min(count - filled, rate - inChunk);
      if (data && inChunk < data.length) {
        out.set(data.subarray(inChunk, Math.min(inChunk + take, data.length)), filled);
      }
      filled += take;
    }
    return out;
  }

  /** Audio recordings have no pairs behind them, so only the waveform applies. */
  drawAudioOnly(seconds: number): void {
    const sample = Math.floor(seconds * 48_000);
    const chunk = this.source(Math.floor(sample / 48_000));
    this.dataReady = chunk !== null;
    if (chunk) this.drawWaveform(sample, styleFor(1, false));
  }

  draw(seconds: number, iqRate: number, speed: number, stepped: boolean): void {
    let pair = pairAt(seconds, iqRate);
    let index = Math.floor(pair / iqRate);
    let chunk = this.source(index);
    if (!chunk && index > 0 && pair === index * iqRate) {
      // Exactly at the end of the recording there's no next second: show the last pair.
      const before = this.source(index - 1);
      if (before && before.arrays.i.length <= iqRate && pair >= (index - 1) * iqRate + before.arrays.i.length) {
        chunk = before;
        index -= 1;
        pair -= 1;
      }
    }
    this.dataReady = chunk !== null;
    if (!chunk) return;

    const k = Math.min(pair - index * iqRate, chunk.arrays.i.length - 1);
    pair = index * iqRate + k;
    const previous = k === 0 && index > 0 ? this.source(index - 1) : null;
    const values = pairValues(chunk, previous, k);
    if (!values) return;
    this.lastPair = pair;
    this.lastValues = values;
    this.updateRange(chunk);

    // On another tab there's nothing to draw into, though the values above still matter.
    if (this.constellation.clientWidth === 0) return;

    const style = styleFor(speed, stepped);
    this.drawConstellation(pair, iqRate, style);
    this.drawPhasor(values);
    this.drawTraces(pair, iqRate, style);
    this.drawWaveform(values.sample, style);

    // At full speed the numbers change too fast to read, so refresh them less often.
    const now = performance.now();
    if (speed < 1 || stepped || now - this.lastTextAt > 100) {
      this.lastText = explain(values, chunk);
      if (this.panel.textContent !== this.lastText) this.panel.textContent = this.lastText;
      this.lastTextAt = now;
    }
  }

  private axes(g: CanvasRenderingContext2D, w: number, h: number, range = this.range): { cx: number; cy: number; radius: number } {
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 16;
    g.strokeStyle = colors.grid;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(cx - radius, Math.round(cy) + 0.5);
    g.lineTo(cx + radius, Math.round(cy) + 0.5);
    g.moveTo(Math.round(cx) + 0.5, cy - radius);
    g.lineTo(Math.round(cx) + 0.5, cy + radius);
    g.stroke();
    g.beginPath();
    g.arc(cx, cy, radius, 0, Math.PI * 2);
    g.stroke();
    g.fillStyle = colors.dim;
    if (display.analysis) this.polarGrid(g, cx, cy, radius, range);
    g.textAlign = "left";
    g.fillText("I", cx + radius + 4, cy - 6);
    g.textAlign = "center";
    g.fillText("Q", cx, 1);
    g.textAlign = "left";
    // Analysis Mode's axes already carry this, and the corner is needed for the panel's title.
    if (!display.analysis) g.fillText(`+/-${range.toPrecision(2)}`, 3, h - 14);
    return { cx, cy, radius };
  }

  /** Grid lines at the half and full range, with numbers on each axis. */
  private polarGrid(g: CanvasRenderingContext2D, cx: number, cy: number, radius: number, range: number): void {
    g.strokeStyle = colors.grid;
    g.beginPath();
    for (const f of [-1, -0.5, 0.5, 1]) {
      g.moveTo(cx + f * radius, cy - radius);
      g.lineTo(cx + f * radius, cy + radius);
      g.moveTo(cx - radius, cy - f * radius);
      g.lineTo(cx + radius, cy - f * radius);
    }
    g.stroke();
    g.textAlign = "right";
    // The bottom of the axis is left bare, as the panel's title sits there.
    for (const f of [1, 0.5, -0.5]) g.fillText(axisValue(f * range), cx - 2, cy - f * radius + 1);
    g.textAlign = "center";
    for (const f of [-1, 1]) g.fillText(axisValue(f * range), cx + f * radius * 0.8, cy + 2);
  }

  /** Raw pairs, straight from the file: every signal in the band at once, before tuning. */
  private drawRawConstellation(burst: RawBurst | null, style: Style): void {
    const { g, w, h } = prepare(this.constellation);
    let peak = 0;
    if (burst) {
      for (let k = 0; k < burst.i.length; k++) peak = Math.max(peak, burst.i[k] ** 2 + burst.q[k] ** 2);
      peak = Math.sqrt(peak);
      if (peak > 0 && (peak > this.rawRange || peak < this.rawRange * 0.38)) this.rawRange = tidyRange(peak * 1.05);
    }
    const { cx, cy, radius } = this.axes(g, w, h, this.rawRange);
    g.fillStyle = colors.accent;
    g.textAlign = "left";
    g.fillText("RAW", 3, 1);
    if (!burst) return;
    const count = Math.min(style.trailPairs, burst.i.length);
    const from = burst.i.length - count;
    this.drawTrail(g, burst.i.subarray(from), burst.q.subarray(from), count, cx, cy, radius / this.rawRange, style);
  }

  private drawConstellation(pair: number, iqRate: number, style: Style): void {
    if (this.rawSource) {
      this.drawRawConstellation(this.rawSource(), style);
      return;
    }
    const { g, w, h } = prepare(this.constellation);
    const { cx, cy, radius } = this.axes(g, w, h);
    const count = Math.min(style.trailPairs, pair + 1);
    const start = pair - count + 1;
    const i = this.read("i", iqRate, start, count);
    const q = this.read("q", iqRate, start, count);
    this.drawTrail(g, i, q, count, cx, cy, radius / this.range, style);
  }

  private drawTrail(
    g: CanvasRenderingContext2D,
    i: Float32Array,
    q: Float32Array,
    count: number,
    cx: number,
    cy: number,
    scale: number,
    style: Style,
  ): void {
    // Older pairs fade out, like phosphor. Grouping by brightness keeps this fast.
    for (let bucket = 0; bucket < TRAIL_BUCKETS; bucket++) {
      const from = Math.floor((bucket * count) / TRAIL_BUCKETS);
      const to = Math.floor(((bucket + 1) * count) / TRAIL_BUCKETS);
      g.fillStyle = colors.fg;
      g.strokeStyle = colors.fg;
      g.globalAlpha = 0.1 + (0.8 * (bucket + 1)) / TRAIL_BUCKETS;
      if (style.markEachPair) {
        g.lineWidth = 1;
        g.beginPath();
        for (let n = Math.max(from - 1, 0); n < to; n++) {
          if (Number.isNaN(i[n])) continue;
          const x = cx + i[n] * scale;
          const y = cy - q[n] * scale;
          if (n === Math.max(from - 1, 0)) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.stroke();
        for (let n = from; n < to; n++) {
          if (!Number.isNaN(i[n])) g.fillRect(cx + i[n] * scale - 1.5, cy - q[n] * scale - 1.5, 3, 3);
        }
      } else {
        g.beginPath();
        for (let n = from; n < to; n++) {
          if (!Number.isNaN(i[n])) g.rect(cx + i[n] * scale, cy - q[n] * scale, 1.5, 1.5);
        }
        g.fill();
      }
    }
    g.globalAlpha = 1;

    // The current pair is the brightest thing on the plot.
    const x = cx + i[count - 1] * scale;
    const y = cy - q[count - 1] * scale;
    g.fillStyle = colors.white;
    g.beginPath();
    g.arc(x, y, 3.5, 0, Math.PI * 2);
    g.fill();
  }

  private drawPhasor(v: PairValues): void {
    const { g, w, h } = prepare(this.phasor);
    const { cx, cy, radius } = this.axes(g, w, h);
    const scale = radius / this.range;
    const x = cx + v.i * scale;
    const y = cy - v.q * scale;
    g.strokeStyle = colors.white;
    g.fillStyle = colors.white;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(cx, cy);
    g.lineTo(x, y);
    g.stroke();
    // An arrowhead, if there's room for one.
    const length = Math.hypot(x - cx, y - cy);
    if (length > 8) {
      const ux = (x - cx) / length;
      const uy = (y - cy) / length;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x - ux * 8 - uy * 4, y - uy * 8 + ux * 4);
      g.lineTo(x - ux * 8 + uy * 4, y - uy * 8 - ux * 4);
      g.closePath();
      g.fill();
    }
    g.fillStyle = colors.bright;
    g.textAlign = "right";
    g.fillText(`LEN ${v.length.toFixed(4)}`, w - 3, 1);
    g.fillText(`ANG ${((v.angleRad * 180) / Math.PI).toFixed(1)}`, w - 3, 14);
    if (display.analysis) g.fillText(dbfs(v.length), w - 3, 27);
    g.textAlign = "left";
  }

  private plotLine(
    g: CanvasRenderingContext2D,
    data: Float32Array,
    w: number,
    h: number,
    range: number,
    marks: boolean,
  ): void {
    const mid = h / 2;
    const yScale = (h / 2 - 6) / range;
    const dx = w / Math.max(data.length - 1, 1);
    g.beginPath();
    let open = false;
    for (let n = 0; n < data.length; n++) {
      if (Number.isNaN(data[n])) {
        open = false;
        continue;
      }
      const x = n * dx;
      const y = mid - data[n] * yScale;
      if (open) g.lineTo(x, y);
      else g.moveTo(x, y);
      open = true;
    }
    g.stroke();
    if (marks) {
      for (let n = 0; n < data.length; n++) {
        if (!Number.isNaN(data[n])) g.fillRect(n * dx - 1.5, mid - data[n] * yScale - 1.5, 3, 3);
      }
    }
  }

  /** Numbers and grid for a trace: amplitude up the side, time from the cursor along the bottom. */
  private traceGrid(g: CanvasRenderingContext2D, w: number, h: number, range: number, windowSeconds: number): void {
    const yScale = (h / 2 - 6) / range;
    g.strokeStyle = colors.grid;
    g.lineWidth = 1;
    g.beginPath();
    for (const f of [-1, -0.5, 0.5, 1]) {
      g.moveTo(0, Math.round(h / 2 - f * range * yScale) + 0.5);
      g.lineTo(w, Math.round(h / 2 - f * range * yScale) + 0.5);
    }
    for (const f of [-0.5, 0.5]) {
      g.moveTo(Math.round(w / 2 + (f * w) / 2) + 0.5, 0);
      g.lineTo(Math.round(w / 2 + (f * w) / 2) + 0.5, h);
    }
    g.stroke();
    g.fillStyle = colors.dim;
    g.textAlign = "left";
    for (const f of [1, 0.5, 0, -0.5, -1]) g.fillText(axisValue(f * range), 3, Math.min(h / 2 - f * range * yScale - 5, h - 13));
    // The window runs from half its length before the cursor to half after, edge to edge.
    const halfMs = (windowSeconds * 1000) / 2;
    g.textAlign = "center";
    for (const f of [-0.5, 0, 0.5]) {
      g.fillText(f === 0 ? "0" : `${f > 0 ? "+" : ""}${Number((f * halfMs).toPrecision(3))} ms`, w / 2 + (f * w) / 2, h - 12);
    }
    g.textAlign = "left";
  }

  private centerLines(g: CanvasRenderingContext2D, w: number, h: number): void {
    g.strokeStyle = colors.grid;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, Math.round(h / 2) + 0.5);
    g.lineTo(w, Math.round(h / 2) + 0.5);
    g.stroke();
    g.strokeStyle = colors.accent;
    g.beginPath();
    g.moveTo(Math.round(w / 2) + 0.5, 0);
    g.lineTo(Math.round(w / 2) + 0.5, h);
    g.stroke();
  }

  private drawTraces(pair: number, iqRate: number, style: Style): void {
    const { g, w, h } = prepare(this.traces);
    if (display.analysis) this.traceGrid(g, w, h, this.range, style.windowSeconds);
    this.centerLines(g, w, h);
    const count = Math.max(Math.round(style.windowSeconds * iqRate), 2) | 1;
    const start = pair - (count - 1) / 2;
    const i = this.read("i", iqRate, start, count);
    const q = this.read("q", iqRate, start, count);

    // Told apart by brightness and dashes as well as color.
    g.strokeStyle = colors.dim;
    g.fillStyle = colors.dim;
    g.lineWidth = 1;
    g.setLineDash([4, 3]);
    this.plotLine(g, q, w, h, this.range, style.markEachPair);
    g.setLineDash([]);
    g.strokeStyle = colors.bright;
    g.fillStyle = colors.bright;
    g.lineWidth = 1.25;
    this.plotLine(g, i, w, h, this.range, style.markEachPair);

    g.fillStyle = colors.bright;
    g.textAlign = "left";
    const legend = display.analysis ? 56 : 3; // clear of the amplitude numbers
    g.fillText("I (solid)", legend, 1);
    g.fillStyle = colors.dim;
    g.fillText("Q (dashed)", legend + 67, 1);
    g.textAlign = "right";
    if (!display.analysis) g.fillText(`${(style.windowSeconds * 1000).toFixed(1)} ms`, w - 3, 1);
    g.textAlign = "left";
  }

  private drawWaveform(sample: number, style: Style): void {
    const { g, w, h } = prepare(this.waveform);
    if (display.analysis) this.traceGrid(g, w, h, 1, style.windowSeconds);
    this.centerLines(g, w, h);
    const rate = 48_000;
    const count = Math.max(Math.round(style.windowSeconds * rate), 2) | 1;
    const start = sample - (count - 1) / 2;
    const audio = this.read("audio", rate, start, count);
    g.strokeStyle = colors.fg;
    g.fillStyle = colors.fg;
    g.lineWidth = 1.25;
    this.plotLine(g, audio, w, h, 1, style.markEachPair);

    // The sample from the PAIR -> SAMPLE panel.
    const value = audio[(count - 1) / 2];
    if (!Number.isNaN(value)) {
      g.fillStyle = colors.white;
      g.beginPath();
      g.arc(w / 2, h / 2 - value * (h / 2 - 6), 3.5, 0, Math.PI * 2);
      g.fill();
    }
    g.fillStyle = colors.dim;
    g.textAlign = "left";
    g.fillText("AUDIO +/-1", display.analysis ? 40 : 3, 1);
  }
}
