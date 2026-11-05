// The Signal Path: the same moment shown at each step from raw band to sound.

import type { BandSlice, Chunk, Tuning } from "./api";
import { Fft } from "./fft";
import { SliceCache } from "./slices";
import { pairAt } from "./transport";
import { colors, display, plotFont } from "./theme";

const FFT_SIZE = 2048;
const RANGE_DB = 70;
const AUDIO_RATE = 48_000;
const TRACE_SECONDS = 0.02;

export interface PathSummary {
  rawLevelAt: (hz: number) => number | null;
  tunedPeakHz: number | null;
  filteredPeakHz: number | null;
  filteredUpper: number;
  filteredLower: number;
  demodPeak: number | null;
  audioPeak: number | null;
  blankFrames: number; // frames that had nothing to show after the first data arrived
}

type Reader = (field: string, rate: number, start: number, count: number) => Float32Array;

interface Curve {
  firstHz: number;
  binHz: number;
  db: ArrayLike<number>;
  count: number;
}

const prepare = (canvas: HTMLCanvasElement) => {
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
};

function rowOf(slice: BandSlice, fraction: number): Curve {
  const row = Math.min(Math.floor(fraction * slice.rowsPerSecond), slice.rows - 1);
  return {
    firstHz: slice.firstHz,
    binHz: slice.binHz,
    db: slice.db.subarray(row * slice.bins, (row + 1) * slice.bins),
    count: slice.bins,
  };
}

export class SignalPath {
  summary: PathSummary | null = null;

  private slices = new SliceCache(16);
  private fft = new Fft(FFT_SIZE);
  private hann = Float64Array.from({ length: FFT_SIZE }, (_, k) => 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / FFT_SIZE));
  private norm = 1 / this.hann.reduce((a, b) => a + b, 0) ** 2;
  private re = new Float64Array(FFT_SIZE);
  private im = new Float64Array(FFT_SIZE);
  private filteredDb = new Float32Array(FFT_SIZE);
  // The last plots, which stay up until newer data arrives, so nothing blinks.
  private lastRaw: Curve | null = null;
  private lastTuned: Curve | null = null;
  private blankFrames = 0;

  constructor(
    private canvases: {
      raw: HTMLCanvasElement;
      tuned: HTMLCanvasElement;
      filtered: HTMLCanvasElement;
      demod: HTMLCanvasElement;
      sound: HTMLCanvasElement;
    },
    private formula: HTMLElement,
  ) {}

  setRecording(id: string | null): void {
    this.slices.setRecording(id);
    this.summary = null;
    this.lastRaw = null;
    this.lastTuned = null;
    this.blankFrames = 0;
  }

  draw(
    seconds: number,
    sampleRate: number,
    duration: number,
    tuning: Tuning,
    iqRate: number,
    chunk: Chunk | null,
    read: Reader,
    centerHz: number | null,
  ): void {
    const index = Math.max(Math.min(Math.floor(seconds), Math.ceil(duration) - 1), 0);
    const fraction = seconds - index;
    const offset = Math.round(tuning.offsetHz);

    // 1. Everything the radio saved.
    const rawSlice = this.slices.get(index, 0, sampleRate);
    if (rawSlice) this.lastRaw = rowOf(rawSlice, fraction);
    const raw = this.lastRaw;

    // 2. The same signals, moved so that the chosen one is at 0 Hz. Nothing is removed yet.
    const span = Math.min(iqRate, sampleRate);
    const limit = sampleRate / 2;
    const center = Math.max(Math.min(offset, limit), -limit);
    const tunedSlice = this.slices.get(index, center, span);
    if (tunedSlice) {
      const row = rowOf(tunedSlice, fraction);
      this.lastTuned = { ...row, firstHz: row.firstHz - center };
    }
    const tuned = this.lastTuned;

    // Ask for the next second before playback gets there, once this one is in hand.
    if (index + 1 < Math.ceil(duration)) {
      if (rawSlice) this.slices.get(index + 1, 0, sampleRate);
      if (tunedSlice) this.slices.get(index + 1, center, span);
    }
    if ((raw || tuned) && !(raw && tuned)) this.blankFrames++;

    // 3. What's left after filtering, from the very pairs the audio was made from.
    const filtered = chunk ? this.filteredCurve(seconds, iqRate, read) : null;

    const bandLabel = centerHz === null ? "offset" : `${((centerHz + offset) / 1e6).toFixed(3)} MHz`;
    this.drawSpectrum(this.canvases.raw, raw, -sampleRate / 2, sampleRate / 2, offset, `tuned to ${bandLabel}`, centerHz);
    this.drawSpectrum(
      this.canvases.tuned,
      tuned,
      -span / 2,
      span / 2,
      0,
      "yours is now at 0 Hz",
    );
    this.drawSpectrum(this.canvases.filtered, filtered, -span / 2, span / 2, 0, "the rest is gone");

    // 4 and 5. From here on it's a signal over time, not a spread of frequencies.
    const pair = pairAt(seconds, iqRate);
    const sample = Math.floor(pair / (iqRate / AUDIO_RATE));
    const count = Math.round(TRACE_SECONDS * AUDIO_RATE);
    const demod = read("demod", AUDIO_RATE, sample - count / 2, count);
    const audio = read("audio", AUDIO_RATE, sample - count / 2, count);
    const demodPeak = this.drawTrace(this.canvases.demod, demod, "before gain", true);
    const audioPeak = this.drawTrace(this.canvases.sound, audio, "what you hear", false);

    let filteredUpper = 0;
    let filteredLower = 0;
    let filteredPeakHz: number | null = null;
    if (filtered) {
      let best = -Infinity;
      const toneHz = tuning.mode === "cw" ? 700 : 0;
      for (let k = 0; k < filtered.count; k++) {
        const hz = filtered.firstHz + k * filtered.binHz - toneHz;
        const db = filtered.db[k];
        if (db > best) {
          best = db;
          filteredPeakHz = hz;
        }
        if (hz > 0) filteredUpper += 10 ** (db / 10);
        else if (hz < 0) filteredLower += 10 ** (db / 10);
      }
    }
    const peakOf = (curve: Curve | null, shift = 0) => {
      if (!curve) return null;
      let best = 0;
      for (let k = 1; k < curve.count; k++) if (curve.db[k] > curve.db[best]) best = k;
      return curve.firstHz + best * curve.binHz - shift;
    };
    this.summary = {
      rawLevelAt: (hz: number) => {
        if (!raw) return null;
        const k = Math.round((hz - raw.firstHz) / raw.binHz);
        return k >= 0 && k < raw.count ? raw.db[k] : null;
      },
      tunedPeakHz: peakOf(tuned),
      filteredPeakHz,
      filteredUpper,
      filteredLower,
      demodPeak,
      audioPeak,
      blankFrames: this.blankFrames,
    };
    this.formula.textContent = FORMULA[tuning.mode] ?? "";
  }

  private filteredCurve(seconds: number, iqRate: number, read: Reader): Curve {
    const pair = pairAt(seconds, iqRate);
    const i = read("i", iqRate, pair - FFT_SIZE / 2, FFT_SIZE);
    const q = read("q", iqRate, pair - FFT_SIZE / 2, FFT_SIZE);
    for (let k = 0; k < FFT_SIZE; k++) {
      this.re[k] = (Number.isNaN(i[k]) ? 0 : i[k]) * this.hann[k];
      this.im[k] = (Number.isNaN(q[k]) ? 0 : q[k]) * this.hann[k];
    }
    this.fft.transform(this.re, this.im);
    const half = FFT_SIZE / 2;
    for (let k = 0; k < FFT_SIZE; k++) {
      const from = (k + half) % FFT_SIZE; // lowest frequency first
      const power = (this.re[from] ** 2 + this.im[from] ** 2) * this.norm;
      this.filteredDb[k] = 10 * Math.log10(Math.max(power, 1e-14));
    }
    return { firstHz: -iqRate / 2, binHz: iqRate / FFT_SIZE, db: this.filteredDb, count: FFT_SIZE };
  }

  private drawSpectrum(
    canvas: HTMLCanvasElement,
    curve: Curve | null,
    lowHz: number,
    highHz: number,
    markHz: number,
    caption: string,
    absoluteHz: number | null = null,
  ): void {
    const { g, w, h } = prepare(canvas);
    if (!curve) {
      g.fillStyle = colors.dim;
      g.fillText("LOADING", 4, 4);
      return;
    }
    let top = -200;
    for (let k = 0; k < curve.count; k++) if (curve.db[k] > top) top = curve.db[k];
    top = Math.min(Math.ceil(top / 10) * 10, 0);
    const bottom = h - 14;
    const xOf = (hz: number) => ((hz - lowHz) / (highHz - lowHz)) * w;
    const yOf = (db: number) => Math.min(2 + ((top - db) / RANGE_DB) * (bottom - 2), bottom);

    g.strokeStyle = colors.grid;
    g.beginPath();
    for (let db = top; db >= top - RANGE_DB; db -= 20) {
      g.moveTo(0, Math.round(yOf(db)) + 0.5);
      g.lineTo(w, Math.round(yOf(db)) + 0.5);
    }
    g.stroke();

    g.beginPath();
    g.moveTo(xOf(curve.firstHz), bottom);
    for (let k = 0; k < curve.count; k++) g.lineTo(xOf(curve.firstHz + k * curve.binHz), yOf(curve.db[k]));
    g.lineTo(xOf(curve.firstHz + (curve.count - 1) * curve.binHz), bottom);
    g.closePath();
    g.fillStyle = colors.fill;
    g.fill();
    g.strokeStyle = colors.fg;
    g.stroke();

    g.strokeStyle = colors.accent;
    g.beginPath();
    g.moveTo(Math.round(xOf(markHz)) + 0.5, 0);
    g.lineTo(Math.round(xOf(markHz)) + 0.5, bottom);
    g.stroke();
    g.fillStyle = colors.dim;
    if (display.analysis) {
      g.textAlign = "left";
      for (let db = top; db >= top - RANGE_DB; db -= 20) g.fillText(`${db} dBFS`, 3, yOf(db) + 1);
      // Three frequencies along the bottom: real ones where the band's center is known.
      const text = (hz: number) => {
        if (absoluteHz !== null) return `${((absoluteHz + hz) / 1e6).toFixed(3)} MHz`;
        return `${hz > 0 ? "+" : ""}${Number((hz / 1000).toPrecision(4))} kHz`;
      };
      g.textAlign = "left";
      g.fillText(text(lowHz), 2, h - 12);
      g.textAlign = "center";
      g.fillText(text((lowHz + highHz) / 2), w / 2, h - 12);
      g.textAlign = "right";
      g.fillText(text(highHz), w - 2, h - 12);
      g.fillText(caption, w - 3, 2);
      g.textAlign = "left";
    } else {
      g.textAlign = "left";
      g.fillText(caption, 3, h - 12);
    }
  }

  /** A signal over time, scaled to fit. Returns its peak. */
  private drawTrace(canvas: HTMLCanvasElement, data: Float32Array, caption: string, autoscale: boolean): number | null {
    const { g, w, h } = prepare(canvas);
    let peak = 0;
    let any = false;
    for (let n = 0; n < data.length; n++) {
      if (Number.isNaN(data[n])) continue;
      any = true;
      peak = Math.max(peak, Math.abs(data[n]));
    }
    g.fillStyle = colors.dim;
    g.textAlign = "left";
    if (display.analysis) {
      const half = (TRACE_SECONDS * 1000) / 2;
      g.fillText(`-${half} ms`, 2, h - 12);
      g.textAlign = "center";
      g.fillText("0", w / 2, h - 12);
      g.textAlign = "right";
      g.fillText(`+${half} ms`, w - 2, h - 12);
      g.fillText(caption, w - 3, 14);
      g.textAlign = "left";
    } else {
      g.fillText(caption, 3, h - 12);
    }
    if (!any) {
      g.fillText("LOADING", 4, 4);
      return null;
    }
    const middle = (h - 14) / 2;
    g.strokeStyle = colors.grid;
    g.beginPath();
    g.moveTo(0, Math.round(middle) + 0.5);
    g.lineTo(w, Math.round(middle) + 0.5);
    g.stroke();
    const scale = (middle - 2) / (autoscale ? Math.max(peak, 1e-6) : 1);
    if (display.analysis) {
      const top = autoscale ? Math.max(peak, 1e-6) : 1;
      g.fillStyle = colors.dim;
      g.strokeStyle = colors.grid;
      g.beginPath();
      for (const f of [-1, -0.5, 0.5, 1]) {
        g.moveTo(0, Math.round(middle - f * top * scale) + 0.5);
        g.lineTo(w, Math.round(middle - f * top * scale) + 0.5);
      }
      g.moveTo(Math.round(w / 2) + 0.5, 0);
      g.lineTo(Math.round(w / 2) + 0.5, h - 14);
      g.stroke();
      for (const f of [1, 0, -1]) {
        const value = f * top;
        g.fillText(value === 0 ? "0" : `${value > 0 ? "+" : ""}${Number(value.toPrecision(2))}`, 3, Math.min(middle - value * scale + 1, h - 26));
      }
    }
    g.strokeStyle = colors.fg;
    g.beginPath();
    let open = false;
    for (let n = 0; n < data.length; n++) {
      if (Number.isNaN(data[n])) {
        open = false;
        continue;
      }
      const x = (n / (data.length - 1)) * w;
      const y = middle - data[n] * scale;
      if (open) g.lineTo(x, y);
      else g.moveTo(x, y);
      open = true;
    }
    g.stroke();
    g.fillStyle = colors.dim;
    g.textAlign = "right";
    g.fillText(autoscale ? `peak ${peak.toPrecision(2)}` : "+/-1", w - 3, 1);
    g.textAlign = "left";
    return peak;
  }
}

const FORMULA: Record<string, string> = {
  am: "AM: arrow length, minus its slow average (the carrier)",
  usb: "USB: keep the upper side, then take I",
  lsb: "LSB: keep the lower side, then take I",
  cw: "CW: narrow filter, lift to a 700 Hz tone, then take I",
  nfm: "NFM: how far the arrow turned since the last pair",
  wfm: "WFM: how far the arrow turned, then 5 pairs become 1 sample",
};
