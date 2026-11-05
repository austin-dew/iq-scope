// The waterfall: frequency across, time down, drawn from tiles on the server's grid.

import { fetchTile, type Recording, type WaterfallInfo } from "./api";
import { colors, display, phosphor, plotFont } from "./theme";

const TILE_CACHE = 160;
const MAX_FETCHES = 3;
const RULER_LEFT = 62;
const RULER_TOP = 20;
const CLICK_SLOP_PX = 4;
const TUNER_GRAB_PX = 8;
const TUNER_GRAB_TOUCH_PX = 20;
const MIN_TIME_SPAN_S = 0.25;
const BYTES_PER_3DB = (3 * 255) / 130; // each frequency zoom halves the noise in a bin

interface Tile {
  bytes: Uint8Array;
  canvas: HTMLCanvasElement | null;
  paletteVersion: number;
}

export interface TuneOverlay {
  offsetHz: number;
  lowHz: number; // edges of the passband relative to offsetHz
  highHz: number;
}

export interface Cursor {
  hz: number; // from the recording's center
  seconds: number;
}

export class Waterfall {
  onCursors: (cursors: Cursor[]) => void = () => {};
  /** Plain clicks place cursors instead of tuning, for screens with no Shift key. */
  placingCursors = false;
  cursors: Cursor[] = [];
  /** The difference between the cursors, shown beside B so it can be read on any tab. */
  cursorNote = "";

  onTune: (offsetHz: number) => void = () => {};
  onSeek: (seconds: number) => void = () => {};
  onPanByUser: () => void = () => {};

  private context: CanvasRenderingContext2D;
  private recording: Recording | null = null;
  private info: WaterfallInfo | null = null;
  private duration = 0;
  private centerHz: number | null = null;

  // The view: what the plot area currently shows.
  private timeStart = 0;
  private timeSpan = 1;
  private lowHz = 0;
  private spanHz = 1;

  private tiles = new Map<string, Tile>();
  private queue: { key: string; level: number; z: number; x: number; t: number }[] = [];
  private loading = new Set<string>();
  private failed = new Set<string>();
  private abort: AbortController | null = null;
  private floorByte = 90;
  private ceilingByte = 200;
  private paletteVersion = 1;
  private autoRangeDone = false;
  private autoFloor = 90;
  private autoCeiling = 200;
  private overlay: TuneOverlay | null = null;
  private slice: { lowHz: number; highHz: number } | null = null;
  private markers: { t: number; label: string; kind: string }[] = [];
  private hover: { x: number; y: number } | null = null;
  private drag: {
    x: number;
    y: number;
    timeStart: number;
    lowHz: number;
    moved: boolean;
    tunerGrabHz: number | null; // set when the press landed on the tuner
  } | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    this.context = canvas.getContext("2d")!;
    canvas.addEventListener("pointerdown", (e) => this.pointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.pointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.pointerUp(e));
    canvas.addEventListener("pointerleave", () => (this.hover = null));
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
  }

  load(recording: Recording | null): void {
    this.abort?.abort();
    this.abort = new AbortController();
    this.tiles.clear();
    this.queue = [];
    this.loading.clear();
    this.failed.clear();
    this.autoRangeDone = false;
    this.autoFloor = 90;
    this.autoCeiling = 200;
    this.recording = recording;
    this.info = recording?.waterfall ?? null;
    this.duration = recording?.duration_s ?? 0;
    this.centerHz = recording?.frequency_hz ?? null;
    this.clearCursors();
    this.fit();
  }

  setCenterHz(hz: number | null): void {
    this.centerHz = hz;
  }

  setMarkers(markers: { t: number; label: string; kind: string }[]): void {
    this.markers = markers;
  }

  /** Mark the part of the band the close-up window is locked onto. */
  setSlice(lowHz: number, highHz: number): void {
    this.slice = { lowHz, highHz };
  }

  setOverlay(overlay: TuneOverlay | null): void {
    this.overlay = overlay;
  }

  setContrast(floorByte: number, ceilingByte: number): void {
    this.floorByte = floorByte;
    this.ceilingByte = Math.max(ceilingByte, floorByte + 1);
    this.paletteVersion++;
  }

  /** The ramp's colors changed, so every tile needs painting again. */
  repaint(): void {
    this.paletteVersion++;
  }

  clearCursors(): void {
    this.cursors = [];
    this.onCursors(this.cursors);
  }

  /** The first click sets A, the second B, and a third starts again with a new A. */
  private placeCursor(x: number, y: number): void {
    const cursor = {
      hz: Math.round(this.xToHz(x)),
      seconds: Math.min(Math.max(this.yToSeconds(y), 0), this.duration),
    };
    this.cursors = this.cursors.length >= 2 ? [cursor] : [...this.cursors, cursor];
    this.onCursors(this.cursors);
  }

  private wantsCursor(e: PointerEvent): boolean {
    return display.analysis && (e.shiftKey || this.placingCursors);
  }

  /** Back to the contrast chosen automatically when this recording was opened. */
  resetContrast(): void {
    this.setContrast(this.autoFloor, this.autoCeiling);
    this.onContrastAuto(this.autoFloor, this.autoCeiling);
  }

  onContrastAuto: (floor: number, ceiling: number) => void = () => {};

  fit(): void {
    if (!this.info) return;
    this.timeStart = 0;
    this.timeSpan = Math.max(this.duration, MIN_TIME_SPAN_S);
    this.lowHz = this.info.low_hz;
    this.spanHz = this.info.span_hz;
  }

  zoomFrequency(factor: number, aroundHz?: number): void {
    if (!this.info) return;
    const minSpan = this.info.span_hz / 2 ** (this.info.max_zoom + 3);
    const span = Math.min(Math.max(this.spanHz * factor, minSpan), this.info.span_hz);
    const anchor = aroundHz ?? this.lowHz + this.spanHz / 2;
    const fraction = (anchor - this.lowHz) / this.spanHz;
    this.lowHz = anchor - fraction * span;
    this.spanHz = span;
    this.clampView();
  }

  zoomTime(factor: number, aroundS?: number): void {
    const span = Math.min(Math.max(this.timeSpan * factor, MIN_TIME_SPAN_S), Math.max(this.duration, MIN_TIME_SPAN_S));
    const anchor = aroundS ?? this.timeStart + this.timeSpan / 2;
    const fraction = (anchor - this.timeStart) / this.timeSpan;
    this.timeStart = anchor - fraction * span;
    this.timeSpan = span;
    this.clampView();
  }

  private clampView(): void {
    if (!this.info) return;
    const fHigh = this.info.low_hz + this.info.span_hz;
    this.lowHz = Math.min(Math.max(this.lowHz, this.info.low_hz), fHigh - this.spanHz);
    this.timeStart = Math.min(Math.max(this.timeStart, 0), Math.max(this.duration - this.timeSpan, 0));
  }

  /** Scroll so the playhead stays visible during playback. */
  follow(seconds: number): void {
    if (this.drag && this.drag.tunerGrabHz === null) return;
    if (seconds < this.timeStart || seconds > this.timeStart + this.timeSpan * 0.9) {
      this.timeStart = seconds - this.timeSpan * 0.1;
      this.clampView();
    }
  }

  /** Bring a frequency into view at the current zoom, for example after typing one. */
  reveal(offsetHz: number): void {
    if (offsetHz < this.lowHz || offsetHz > this.lowHz + this.spanHz) {
      this.lowHz = offsetHz - this.spanHz / 2;
      this.clampView();
    }
  }

  // --- Coordinates -------------------------------------------------------------------

  private get plotWidth(): number {
    return Math.max(this.canvas.clientWidth - RULER_LEFT, 1);
  }

  private get plotHeight(): number {
    return Math.max(this.canvas.clientHeight - RULER_TOP, 1);
  }

  private xToHz(x: number): number {
    return this.lowHz + ((x - RULER_LEFT) / this.plotWidth) * this.spanHz;
  }

  private hzToX(hz: number): number {
    return RULER_LEFT + ((hz - this.lowHz) / this.spanHz) * this.plotWidth;
  }

  private yToSeconds(y: number): number {
    return this.timeStart + ((y - RULER_TOP) / this.plotHeight) * this.timeSpan;
  }

  private secondsToY(seconds: number): number {
    return RULER_TOP + ((seconds - this.timeStart) / this.timeSpan) * this.plotHeight;
  }

  // --- Pointer -----------------------------------------------------------------------

  private localPoint(e: PointerEvent | WheelEvent): { x: number; y: number } {
    const box = this.canvas.getBoundingClientRect();
    return { x: e.clientX - box.left, y: e.clientY - box.top };
  }

  private pointerDown(e: PointerEvent): void {
    if (!this.info) return;
    const p = this.localPoint(e);
    this.canvas.setPointerCapture(e.pointerId);
    const onTuner =
      !this.wantsCursor(e) && this.isOnTuner(p.x, p.y, e.pointerType === "touch" ? TUNER_GRAB_TOUCH_PX : TUNER_GRAB_PX);
    this.drag = {
      x: p.x,
      y: p.y,
      timeStart: this.timeStart,
      lowHz: this.lowHz,
      moved: false,
      // Keeps the tuner from jumping to the cursor when grabbed slightly off-center.
      tunerGrabHz: onTuner ? this.overlay!.offsetHz - this.xToHz(p.x) : null,
    };
  }

  /** Is this point on the tuning line or inside its shaded passband? */
  private isOnTuner(x: number, y: number, grabPx: number): boolean {
    if (!this.overlay || this.recording?.kind !== "iq" || x < RULER_LEFT || y < RULER_TOP) return false;
    const { offsetHz, lowHz, highHz } = this.overlay;
    const line = this.hzToX(offsetHz);
    const edgeA = this.hzToX(offsetHz + lowHz);
    const edgeB = this.hzToX(offsetHz + highHz);
    return Math.abs(x - line) <= grabPx || (x >= Math.min(edgeA, edgeB) && x <= Math.max(edgeA, edgeB));
  }

  private pointerMove(e: PointerEvent): void {
    const p = this.localPoint(e);
    this.hover = p;
    if (!this.drag) {
      this.canvas.classList.toggle("over-tuner", this.isOnTuner(p.x, p.y, TUNER_GRAB_PX));
      return;
    }
    const dx = p.x - this.drag.x;
    const dy = p.y - this.drag.y;
    if (!this.drag.moved && Math.hypot(dx, dy) <= CLICK_SLOP_PX) return;
    if (this.drag.x < RULER_LEFT) {
      this.drag.moved = true;
      this.onSeek(Math.min(Math.max(this.yToSeconds(p.y), 0), this.duration));
      return;
    }
    if (this.drag.tunerGrabHz !== null) {
      // The tuner follows the cursor left and right, staying within what's on screen.
      this.drag.moved = true;
      const x = Math.min(Math.max(p.x, RULER_LEFT), this.canvas.clientWidth);
      this.onTune(this.snap(this.xToHz(x) + this.drag.tunerGrabHz));
      return;
    }
    this.drag.moved = true;
    this.onPanByUser();
    this.lowHz = this.drag.lowHz - (dx / this.plotWidth) * this.spanHz;
    this.timeStart = this.drag.timeStart - (dy / this.plotHeight) * this.timeSpan;
    this.clampView();
  }

  private pointerUp(e: PointerEvent): void {
    const drag = this.drag;
    this.drag = null;
    if (!drag || drag.moved || !this.info) return;
    const p = this.localPoint(e);
    if (p.y < RULER_TOP) return;
    if (p.x < RULER_LEFT) {
      this.onSeek(Math.min(Math.max(this.yToSeconds(p.y), 0), this.duration));
      return;
    }
    if (this.wantsCursor(e)) {
      this.placeCursor(p.x, p.y);
      return;
    }
    if (this.recording?.kind === "iq") this.onTune(this.snap(this.xToHz(p.x)));
  }

  /** Round a click to a step suited to the zoom, so frequencies stay readable. */
  private snap(hz: number): number {
    const hzPerPixel = this.spanHz / this.plotWidth;
    const step = hzPerPixel > 500 ? 1000 : hzPerPixel > 50 ? 100 : 10;
    return Math.round(hz / step) * step;
  }

  private wheel(e: WheelEvent): void {
    if (!this.info) return;
    e.preventDefault();
    const p = this.localPoint(e);
    const factor = e.deltaY > 0 ? 1.25 : 0.8;
    if (e.shiftKey || p.x < RULER_LEFT) {
      this.onPanByUser();
      this.zoomTime(factor, this.yToSeconds(p.y));
    } else {
      this.zoomFrequency(factor, this.xToHz(p.x));
    }
  }

  // --- Tiles -------------------------------------------------------------------------

  private chooseLevel(): number {
    const secondsPerPixel = this.timeSpan / this.plotHeight;
    const level = Math.ceil(Math.log2(Math.max(secondsPerPixel / this.info!.base_row_s, 1)));
    return Math.min(level, this.info!.max_level);
  }

  private chooseZoom(): number {
    const info = this.info!;
    const hzPerPixel = this.spanHz / this.plotWidth;
    const fullBinHz = info.span_hz / info.bins;
    const z = Math.ceil(Math.log2(Math.max(fullBinHz / hzPerPixel, 1)));
    return Math.min(z, info.max_zoom);
  }

  private visibleTiles(level: number, z: number): { x: number; t: number }[] {
    const info = this.info!;
    const tileSeconds = info.tile_rows * info.base_row_s * 2 ** level;
    const tileHz = info.span_hz / 2 ** z;
    const lastT = Math.max(Math.ceil(this.duration / tileSeconds) - 1, 0);
    const firstT = Math.max(Math.floor(this.timeStart / tileSeconds), 0);
    const endT = Math.min(Math.floor((this.timeStart + this.timeSpan) / tileSeconds), lastT);
    const firstX = Math.max(Math.floor((this.lowHz - info.low_hz) / tileHz), 0);
    const endX = Math.min(Math.floor((this.lowHz + this.spanHz - info.low_hz - 1e-6) / tileHz), 2 ** z - 1);
    const out = [];
    for (let t = firstT; t <= endT; t++) for (let x = firstX; x <= endX; x++) out.push({ x, t });
    return out;
  }

  private request(level: number, z: number, x: number, t: number): void {
    const key = `${level}/${z}/${x}/${t}`;
    if (this.tiles.has(key) || this.loading.has(key) || this.failed.has(key)) return;
    this.queue.push({ key, level, z, x, t });
  }

  private pumpFetches(): void {
    while (this.loading.size < MAX_FETCHES && this.queue.length > 0) {
      const job = this.queue.shift()!;
      if (this.tiles.has(job.key) || this.loading.has(job.key)) continue;
      this.loading.add(job.key);
      const id = this.recording!.id;
      const signal = this.abort?.signal;
      fetchTile(id, job.level, job.z, job.x, job.t, signal)
        .then((bytes) => {
          if (this.recording?.id !== id) return;
          this.tiles.set(job.key, { bytes, canvas: null, paletteVersion: 0 });
          if (!this.autoRangeDone) this.autoRange(bytes);
          this.evict();
        })
        .catch(() => {
          if (!signal?.aborted) {
            this.failed.add(job.key);
            setTimeout(() => this.failed.delete(job.key), 5000);
          }
        })
        .finally(() => this.loading.delete(job.key));
    }
  }

  private evict(): void {
    while (this.tiles.size > TILE_CACHE) {
      const oldest = this.tiles.keys().next().value!;
      if (oldest.startsWith(`${this.info!.max_level}/0/`)) {
        // Keep the overview: it's the backdrop while finer tiles load.
        const tile = this.tiles.get(oldest)!;
        this.tiles.delete(oldest);
        this.tiles.set(oldest, tile);
      }
      const victim = this.tiles.keys().next().value!;
      this.tiles.delete(victim);
    }
  }

  /** Set the initial contrast from the first tile: floor near the noise, ceiling above it. */
  private autoRange(bytes: Uint8Array): void {
    const counts = new Uint32Array(256);
    let total = 0;
    for (let i = 0; i < bytes.length; i++) {
      if (bytes[i] > 0) {
        counts[bytes[i]]++;
        total++;
      }
    }
    if (total === 0) return;
    let seen = 0;
    let median = 0;
    let top = 255;
    for (let v = 0; v < 256; v++) {
      seen += counts[v];
      if (seen < total * 0.5) median = v;
      if (seen < total * 0.999) top = v;
    }
    this.autoRangeDone = true;
    const floor = Math.max(median - 8, 1);
    const ceiling = Math.max(Math.min(top + 8, 255), floor + 30);
    this.autoFloor = floor;
    this.autoCeiling = ceiling;
    this.setContrast(floor, ceiling);
    this.onContrastAuto(floor, ceiling);
  }

  private tileCanvas(tile: Tile, z: number): HTMLCanvasElement {
    if (tile.canvas && tile.paletteVersion === this.paletteVersion) return tile.canvas;
    const info = this.info!;
    const canvas = tile.canvas ?? document.createElement("canvas");
    canvas.width = info.bins;
    canvas.height = info.tile_rows;
    const image = new ImageData(info.bins, info.tile_rows);
    const pixels = new Uint32Array(image.data.buffer);

    const floor = this.floorByte - z * BYTES_PER_3DB;
    const range = this.ceilingByte - this.floorByte;
    const palette = new Uint32Array(256);
    for (let v = 0; v < 256; v++) {
      const [r, g, b] = phosphor((v - floor) / range);
      palette[v] = (255 << 24) | (b << 16) | (g << 8) | r;
    }
    const bytes = tile.bytes;
    for (let i = 0; i < bytes.length; i++) pixels[i] = palette[bytes[i]];
    canvas.getContext("2d")!.putImageData(image, 0, 0);
    tile.canvas = canvas;
    tile.paletteVersion = this.paletteVersion;
    return canvas;
  }

  private drawTile(level: number, z: number, x: number, t: number): boolean {
    const info = this.info!;
    const tile = this.tiles.get(`${level}/${z}/${x}/${t}`);
    if (!tile) return false;
    const tileSeconds = info.tile_rows * info.base_row_s * 2 ** level;
    const tileHz = info.span_hz / 2 ** z;
    const left = this.hzToX(info.low_hz + x * tileHz);
    const right = this.hzToX(info.low_hz + (x + 1) * tileHz);
    const top = this.secondsToY(t * tileSeconds);
    const bottom = this.secondsToY((t + 1) * tileSeconds);
    this.context.drawImage(this.tileCanvas(tile, z), left, top, right - left, bottom - top);
    return true;
  }

  // --- Drawing -----------------------------------------------------------------------

  draw(playheadSeconds: number): void {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const g = this.context;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = colors.plot;
    g.fillRect(0, 0, width, height);
    if (!this.info) return;

    g.save();
    g.beginPath();
    g.rect(RULER_LEFT, RULER_TOP, width - RULER_LEFT, height - RULER_TOP);
    g.clip();
    g.imageSmoothingEnabled = false;

    // The whole-recording overview is the backdrop while sharper tiles arrive.
    const top = this.info.max_level;
    this.request(top, 0, 0, 0);
    this.drawTile(top, 0, 0, 0);

    const level = this.chooseLevel();
    const z = this.chooseZoom();
    // A coarser level is the next layer, if one has already been loaded.
    for (let coarser = Math.min(level + 3, top - 1); coarser > level; coarser--) {
      for (const { x, t } of this.visibleTiles(coarser, 0)) this.drawTile(coarser, 0, x, t);
    }
    if (z > 0) {
      for (const { x, t } of this.visibleTiles(level, 0)) this.drawTile(level, 0, x, t);
    }
    this.queue = [];
    for (const { x, t } of this.visibleTiles(level, z)) {
      if (!this.drawTile(level, z, x, t)) this.request(level, z, x, t);
    }
    this.pumpFetches();

    if (display.analysis) this.drawGrid(g, width, height);
    this.drawOverlay(g, height);
    this.drawMarkers(g, width);
    if (display.analysis) this.drawCursors(g, width, height);
    this.drawPlayhead(g, playheadSeconds, width);
    g.restore();

    this.drawRulers(g, width, height, playheadSeconds);
    if (this.hover) this.drawHover(g, width);
  }

  /** Faint lines continuing the rulers' ticks across the plot. */
  private drawGrid(g: CanvasRenderingContext2D, width: number, height: number): void {
    g.strokeStyle = colors.grid;
    g.globalAlpha = 0.7;
    g.lineWidth = 1;
    g.beginPath();
    const hzStep = niceStep(this.spanHz / Math.max(this.plotWidth / 110, 1));
    for (let hz = Math.ceil(this.lowHz / hzStep) * hzStep; hz <= this.lowHz + this.spanHz; hz += hzStep) {
      g.moveTo(Math.round(this.hzToX(hz)) + 0.5, RULER_TOP);
      g.lineTo(Math.round(this.hzToX(hz)) + 0.5, height);
    }
    const timeStep = niceTimeStep(this.timeSpan / Math.max(this.plotHeight / 42, 1));
    for (let s = Math.ceil(this.timeStart / timeStep) * timeStep; s <= this.timeStart + this.timeSpan; s += timeStep) {
      g.moveTo(RULER_LEFT, Math.round(this.secondsToY(s)) + 0.5);
      g.lineTo(width, Math.round(this.secondsToY(s)) + 0.5);
    }
    g.stroke();
    g.globalAlpha = 1;
  }

  private drawCursors(g: CanvasRenderingContext2D, width: number, height: number): void {
    g.font = plotFont(12);
    g.textAlign = "left";
    g.textBaseline = "top";
    g.lineWidth = 1;
    this.cursors.forEach((cursor, n) => {
      const x = Math.round(this.hzToX(cursor.hz)) + 0.5;
      const y = Math.round(this.secondsToY(cursor.seconds)) + 0.5;
      g.strokeStyle = colors.white;
      g.setLineDash(n === 0 ? [6, 3] : [2, 3]); // told apart by dash as well as by name
      g.beginPath();
      g.moveTo(x, RULER_TOP);
      g.lineTo(x, height);
      g.moveTo(RULER_LEFT, y);
      g.lineTo(width, y);
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = colors.white;
      g.fillRect(x + 2, y + 2, 12, 14);
      g.fillStyle = colors.plot;
      g.fillText(n === 0 ? "A" : "B", x + 5, y + 3);
      if (n === 1 && this.cursorNote) {
        const noteWidth = g.measureText(this.cursorNote).width + 8;
        const noteX = Math.min(x + 16, width - noteWidth - 2);
        g.fillStyle = colors.scrim;
        g.fillRect(noteX, y + 2, noteWidth, 14);
        g.fillStyle = colors.white;
        g.fillText(this.cursorNote, noteX + 4, y + 3);
      }
    });
    g.textBaseline = "middle";
  }

  private drawOverlay(g: CanvasRenderingContext2D, height: number): void {
    if (!this.overlay) return;
    const x = this.hzToX(this.overlay.offsetHz);
    const a = this.hzToX(this.overlay.offsetHz + this.overlay.lowHz);
    const b = this.hzToX(this.overlay.offsetHz + this.overlay.highHz);
    g.fillStyle = colors.accentFill;
    g.fillRect(Math.min(a, b), RULER_TOP, Math.max(Math.abs(b - a), 1), height - RULER_TOP);
    g.strokeStyle = colors.accentLine;
    g.lineWidth = 1;
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(Math.round(x) + 0.5, RULER_TOP);
    g.lineTo(Math.round(x) + 0.5, height);
    g.stroke();
  }

  private drawMarkers(g: CanvasRenderingContext2D, width: number): void {
    g.font = plotFont(11);
    g.textBaseline = "bottom";
    g.textAlign = "left";
    g.lineWidth = 1;
    for (const marker of this.markers) {
      const y = this.secondsToY(marker.t);
      if (y < RULER_TOP || y > this.canvas.clientHeight) continue;
      const color = marker.kind === "frequency" ? colors.accent : colors.bright;
      g.strokeStyle = color;
      g.setLineDash(marker.kind === "frequency" ? [] : [5, 4]);
      g.beginPath();
      g.moveTo(RULER_LEFT, Math.round(y) + 0.5);
      g.lineTo(width, Math.round(y) + 0.5);
      g.stroke();
      g.setLineDash([]);
      if (marker.label) {
        const textWidth = g.measureText(marker.label).width;
        g.fillStyle = colors.scrim;
        g.fillRect(RULER_LEFT + 2, y - 14, textWidth + 6, 14);
        g.fillStyle = color;
        g.fillText(marker.label, RULER_LEFT + 5, y - 1);
      }
    }
    g.textBaseline = "middle";
  }

  private drawPlayhead(g: CanvasRenderingContext2D, seconds: number, width: number): void {
    const y = this.secondsToY(seconds);
    g.strokeStyle = colors.white;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(RULER_LEFT, Math.round(y) + 0.5);
    g.lineTo(width, Math.round(y) + 0.5);
    g.stroke();
  }

  private labelFor(offsetHz: number): string {
    if (this.recording?.kind === "audio") return `${(offsetHz / 1000).toFixed(1)} kHz`;
    if (this.centerHz !== null) return `${((this.centerHz + offsetHz) / 1e6).toFixed(this.spanHz < 200_000 ? 4 : 3)}`;
    return `${offsetHz >= 0 ? "+" : ""}${(offsetHz / 1000).toFixed(1)}k`;
  }

  private drawRulers(g: CanvasRenderingContext2D, width: number, height: number, playhead: number): void {
    g.fillStyle = colors.ruler;
    g.fillRect(0, 0, width, RULER_TOP);
    g.fillRect(0, 0, RULER_LEFT, height);
    g.font = plotFont(11);
    g.fillStyle = colors.fg;
    g.strokeStyle = colors.dim;
    g.lineWidth = 1;

    // Frequency ruler.
    const hzStep = niceStep(this.spanHz / Math.max(this.plotWidth / 110, 1));
    g.textBaseline = "middle";
    g.textAlign = "center";
    for (let hz = Math.ceil(this.lowHz / hzStep) * hzStep; hz <= this.lowHz + this.spanHz; hz += hzStep) {
      const x = this.hzToX(hz);
      // A label near the edge moves inward so it's never cut off; its tick stays true.
      const text = this.labelFor(hz);
      const half = g.measureText(text).width / 2 + 2;
      g.fillText(text, Math.min(Math.max(x, RULER_LEFT + half), width - half), RULER_TOP / 2);
      g.beginPath();
      g.moveTo(Math.round(x) + 0.5, RULER_TOP - 3);
      g.lineTo(Math.round(x) + 0.5, RULER_TOP);
      g.stroke();
    }

    // Which slice the close-up window is showing.
    if (this.slice && this.recording?.kind === "iq") {
      const a = Math.max(this.hzToX(this.slice.lowHz), RULER_LEFT);
      const b = Math.min(this.hzToX(this.slice.highHz), width);
      if (b > a - 2) {
        g.fillStyle = colors.accent;
        g.fillRect(a - 1, RULER_TOP - 3, Math.max(b - a, 1) + 2, 3);
        g.fillStyle = colors.fg;
      }
    }

    // Time ruler.
    const timeStep = niceTimeStep(this.timeSpan / Math.max(this.plotHeight / 42, 1));
    g.textAlign = "right";
    for (let s = Math.ceil(this.timeStart / timeStep) * timeStep; s <= this.timeStart + this.timeSpan; s += timeStep) {
      const y = this.secondsToY(s);
      g.fillText(formatTimeShort(s, timeStep), RULER_LEFT - 6, Math.min(Math.max(y, RULER_TOP + 7), height - 7));
      g.beginPath();
      g.moveTo(RULER_LEFT - 3, Math.round(y) + 0.5);
      g.lineTo(RULER_LEFT, Math.round(y) + 0.5);
      g.stroke();
    }

    // Where the playhead is, drawn on the ruler as a seek handle.
    const y = this.secondsToY(playhead);
    if (y >= RULER_TOP && y <= height) {
      g.fillStyle = colors.white;
      g.fillRect(RULER_LEFT - 8, Math.round(y) - 2, 8, 5);
    }
  }

  private drawHover(g: CanvasRenderingContext2D, width: number): void {
    const p = this.hover!;
    if (p.x < RULER_LEFT || p.y < RULER_TOP || this.drag?.moved) return;
    const hz = this.xToHz(p.x);
    const seconds = this.yToSeconds(p.y);
    let text = `${this.labelFor(hz)}  ${formatTimeShort(seconds, 0.01)}`;
    if (display.analysis) {
      const sample = Math.floor(Math.max(seconds, 0) * (this.recording?.sample_rate ?? 0));
      const offset = this.recording?.kind === "iq" ? `${hz >= 0 ? "+" : ""}${(hz / 1000).toFixed(3)} kHz  ` : "";
      text = `${this.labelFor(hz)}  ${offset}${formatTimeShort(seconds, 0.01)}  #${sample.toLocaleString("en-US")}`;
    }
    g.font = plotFont(12);
    const w = g.measureText(text).width + 10;
    const x = Math.min(p.x + 10, width - w - 2);
    const y = Math.max(p.y - 22, RULER_TOP + 2);
    g.fillStyle = colors.scrim;
    g.fillRect(x, y, w, 18);
    g.fillStyle = colors.bright;
    g.textAlign = "left";
    g.textBaseline = "middle";
    g.fillText(text, x + 5, y + 9);
  }
}

function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-9)));
  for (const m of [1, 2, 5, 10]) if (m * power >= rough) return m * power;
  return 10 * power;
}

function niceTimeStep(rough: number): number {
  for (const s of [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]) {
    if (s >= rough) return s;
  }
  return 14400;
}

function formatTimeShort(seconds: number, step: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  const digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  const text = rest.toFixed(digits).padStart(digits ? 3 + digits : 2, "0");
  if (minutes >= 60) return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${text}`;
  return `${minutes}:${text}`;
}
