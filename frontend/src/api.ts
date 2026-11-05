// Types and requests for the server's API.

import type { Marker } from "./markers";

export interface ExampleInfo {
  title: string;
  mode: string;
  description: string;
  offset_hz: number;
  order: number;
  segments?: { mode: string; start_s: number; end_s: number }[];
}

export interface WaterfallInfo {
  bins: number;
  tile_rows: number;
  base_row_s: number;
  max_level: number;
  max_zoom: number;
  db_min: number;
  db_max: number;
  low_hz: number;
  span_hz: number;
}

export interface Recording {
  id: string;
  name: string;
  folder: string;
  available: boolean;
  kind: "iq" | "audio" | "unknown" | "unreadable";
  kind_reason: string;
  ingest: { done: number; total: number };
  warnings: string[];
  size_bytes: number;
  sample_rate?: number;
  duration_s?: number;
  frames?: number;
  channels?: number;
  sample_type?: string;
  container?: string;
  frequency_hz?: number | null;
  frequency_source?: string | null;
  frequency_note?: string | null;
  started_at?: string | null;
  recorded_with?: string | null;
  lr_match?: boolean;
  waterfall?: WaterfallInfo;
  markers?: Marker[];
  example?: ExampleInfo | null;
}

export interface ModeInfo {
  default_bw_hz: number;
  iq_rate: number;
}

export interface ModesResponse {
  ssb_low_hz: number;
  modes: Record<string, ModeInfo>;
}

export interface Health {
  status: string;
  share: { mounted: boolean; readable: boolean; detail: string };
}

export interface ChunkHeader {
  index: number;
  mode: string;
  audio_rate: number;
  iq_rate: number;
  pairs_per_sample: number;
  gain_block: number;
  bw_hz: number;
  offset_hz: number;
  arrays: { name: string; length: number }[];
}

export interface Chunk {
  header: ChunkHeader;
  arrays: Record<string, Float32Array>;
}

export interface Tuning {
  offsetHz: number;
  mode: string;
  bwHz: number;
}

export interface BandSlice {
  rows: number;
  bins: number;
  rowsPerSecond: number;
  firstHz: number;
  binHz: number;
  db: Float32Array;
}

export async function fetchSlice(id: string, index: number, centerHz: number, spanHz: number): Promise<BandSlice> {
  const query = new URLSearchParams({ index: `${index}`, center_hz: `${Math.round(centerHz)}`, span_hz: `${spanHz}` });
  const response = await fetch(`/api/recordings/${encodeURIComponent(id)}/slice?${query}`);
  if (!response.ok) throw new Error(await explain(response));
  const buffer = await response.arrayBuffer();
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  return {
    rows: header.rows,
    bins: header.bins,
    rowsPerSecond: header.rows_per_second,
    firstHz: header.first_hz,
    binHz: header.bin_hz,
    db: new Float32Array(buffer, 4 + headerLength, header.rows * header.bins),
  };
}

/** A request the server refused for a reason the user should read, and retrying won't fix. */
export class RefusedError extends Error {}

async function explain(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await explain(response));
  return response.json() as Promise<T>;
}

export const fetchRecordings = () =>
  getJson<{ online: boolean; complete?: boolean; arriving?: { name: string; size_bytes: number }[]; recordings: Recording[] }>("/api/recordings");
export const fetchHealth = () => getJson<Health>("/api/health");
export const fetchModes = () => getJson<ModesResponse>("/api/modes");

export async function saveSidecar(
  id: string,
  values: { kind?: "iq" | "audio"; frequency_hz?: number | null; markers?: Marker[] },
): Promise<Recording> {
  const response = await fetch(`/api/recordings/${encodeURIComponent(id)}/sidecar`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
  if (!response.ok) throw new Error(await explain(response));
  return response.json() as Promise<Recording>;
}

export async function fetchTile(
  id: string,
  level: number,
  z: number,
  x: number,
  t: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const query = new URLSearchParams({ level: `${level}`, z: `${z}`, x: `${x}`, t: `${t}` });
  const response = await fetch(`/api/recordings/${encodeURIComponent(id)}/waterfall?${query}`, { signal });
  if (!response.ok) throw new Error(await explain(response));
  return new Uint8Array(await response.arrayBuffer());
}

export async function fetchChunk(
  id: string,
  index: number,
  tuning: Tuning,
  signal?: AbortSignal,
): Promise<Chunk> {
  const query = new URLSearchParams({
    index: `${index}`,
    offset_hz: `${Math.round(tuning.offsetHz)}`,
    mode: tuning.mode,
    bw_hz: `${tuning.bwHz}`,
    // One request serves both the sound and the panels, so they can't disagree.
    fields: "audio,i,q,demod,gain_nodes,carrier,discriminator",
  });
  const response = await fetch(`/api/recordings/${encodeURIComponent(id)}/chunk?${query}`, { signal });
  if (response.status === 422 || response.status === 409) throw new RefusedError(await explain(response));
  if (!response.ok) throw new Error(await explain(response));

  // Frame layout: header length, JSON header (padded to 4 bytes), then Float32 arrays.
  const buffer = await response.arrayBuffer();
  const headerLength = new DataView(buffer).getUint32(0, true);
  const text = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength));
  const header = JSON.parse(text) as ChunkHeader;
  const arrays: Record<string, Float32Array> = {};
  let position = 4 + headerLength;
  for (const { name, length } of header.arrays) {
    arrays[name] = new Float32Array(buffer, position, length);
    position += length * 4;
  }
  return { header, arrays };
}

export interface RawBurst {
  startPair: number;
  i: Float32Array;
  q: Float32Array;
}

export async function fetchRaw(id: string, start: number, count: number): Promise<RawBurst> {
  const query = new URLSearchParams({ start: `${start}`, count: `${count}` });
  const response = await fetch(`/api/recordings/${encodeURIComponent(id)}/raw?${query}`);
  if (!response.ok) throw new Error(await explain(response));
  const buffer = await response.arrayBuffer();
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const at = 4 + headerLength;
  return {
    startPair: header.start_pair,
    i: new Float32Array(buffer, at, header.count),
    q: new Float32Array(buffer, at + header.count * 4, header.count),
  };
}

export const rescan = async (): Promise<void> => {
  const response = await fetch("/api/rescan", { method: "POST", headers: { "X-IQScope": "1" } });
  if (!response.ok) throw new Error(await explain(response));
};

// --- Uploads ------------------------------------------------------------------------------

export interface UploadInfo {
  id: string;
  name: string;
  original_name: string;
  size: number;
  received: number;
  part_bytes: number;
  part_hashes: string[];
}

/** Why the archive turned a request down, and whether trying again could help. */
export class UploadFailure extends Error {
  constructor(
    message: string,
    public status: number,
    public retry: boolean,
    public received: number | null,
  ) {
    super(message);
  }
}

function uploadFailure(status: number, text: string): UploadFailure {
  let detail = `Request failed (${status}).`;
  let retry = status >= 500 && status !== 507;
  let received: number | null = null;
  try {
    const body = JSON.parse(text);
    if (typeof body.detail === "string") detail = body.detail;
    if (typeof body.retry === "boolean") retry = body.retry;
    if (typeof body.received === "number") received = body.received;
  } catch {
    // Not from the app itself, such as a proxy's error page; the status has to do.
  }
  return new UploadFailure(detail, status, retry, received);
}

async function uploadJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "X-IQScope": "1", ...(init?.body ? { "Content-Type": "application/json" } : {}) },
  });
  if (!response.ok) throw uploadFailure(response.status, await response.text());
  return (response.status === 204 ? null : await response.json()) as T;
}

export const listUploads = async (): Promise<UploadInfo[]> =>
  (await uploadJson<{ uploads: UploadInfo[] }>("/api/uploads")).uploads;
export const uploadState = (id: string) => uploadJson<UploadInfo>(`/api/uploads/${id}`);
export const startUpload = (name: string, size: number) =>
  uploadJson<UploadInfo>("/api/uploads", { method: "POST", body: JSON.stringify({ name, size }) });
export const cancelUpload = (id: string) => uploadJson<void>(`/api/uploads/${id}`, { method: "DELETE" });
export const finishUpload = (id: string, piecesSha256: string) =>
  uploadJson<Recording>(`/api/uploads/${id}/finish`, {
    method: "POST",
    body: JSON.stringify({ pieces_sha256: piecesSha256 }),
  });

/** One piece. XMLHttpRequest, because fetch can't report how much it has sent so far. */
export function sendUploadPiece(
  id: string,
  offset: number,
  sha256: string,
  data: ArrayBuffer,
  signal: AbortSignal,
  onProgress: (bytes: number) => void,
): Promise<number> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", `/api/uploads/${id}/part?offset=${offset}&sha256=${sha256}`);
    request.setRequestHeader("X-IQScope", "1");
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.upload.onprogress = (event) => onProgress(event.loaded);
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) resolve(JSON.parse(request.responseText).received);
      else reject(uploadFailure(request.status, request.responseText));
    };
    request.onerror = () => reject(new Error("CONNECTION LOST."));
    request.ontimeout = () => reject(new Error("CONNECTION TIMED OUT."));
    request.onabort = () => reject(new Error("CANCELLED."));
    signal.addEventListener("abort", () => request.abort(), { once: true });
    request.send(data);
  });
}
