// PAIR -> SAMPLE: how the current pair became the current audio sample.
//
// Every number here comes from the arrays the server used to make the audio. The only
// thing computed in the browser is the gain between its 10 ms points, which uses the
// same straight-line rule as the server, and the arrow length, which is just Pythagoras.

import type { Chunk } from "./api";

export interface PairValues {
  mode: string;
  pair: number; // absolute tuned pair number
  sample: number; // absolute audio sample number
  i: number;
  q: number;
  length: number;
  angleRad: number;
  carrier: number | null;
  turnRad: number | null; // FM: angle from the previous pair to this one
  discriminator: number | null;
  demod: number;
  gain: number;
  audio: number;
  limited: boolean; // the audio hit full scale, so demod x gain was cut back
  firstPairOfSample: number;
  pairsPerSample: number;
}

/** Gain at a sample of this chunk: straight lines between the 10 ms points. */
export function gainAt(nodes: Float32Array, block: number, sample: number): number {
  const position = sample / block;
  const before = Math.min(Math.floor(position), nodes.length - 1);
  const after = Math.min(before + 1, nodes.length - 1);
  return nodes[before] + (nodes[after] - nodes[before]) * (position - before);
}

export function pairValues(chunk: Chunk, previous: Chunk | null, pairInChunk: number): PairValues | null {
  const { header, arrays } = chunk;
  const k = pairInChunk;
  if (k < 0 || k >= arrays.i.length) return null;
  const perSample = header.pairs_per_sample;
  const s = Math.floor(k / perSample);
  if (s >= arrays.audio.length) return null;

  const i = arrays.i[k];
  const q = arrays.q[k];
  let turnRad: number | null = null;
  if (arrays.discriminator) {
    const prevI = k > 0 ? arrays.i[k - 1] : previous?.arrays.i.at(-1);
    const prevQ = k > 0 ? arrays.q[k - 1] : previous?.arrays.q.at(-1);
    if (prevI !== undefined && prevQ !== undefined) {
      // The angle of z[n] * conj(z[n-1]).
      turnRad = Math.atan2(q * prevI - i * prevQ, i * prevI + q * prevQ);
    }
  }
  const demod = arrays.demod[s];
  const gain = gainAt(arrays.gain_nodes, header.gain_block, s);
  const audio = arrays.audio[s];
  return {
    mode: header.mode,
    pair: header.index * header.iq_rate + k,
    sample: header.index * header.audio_rate + s,
    i,
    q,
    length: Math.hypot(i, q),
    angleRad: Math.atan2(q, i),
    carrier: arrays.carrier ? arrays.carrier[k] : null,
    turnRad,
    discriminator: arrays.discriminator ? arrays.discriminator[k] : null,
    demod,
    gain,
    audio,
    limited: Math.abs(demod * gain) > 1,
    firstPairOfSample: s * perSample,
    pairsPerSample: perSample,
  };
}

const signed = (x: number, digits = 5) => `${x >= 0 ? "+" : "-"}${Math.abs(x).toFixed(digits)}`;
const plain = (x: number, digits = 5) => x.toFixed(digits);
const count = (x: number) => x.toLocaleString("en-US");

function gainLines(v: PairValues): string[] {
  const lines = [`     x gain     ${plain(v.gain, 2)}`];
  lines.push(`     AUDIO     ${signed(v.audio)}  <-- on waveform`);
  if (v.limited) lines.push("     (limited to full scale)");
  return lines;
}

/** The text of the panel. */
export function explain(v: PairValues, chunk: Chunk): string {
  const head = `PAIR #${count(v.pair)}   I = ${signed(v.i)}   Q = ${signed(v.q)}`;
  switch (v.mode) {
    case "am":
      return [
        head,
        `AM:  sqrt(I^2 + Q^2) = ${plain(v.length)}`,
        `     - carrier  ${plain(v.carrier ?? 0)}`,
        ...gainLines(v),
      ].join("\n");
    case "usb":
    case "lsb":
      return [
        head,
        `${v.mode.toUpperCase()}: pair shown after the sideband filter,`,
        "     which uses many neighboring pairs",
        `     I         ${signed(v.i)}`,
        ...gainLines(v),
      ].join("\n");
    case "cw":
      return [
        head,
        "CW:  pair shown after the narrow filter and a",
        "     700 Hz shift, so the arrow spins 700 turns/s",
        `     I         ${signed(v.i)}`,
        ...gainLines(v),
      ].join("\n");
    case "nfm":
      return [
        head,
        `NFM: turn since the last pair  ${v.turnRad === null ? "--" : signed(v.turnRad)} rad`,
        `     as share of full swing   ${signed(v.discriminator ?? 0)}`,
        `     after de-emphasis        ${signed(v.demod)}`,
        ...gainLines(v),
      ].join("\n");
    default: {
      const { i, q, discriminator } = chunk.arrays;
      const lines = [`WFM: 5 pairs make this one audio sample #${count(v.sample)}`];
      const chunkStart = chunk.header.index * chunk.header.iq_rate;
      for (let k = v.firstPairOfSample; k < v.firstPairOfSample + v.pairsPerSample && k < i.length; k++) {
        const here = chunkStart + k === v.pair;
        lines.push(
          `${here ? ">" : " "} #${count(chunkStart + k)}  I ${signed(i[k], 4)}  Q ${signed(q[k], 4)}  swing ${signed(discriminator![k], 3)}`,
        );
      }
      lines.push(`     de-emphasis + 5:1   ${signed(v.demod)}`, ...gainLines(v));
      return lines.join("\n");
    }
  }
}
