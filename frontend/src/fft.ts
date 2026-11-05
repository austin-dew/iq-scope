// A small FFT for the Signal Path's filtered stage, kept here so the app loads nothing else.

export class Fft {
  private cosine: Float64Array;
  private sine: Float64Array;
  private reversed: Uint32Array;

  constructor(readonly size: number) {
    const bits = Math.log2(size);
    if (!Number.isInteger(bits)) throw new Error("FFT size must be a power of two");
    this.cosine = new Float64Array(size / 2);
    this.sine = new Float64Array(size / 2);
    for (let k = 0; k < size / 2; k++) {
      this.cosine[k] = Math.cos((2 * Math.PI * k) / size);
      this.sine[k] = -Math.sin((2 * Math.PI * k) / size);
    }
    this.reversed = new Uint32Array(size);
    for (let k = 0; k < size; k++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (k & (1 << b)) r |= 1 << (bits - 1 - b);
      this.reversed[k] = r;
    }
  }

  /** Transform in place: re and im go in as time and come out as frequency. */
  transform(re: Float64Array, im: Float64Array): void {
    const n = this.size;
    for (let k = 0; k < n; k++) {
      const r = this.reversed[k];
      if (r > k) {
        [re[k], re[r]] = [re[r], re[k]];
        [im[k], im[r]] = [im[r], im[k]];
      }
    }
    for (let span = 2; span <= n; span *= 2) {
      const half = span / 2;
      const stride = n / span;
      for (let start = 0; start < n; start += span) {
        for (let k = 0; k < half; k++) {
          const c = this.cosine[k * stride];
          const s = this.sine[k * stride];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * c - im[b] * s;
          const ti = re[b] * s + im[b] * c;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}
