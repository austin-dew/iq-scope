// Runs on the browser's audio thread. It plays queued blocks in order and reports which
// sample it has reached. That report is the master clock: every visual derives its
// position from it, so nothing on screen can drift away from the sound.
//
// Blocks carry absolute sample numbers, and a generation number marks everything
// belonging to one seek or retune. Old blocks that arrive late are simply ignored.

const REPORT_EVERY_FRAMES = 1024;

class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.generation = 0;
    this.position = 0;
    this.endSample = Infinity;
    this.sinceReport = 0;
    this.finished = false;
    this.port.onmessage = (event) => this.receive(event.data);
  }

  receive(message) {
    if (message.type === "reset") {
      this.generation = message.generation;
      this.position = message.sample;
      this.endSample = Infinity;
      this.queue = [];
      this.finished = false;
    } else if (message.type === "chunk" && message.generation === this.generation) {
      // A retune sends the same second again with new tuning, and it replaces the old
      // copy, even while that copy is playing. Until then the old copy keeps playing,
      // so changing the tuning never leaves a gap.
      const end = message.start + message.samples.length;
      if (message.last) this.endSample = end;
      if (end <= this.position) return;
      this.queue = this.queue.filter((block) => block.start !== message.start);
      this.queue.push({ start: message.start, samples: message.samples });
      this.queue.sort((x, y) => x.start - y.start);
    }
  }

  process(_inputs, outputs) {
    const channels = outputs[0];
    const out = channels[0];
    const blockStartPosition = this.position;
    let written = 0;

    while (written < out.length && this.queue.length > 0) {
      const head = this.queue[0];
      const into = this.position - head.start;
      if (into >= head.samples.length) {
        this.queue.shift();
        continue;
      }
      if (into < 0) break; // the next second hasn't arrived, so wait for it
      const count = Math.min(out.length - written, head.samples.length - into);
      out.set(head.samples.subarray(into, into + count), written);
      written += count;
      this.position += count;
    }
    out.fill(0, written);
    for (let c = 1; c < channels.length; c++) channels[c].set(out);

    const starved = written < out.length;
    this.sinceReport += out.length;
    if (this.sinceReport >= REPORT_EVERY_FRAMES || starved) {
      this.sinceReport = 0;
      this.port.postMessage({
        type: "clock",
        generation: this.generation,
        sample: blockStartPosition,
        time: currentTime,
        starved,
      });
    }

    if (!this.finished && this.position >= this.endSample) {
      this.finished = true;
      this.port.postMessage({ type: "ended", generation: this.generation });
    }
    return true;
  }
}

registerProcessor("player", PlayerProcessor);
