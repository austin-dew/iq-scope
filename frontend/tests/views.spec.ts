import { expect, test, type Page } from "@playwright/test";

const IQ_FILE = "baseband_7200000Hz_12-00-00_03-11-2026.wav";
const AM_KHZ = "7100"; // an AM signal sits at -100 kHz
const USB_KHZ = "7250"; // a USB signal sits at +50 kHz

interface V {
  mode: string; pair: number; sample: number; i: number; q: number; length: number;
  carrier: number | null; turnRad: number | null; discriminator: number | null;
  demod: number; gain: number; audio: number; limited: boolean;
  firstPairOfSample: number; pairsPerSample: number;
}

const values = (page: Page) =>
  page.evaluate(() => (window as any).iqScopeDebug.views.lastValues) as Promise<V | null>;
const pair = async (page: Page) => (await values(page))?.pair;
const seconds = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.transport.seconds) as Promise<number>;
const playing = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.engine.isPlaying) as Promise<boolean>;
const ctxState = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.engine.context?.state ?? "none") as Promise<string>;

async function open(page: Page, khz: string, mode: string): Promise<void> {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId(`mode-${mode}`).click();
  await page.getByTestId("tune").fill(khz);
  await page.getByTestId("tune").press("Enter");
  await expect.poll(() => values(page), { timeout: 20_000 }).not.toBeNull();
  await expect.poll(async () => (await values(page))?.mode).toBe(mode);
}

async function goTo(page: Page, s: number): Promise<void> {
  await page.evaluate((t) => (window as any).iqScopeDebug.transport.seek(t), s);
  await expect.poll(async () => Math.abs(((await values(page))?.sample ?? -1e9) / 48000 - s), { timeout: 15_000 }).toBeLessThan(0.001);
}

/** The audio as the server reports it, fetched separately from what's on screen. */
const serverChunk = (page: Page, index: number, offset: number, mode: string, bw: number) =>
  page.evaluate(async ({ index, offset, mode, bw }) => {
    const list = await (await fetch("/api/recordings")).json();
    const id = list.recordings.find((r: any) => r.name.startsWith("baseband_7200000Hz")).id;
    const q = new URLSearchParams({ index: `${index}`, offset_hz: `${offset}`, mode, bw_hz: `${bw}`, fields: "audio,i,q" });
    const buffer = await (await fetch(`/api/recordings/${id}/chunk?${q}`)).arrayBuffer();
    const size = new DataView(buffer).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, size)));
    let at = 4 + size; const out: Record<string, number[]> = {};
    for (const { name, length } of header.arrays) { out[name] = Array.from(new Float32Array(buffer, at, length)); at += length * 4; }
    return out;
  }, { index, offset, mode, bw });

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => { throw error; });
});

// --- PAIR -> SAMPLE shows the values the audio actually used ---------------------------

const formulas: [string, string, number, (v: V) => number][] = [
  ["am", AM_KHZ, -100_000, (v) => (v.length - v.carrier!) * v.gain],
  ["usb", USB_KHZ, 50_000, (v) => v.i * v.gain],
  ["lsb", USB_KHZ, 50_000, (v) => v.i * v.gain],
  ["cw", USB_KHZ, 50_000, (v) => v.i * v.gain],
  ["nfm", USB_KHZ, 50_000, (v) => v.demod * v.gain],
  ["wfm", USB_KHZ, 50_000, (v) => v.demod * v.gain],
];

for (const [mode, khz, offset, formula] of formulas) {
  test(`${mode}: the numbers shown reproduce the audio sample`, async ({ page }) => {
    await open(page, khz, mode);
    const bw = (await page.evaluate(() => (window as any).iqScopeDebug.state().bwHz)) as number;
    const server = await serverChunk(page, 3, offset, mode, bw);
    for (const t of [3.1, 3.37501, 3.62, 3.9]) {
      await goTo(page, t);
      const v = (await values(page))!;
      const computed = Math.max(-1, Math.min(1, formula(v)));
      expect(Math.abs(computed - v.audio)).toBeLessThan(2e-3);
      // And it's the very sample and pair the server made, not a similar one.
      expect(v.audio).toBeCloseTo(server.audio[v.sample - 3 * 48000], 5);
      const iqRate = mode === "wfm" ? 240000 : 48000;
      expect(v.i).toBeCloseTo(server.i[v.pair - 3 * iqRate], 5);
      expect(v.q).toBeCloseTo(server.q[v.pair - 3 * iqRate], 5);
    }
  });
}

test("the panel text states the same numbers", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.25);
  const v = (await values(page))!;
  const text = await page.getByTestId("pair-sample").textContent();
  expect(text).toContain(`PAIR #${v.pair.toLocaleString("en-US")}`);
  expect(text).toContain(v.length.toFixed(5));
  expect(text).toContain(v.carrier!.toFixed(5));
  expect(text).toContain(v.gain.toFixed(2));
  expect(text).toContain(Math.abs(v.audio).toFixed(5));
});

test("FM's turn angle comes from the shown pairs and matches the discriminator", async ({ page }) => {
  await open(page, USB_KHZ, "nfm");
  await goTo(page, 3.5);
  for (let n = 0; n < 5; n++) {
    const before = (await pair(page))!;
    await page.getByTestId("step-forward").click();
    await expect.poll(() => pair(page)).toBe(before + 1);
    const v = (await values(page))!;
    expect(v.turnRad).not.toBeNull();
    const turnShare = (v.turnRad! * 48000) / (2 * Math.PI * 2500);
    expect(Math.abs(turnShare - v.discriminator!)).toBeLessThan(5e-3);
  }
});

test("WFM lists the five pairs behind one audio sample", async ({ page }) => {
  await open(page, USB_KHZ, "wfm");
  await goTo(page, 3.5);
  const v = (await values(page))!;
  expect(v.pairsPerSample).toBe(5);
  const text = (await page.getByTestId("pair-sample").textContent())!;
  expect((text.match(/#[\d,]+ {2}I /g) ?? []).length).toBe(5);
  await page.getByTestId("step-forward").click();
  await expect.poll(() => pair(page)).toBe(v.pair + 1);
});

// --- STEP -----------------------------------------------------------------------------

test("step moves exactly one pair, both ways, by button and key", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  const start = (await pair(page))!;
  await page.getByTestId("step-forward").click();
  await expect.poll(() => pair(page)).toBe(start + 1);
  await page.getByTestId("step-forward").click();
  await expect.poll(() => pair(page)).toBe(start + 2);
  await page.getByTestId("step-back").click();
  await expect.poll(() => pair(page)).toBe(start + 1);
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => pair(page)).toBe(start + 2);
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect.poll(() => pair(page)).toBe(start);
  await page.keyboard.press("Shift+ArrowRight");
  await expect.poll(() => pair(page)).toBe(start + 100);
  await expect(page.getByTestId("muted")).toBeVisible();
});

test("a thousand steps land exactly a thousand pairs on, hours of rounding aside", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  const start = (await pair(page))!;
  await page.evaluate(() => {
    const d = (window as any).iqScopeDebug;
    for (let n = 0; n < 1000; n++) d.transport.step(1, 48000);
  });
  await expect.poll(() => pair(page)).toBe(start + 1000);
});

test("stepping crosses a chunk boundary in both directions, with the right values", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  const bw = (await page.evaluate(() => (window as any).iqScopeDebug.state().bwHz)) as number;
  const [a, b] = [await serverChunk(page, 2, -100_000, "am", bw), await serverChunk(page, 3, -100_000, "am", bw)];
  await page.evaluate(() => (window as any).iqScopeDebug.transport.seek(47998 / 48000 + 2));
  await expect.poll(() => pair(page), { timeout: 15_000 }).toBe(2 * 48000 + 47998);

  await page.keyboard.press("ArrowRight");
  await expect.poll(() => pair(page)).toBe(2 * 48000 + 47999);
  expect((await values(page))!.i).toBeCloseTo(a.i[47999], 5);

  await page.keyboard.press("ArrowRight");
  await expect.poll(() => pair(page), { timeout: 15_000 }).toBe(3 * 48000);
  expect((await values(page))!.i).toBeCloseTo(b.i[0], 5);

  await page.keyboard.press("ArrowLeft");
  await expect.poll(() => pair(page), { timeout: 15_000 }).toBe(2 * 48000 + 47999);
  expect((await values(page))!.i).toBeCloseTo(a.i[47999], 5);
});

test("stepping stops playback and mutes, and play picks up where stepping stopped", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
  await page.getByTestId("step-forward").click();
  await expect(page.getByTestId("muted")).toBeVisible();
  expect(await playing(page)).toBe(false);
  await expect.poll(() => ctxState(page)).toBe("suspended");
  const held = await seconds(page);
  await page.waitForTimeout(400);
  expect(await seconds(page)).toBe(held);

  await page.getByTestId("play").click();
  await expect(page.getByTestId("muted")).toBeHidden();
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(held + 0.3);
  expect(await seconds(page)).toBeLessThan(held + 6);
});

test("stepping stops cleanly at both ends of the recording", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 0.0001);
  for (let n = 0; n < 10; n++) await page.keyboard.press("ArrowLeft");
  await expect.poll(() => pair(page)).toBe(0);
  await page.evaluate(() => (window as any).iqScopeDebug.transport.seek(20));
  await expect.poll(() => pair(page), { timeout: 15_000 }).toBe(20 * 48000 - 1);
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(200);
  expect(await pair(page)).toBe(20 * 48000 - 1);
});

// --- Slow motion ------------------------------------------------------------------------

async function measuredRate(page: Page, ms: number): Promise<number> {
  const a = await page.evaluate(() => [performance.now(), (window as any).iqScopeDebug.transport.seconds]);
  await page.waitForTimeout(ms);
  const b = await page.evaluate(() => [performance.now(), (window as any).iqScopeDebug.transport.seconds]);
  return (b[1] - a[1]) / ((b[0] - a[0]) / 1000);
}

for (const [label, speed] of [["1/10", 0.1], ["1/1000", 0.001]] as const) {
  test(`${label} runs at the right rate with the sound off`, async ({ page }) => {
    await open(page, AM_KHZ, "am");
    await page.getByTestId("play").click();
    await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
    await page.getByTestId(`speed-${speed}`).click();
    await expect(page.getByTestId("muted")).toHaveText("AUDIO MUTED");
    expect(await playing(page)).toBe(false);
    await expect.poll(() => ctxState(page)).toBe("suspended");
    await expect(page.getByTestId("play")).toHaveText("PAUSE");
    await page.waitForTimeout(1500); // let the neighboring chunks arrive
    expect(await measuredRate(page, 3000) / speed).toBeGreaterThan(0.95);
    expect(await measuredRate(page, 3000) / speed).toBeLessThan(1.05);
  });
}

test("going back to 1x resumes the sound from the same spot", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
  await page.getByTestId("speed-0.1").click();
  await page.waitForTimeout(800);
  const here = await seconds(page);
  await page.getByTestId("speed-1").click();
  await expect(page.getByTestId("muted")).toBeHidden();
  await expect.poll(() => playing(page)).toBe(true);
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(here + 0.2);
  expect(await seconds(page)).toBeLessThan(here + 4);
  expect(await seconds(page)).toBeGreaterThanOrEqual(here);
});

test("slow motion pauses, resumes, and retunes without losing its place", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.2);
  await page.getByTestId("speed-0.001").click();
  await page.getByTestId("play").click();
  await page.waitForTimeout(500);
  await page.getByTestId("play").click();
  await page.waitForTimeout(200); // the panels show the stopped position from the next frame
  const held = (await pair(page))!;
  await page.waitForTimeout(300);
  expect(await pair(page)).toBe(held);
  await page.getByTestId("tune").fill(USB_KHZ);
  await page.getByTestId("tune").press("Enter");
  await page.getByTestId("mode-usb").click();
  await expect.poll(async () => (await values(page))?.mode, { timeout: 15_000 }).toBe("usb");
  expect(await pair(page)).toBe(held);
});

test("playing past the end in slow motion stops there", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 19.99);
  await page.getByTestId("speed-0.1").click();
  await page.getByTestId("play").click();
  await expect(page.getByTestId("status")).toHaveText("END OF RECORDING", { timeout: 15_000 });
  await expect(page.getByTestId("play")).toHaveText("PLAY");
});

test("scrubbing in slow motion moves the panels and stays muted", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await page.getByTestId("speed-0.1").click();
  await page.getByTestId("scrub").evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    el.value = "9";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => pair(page), { timeout: 15_000 }).toBe(9 * 48000);
  await expect(page.getByTestId("muted")).toBeVisible();
  expect(await playing(page)).toBe(false);
});

// --- Sync at full speed -------------------------------------------------------------------

test("the panels stay within a video frame of the audio during playback", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
  const worst = await page.evaluate(async () => {
    const d = (window as any).iqScopeDebug;
    let worst = 0;
    const end = performance.now() + 3000;
    while (performance.now() < end) {
      await new Promise((r) => requestAnimationFrame(r));
      const drawn = d.views.lastPair;
      if (drawn !== null) worst = Math.max(worst, Math.abs(drawn / 48000 - d.transport.seconds));
    }
    return worst;
  });
  expect(worst).toBeLessThan(2 / 60); // the check runs a frame after the drawing
});

test("during a retune the panels show the tuning being heard", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
  const check = () => page.evaluate(() => {
    const d = (window as any).iqScopeDebug;
    const index = Math.floor(d.views.lastPair / 48000);
    const chunk = d.engine.chunkFor(index);
    const v = d.views.lastValues;
    return chunk ? { same: chunk.arrays.audio[v.sample - index * 48000] === v.audio } : { same: true };
  });
  await page.getByTestId("mode-usb").click();
  for (let n = 0; n < 15; n++) {
    expect((await check()).same).toBe(true);
    await page.waitForTimeout(100);
  }
});

// --- What's on screen ----------------------------------------------------------------------

// --- The close-up spectrum --------------------------------------------------------------

const closeUp = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.closeUp.summary);
const tunedOffset = (page: Page) =>
  page.evaluate(() => (window as any).iqScopeDebug.state().offsetHz) as Promise<number>;

async function ready(page: Page): Promise<void> {
  await expect.poll(async () => (await closeUp(page))?.peakDb ?? null, { timeout: 15_000 }).not.toBeNull();
}

/** Where a frequency falls on the panel. */
async function pointAt(page: Page, hz: number) {
  const box = (await page.getByTestId("spectrum").boundingBox())!;
  const c = await closeUp(page);
  const hzPerPixel = c.spanHz / box.width;
  return { x: box.x + box.width / 2 + (hz - c.lockedCenterHz) / hzPerPixel, y: box.y + box.height / 2, hzPerPixel, box };
}

test("AM shows its carrier on the tuner at the right level, with balanced sidebands", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const s = await closeUp(page);
  expect(Math.abs(s.peakFromTuningHz)).toBeLessThan(30);
  expect(Math.abs(s.peakDb + 12)).toBeLessThan(1.5); // carrier amplitude 0.25
  expect(s.upperPower / s.lowerPower).toBeGreaterThan(0.5);
  expect(s.upperPower / s.lowerPower).toBeLessThan(2);
});

test("USB energy shows above the tuner, and the filter moves to the other side for LSB", async ({ page }) => {
  await open(page, USB_KHZ, "usb");
  await goTo(page, 3.5);
  await ready(page);
  const usb = await closeUp(page);
  expect(usb.peakFromTuningHz).toBeGreaterThan(300);
  expect(usb.peakFromTuningHz).toBeLessThan(3000);
  await page.getByTestId("mode-lsb").click();
  await expect.poll(async () => (await closeUp(page)).peakDb, { timeout: 15_000 }).toBeLessThan(usb.peakDb - 30);
  // The band itself hasn't changed: same slice, same data.
  expect((await closeUp(page)).lockedCenterHz).toBe(usb.lockedCenterHz);
});

test("the close-up shows the radio signal itself, so CW's carrier sits on the tuner", async ({ page }) => {
  await open(page, AM_KHZ, "cw");
  await goTo(page, 3.5);
  await ready(page);
  expect(Math.abs((await closeUp(page)).peakFromTuningHz)).toBeLessThan(30);
});

test("WFM widens the window to fit its channel", async ({ page }) => {
  await open(page, USB_KHZ, "wfm");
  await goTo(page, 3.5);
  await ready(page);
  expect((await closeUp(page)).spanHz).toBeGreaterThanOrEqual(250_000);
});

test("clicking moves the tuner there and the window stays locked", async ({ page }) => {
  await open(page, "7100.5", "am"); // the carrier is 500 Hz below the tuner
  await goTo(page, 3.5);
  await ready(page);
  const locked = (await closeUp(page)).lockedCenterHz;
  const p = await pointAt(page, -100_000);
  await page.mouse.click(p.x, p.y);
  await expect.poll(async () => Math.abs((await tunedOffset(page)) + 100_000)).toBeLessThanOrEqual(30);
  await page.waitForTimeout(800);
  const after = await closeUp(page);
  expect(after.lockedCenterHz).toBe(locked); // no snapping back to the center
  expect(Math.abs(after.tunedHz + 100_000)).toBeLessThanOrEqual(30);
  expect(Math.abs(after.peakFromTuningHz)).toBeLessThan(40);
});

test("the tuner follows a drag exactly, and the window never moves, even on release", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const locked = (await closeUp(page)).lockedCenterHz;
  const start = await tunedOffset(page);
  const a = await pointAt(page, start);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  for (const dx of [40, 120, -60, 90]) {
    await page.mouse.move(a.x + dx, a.y, { steps: 4 });
    const tuned = await tunedOffset(page);
    expect(Math.abs(tuned - (start + dx * a.hzPerPixel))).toBeLessThanOrEqual(10);
    expect((await closeUp(page)).lockedCenterHz).toBe(locked);
  }
  await page.waitForTimeout(1200); // new audio arrives: nothing may creep
  const held = await tunedOffset(page);
  await page.mouse.up();
  await page.waitForTimeout(500);
  expect(await tunedOffset(page)).toBe(held);
  expect((await closeUp(page)).lockedCenterHz).toBe(locked);
});

test("a drag stops at the edge of the window rather than moving it", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const c = await closeUp(page);
  const a = await pointAt(page, c.tunedHz);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(a.box.x + a.box.width + 400, a.y, { steps: 6 });
  await page.mouse.up();
  expect(await tunedOffset(page)).toBeLessThanOrEqual(c.lockedCenterHz + c.spanHz / 2);
  expect(await tunedOffset(page)).toBeGreaterThan(c.lockedCenterHz + c.spanHz / 2 - 20);
  expect((await closeUp(page)).lockedCenterHz).toBe(c.lockedCenterHz);
});

test("Shift-drag tunes ten times finer, in 1 Hz steps, without jumping to the cursor", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const start = await tunedOffset(page);
  const a = await pointAt(page, start + 1000);
  await page.keyboard.down("Shift");
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  expect(await tunedOffset(page)).toBe(start);
  await page.mouse.move(a.x + 40, a.y, { steps: 8 });
  expect(Math.abs((await tunedOffset(page)) - (start + 40 * a.hzPerPixel * 0.1))).toBeLessThanOrEqual(1);
  await page.mouse.up();
  await page.keyboard.up("Shift");
});

test("scrolling nudges by 10 Hz, or 1 Hz with Shift, and the window stays locked", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const start = await tunedOffset(page);
  const locked = (await closeUp(page)).lockedCenterHz;
  const a = await pointAt(page, start);
  await page.mouse.move(a.x, a.y);
  await page.mouse.wheel(0, -100);
  await expect.poll(() => tunedOffset(page)).toBe(start + 10);
  await page.mouse.wheel(0, 100);
  await page.mouse.wheel(0, 100);
  await expect.poll(() => tunedOffset(page)).toBe(start - 10);
  await page.keyboard.down("Shift");
  await page.mouse.wheel(0, -100);
  await page.keyboard.up("Shift");
  await expect.poll(() => tunedOffset(page)).toBe(start - 9);
  expect((await closeUp(page)).lockedCenterHz).toBe(locked);
});

test("the window moves only when the tuner leaves it some other way, or on CENTER", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  const locked = (await closeUp(page)).lockedCenterHz;
  await page.getByRole("button", { name: "+1k" }).click();
  expect((await closeUp(page)).lockedCenterHz).toBe(locked); // still inside: stays

  await page.getByTestId("slice-center").click();
  await expect.poll(async () => (await closeUp(page)).lockedCenterHz).toBe(locked + 1000);

  await page.getByTestId("tune").fill(USB_KHZ); // far away: it has to follow
  await page.getByTestId("tune").press("Enter");
  await expect.poll(async () => (await closeUp(page)).lockedCenterHz).toBe(50_000);
});

test("+ and - change how much of the band the window shows", async ({ page }) => {
  await open(page, USB_KHZ, "usb");
  await goTo(page, 3.5);
  await ready(page);
  expect((await closeUp(page)).spanHz).toBe(10_000);
  await page.getByTestId("slice-wide").click();
  await expect.poll(async () => (await closeUp(page)).spanHz).toBe(20_000);
  await page.getByTestId("slice-narrow").click();
  await page.getByTestId("slice-narrow").click();
  await expect.poll(async () => (await closeUp(page)).spanHz).toBe(5_000);
  await ready(page);
});

test("moving the tuner in the window asks the server for no new spectrum", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  await ready(page);
  await page.waitForTimeout(1500);
  let requests = 0;
  page.on("request", (r) => { if (r.url().includes("/slice?")) requests++; });
  const a = await pointAt(page, (await closeUp(page)).tunedHz);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(a.x + 100, a.y, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  expect(requests).toBe(0);
});

test("sound keeps going while tuning in the window", async ({ page }) => {
  await open(page, USB_KHZ, "usb");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(2);
  await page.waitForTimeout(1500);
  const a = await pointAt(page, await tunedOffset(page));
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  const t0 = Date.now();
  const p0 = await seconds(page);
  for (let n = 1; n <= 20; n++) {
    await page.mouse.move(a.x + n * 5, a.y);
    await page.waitForTimeout(100);
  }
  const rate = ((await seconds(page)) - p0) / ((Date.now() - t0) / 1000);
  await page.mouse.up();
  expect(rate).toBeGreaterThan(0.9);
  expect(await playing(page)).toBe(true);
});

test("every panel draws, and audio recordings don't get them", async ({ page }) => {
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  for (const id of ["constellation", "phasor", "traces", "waveform", "spectrum"]) {
    await expect.poll(() => page.getByTestId(id).evaluate((c: HTMLCanvasElement) => {
      const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
      let bright = 0;
      for (let i = 1; i < d.length; i += 4) if (d[i] > 100) bright++;
      return bright;
    })).toBeGreaterThan(10);
  }
  await page.getByTestId("file-audio_7000000Hz_12-00-00_03-11-2026.wav").click();
  await expect(page.getByTestId("constellation")).toBeHidden();
  await expect(page.getByTestId("waveform")).toBeVisible();
  await expect(page.getByTestId("step-forward")).toBeHidden();
});

test("the panels fit at phone width with nothing overflowing", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await open(page, AM_KHZ, "am");
  await goTo(page, 3.5);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  for (const id of ["constellation", "phasor", "traces", "waveform", "spectrum"]) await expect(page.getByTestId(id)).toBeVisible();
  const box = (await page.getByTestId("constellation").boundingBox())!;
  expect(box.height).toBeGreaterThan(120);
});
