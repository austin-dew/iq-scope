import { expect, test, type Page } from "@playwright/test";

const IQ_FILE = "baseband_7200000Hz_12-00-00_03-11-2026.wav";

interface Debug {
  engine: { seconds: number; clockSeconds: number | null; latencySeconds: number; isPlaying: boolean };
  waterfall: unknown;
  state: () => { offsetHz: number; mode: string; bwHz: number; current?: string };
}

const debug = <T>(page: Page, read: (d: Debug) => T) =>
  page.evaluate(`(${read.toString()})(window.iqScopeDebug)`) as Promise<T>;

const seconds = (page: Page) => debug(page, (d) => d.engine.seconds);

async function openIqFile(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
}

async function waitForSound(page: Page): Promise<void> {
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.2);
}

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => {
    throw error;
  });
});

test("loads only from this server and lists recordings honestly", async ({ page }) => {
  const outside: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol.startsWith("http") && url.host !== "127.0.0.1:18852") outside.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByTestId(`file-${IQ_FILE}`)).toContainText("IQ");
  await expect(page.getByTestId(`file-${IQ_FILE}`)).toContainText("7.200 MHz");
  await expect(page.getByTestId("file-audio_7000000Hz_12-00-00_03-11-2026.wav")).toContainText("AUDIO");
  await expect(page.getByTestId("file-mystery.wav")).toContainText("?");
  await expect(page.getByTestId("file-broken.wav")).toContainText("ERR");
  await expect(page.getByTestId("archive")).toHaveText("ARCHIVE ONLINE");
  expect(outside).toEqual([]);
});

test("a strange filename appears only as text", async ({ page }) => {
  await page.goto("/");
  const name = '<img src=x onerror=alert(1)> "odd".wav';
  const odd = page.getByTestId(`file-${name}`);
  await expect(odd).toContainText(name);
  expect(await odd.locator("img").count()).toBe(0);
});

test("plays, and the on-screen position follows the audio clock", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("play").click();
  await waitForSound(page);
  await expect(page.getByTestId("play")).toHaveText("PAUSE");

  // Over each half second, the on-screen position must match the audio clock, less the
  // output latency. The clock moves in steps longer than a video frame, so each half
  // second is averaged to compare against the middle of those steps.
  const gaps = await page.evaluate(async () => {
    const d = (window as unknown as { iqScopeDebug: Debug }).iqScopeDebug;
    const out: number[] = [];
    const end = performance.now() + 4000;
    while (performance.now() < end) {
      await new Promise((r) => requestAnimationFrame(r));
      const clock = d.engine.clockSeconds;
      if (clock !== null) out.push(clock - d.engine.latencySeconds - d.engine.seconds);
    }
    return out;
  });
  expect(gaps.length).toBeGreaterThan(50);
  const window = Math.floor(gaps.length / 8);
  const means = Array.from({ length: 8 }, (_, i) => {
    const part = gaps.slice(i * window, (i + 1) * window);
    return part.reduce((a, b) => a + b, 0) / part.length;
  });
  for (const mean of means) expect(Math.abs(mean)).toBeLessThan(1 / 60);

  // Motion on screen must be steady: within one video frame of a straight line in time.
  const motion = await page.evaluate(async () => {
    const d = (window as unknown as { iqScopeDebug: Debug }).iqScopeDebug;
    const rows: [number, number][] = [];
    const end = performance.now() + 3000;
    while (performance.now() < end) {
      await new Promise((r) => requestAnimationFrame(r));
      rows.push([performance.now() / 1000, d.engine.seconds]);
    }
    return rows;
  });
  const drift = motion.map(([wall, pos]) => pos - motion[0][1] - (wall - motion[0][0]));
  const middle = (Math.max(...drift) + Math.min(...drift)) / 2;
  expect(Math.max(...drift.map((v) => Math.abs(v - middle)))).toBeLessThan(1 / 60);

  const startPos = await seconds(page);
  const startWall = Date.now();
  await page.waitForTimeout(3000);
  const rate = ((await seconds(page)) - startPos) / ((Date.now() - startWall) / 1000);
  expect(rate).toBeGreaterThan(0.97);
  expect(rate).toBeLessThan(1.03);
});

test("pausing holds the position and resuming continues from it", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("play").click();
  await waitForSound(page);
  await page.getByTestId("play").click();
  await expect(page.getByTestId("play")).toHaveText("PLAY");
  const held = await seconds(page);
  await page.waitForTimeout(600);
  expect(await seconds(page)).toBeCloseTo(held, 3);
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(held + 0.3);
  expect(await seconds(page)).toBeLessThan(held + 6);
});

test("clicking the waterfall tunes to where you clicked", async ({ page }) => {
  await openIqFile(page);
  const canvas = page.getByTestId("waterfall");
  const box = (await canvas.boundingBox())!;
  const plotWidth = box.width - 62;
  // +50 kHz in a 1.024 MHz band, measured across the plot area.
  const x = 62 + plotWidth * ((50_000 + 512_000) / 1_024_000);
  await canvas.click({ position: { x, y: box.height / 2 } });
  const state = await debug(page, (d) => d.state());
  expect(Math.abs(state.offsetHz - 50_000)).toBeLessThan(2000);
  await expect(page.getByTestId("title-frequency")).toContainText("7.25");
});

const RULER_LEFT = 62;

async function hzToX(page: Page, offsetHz: number): Promise<number> {
  const box = (await page.getByTestId("waterfall").boundingBox())!;
  return RULER_LEFT + (box.width - RULER_LEFT) * ((offsetHz + 512_000) / 1_024_000);
}

async function dragOnWaterfall(page: Page, fromX: number, toX: number, y: number, steps = 8): Promise<void> {
  const box = (await page.getByTestId("waterfall").boundingBox())!;
  await page.mouse.move(box.x + fromX, box.y + y);
  await page.mouse.down();
  await page.mouse.move(box.x + toX, box.y + y, { steps });
  await page.mouse.up();
}

test("dragging the tuner makes it follow the cursor, without moving the view", async ({ page }) => {
  await openIqFile(page);
  const y = 300;
  await dragOnWaterfall(page, await hzToX(page, 0), await hzToX(page, 100_000), y);
  const state = await debug(page, (d) => d.state());
  expect(Math.abs(state.offsetHz - 100_000)).toBeLessThan(2000);
  await expect(page.getByTestId("title-frequency")).toContainText("7.30");

  // Grabbing it slightly off-center must not make it jump by the difference.
  await dragOnWaterfall(page, (await hzToX(page, 100_000)) + 5, await hzToX(page, 0), y);
  const again = await debug(page, (d) => d.state());
  const fivePixelsHz = (5 / ((await page.getByTestId("waterfall").boundingBox())!.width - RULER_LEFT)) * 1_024_000;
  expect(Math.abs(again.offsetHz + fivePixelsHz)).toBeLessThan(2000);
});

test("the tuner stays within the band and follows only left and right", async ({ page }) => {
  await openIqFile(page);
  const box = (await page.getByTestId("waterfall").boundingBox())!;
  const start = await hzToX(page, 0);
  await page.mouse.move(box.x + start, box.y + 300);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width + 300, box.y + 60, { steps: 6 });
  await page.mouse.up();
  const state = await debug(page, (d) => d.state());
  expect(state.offsetHz).toBeLessThanOrEqual(512_000);
  expect(state.offsetHz).toBeGreaterThan(500_000);
  expect(await seconds(page)).toBe(0);
});

test("dragging away from the tuner pans the view and leaves the tuning alone", async ({ page }) => {
  await openIqFile(page);
  await page.getByRole("button", { name: "+" }).first().click();
  const before = await debug(page, (d) => d.state());
  await dragOnWaterfall(page, await hzToX(page, 150_000), await hzToX(page, 100_000), 300);
  expect((await debug(page, (d) => d.state())).offsetHz).toBe(before.offsetHz);
});

test("sound keeps going while the tuner is dragged", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);

  const box = (await page.getByTestId("waterfall").boundingBox())!;
  await page.mouse.move(box.x + (await hzToX(page, 0)), box.y + 300);
  await page.mouse.down();
  const positions: number[] = [];
  const startedAt = Date.now();
  for (let i = 1; i <= 20; i++) {
    await page.mouse.move(box.x + (await hzToX(page, i * 3000)), box.y + 300);
    positions.push(await seconds(page));
    await page.waitForTimeout(100);
  }
  const elapsed = (Date.now() - startedAt) / 1000;
  await page.mouse.up();

  // Throughout the drag the playhead keeps real time, with no restart or freeze.
  const moved = positions[positions.length - 1] - positions[0];
  expect(moved / elapsed).toBeGreaterThan(0.9);
  expect(moved / elapsed).toBeLessThan(1.1);
  for (let i = 1; i < positions.length; i++) expect(positions[i]).toBeGreaterThanOrEqual(positions[i - 1]);
  expect(await debug(page, (d) => d.engine.isPlaying)).toBe(true);

  // The final tuning is what ends up playing, and the sound continues.
  const state = await debug(page, (d) => d.state());
  expect(Math.abs(state.offsetHz - 60_000)).toBeLessThan(2000);
  const after = await seconds(page);
  await expect.poll(() => seconds(page), { timeout: 10_000 }).toBeGreaterThan(after + 1);
});

test("sound keeps going during a drag even when the server is slow", async ({ page }) => {
  // The real server takes about 0.5 s per chunk, more when it's busy. This machine is faster, so this adds
  // enough for about 0.8 s. Beyond 1 s a chunk can't keep up with real time at all.
  const offsets: { offset: number; at: number }[] = [];
  await page.route("**/chunk?*", async (route) => {
    offsets.push({ offset: Number(new URL(route.request().url()).searchParams.get("offset_hz")), at: Date.now() });
    await new Promise((resolve) => setTimeout(resolve, 600));
    await route.continue();
  });
  await openIqFile(page);
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 30_000 }).toBeGreaterThan(1);
  await page.waitForTimeout(2500); // let the normal buffer fill, as it would for a user

  const box = (await page.getByTestId("waterfall").boundingBox())!;
  await page.mouse.move(box.x + (await hzToX(page, 0)), box.y + 300);
  await page.mouse.down();
  const positions: number[] = [];
  const times: number[] = [];
  // Worked out once, so the drag takes a known time and stays well short of the file's end,
  // where there would be nothing left to fetch.
  const zero = await hzToX(page, 0);
  const pixelsPerHz = ((await hzToX(page, 100_000)) - zero) / 100_000;
  for (let i = 1; i <= 50; i++) {
    await page.mouse.move(box.x + zero + i * 2000 * pixelsPerHz, box.y + 300);
    positions.push(await seconds(page));
    times.push(Date.now());
    await page.waitForTimeout(100);
  }
  const elapsed = (times[times.length - 1] - times[0]) / 1000;
  const dragEnded = Date.now();
  await page.mouse.up();

  // A long drag with slow chunks: the playhead must never stall.
  const moved = positions[positions.length - 1] - positions[0];
  expect(moved / elapsed).toBeGreaterThan(0.95);
  let longestStall = 0;
  let stallStart = positions[0];
  let stallIndex = 0;
  positions.forEach((position, i) => {
    if (position > stallStart + 0.02) {
      stallStart = position;
      stallIndex = i;
    }
    longestStall = Math.max(longestStall, i - stallIndex);
  });
  expect(longestStall).toBeLessThanOrEqual(2);
  expect(await debug(page, (d) => d.engine.isPlaying)).toBe(true);

  // Once settled, everything fetched uses the final tuning, including what's ahead.
  await page.waitForTimeout(3500);
  const final = (await debug(page, (d) => d.state())).offsetHz;
  expect(Math.abs(final - 100_000)).toBeLessThan(2000);
  const settled = offsets.filter((o) => o.at > dragEnded + 1000);
  expect(settled.length).toBeGreaterThan(0);
  for (const o of settled) expect(o.offset).toBe(final);
});

test("a retune that can't work stops playback and says why", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("play").click();
  await waitForSound(page);
  await page.getByTestId("mode-wfm").click();
  await page.getByTestId("tune").fill("7650");
  await page.getByTestId("tune").press("Enter");
  await expect(page.getByTestId("status")).toContainText("only reaches", { timeout: 15_000 });
  await expect(page.getByTestId("play")).toHaveText("PLAY");
});

test("RESET puts the floor and ceiling back for this recording", async ({ page }) => {
  await openIqFile(page);
  const floor = page.locator("#floor");
  const ceiling = page.locator("#ceiling");
  // Wait for the automatic values, which differ from the page's starting ones.
  await expect.poll(() => floor.inputValue(), { timeout: 20_000 }).not.toBe("90");
  const autoFloor = await floor.inputValue();
  const autoCeiling = await ceiling.inputValue();

  await floor.fill("20");
  await ceiling.fill("240");
  expect(await floor.inputValue()).toBe("20");

  await page.getByTestId("contrast-reset").click();
  expect(await floor.inputValue()).toBe(autoFloor);
  expect(await ceiling.inputValue()).toBe(autoCeiling);
});

test("retune and mode change keep the same position and keep playing", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("tune").fill("7250");
  await page.getByTestId("tune").press("Enter");
  await page.getByTestId("mode-usb").click();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1.5);

  const before = await seconds(page);
  await page.getByTestId("mode-lsb").click();
  await page.waitForTimeout(150);
  const after = await seconds(page);
  expect(Math.abs(after - before)).toBeLessThan(1);
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(after + 0.3);
  expect((await debug(page, (d) => d.state())).mode).toBe("lsb");
  expect(await debug(page, (d) => d.engine.isPlaying)).toBe(true);
});

test("scrubbing moves at once while paused, and playback resumes from the new spot", async ({ page }) => {
  await openIqFile(page);
  const scrub = page.getByTestId("scrub");
  await scrub.evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    el.value = "12";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(await seconds(page)).toBeCloseTo(12, 3);
  await expect(page.getByTestId("time")).toContainText("00:12.000");
  await scrub.evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(12.3);
  expect(await seconds(page)).toBeLessThan(17);
});

test("scrubbing during playback continues from the release point", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("play").click();
  await waitForSound(page);
  const scrub = page.getByTestId("scrub");
  await scrub.evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    el.value = "15";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(15.2);
  expect(await debug(page, (d) => d.engine.isPlaying)).toBe(true);
});

test("playing to the end stops cleanly at the end", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("scrub").evaluate((el: HTMLInputElement) => {
    el.value = "18.5";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.getByTestId("play").click();
  await expect(page.getByTestId("status")).toHaveText("END OF RECORDING", { timeout: 20_000 });
  expect(await seconds(page)).toBeCloseTo(20, 1);
  await expect(page.getByTestId("play")).toHaveText("PLAY");
});

test("tuning that can't work is refused with the reason shown", async ({ page }) => {
  await openIqFile(page);
  await page.getByTestId("mode-wfm").click();
  await page.getByTestId("tune").fill("7650");
  await page.getByTestId("tune").press("Enter");
  await page.getByTestId("play").click();
  await expect(page.getByTestId("status")).toContainText("only reaches", { timeout: 15_000 });
  await expect(page.getByTestId("play")).toHaveText("PLAY");
});

test("an audio recording says it has no IQ and can't be tuned, but it can be played", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-audio_7000000Hz_12-00-00_03-11-2026.wav").click();
  await expect(page.locator("#banner")).toContainText("AUDIO RECORDING: NO IQ DATA");
  await expect(page.locator("#banner")).toContainText("playing one channel");
  await expect(page.getByTestId("play")).toBeEnabled();
  await expect(page.getByTestId("tune")).toBeHidden(); // nothing to tune in sound
  await expect(page.getByTestId("mode-am")).toBeHidden();
});

test("an unknown file asks, and remembers the answer", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-mystery.wav").click();
  await expect(page.locator("#question")).toContainText("WHAT IS THIS FILE?");
  await page.getByRole("button", { name: "IT'S AUDIO" }).click();
  await expect(page.getByTestId("file-mystery.wav")).toContainText("AUDIO");
  await page.reload();
  await expect(page.getByTestId("file-mystery.wav")).toContainText("AUDIO");
});

test("the waterfall really draws the signal", async ({ page }) => {
  await openIqFile(page);
  await expect.poll(
    () =>
      page.getByTestId("waterfall").evaluate((c: HTMLCanvasElement) => {
        const g = c.getContext("2d")!;
        const d = g.getImageData(c.width / 2, c.height / 2, c.width / 3, 40).data;
        let brightest = 0;
        for (let i = 1; i < d.length; i += 4) brightest = Math.max(brightest, d[i]);
        return brightest;
      }),
    { timeout: 20_000 },
  ).toBeGreaterThan(120);
});

test("the layout holds together at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await openIqFile(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await expect(page.getByTestId("waterfall")).toBeVisible();
  await expect(page.getByTestId("play")).toBeVisible();
});
