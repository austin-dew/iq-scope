import { expect, test, type Page } from "@playwright/test";

const IQ_FILE = "baseband_7200000Hz_12-00-00_03-11-2026.wav";
const IQ_RATE = 1_024_000;
const seconds = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.transport.seconds) as Promise<number>;
const playing = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.engine.isPlaying) as Promise<boolean>;
const seek = (page: Page, t: number) => page.evaluate((s) => (window as any).iqScopeDebug.transport.seek(s), t);
const state = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.state());
const cssVar = (page: Page, name: string) =>
  page.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), name);

async function openFile(page: Page, name = IQ_FILE): Promise<void> {
  await page.goto("/");
  await page.getByTestId(`file-${name}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
}

/** A place on the waterfall's plot, in page coordinates, away from the tuner. */
async function plotPoint(page: Page, fx: number, fy: number): Promise<{ x: number; y: number }> {
  const box = (await page.getByTestId("waterfall").boundingBox())!;
  return { x: box.x + 62 + fx * (box.width - 62), y: box.y + 20 + fy * (box.height - 20) };
}

async function shiftClick(page: Page, x: number, y: number): Promise<void> {
  await page.mouse.move(x, y);
  await page.keyboard.down("Shift");
  await page.mouse.down();
  await page.mouse.up();
  await page.keyboard.up("Shift");
}

/** A summary of what a canvas shows, to tell whether it changed. */
const picture = (page: Page, id: string) =>
  page.evaluate((canvasId) => {
    const c = document.getElementById(canvasId) as HTMLCanvasElement;
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let k = 0; k < d.length; k += 4) if (d[k] + d[k + 1] + d[k + 2] > 40) lit++;
    return lit;
  }, id);

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => { throw error; });
});

// --- The look ---------------------------------------------------------------------------

test("both fonts load from this server and nothing comes from anywhere else", async ({ page, baseURL }) => {
  const origins = new Set<string>();
  page.on("request", (r) => {
    if (!r.url().startsWith("data:")) origins.add(new URL(r.url()).origin);
  });
  await openFile(page);
  await page.evaluate(() => document.fonts.ready);
  await expect.poll(() => page.evaluate(() => document.fonts.check('26px "VT323"') && document.fonts.check('14px "Share Tech Mono"'))).toBe(true);
  const loaded = await page.evaluate(() => [...document.fonts].filter((f) => f.status === "loaded").map((f) => f.family.replaceAll('"', "")).sort());
  expect(loaded).toEqual(["Share Tech Mono", "VT323"]);
  expect([...origins]).toEqual([new URL(baseURL!).origin]);
  expect((await page.request.get("/fonts/LICENSE-vt323.txt")).status()).toBe(200);
});

test("amber switches the page and the plots, including the waterfall, and is remembered", async ({ page }) => {
  await openFile(page);
  const greenFg = await cssVar(page, "--fg");
  const redVersusGreen = () =>
    page.evaluate(() => {
      const c = document.querySelector<HTMLCanvasElement>("#waterfall")!;
      const d = c.getContext("2d")!.getImageData(70, 30, c.width - 80, c.height - 40).data;
      let red = 0;
      let green = 0;
      for (let k = 0; k < d.length; k += 4) {
        red += d[k];
        green += d[k + 1];
      }
      return red / Math.max(green, 1);
    });
  await expect.poll(redVersusGreen, { timeout: 20_000 }).toBeLessThan(0.7);

  await page.getByTestId("open-settings").click();
  await page.getByTestId("theme-amber").click();
  await expect(page.getByTestId("theme-amber")).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("close-settings").click();
  expect(await cssVar(page, "--fg")).not.toBe(greenFg);
  await expect.poll(redVersusGreen).toBeGreaterThan(1.1);

  await page.reload();
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("amber");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect.poll(redVersusGreen, { timeout: 20_000 }).toBeGreaterThan(1.1);
});

const THEMES = ["green", "amber", "white", "blue", "magenta"];

/** How much brighter one color is than another: WCAG's contrast ratio. */
function contrast(a: string, b: string): number {
  const luminance = (hex: string) => {
    const [r, g, c] = [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16) / 255).map((v) =>
      v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4,
    );
    return 0.2126 * r + 0.7152 * g + 0.0722 * c;
  };
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

async function chooseTheme(page: Page, theme: string): Promise<void> {
  await page.getByTestId("open-settings").click();
  await page.getByTestId(`theme-${theme}`).click();
  await expect(page.getByTestId(`theme-${theme}`)).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("close-settings").click();
}

test("in every color, text stays readable and the page and plots use the same shades", async ({ page }) => {
  await page.goto("/");
  for (const theme of THEMES) {
    await chooseTheme(page, theme);
    const plot = await page.evaluate(() => ({ ...(window as any).iqScopeDebug.colors }));
    expect(await cssVar(page, "--fg"), theme).toBe(plot.fg);
    expect(await cssVar(page, "--fg-bright"), theme).toBe(plot.bright);
    expect(await cssVar(page, "--dim"), theme).toBe(plot.dim);
    expect(await cssVar(page, "--plot"), theme).toBe(plot.plot);
    const bg = await cssVar(page, "--bg");
    expect(contrast(plot.fg, bg), `${theme} text`).toBeGreaterThan(7);
    expect(contrast(plot.bright, bg), `${theme} bright text`).toBeGreaterThan(7);
    expect(contrast(plot.dim, bg), `${theme} labels`).toBeGreaterThan(4.5);
    expect(contrast(await cssVar(page, "--alert"), bg), `${theme} alerts`).toBeGreaterThan(4.5);
    expect(contrast(plot.accent, plot.plot), `${theme} tuner line`).toBeGreaterThan(4.5);
    // The tuner has to stand out from the trace it sits on.
    expect(plot.accent, theme).not.toBe(plot.fg);
    expect(contrast(plot.grid, plot.plot), `${theme} grid`).toBeGreaterThan(1.2);
  }
});

test("white, blue and magenta recolor the waterfall too, and are remembered", async ({ page }) => {
  await openFile(page);
  const tint = () =>
    page.evaluate(() => {
      const c = document.querySelector<HTMLCanvasElement>("#waterfall")!;
      const d = c.getContext("2d")!.getImageData(70, 30, c.width - 80, c.height - 40).data;
      const sum = [0, 0, 0];
      for (let k = 0; k < d.length; k += 4) for (let n = 0; n < 3; n++) sum[n] += d[k + n];
      return { redPerGreen: sum[0] / Math.max(sum[1], 1), bluePerGreen: sum[2] / Math.max(sum[1], 1) };
    });
  await chooseTheme(page, "blue");
  await expect.poll(async () => (await tint()).bluePerGreen, { timeout: 20_000 }).toBeGreaterThan(1.3);
  expect((await tint()).redPerGreen).toBeLessThan(0.5);

  await chooseTheme(page, "white");
  await expect.poll(async () => (await tint()).redPerGreen).toBeGreaterThan(0.85);
  expect((await tint()).bluePerGreen).toBeGreaterThan(0.85);

  await chooseTheme(page, "magenta");
  await expect.poll(async () => (await tint()).redPerGreen).toBeGreaterThan(1.5);
  expect((await tint()).bluePerGreen).toBeGreaterThan(1.2);

  await chooseTheme(page, "white");
  await page.reload();
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("white");
  expect(await page.evaluate(() => (window as any).iqScopeDebug.colors.fg)).toBe(await cssVar(page, "--fg"));
});

test("a saved color this version doesn't know falls back to green", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.setItem("iqscope.settings", JSON.stringify({ theme: "<b>plaid" })));
  await page.reload();
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("green");
  expect(await page.evaluate(() => (window as any).iqScopeDebug.colors.fg)).toBe(await cssVar(page, "--fg"));
});

test("screen effects stay off the plots and readouts, and can be turned off", async ({ page }) => {
  await openFile(page);
  const check = () =>
    page.evaluate(() => {
      const bad: string[] = [];
      for (const id of ["waterfall", "constellation", "phasor", "spectrum", "traces", "waveform"]) {
        const canvas = document.getElementById(id)!;
        const box = canvas.getBoundingClientRect();
        // Sample across the plot: the plot itself must be what's on top everywhere.
        for (const [fx, fy] of [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]]) {
          const top = document.elementFromPoint(box.left + fx * box.width, box.top + fy * box.height);
          if (top !== canvas && !top?.closest("figcaption, .closeup-bar")) bad.push(`${id} is covered by ${top?.tagName}.${top?.className}`);
        }
        for (let e: HTMLElement | null = canvas; e; e = e.parentElement) {
          const style = getComputedStyle(e);
          if (style.filter !== "none" || style.mixBlendMode !== "normal" || style.transform !== "none") bad.push(`${e.tagName} alters ${id}`);
        }
      }
      for (const selector of ["#pair-sample", "#time", "#title-frequency", "#tune", "#analysis-numbers"]) {
        if (getComputedStyle(document.querySelector(selector)!).textShadow !== "none") bad.push(`${selector} glows`);
      }
      // Nothing is laid over the page as a whole.
      for (const e of [document.documentElement, document.body]) {
        for (const part of ["::before", "::after"]) {
          if (getComputedStyle(e, part).content !== "none") bad.push(`${e.tagName}${part} exists`);
        }
      }
      return bad;
    });
  expect(await check()).toEqual([]);
  const body = () => page.evaluate(() => {
    const s = getComputedStyle(document.body);
    return { radius: s.borderTopLeftRadius, shadow: s.boxShadow, title: getComputedStyle(document.querySelector("h1")!).textShadow };
  });
  expect((await body()).radius).not.toBe("0px");
  expect((await body()).title).not.toBe("none");

  await page.getByTestId("open-settings").click();
  await page.getByTestId("effects-off").click();
  await page.getByTestId("close-settings").click();
  expect(await body()).toEqual({ radius: "0px", shadow: "none", title: "none" });
  expect(await check()).toEqual([]);
  await page.reload();
  expect(await page.evaluate(() => document.documentElement.dataset.effects)).toBe("off");
});

test("reduced motion: no boot, no blinking, and the settings say why", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("boot")).toBeHidden();
  expect(await page.evaluate(() => getComputedStyle(document.querySelector("h1")!, "::after").content)).toBe("none");
  await page.getByTestId("open-settings").click();
  await expect(page.getByTestId("motion-note")).toBeVisible();
});

test.describe("with motion allowed", () => {
  test.use({ contextOptions: { reducedMotion: "no-preference" } });

  test("the boot sequence reports real results, ends itself, and runs once per session", async ({ page }) => {
    await page.goto("/");
    const boot = page.getByTestId("boot");
    await expect(boot).toBeVisible();
    await expect(boot).toContainText("ARCHIVE LINK");
    await expect(boot).toContainText(/RECORDINGS \.+ \d+ FOUND/);
    await expect(boot).toBeHidden({ timeout: 6000 });
    await page.reload();
    await expect(page.locator("h1")).toBeVisible();
    await expect(boot).toBeHidden();
    await page.waitForTimeout(300);
    await expect(boot).toBeHidden();
  });

  test("a key skips the boot sequence and does nothing else", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("boot")).toBeVisible();
    await page.keyboard.press("Space");
    await expect(page.getByTestId("boot")).toBeHidden();
    expect(await playing(page)).toBe(false);
  });

  test("a click skips the boot sequence and doesn't reach what's behind it", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("boot")).toBeVisible();
    await page.mouse.click(600, 300);
    await expect(page.getByTestId("boot")).toBeHidden();
    expect(await page.evaluate(() => document.querySelector("dialog")!.open)).toBe(false);
  });

  test("turning off effects or the boot setting stops the boot sequence next session", async ({ page, context }) => {
    await page.goto("/");
    await page.keyboard.press("Enter");
    await page.getByTestId("open-settings").click();
    await page.getByTestId("boot-off").click();
    await page.getByTestId("close-settings").click();
    const next = await context.newPage();
    await next.goto("/");
    await expect(next.locator("h1")).toBeVisible();
    await expect(next.getByTestId("boot")).toBeHidden();
  });
});

test("when the archive is down, the reason is on the screen in plain words", async ({ page }) => {
  const detail = "The share didn't answer within 3 s.";
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: { status: "degraded", share: { mounted: true, readable: false, detail } } }),
  );
  await page.goto("/");
  await expect(page.getByTestId("archive")).toHaveText("ARCHIVE OFFLINE");
  await expect(page.getByTestId("archive-note")).toHaveText(`ARCHIVE OFFLINE: ${detail}`);
  await page.unroute("**/api/health");
  await expect(page.getByTestId("archive-note")).toBeHidden({ timeout: 10_000 });
});

test("settings keep the keyboard to themselves and close with Escape", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("open-settings").click();
  await page.keyboard.press("Space");
  await page.keyboard.press("KeyM");
  expect(await playing(page)).toBe(false);
  await expect(page.getByTestId("markers")).toBeHidden();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("settings")).toBeHidden();
});

// --- Analysis Mode ----------------------------------------------------------------------

test("Analysis Mode asks the server for nothing new, and playback doesn't notice it", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  const tuningBefore = await state(page);
  const before = await seconds(page);
  const asked: string[] = [];
  page.on("request", (r) => {
    if (/\/raw\?/.test(r.url())) asked.push(r.url());
  });
  await page.getByTestId("analysis-toggle").click();
  await expect(page.getByTestId("analysis-toggle")).toHaveText("[ ANALYSIS: ON ]");
  await page.waitForTimeout(400);
  expect(await playing(page)).toBe(true);
  expect(await seconds(page)).toBeGreaterThan(before);
  expect(await state(page)).toEqual(tuningBefore);
  await page.getByTestId("analysis-toggle").click();
  await expect(page.getByTestId("analysis-bar")).toBeHidden();
  expect(await playing(page)).toBe(true);
  expect(asked).toEqual([]);
});

test("Analysis Mode is remembered in this browser", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  await page.reload();
  await expect(page.getByTestId("analysis-toggle")).toHaveText("[ ANALYSIS: ON ]");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("analysis-bar")).toBeVisible();
});

test("every plot gains axes and grid", async ({ page }) => {
  await openFile(page);
  await seek(page, 2);
  await expect.poll(() => page.evaluate(() => (window as any).iqScopeDebug.views.dataReady), { timeout: 20_000 }).toBe(true);
  await page.waitForTimeout(600);
  const canvases = ["constellation", "phasor", "traces", "waveform"];
  const off: Record<string, number> = {};
  for (const id of canvases) off[id] = await picture(page, id);
  await page.getByTestId("analysis-toggle").click();
  await page.waitForTimeout(200);
  for (const id of canvases) expect(await picture(page, id), id).toBeGreaterThan(off[id]);

  // And the Signal Path, once its plots have data.
  await page.getByTestId("analysis-toggle").click();
  await page.getByTestId("tab-path").click();
  const paths = ["path-raw", "path-tuned", "path-filtered", "path-demod", "path-sound"];
  await expect.poll(() => picture(page, "path-tuned"), { timeout: 20_000 }).toBeGreaterThan(200);
  await page.waitForTimeout(600);
  const pathOff: Record<string, number> = {};
  for (const id of paths) pathOff[id] = await picture(page, id);
  await page.getByTestId("analysis-toggle").click();
  await page.waitForTimeout(200);
  for (const id of paths) expect(await picture(page, id), id).toBeGreaterThan(pathOff[id]);
});

test("trace time labels match where those times really are", async ({ page }) => {
  // Note every label drawn, with its position, as a canvas keeps no record of its text.
  await page.addInitScript(() => {
    const seen: { text: string; x: number; width: number }[] = [];
    (window as any).labels = seen;
    const original = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (text: string, x: number, y: number) {
      if (this.canvas.id === "traces") seen.push({ text, x, width: this.canvas.clientWidth });
      return original.call(this, text, x, y);
    };
  });
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  await seek(page, 2);
  await expect.poll(() => page.evaluate(() => (window as any).labels.some((l: any) => l.text === "+5 ms")), { timeout: 20_000 }).toBe(true);
  const labels = await page.evaluate(() => (window as any).labels.slice(-40));
  const plus = labels.find((l: any) => l.text === "+5 ms");
  const minus = labels.find((l: any) => l.text === "-5 ms");
  // A 20 ms window centered on the cursor: +5 ms is three-quarters of the way across.
  expect(plus.x / plus.width).toBeCloseTo(0.75, 2);
  expect(minus.x / minus.width).toBeCloseTo(0.25, 2);
});

test("the numbers strip shows true sample numbers, frequency, offset and uncalibrated levels", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  await page.getByTestId("tune").fill("7212.5");
  await page.getByTestId("tune").press("Enter");
  await seek(page, 2);
  const numbers = page.getByTestId("analysis-numbers");
  await expect(numbers).toContainText(`RAW    #${(2 * IQ_RATE).toLocaleString("en-US")}`);
  await expect(numbers).toContainText("TUNED 7.212 500 MHz");
  await expect(numbers).toContainText("OFFSET +12,500 Hz");
  await expect(numbers).toContainText("PAIR   #96,000", { timeout: 20_000 });
  await expect(numbers).toContainText("AUDIO  #96,000");
  await expect(numbers).toContainText(/FILTER -\d+\.\d dBFS/);
  await expect(page.getByTestId("analysis-bar")).toContainText("NOT CALIBRATED");
});

test("cursors read frequency and time differences, and don't tune", async ({ page }) => {
  await openFile(page);
  const readout = page.getByTestId("cursor-readout");
  await page.getByTestId("analysis-toggle").click();
  const before = (await state(page)).offsetHz;
  // A quarter of the band and a quarter of the 20 s recording apart.
  const a = await plotPoint(page, 0.25, 0.25);
  const b = await plotPoint(page, 0.5, 0.5);
  await shiftClick(page, a.x, a.y);
  await expect(readout).toContainText("A ");
  await expect(readout).toContainText("B: SHIFT+CLICK AGAIN");
  await shiftClick(page, b.x, b.y);
  const text = (await readout.textContent())!;
  const dF = Number(/dF \+([\d,]+) Hz/.exec(text)![1].replaceAll(",", ""));
  const dT = Number(/dT \+([\d.]+) s/.exec(text)![1]);
  expect(Math.abs(dF - IQ_RATE / 4)).toBeLessThan(IQ_RATE / 400);
  expect(Math.abs(dT - 5)).toBeLessThan(0.1);
  expect((await state(page)).offsetHz).toBe(before);
  // The same difference is shown beside cursor B, for reading it on any tab.
  const note = await page.evaluate(() => (window as any).iqScopeDebug.waterfall.cursorNote);
  expect(note).toContain(/dF \+[\d,]+ Hz/.exec(text)![0]);
  expect(note).toContain(/dT \+[\d.]+ s/.exec(text)![0]);

  // A third starts again, and CLEAR empties them.
  await shiftClick(page, a.x, a.y);
  await expect(readout).toContainText("B: SHIFT+CLICK AGAIN");
  await page.getByTestId("cursor-clear").click();
  await expect(readout).toContainText("SHIFT+CLICK THE WATERFALL");

  // An ordinary click still tunes.
  await page.mouse.click(a.x, a.y);
  await expect.poll(async () => (await state(page)).offsetHz).not.toBe(before);
});

test("PLACE CURSORS works with no Shift key, and Shift means nothing outside Analysis Mode", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  const before = (await state(page)).offsetHz;
  const a = await plotPoint(page, 0.3, 0.3);
  await page.getByTestId("cursor-arm").click();
  await page.mouse.click(a.x, a.y);
  await expect(page.getByTestId("cursor-readout")).toContainText("A ");
  expect((await state(page)).offsetHz).toBe(before);
  await page.getByTestId("cursor-arm").click();
  await page.getByTestId("analysis-toggle").click();
  await shiftClick(page, a.x, a.y);
  await expect.poll(async () => (await state(page)).offsetHz).not.toBe(before);
});

test("the raw constellation shows the file's own pairs, and asks for them only while on", async ({ page }) => {
  await openFile(page);
  let requests = 0;
  page.on("request", (r) => {
    if (/\/raw\?/.test(r.url())) requests++;
  });
  await page.getByTestId("analysis-toggle").click();
  await seek(page, 2);
  await expect.poll(() => page.evaluate(() => (window as any).iqScopeDebug.views.dataReady), { timeout: 20_000 }).toBe(true);
  await page.waitForTimeout(300);
  expect(requests).toBe(0);
  const tunedPair = await page.evaluate(() => (window as any).iqScopeDebug.views.lastValues);

  await page.getByTestId("constellation-raw").click();
  await expect.poll(() => requests).toBeGreaterThan(0);
  await expect.poll(() =>
    page.evaluate(() => {
      const b = (window as any).iqScopeDebug.rawPairs.burst;
      return b ? b.startPair + b.i.length - 1 : null;
    }),
  ).toBe(2 * IQ_RATE);
  await expect(page.getByTestId("analysis-numbers")).toContainText(/BAND   -\d+\.\d dBFS/);

  // What the sound is made from is untouched, and a paused playhead means one request, not many.
  expect(await page.evaluate(() => (window as any).iqScopeDebug.views.lastValues)).toEqual(tunedPair);
  const settled = requests;
  await page.waitForTimeout(500);
  expect(requests).toBe(settled);

  await seek(page, 3);
  await expect.poll(() => requests).toBeGreaterThan(settled);
  await page.getByTestId("constellation-tuned").click();
  await page.waitForTimeout(150);
  const after = requests;
  await seek(page, 4);
  await page.waitForTimeout(300);
  expect(requests).toBe(after);
});

test("raw requests during playback leave the sound running", async ({ page }) => {
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  await page.getByTestId("constellation-raw").click();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  const before = await seconds(page);
  await page.waitForTimeout(1500);
  expect(await playing(page)).toBe(true);
  expect(await seconds(page)).toBeGreaterThan(before + 1);
});

test("an audio recording gets its numbers, with no pairs to pretend about", async ({ page }) => {
  let requests = 0;
  page.on("request", (r) => {
    if (/\/raw\?/.test(r.url())) requests++;
  });
  await openFile(page, "audio_44100.wav");
  await page.getByTestId("analysis-toggle").click();
  await expect(page.getByTestId("constellation-source")).toBeHidden();
  await seek(page, 2);
  const numbers = page.getByTestId("analysis-numbers");
  await expect(numbers).toContainText("SAMPLE #88,200");
  await expect(numbers).toContainText("RATE   44,100 S/s");
  await expect(numbers).not.toContainText("PAIR");
  expect(requests).toBe(0);
});

test("Analysis Mode takes no height from the waterfall, and its numbers all fit", async ({ page }) => {
  for (const [width, height] of [[1896, 950], [1500, 950], [1400, 900], [1380, 800]]) {
    await page.setViewportSize({ width, height });
    await page.evaluate(() => localStorage.clear()).catch(() => {});
    await openFile(page);
    const off = (await page.getByTestId("waterfall").boundingBox())!.height;
    await page.getByTestId("analysis-toggle").click();
    await page.getByTestId("constellation-raw").click();
    await expect(page.getByTestId("analysis-numbers")).toContainText("BAND", { timeout: 20_000 });
    expect((await page.getByTestId("waterfall").boundingBox())!.height).toBe(off);

    const a = await plotPoint(page, 0.3, 0.3);
    const b = await plotPoint(page, 0.6, 0.6);
    await shiftClick(page, a.x, a.y);
    await shiftClick(page, b.x, b.y);
    const scroll = await page.locator(".readout-body").evaluate((e) => e.scrollHeight - e.clientHeight);
    expect(scroll, `readout overflows at ${width} wide`).toBeLessThanOrEqual(0);
    const closeup = (await page.locator(".panel.spectrum").boundingBox())!;
    expect(closeup.width).toBeGreaterThan(220);
    await expect(page.getByTestId("pair-sample")).not.toHaveText("WAITING FOR DATA", { timeout: 20_000 });
    const clipped = await page.getByTestId("pair-sample").evaluate((e) => e.scrollWidth - e.clientWidth);
    expect(clipped, `PAIR > SAMPLE is cut off at ${width} wide`).toBeLessThanOrEqual(0);
    const transport = (await page.getByTestId("play").boundingBox())!;
    expect(transport.y + transport.height).toBeLessThanOrEqual(height);
  }
});

test("everything still fits at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await openFile(page);
  await page.getByTestId("analysis-toggle").click();
  await page.getByTestId("constellation-raw").click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await page.getByTestId("open-settings").click();
  await expect(page.getByTestId("theme-magenta")).toBeVisible();
  const box = (await page.getByTestId("settings").boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
});
