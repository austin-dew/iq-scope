import { expect, test, type Page } from "@playwright/test";

const IQ_FILE = "baseband_7200000Hz_12-00-00_03-11-2026.wav";
const state = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.state());
const seconds = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.transport.seconds) as Promise<number>;
const playing = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.engine.isPlaying) as Promise<boolean>;
const seek = (page: Page, t: number) => page.evaluate((s) => (window as any).iqScopeDebug.transport.seek(s), t);

async function openIq(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
}

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => { throw error; });
  // Every test starts with a clean slate of markers on the shared fixture file.
  await page.goto("/");
  await page.evaluate(async () => {
    const list = await (await fetch("/api/recordings")).json();
    for (const r of list.recordings) {
      if ((r.markers ?? []).length && r.kind !== "unreadable") {
        await fetch(`/api/recordings/${r.id}/sidecar`, {
          method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ markers: [] }),
        });
      }
    }
  });
});

// --- Tabs ---------------------------------------------------------------------------------

test("tabs swap the lower panels while the waterfall and playback stay put", async ({ page }) => {
  await openIq(page);
  await expect(page.getByTestId("constellation")).toBeVisible();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  for (const [name, id] of [["path", "path"], ["notes", "notes"], ["markers", "markers"]]) {
    await page.getByTestId(`tab-${name}`).click();
    await expect(page.getByTestId(id)).toBeVisible();
    await expect(page.getByTestId("constellation")).toBeHidden();
    await expect(page.getByTestId("waterfall")).toBeVisible();
    expect(await playing(page)).toBe(true);
  }
  await page.getByTestId("tab-panels").click();
  await expect(page.getByTestId("constellation")).toBeVisible();
});

// --- Signal Path --------------------------------------------------------------------------

const path = (page: Page) => page.evaluate(() => {
  const s = (window as any).iqScopeDebug.signalPath.summary;
  if (!s) return null;
  return {
    rawAtAm: s.rawLevelAt(-100_000), rawAtUsbEmptyGap: s.rawLevelAt(300_000),
    tuned: s.tunedPeakHz, filtered: s.filteredPeakHz, upper: s.filteredUpper, lower: s.filteredLower,
    demodPeak: s.demodPeak, audioPeak: s.audioPeak,
  };
});

async function openPath(page: Page, khz: string, mode: string): Promise<void> {
  await openIq(page);
  await page.getByTestId(`mode-${mode}`).click();
  await page.getByTestId("tune").fill(khz);
  await page.getByTestId("tune").press("Enter");
  await page.getByTestId("tab-path").click();
  await seek(page, 3.5);
  await expect.poll(async () => { const p = await path(page); return p && p.filtered !== null && p.tuned !== null && p.demodPeak !== null && p.audioPeak !== null; }, { timeout: 20_000 }).toBe(true);
  await page.waitForTimeout(400);
}

test("each stage shows what it should for an AM signal", async ({ page }) => {
  await openPath(page, "7100", "am");
  const p = (await path(page))!;
  // Stage 1: the carrier is where it really is in the band, at its known level.
  expect(Math.abs(p.rawAtAm + 12)).toBeLessThan(3);
  expect(p.rawAtUsbEmptyGap).toBeLessThan(-30);
  // Stages 2 and 3: tuning has moved it to 0 Hz, and filtering keeps it there.
  expect(Math.abs(p.tuned!)).toBeLessThan(60);
  expect(Math.abs(p.filtered!)).toBeLessThan(30);
  // Stages 4 and 5: demodulating gives a small signal, and gain brings it up to listening level.
  expect(p.demodPeak).toBeGreaterThan(0.05);
  expect(p.audioPeak).toBeGreaterThan(p.demodPeak!);
  expect(p.audioPeak).toBeLessThanOrEqual(1);
  await expect(page.getByTestId("path-formula")).toContainText("AM: arrow length");
});

test("the filtered stage has removed the other sideband for USB", async ({ page }) => {
  await openPath(page, "7250", "usb");
  const p = (await path(page))!;
  expect(p.filtered!).toBeGreaterThan(200);
  expect(p.upper / p.lower).toBeGreaterThan(1000);
  await expect(page.getByTestId("path-formula")).toContainText("USB");
});

test("CW's filtered stage shows the carrier at the tuning, lifted by its tone", async ({ page }) => {
  await openPath(page, "7100", "cw");
  expect(Math.abs((await path(page))!.filtered!)).toBeLessThan(30);
});

test("the Signal Path is hidden for audio recordings", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("tab-path").click();
  await page.getByTestId("file-audio_44100.wav").click();
  await expect(page.getByTestId("tab-path")).toBeHidden();
  await expect(page.getByTestId("path")).toBeHidden();
});

test("panels 1 and 2 never blank out during playback, even with a slow server", async ({ page }) => {
  await page.route("**/slice?*", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.continue();
  });
  await openIq(page);
  await page.getByTestId("tab-path").click();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(1);
  const frames = await page.evaluate(async () => {
    const c = (window as any).iqScopeDebug;
    const canvases = ["path-raw", "path-tuned"].map((id) => document.getElementById(id) as HTMLCanvasElement);
    const lit = (canvas: HTMLCanvasElement) => {
      const d = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
      let n = 0;
      for (let i = 1; i < d.length; i += 4) if (d[i] > 180) n++;
      return n;
    };
    let empty = 0;
    let checked = 0;
    const end = performance.now() + 5000;
    while (performance.now() < end) {
      await new Promise((r) => requestAnimationFrame(r));
      checked++;
      for (const canvas of canvases) if (lit(canvas) < 40) empty++;
    }
    return { empty, checked, blank: c.signalPath.summary.blankFrames };
  });
  expect(frames.checked).toBeGreaterThan(50);
  expect(frames.empty).toBe(0);
});

// --- Notes --------------------------------------------------------------------------------

test("the notes follow the mode and cover the common artifacts", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("tab-notes").click();
  const notes = page.getByTestId("notes");
  await expect(notes).toContainText("LENGTH");
  await expect(notes).toContainText("DC spike");
  await expect(notes).toContainText("8 bits");
  await expect(notes).toContainText("Overload");
  await page.getByTestId("mode-nfm").click();
  await expect(notes).toContainText("SPIN SPEED");
  await page.getByTestId("mode-cw").click();
  await expect(notes).toContainText("ON and OFF");
  await page.getByTestId("file-audio_44100.wav").click();
  await expect(notes).toContainText("This file holds sound, not IQ");
  await expect(notes).not.toContainText("DC spike");
});

// --- Examples -----------------------------------------------------------------------------

async function openExamples(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId("show-examples").click();
  await expect(page.getByTestId("file-example_switch_7200000Hz.wav")).toBeVisible({ timeout: 45_000 });
}

test("examples are listed separately and every mode is there", async ({ page }) => {
  await openExamples(page);
  for (const m of ["am", "usb", "lsb", "cw", "nfm", "wfm"]) {
    await expect(page.getByTestId(`file-example_${m}_7200000Hz.wav`)).toContainText(m.toUpperCase());
  }
  await expect(page.getByTestId("file-list").locator(".file-name")).toHaveText(["AM", "USB", "LSB", "CW", "NFM", "WFM", "ALL MODES IN TURN"]);
  await expect(page.getByTestId(`file-${IQ_FILE}`)).toHaveCount(0);
  await page.getByTestId("show-files").click();
  await expect(page.getByTestId(`file-${IQ_FILE}`)).toBeVisible();
  await expect(page.getByTestId("file-example_am_7200000Hz.wav")).toHaveCount(0);
});

test("an example opens tuned and ready, in its own mode, and plays", async ({ page }) => {
  await openExamples(page);
  await page.getByTestId("file-example_lsb_7200000Hz.wav").click();
  await expect.poll(async () => (await state(page)).mode).toBe("lsb");
  expect((await state(page)).offsetHz).toBe(50_000);
  await expect(page.getByTestId("title-frequency")).toContainText("7.250");
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
});

test("in the clip that switches, the mode follows along and can still be changed by hand", async ({ page }) => {
  await openExamples(page);
  await page.getByTestId("file-example_switch_7200000Hz.wav").click();
  await expect.poll(async () => (await state(page)).mode).toBe("am");
  for (const [t, m] of [[3, "usb"], [5, "lsb"], [7, "cw"], [9, "nfm"], [11, "wfm"], [1, "am"]] as const) {
    await seek(page, t);
    await expect.poll(async () => (await state(page)).mode).toBe(m);
  }
  await page.getByTestId("mode-lsb").click();
  await page.waitForTimeout(200);
  expect((await state(page)).mode).toBe("lsb"); // stays yours until the next segment
});

// --- Markers ------------------------------------------------------------------------------

test("M marks the moment and is ready for a label, which survives a reload", async ({ page }) => {
  await openIq(page);
  await seek(page, 4.25);
  await page.keyboard.press("m");
  await expect(page.getByTestId("markers")).toBeVisible();
  await expect(page.getByTestId("marker-label")).toBeFocused();
  await page.keyboard.type("Preacher");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("marker-go")).toHaveText("00:04.250");
  await page.waitForTimeout(400);

  await page.reload();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("tab-markers").click();
  await expect(page.getByTestId("marker-label")).toHaveValue("Preacher");
});

test("typing M in a label is just the letter M, and space stays a space", async ({ page }) => {
  await openIq(page);
  await page.keyboard.press("m");
  await expect(page.getByTestId("marker-label")).toBeFocused();
  await page.keyboard.type("m m");
  await expect(page.getByTestId("marker-label")).toHaveValue("m m");
  await expect(page.getByTestId("marker-label")).toHaveCount(1);
  expect(await playing(page)).toBe(false);
});

test("jumping and deleting work, and markers stay in time order", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("tab-markers").click();
  for (const t of [9, 2, 5]) {
    await seek(page, t);
    await page.getByTestId("add-marker").click();
    await page.getByTestId("marker-label").first().blur();
    await page.waitForTimeout(150);
  }
  await expect(page.getByTestId("marker-go")).toHaveText(["00:02.000", "00:05.000", "00:09.000"]);
  await page.getByTestId("marker-go").nth(1).click();
  expect(await seconds(page)).toBeCloseTo(5, 3);
  await page.getByTestId("marker-delete").nth(0).click();
  await expect(page.getByTestId("marker-go")).toHaveText(["00:05.000", "00:09.000"]);
  await page.waitForTimeout(400);
  await page.reload();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("tab-markers").click();
  await expect(page.getByTestId("marker-go")).toHaveCount(2);
});

test("markers added faster than the server replies all survive", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("tab-markers").click();
  await page.route("**/sidecar", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.continue();
  });
  for (const t of [1, 2, 3, 4, 5]) {
    await seek(page, t);
    await page.getByTestId("add-marker").click();
  }
  await expect(page.getByTestId("marker-go")).toHaveCount(5);
  await page.waitForTimeout(2000); // every save finishes, and the list doesn't shrink back
  await expect(page.getByTestId("marker-go")).toHaveCount(5);
  await page.reload();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("tab-markers").click();
  await expect(page.getByTestId("marker-go")).toHaveCount(5);
});

test("a background refresh doesn't disturb a label being typed", async ({ page }) => {
  await openIq(page);
  await page.keyboard.press("m");
  await page.keyboard.type("half a la");
  await page.waitForTimeout(5500); // past the next refresh from the server
  await expect(page.getByTestId("marker-label")).toBeFocused();
  await page.keyboard.type("bel");
  await expect(page.getByTestId("marker-label")).toHaveValue("half a label");
});

test("a strange marker label stays plain text", async ({ page }) => {
  await openIq(page);
  await page.keyboard.press("m");
  await page.keyboard.type('<img src=x onerror="alert(1)">');
  await page.keyboard.press("Enter");
  await page.waitForTimeout(300);
  await page.reload();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("tab-markers").click();
  await expect(page.getByTestId("marker-label")).toHaveValue('<img src=x onerror="alert(1)">');
  expect(await page.locator("#markers img").count()).toBe(0);
});

test("frequency marks name the band correctly on each side of a retune", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("tab-markers").click();
  await expect(page.getByTestId("title-frequency")).toContainText("7.200");

  await seek(page, 5);
  await page.getByTestId("mark-frequency").fill("20100");
  await page.getByTestId("add-frequency").click();
  await seek(page, 10);
  await page.getByTestId("mark-frequency").fill("28200");
  await page.getByTestId("mark-frequency").press("Enter");
  await page.waitForTimeout(300);

  for (const [t, name, khz] of [[1, "7.200", "7200.000"], [6, "20.100", "20100.000"], [12, "28.200", "28200.000"], [4.9, "7.200", "7200.000"]] as const) {
    await seek(page, t);
    await expect(page.getByTestId("title-frequency")).toContainText(name);
    await expect(page.getByTestId("tune")).toHaveValue(khz);
  }

  // Typing a frequency tunes relative to the center in force at this moment.
  await seek(page, 6);
  await expect(page.getByTestId("title-frequency")).toContainText("20.100");
  await page.getByTestId("tune").fill("20150");
  await page.getByTestId("tune").press("Enter");
  expect((await state(page)).offsetHz).toBe(50_000);

  await page.reload();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await seek(page, 12);
  await expect(page.getByTestId("title-frequency")).toContainText("28.2");
});

test("a frequency mark doesn't change the audio, only what the band is called", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("tab-markers").click();
  await seek(page, 1);
  await page.getByTestId("mark-frequency").fill("14000");
  await page.getByTestId("add-frequency").click();
  await page.waitForTimeout(300);
  await seek(page, 3);
  expect((await state(page)).offsetHz).toBe(0);
});

test("markers work on audio recordings", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-audio_44100.wav").click();
  await page.keyboard.press("m");
  await expect(page.getByTestId("marker-label")).toBeFocused();
  await page.keyboard.type("thunder");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("marker-label")).toHaveValue("thunder");
});

// --- Audio-only playback ------------------------------------------------------------------

test("an audio recording plays in real time, scrubs, stops at the end, and draws its waveform", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-audio_44100.wav").click();
  await expect(page.getByTestId("play")).toBeEnabled();
  await expect(page.getByTestId("speeds")).toHaveCount(0);
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  const a = await page.evaluate(() => [performance.now(), (window as any).iqScopeDebug.transport.seconds]);
  await page.waitForTimeout(1500);
  const b = await page.evaluate(() => [performance.now(), (window as any).iqScopeDebug.transport.seconds]);
  expect((b[1] - a[1]) / ((b[0] - a[0]) / 1000)).toBeGreaterThan(0.95);

  await expect.poll(() => page.getByTestId("waveform").evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let bright = 0;
    for (let i = 1; i < d.length; i += 4) if (d[i] > 150) bright++;
    return bright;
  })).toBeGreaterThan(50);

  await page.getByTestId("scrub").evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    el.value = "5.5";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.getByTestId("status")).toHaveText("END OF RECORDING", { timeout: 10_000 });
  expect(await seconds(page)).toBeCloseTo(6, 1);
  await expect(page.getByTestId("play")).toHaveText("PLAY");
});

test("a mono audio recording says nothing about channels and still plays", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-GigaMono.wav").click();
  await expect(page.locator("#banner")).not.toContainText("one channel");
  await page.keyboard.press("Space");
  await expect.poll(() => seconds(page), { timeout: 15_000 }).toBeGreaterThan(0.2);
});

test("arrow keys do nothing on audio recordings, which have no pairs to step", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-audio_44100.wav").click();
  await page.keyboard.press("ArrowRight");
  expect(await seconds(page)).toBe(0);
});

test("switching from audio back to IQ brings everything back", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-audio_44100.wav").click();
  await expect(page.getByTestId("constellation")).toBeHidden();
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("constellation")).toBeVisible();
  await expect(page.getByTestId("speed-0.1")).toBeVisible();
  await expect(page.getByTestId("step-forward")).toBeEnabled();
  await expect(page.getByTestId("tab-path")).toBeVisible();
});

test("the new tabs fit at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await openIq(page);
  for (const name of ["path", "notes", "markers", "panels"]) {
    await page.getByTestId(`tab-${name}`).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  }
});

test("the tabs sit directly above the close-up, at any wide window", async ({ page }) => {
  for (const [width, height] of [[1280, 800], [1400, 900], [1500, 950], [1920, 1080], [2560, 1440]]) {
    await page.setViewportSize({ width, height });
    await openIq(page);
    const scope = (await page.getByTestId("tab-panels").boundingBox())!;
    const markers = (await page.getByTestId("tab-markers").boundingBox())!;
    const closeup = (await page.locator(".panel.spectrum").boundingBox())!;
    // Nothing fits between them: they end just above the panel, on its right-hand side, all on one line.
    expect(closeup.y - (markers.y + markers.height)).toBeGreaterThanOrEqual(0);
    expect(closeup.y - (markers.y + markers.height)).toBeLessThan(14);
    expect(Math.abs(markers.x + markers.width - (closeup.x + closeup.width))).toBeLessThan(4);
    expect(Math.abs(scope.y - markers.y)).toBeLessThan(2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  }
});

test("volume changes the sound's level, is remembered, and doesn't touch the position", async ({ page }) => {
  await openIq(page);
  const gain = () => page.evaluate(() => {
    const e = (window as any).iqScopeDebug.engine;
    return [e.volume, e.output ? e.output.gain.value : null];
  });
  await page.getByTestId("volume").fill("50");
  expect((await gain())[0]).toBeCloseTo(0.25, 5);

  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  await expect.poll(async () => (await gain())[1], { timeout: 2000 }).toBeCloseTo(0.25, 2);
  const before = await seconds(page);
  await page.getByTestId("volume").fill("0");
  await expect.poll(async () => Math.abs((await gain())[1])).toBeLessThan(0.01);
  await page.waitForTimeout(300);
  expect(await seconds(page)).toBeGreaterThan(before);
  expect(await playing(page)).toBe(true);

  await page.getByTestId("volume").fill("70");
  await page.reload();
  await expect(page.getByTestId("volume")).toHaveValue("70");
  expect((await gain())[0]).toBeCloseTo(0.49, 5);
});

test("arrow keys on the volume slider change the volume, not the position", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("volume").fill("50");
  await page.getByTestId("volume").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("volume")).toHaveValue("51");
  expect(await seconds(page)).toBe(0);
});

test("space still plays after using the volume slider", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("volume").focus();
  await page.keyboard.press("Space");
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.3);
});

test("scrubbing straight after opening a file, before ever pressing play, is safe", async ({ page }) => {
  await openIq(page);
  await page.getByTestId("scrub").evaluate((el: HTMLInputElement) => {
    el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    for (const v of ["3", "4", "5"]) {
      el.value = v;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(5.2);
});

test("pressing play twice quickly on a fresh page starts once and plays", async ({ page }) => {
  await openIq(page);
  await page.evaluate(() => {
    const t = (window as any).iqScopeDebug.transport;
    t.play();
    t.play();
  });
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  expect(await playing(page)).toBe(true);
});
