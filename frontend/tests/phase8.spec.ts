import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";

const IQ_FILE = "baseband_7200000Hz_12-00-00_03-11-2026.wav";
const RECORDINGS = join(tmpdir(), "iqscope-ui-fixture/radio/recordings");
const UPLOADS = join(RECORDINGS, "uploads");
const PIECE = 2 ** 20;
const SCRATCH = join(tmpdir(), "iqscope-upload-sources");
const seconds = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.transport.seconds) as Promise<number>;
const playing = (page: Page) => page.evaluate(() => (window as any).iqScopeDebug.engine.isPlaying) as Promise<boolean>;

/** A real two-channel WAV of about this size, with contents depending on the seed. */
function makeWav(name: string, megabytes: number, seed = 1): { path: string; name: string; bytes: Buffer } {
  mkdirSync(SCRATCH, { recursive: true });
  const dataBytes = Math.floor((megabytes * 2 ** 20) / 4) * 4;
  const bytes = Buffer.alloc(44 + dataBytes);
  bytes.write("RIFF", 0);
  bytes.writeUInt32LE(36 + dataBytes, 4);
  bytes.write("WAVEfmt ", 8);
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(2, 22);
  bytes.writeUInt32LE(240_000, 24);
  bytes.writeUInt32LE(960_000, 28);
  bytes.writeUInt16LE(4, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36);
  bytes.writeUInt32LE(dataBytes, 40);
  let x = seed * 2654435761;
  for (let k = 44; k < bytes.length; k += 2) {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    bytes.writeInt16LE((x >> 8) % 2000, k);
  }
  const path = join(SCRATCH, `${seed}-${name}`);
  writeFileSync(path, bytes);
  // The file chooser sees the file's own name, so keep a folder per seed for the real name.
  const folder = join(SCRATCH, `${seed}`);
  mkdirSync(folder, { recursive: true });
  const named = join(folder, name);
  writeFileSync(named, bytes);
  rmSync(path);
  return { path: named, name, bytes };
}

const sha = (data: Buffer) => createHash("sha256").update(data).digest("hex");

async function openUpload(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId("show-upload").click();
  await expect(page.getByTestId("upload")).toBeVisible();
}

const job = (page: Page) => page.getByTestId("upload-job");

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => { throw error; });
  rmSync(UPLOADS, { recursive: true, force: true });
  rmSync(SCRATCH, { recursive: true, force: true });
  // Clear what earlier tests left half-sent, so each starts with nothing unfinished.
  await page.goto("/");
  await page.evaluate(async () => {
    const { uploads } = await (await fetch("/api/uploads")).json();
    for (const u of uploads) await fetch(`/api/uploads/${u.id}`, { method: "DELETE", headers: { "X-IQScope": "1" } });
  });
});

// --- Upload -------------------------------------------------------------------------------

test("a file goes up in pieces, arrives identical, and can be opened right away", async ({ page }) => {
  const source = makeWav("baseband_7329000Hz_21-56-25_03-11-2026.wav", 3.5);
  const pieces: number[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/part?")) pieces.push(Number(new URL(r.url()).searchParams.get("offset")));
  });
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 30_000 });
  await expect(job(page)).toContainText(`SAVED AS uploads/${source.name}`);
  expect(pieces).toEqual([0, PIECE, 2 * PIECE, 3 * PIECE]);
  expect(sha(readFileSync(join(UPLOADS, source.name)))).toBe(sha(source.bytes));
  expect(readdirSync(UPLOADS).filter((n) => n.endsWith(".part"))).toEqual([]);

  await page.getByTestId("show-files").click();
  await page.getByTestId(`file-${source.name}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
  await expect(page.getByTestId("title-frequency")).toContainText("7.329");
});

test("several files go one after another", async ({ page }) => {
  const a = makeWav("first.wav", 1.5, 1);
  const b = makeWav("second.wav", 1.5, 2);
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles([a.path, b.path]);
  await expect(job(page).nth(1)).toContainText("DONE", { timeout: 30_000 });
  await expect(job(page).nth(0)).toContainText("DONE");
  expect(readdirSync(UPLOADS).sort()).toEqual(["first.wav", "second.wav"]);
});

test("a file that isn't a WAV is refused, by name and by contents", async ({ page }) => {
  mkdirSync(SCRATCH, { recursive: true });
  const flac = join(SCRATCH, "song.flac");
  writeFileSync(flac, "not a wav");
  const fake = join(SCRATCH, "fake.wav");
  writeFileSync(fake, "plain text with a wav name ".repeat(100));
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(flac);
  await expect(job(page).first()).toContainText("ONLY .WAV FILES");
  await page.getByTestId("upload-input").setInputFiles(fake);
  await expect(job(page).nth(1)).toContainText("isn't really a WAV", { timeout: 10_000 });
  expect(existsSync(join(UPLOADS, "fake.wav"))).toBe(false);
});

test("connection drops are survived by trying again, with no gaps in the file", async ({ page }) => {
  const source = makeWav("flaky.wav", 3.5);
  let attempts = 0;
  await page.route("**/part?*", (route) => {
    attempts++;
    if (attempts % 2 === 1) return route.abort("connectionreset");
    return route.continue();
  });
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("CONNECTION TROUBLE", { timeout: 10_000 });
  await expect(job(page)).toContainText("DONE", { timeout: 60_000 });
  expect(attempts).toBe(8);
  expect(sha(readFileSync(join(UPLOADS, "flaky.wav")))).toBe(sha(source.bytes));
});

test("a reply lost after a piece landed doesn't send it twice into the file", async ({ page }) => {
  const source = makeWav("lostreply.wav", 2.5);
  let dropped = false;
  await page.route("**/part?*", async (route) => {
    if (!dropped) {
      dropped = true;
      await route.fetch(); // it reaches the server, but the answer never comes back
      return route.abort("connectionreset");
    }
    return route.continue();
  });
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 60_000 });
  expect(sha(readFileSync(join(UPLOADS, "lostreply.wav")))).toBe(sha(source.bytes));
});

test("a piece damaged on the way is caught and sent again", async ({ page }) => {
  const source = makeWav("damaged.wav", 2.5);
  let damaged = 0;
  await page.route("**/part?*", (route) => {
    const body = route.request().postDataBuffer();
    if (damaged === 0 && body) {
      damaged++;
      const bad = Buffer.from(body);
      bad[1000] ^= 0xff;
      return route.continue({ postData: bad });
    }
    return route.continue();
  });
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 60_000 });
  expect(damaged).toBe(1);
  expect(sha(readFileSync(join(UPLOADS, "damaged.wav")))).toBe(sha(source.bytes));
});

test("after a reload, choosing the same file carries on from where it stopped", async ({ page }) => {
  const source = makeWav("resume.wav", 4.5);
  let sent = 0;
  await page.route("**/part?*", (route) => (++sent <= 2 ? route.continue() : route.abort("connectionreset")));
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("CONNECTION TROUBLE", { timeout: 20_000 });
  await page.unroute("**/part?*");
  await page.reload();

  await page.getByTestId("show-upload").click();
  await expect(page.getByTestId("upload-unfinished-item")).toContainText("UNFINISHED: 2.0 MiB OF 4.5 MiB");
  expect(existsSync(join(UPLOADS, "resume.wav"))).toBe(false);

  const offsets: number[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/part?")) offsets.push(Number(new URL(r.url()).searchParams.get("offset")));
  });
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 30_000 });
  expect(offsets).toEqual([2 * PIECE, 3 * PIECE, 4 * PIECE]);
  expect(sha(readFileSync(join(UPLOADS, "resume.wav")))).toBe(sha(source.bytes));
  await expect(page.getByTestId("upload-unfinished-item")).toHaveCount(0);
});

test("a different file that happens to share a name and size starts afresh", async ({ page }) => {
  const original = makeWav("same.wav", 3.5, 1);
  const other = makeWav("same.wav", 3.5, 2);
  let sent = 0;
  await page.route("**/part?*", (route) => (++sent <= 1 ? route.continue() : route.abort("connectionreset")));
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(original.path);
  await expect(job(page)).toContainText("CONNECTION TROUBLE", { timeout: 20_000 });
  await page.unroute("**/part?*");
  await page.reload();
  await page.getByTestId("show-upload").click();

  await page.getByTestId("upload-input").setInputFiles(other.path);
  await expect(job(page)).toContainText("DONE", { timeout: 30_000 });
  expect(sha(readFileSync(join(UPLOADS, "same.wav")))).toBe(sha(other.bytes));
  // The first one's piece is still offered to continue or discard; it was never mixed in.
  await expect(page.getByTestId("upload-unfinished-item")).toHaveCount(1);
  await page.getByTestId("upload-discard").click();
  await expect(page.getByTestId("upload-unfinished-item")).toHaveCount(0);
  expect(readdirSync(UPLOADS).filter((n) => n.endsWith(".part"))).toEqual([]);
});

test("cancel stops it and removes what had arrived", async ({ page }) => {
  const source = makeWav("cancelled.wav", 4.5);
  let sent = 0;
  await page.route("**/part?*", (route) => (++sent <= 1 ? route.continue() : route.abort("connectionreset")));
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("CONNECTION TROUBLE", { timeout: 20_000 });
  await page.getByTestId("upload-cancel").click();
  await expect(job(page)).toHaveCount(0);
  await expect.poll(() => readdirSync(UPLOADS).length, { timeout: 5000 }).toBe(0);
  await expect(page.getByTestId("upload-unfinished-item")).toHaveCount(0);
});

test("a strange filename can only ever show up as text", async ({ page }) => {
  const source = makeWav('<img src=x onerror=alert(1)>.wav', 0.5);
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 30_000 });
  await expect(job(page).locator(".file-name")).toHaveText('<img src=x onerror=alert(1)>.wav');
  expect(await page.locator("#upload img").count()).toBe(0);
  expect(readdirSync(UPLOADS)).toEqual(["img_src_x_onerror_alert_1.wav"]);
});

test("sound keeps playing during an upload", async ({ page }) => {
  const source = makeWav("background.wav", 6.5);
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.5);
  await page.getByTestId("show-upload").click();
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("DONE", { timeout: 30_000 });
  const before = await seconds(page);
  await page.waitForTimeout(500);
  expect(await playing(page)).toBe(true);
  expect(await seconds(page)).toBeGreaterThan(before);
});

test("the upload tab fits at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await openUpload(page);
  await expect(page.getByTestId("upload-drop")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
});

// --- The watched folder -------------------------------------------------------------------

test("a file still being copied shows as COPYING, then joins the list", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("arriving")).toBeHidden();
  mkdirSync(join(RECORDINGS, "laptop"), { recursive: true });
  const path = join(RECORDINGS, "laptop", "incoming_1.wav");
  const source = makeWav("x.wav", 0.5);
  writeFileSync(path, source.bytes);
  try {
    await page.getByTestId("rescan").click();
    await expect(page.getByTestId("arriving")).toContainText("COPYING... incoming_1.wav", { timeout: 6000 });
    await expect(page.getByTestId("file-incoming_1.wav")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("arriving")).toBeHidden();
  } finally {
    rmSync(path, { force: true });
  }
});

test("RESCAN finds an already-finished file at once", async ({ page }) => {
  await page.goto("/");
  const path = join(RECORDINGS, "laptop", "copied_earlier.wav");
  writeFileSync(path, makeWav("x.wav", 0.5).bytes);
  const old = new Date(Date.now() - 3_600_000);
  utimesSync(path, old, old);
  try {
    await page.getByTestId("rescan").click();
    await expect(page.getByTestId("file-copied_earlier.wav")).toBeVisible({ timeout: 1500 });
    await expect(page.getByTestId("rescan-status")).toHaveText("");
  } finally {
    rmSync(path, { force: true });
  }
});

test("if the open recording is deleted, the page says so and is left in a clean state", async ({ page }) => {
  const path = join(RECORDINGS, "laptop", "soon_gone.wav");
  writeFileSync(path, makeWav("x.wav", 0.5).bytes);
  const old = new Date(Date.now() - 3_600_000);
  utimesSync(path, old, old);
  await page.goto("/");
  await page.getByTestId("rescan").click();
  await page.getByTestId("file-soon_gone.wav").click();
  await expect(page.getByTestId("play")).toBeEnabled();
  rmSync(path);
  await expect(page.locator("#banner")).toContainText("soon_gone.wav IS NO LONGER IN THE ARCHIVE", { timeout: 15_000 });
  await expect(page.getByTestId("play")).toBeDisabled();
  await expect(page.getByTestId("file-info")).toHaveText("SELECT A RECORDING");
  // Everything else still works.
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
  await expect(page.locator("#banner")).toBeHidden();
});

test("a list that's merely incomplete, as after a restart, doesn't close anything", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
  await page.route("**/api/recordings", (route) =>
    route.fulfill({ json: { online: true, complete: false, arriving: [], recordings: [] } }),
  );
  await page.waitForTimeout(6000);
  await expect(page.getByTestId("play")).toBeEnabled();
  await expect(page.locator("#banner")).not.toContainText("NO LONGER");
});

// --- Failures -----------------------------------------------------------------------------

test("the archive dropping out during playback says why, and recovers by itself", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.3);

  // What the real server answers when the share is gone.
  const reason = "The share didn't answer within 3 s.";
  await page.route("**/chunk?*", (route) =>
    route.fulfill({ status: 503, json: { detail: "Couldn't read the recording: Host is down." } }),
  );
  await page.route("**/slice?*", (route) => route.fulfill({ status: 503, json: { detail: "Couldn't read the recording." } }));
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: { status: "degraded", share: { mounted: true, readable: false, detail: reason } } }),
  );
  await page.route("**/api/recordings", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const r of body.recordings) r.available = false;
    return route.fulfill({ json: { ...body, online: false, complete: false } });
  });
  await expect(page.getByTestId("archive-note")).toHaveText(`ARCHIVE OFFLINE: ${reason}`, { timeout: 15_000 });
  await expect(page.getByTestId("status")).toContainText("Host is down", { timeout: 30_000 });
  // Sound that's going can still be paused. Nothing has crashed or been unloaded.
  await expect(page.getByTestId("play")).toBeEnabled();
  const stuck = await seconds(page);

  await page.unroute("**/chunk?*");
  await page.unroute("**/slice?*");
  await page.unroute("**/api/health");
  await page.unroute("**/api/recordings");
  await expect(page.getByTestId("archive-note")).toBeHidden({ timeout: 15_000 });
  await expect.poll(() => seconds(page), { timeout: 30_000 }).toBeGreaterThan(stuck + 0.5);
  await expect(page.getByTestId("status")).not.toContainText("Host is down");
  expect(await playing(page)).toBe(true);
});

test("the server going away is reported, and the page comes back with it", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId(`file-${IQ_FILE}`).click();
  await expect(page.getByTestId("play")).toBeEnabled();
  await page.route("**/api/**", (route) => route.abort("connectionrefused"));
  await expect(page.getByTestId("archive")).toHaveText("SERVER NOT RESPONDING", { timeout: 15_000 });
  await expect(page.getByTestId("archive-note")).toContainText("can't reach the app");
  await page.unroute("**/api/**");
  await expect(page.getByTestId("archive")).toHaveText("ARCHIVE ONLINE", { timeout: 15_000 });
  await expect(page.getByTestId("archive-note")).toBeHidden();
  await page.getByTestId("play").click();
  await expect.poll(() => seconds(page), { timeout: 20_000 }).toBeGreaterThan(0.3);
});

test("an upload waits out an archive outage and then finishes", async ({ page }) => {
  const source = makeWav("patient.wav", 2.5);
  let failing = true;
  await page.route("**/part?*", (route) =>
    failing
      ? route.fulfill({ status: 503, json: { detail: "Couldn't write to the archive: Host is down.", retry: true } })
      : route.continue(),
  );
  await openUpload(page);
  await page.getByTestId("upload-input").setInputFiles(source.path);
  await expect(job(page)).toContainText("Host is down", { timeout: 10_000 });
  failing = false;
  await expect(job(page)).toContainText("DONE", { timeout: 60_000 });
  expect(sha(readFileSync(join(UPLOADS, "patient.wav")))).toBe(sha(source.bytes));
});
