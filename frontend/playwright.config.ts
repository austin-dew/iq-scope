import { resolve } from "node:path";
import { defineConfig } from "@playwright/test";

const PORT = 18852;

export default defineConfig({
  testDir: "tests",
  timeout: 60_000,
  workers: 1,
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1400, height: 900 },
    // The boot sequence stays out of the way of tests that are about something else.
    contextOptions: { reducedMotion: "reduce" },
    launchOptions: { args: ["--autoplay-policy=no-user-gesture-required"] },
  },
  webServer: {
    command: `${resolve("../.venv/bin/python")} tests/serve_fixture.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
