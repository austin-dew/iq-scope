// Display settings, remembered per browser. None of them touch the signal or the sound.

import { display, THEMES, usePalette, type ThemeName } from "./theme";

const KEY = "iqscope.settings";

export interface Settings {
  theme: ThemeName;
  effects: boolean;
  boot: boolean;
  analysis: boolean;
}

function load(): Settings {
  let saved: Partial<Settings> = {};
  try {
    saved = JSON.parse(localStorage.getItem(KEY) ?? "{}") ?? {};
  } catch {
    // Unreadable or blocked storage just means the defaults.
  }
  return {
    theme: THEMES.includes(saved.theme as ThemeName) ? (saved.theme as ThemeName) : "green",
    effects: saved.effects !== false,
    boot: saved.boot !== false,
    analysis: saved.analysis === true,
  };
}

export const settings: Settings = load();

export function applySettings(): void {
  const root = document.documentElement;
  root.dataset.theme = settings.theme;
  root.dataset.effects = settings.effects ? "on" : "off";
  root.classList.toggle("analysis", settings.analysis);
  usePalette(settings.theme);
  display.analysis = settings.analysis;
}

export function changeSettings(change: Partial<Settings>): void {
  Object.assign(settings, change);
  applySettings();
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // It still applies now; it just won't be remembered.
  }
}
