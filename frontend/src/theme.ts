// The colors plots draw with. The page's own colors are CSS variables; a canvas can't use
// those directly, so both sets are switched together by the data-theme attribute's setter.

export const THEMES = ["green", "amber", "white", "blue", "magenta"] as const;
export type ThemeName = (typeof THEMES)[number];

export interface Palette {
  bright: string;
  fg: string;
  dim: string;
  grid: string;
  white: string;
  accent: string; // the tuner and retune marks: must stand out from the phosphor color
  plot: string; // behind a plot
  ruler: string; // behind the waterfall's rulers
  fill: string; // under a spectrum curve
  accentFill: string; // the passband
  accentLine: string;
  scrim: string; // behind text drawn over a plot
  rampMid: [number, number, number]; // the waterfall, 3/4 of the way up
  rampPower: [number, number, number]; // how quickly each channel rises to rampMid
  rampTop: [number, number, number];
}

const PALETTES: Record<ThemeName, Palette> = {
  green: {
    bright: "#c8ffd8",
    fg: "#5dff8a",
    dim: "#2f9e52",
    grid: "#17402a",
    white: "#f4fff7",
    accent: "#ffc450",
    plot: "#020603",
    ruler: "#050c07",
    fill: "rgba(93, 255, 138, 0.3)",
    accentFill: "rgba(255, 196, 80, 0.14)",
    accentLine: "rgba(255, 196, 80, 0.95)",
    scrim: "rgba(2, 6, 3, 0.88)",
    rampMid: [40, 235, 70],
    rampPower: [2, 1, 2],
    rampTop: [230, 255, 240],
  },
  amber: {
    bright: "#ffe7bd",
    fg: "#ffb638",
    dim: "#ab7420",
    grid: "#47310f",
    white: "#fff8ec",
    accent: "#7fe3ff",
    plot: "#060402",
    ruler: "#0d0905",
    fill: "rgba(255, 182, 56, 0.3)",
    accentFill: "rgba(127, 227, 255, 0.14)",
    accentLine: "rgba(127, 227, 255, 0.95)",
    scrim: "rgba(6, 4, 2, 0.88)",
    rampMid: [255, 165, 25],
    rampPower: [1, 1.4, 2.5],
    rampTop: [255, 245, 225],
  },
  white: {
    bright: "#ffffff",
    fg: "#d8e4e0",
    dim: "#8b9894",
    grid: "#2c3331",
    white: "#ffffff",
    accent: "#ffc450",
    plot: "#040505",
    ruler: "#0a0c0c",
    fill: "rgba(216, 228, 224, 0.3)",
    accentFill: "rgba(255, 196, 80, 0.14)",
    accentLine: "rgba(255, 196, 80, 0.95)",
    scrim: "rgba(4, 5, 5, 0.88)",
    rampMid: [190, 205, 200],
    rampPower: [1, 1, 1],
    rampTop: [255, 255, 255],
  },
  blue: {
    bright: "#cfeeff",
    fg: "#62c6ff",
    dim: "#3f8cbb",
    grid: "#163347",
    white: "#f2fbff",
    accent: "#ffc450",
    plot: "#020507",
    ruler: "#050b10",
    fill: "rgba(98, 198, 255, 0.3)",
    accentFill: "rgba(255, 196, 80, 0.14)",
    accentLine: "rgba(255, 196, 80, 0.95)",
    scrim: "rgba(2, 5, 7, 0.88)",
    rampMid: [35, 150, 255],
    rampPower: [2, 1.2, 1],
    rampTop: [225, 245, 255],
  },
  magenta: {
    bright: "#ffd6f0",
    fg: "#ff6ec7",
    dim: "#c7559d",
    grid: "#4a1a3b",
    white: "#fff5fb",
    accent: "#7fe3ff",
    plot: "#060206",
    ruler: "#0e050c",
    fill: "rgba(255, 110, 199, 0.3)",
    accentFill: "rgba(127, 227, 255, 0.14)",
    accentLine: "rgba(127, 227, 255, 0.95)",
    scrim: "rgba(6, 2, 6, 0.88)",
    rampMid: [255, 60, 190],
    rampPower: [1, 2, 1.1],
    rampTop: [255, 235, 248],
  },
};

/** Always the current palette: plots read it each frame, so a switch shows at once. */
export const colors: Palette = { ...PALETTES.green };

export function usePalette(name: ThemeName): void {
  Object.assign(colors, PALETTES[name]);
}

/** Share Tech Mono stays clear at small sizes; the rest are fallbacks while it loads. */
export const plotFont = (px = 11): string => `${px}px "Share Tech Mono", ui-monospace, monospace`;

/** Black through the phosphor color to near-white. */
export function phosphor(level: number): [number, number, number] {
  const v = Math.min(Math.max(level, 0), 1);
  const { rampMid: mid, rampPower: power, rampTop: top } = colors;
  if (v < 0.75) {
    const g = v / 0.75;
    return [0, 1, 2].map((c) => Math.round(mid[c] * g ** power[c])) as [number, number, number];
  }
  const w = (v - 0.75) / 0.25;
  return [0, 1, 2].map((c) => Math.round(mid[c] + w * (top[c] - mid[c]))) as [number, number, number];
}

/** What the plots should show beyond the basics. It never changes what data they use. */
export const display = { analysis: false };
