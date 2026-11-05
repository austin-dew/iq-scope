import { defineConfig } from "vite";

export default defineConfig({
  base: "/",
  build: {
    outDir: "../backend/iqscope/static",
    emptyOutDir: true,
    assetsInlineLimit: 0,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8850" },
  },
});
