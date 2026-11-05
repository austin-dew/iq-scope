// Runs before the first paint, so the page never flashes the wrong color or shows itself
// ahead of the boot sequence. settings.ts owns these same values once the app has started.
(function () {
  var saved = {};
  try { saved = JSON.parse(localStorage.getItem("iqscope.settings")) || {}; } catch (e) { /* defaults */ }
  var root = document.documentElement;
  root.dataset.theme = ["amber", "white", "blue", "magenta"].indexOf(saved.theme) >= 0 ? saved.theme : "green";
  root.dataset.effects = saved.effects === false ? "off" : "on";
  var booted = false;
  try {
    booted = sessionStorage.getItem("iqscope.booted") === "1";
    sessionStorage.setItem("iqscope.booted", "1");
  } catch (e) { /* no storage: the boot sequence still runs, and can still be skipped */ }
  var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!booted && !still && saved.boot !== false && saved.effects !== false) root.dataset.boot = "1";
})();
