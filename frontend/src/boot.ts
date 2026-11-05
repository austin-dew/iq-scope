// A short boot sequence, at most once per session. theme-early.js decides whether it runs.
// Its status lines report real checks, and nothing waits on it: the app loads behind it.

export interface BootFacts {
  archive: string;
  recordings: string;
}

const LINE_MS = 190;
const HOLD_MS = 450;
const FACTS_WAIT_MS = 900;

export function runBoot(facts: Promise<BootFacts>): void {
  const root = document.documentElement;
  if (root.dataset.boot !== "1") return;
  const screen = document.getElementById("boot")!;
  const output = document.getElementById("boot-lines")!;
  let finished = false;

  const finish = () => {
    if (finished) return;
    finished = true;
    delete root.dataset.boot;
    window.removeEventListener("keydown", onKey, true);
    screen.removeEventListener("click", finish);
  };
  // The key that skips this does nothing else, so Space can't also start playback.
  const onKey = (event: KeyboardEvent) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    finish();
  };
  window.addEventListener("keydown", onKey, true);
  screen.addEventListener("click", finish);

  const say = (text: string) => {
    const line = document.createElement("div");
    line.textContent = text;
    output.append(line);
  };
  const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  void (async () => {
    say("CONELRAD SIGNAL ANALYZER");
    await wait(LINE_MS);
    say("PRIVATE TERMINAL");
    await wait(LINE_MS);
    const known = await Promise.race([facts, wait(FACTS_WAIT_MS).then(() => null)]);
    if (finished) return;
    say(`ARCHIVE LINK ...... ${known?.archive ?? "STILL CHECKING"}`);
    await wait(LINE_MS);
    say(`RECORDINGS ........ ${known?.recordings ?? "STILL CHECKING"}`);
    await wait(LINE_MS);
    say("READY");
    await wait(HOLD_MS);
    finish();
  })();
}
