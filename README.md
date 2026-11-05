# IQ Scope

A personal web app for playing RTL-SDR recordings while showing the IQ data behind the sound:
waterfall, constellation, phasor, I/Q traces, and a live readout of how each pair of numbers
becomes one audio sample. It was built for learning how radio demodulation actually works.

In the app it calls itself the CONELRAD SIGNAL ANALYZER, after the real 1950s US emergency
broadcast system. Its look is inspired by atomic-age computer terminals, like those in the Fallout
games. It isn't affiliated with or endorsed by Bethesda, and it uses no game assets.

## What it does

- Plays baseband WAV recordings in AM, USB, LSB, CW, NFM and WFM, with click-to-tune
- Handles long files: RIFF, RF64 and BW64, including data sizes wrapped at 4 GiB
- Slow motion down to 1/1000 speed, and stepping one IQ pair at a time
- A Signal Path view, from raw band to sound, and notes explaining what you're looking at
- Generated example clips of every mode, so it works before you have a recording
- Markers and frequency marks, saved in a sidecar file next to each recording
- Analysis Mode: axes, sample numbers, dBFS levels, cursors, and raw pre-tuning pairs
- Audio-only WAVs still play, clearly marked as having no IQ data
- Resumable uploads, checked piece by piece, plus a watched folder
- Five colors, and CRT effects that stay off the data

## Running it

It's designed for Docker on a home server, reading recordings from a NAS over a CIFS mount:

```text
browser --> tailscale serve :8443 --> 127.0.0.1:8850 --> container :8000
                                                           |-- /radio  = /mnt/radio on the host
                                                           `-- /cache  = /srv/iq-scope/cache on the host
```

HTTPS is required, as browsers only allow the audio engine in a secure context. Tailscale's
`serve` gives you that without managing certificates.

```bash
sudo install -d -o 1000 -g 1000 /srv/iq-scope/cache
# Recordings go in /mnt/radio/recordings/, in any subfolders.
docker build --target test .        # runs the tests inside the container
docker compose up -d --build
tailscale serve --bg --https=8443 http://127.0.0.1:8850
```

`compose.yaml` publishes on localhost only, and limits the app to 2 CPUs and 2 GB.

For a CIFS mount, make the empty mountpoint immutable (`chattr +i`), so nothing lands on the
host's own disk when the share is down. Use `nofail` in `/etc/fstab`. A mount that fails at boot
is never retried by itself, so `deploy/` has a systemd timer that tries again every minute:

```bash
sudo cp deploy/iq-scope-remount.service deploy/iq-scope-remount.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now iq-scope-remount.timer
```

**There's no login.** Anyone who can reach the app can listen to everything, upload files, and
edit markers. Keep it on a private network.

### Showing it to someone

Tailscale Funnel can publish it for a while and then stop, on a port of its own:

```bash
sudo tailscale funnel --https=10000 8850       # on until Ctrl+C
sudo tailscale funnel --bg --https=10000 8850  # or in the background...
sudo tailscale funnel --https=10000 off        # ...and off again
```

While it's on, the warning above applies to the whole internet, so turn it off afterward. Don't
use `tailscale funnel reset` if you serve anything else: it clears all of it.

## Adding recordings

**Record IQ, not audio.** In SDR++, set the Recorder module to **Baseband**. Its audio mode saves
sound that's already been demodulated, and the radio signal can't be recovered from it.

- **Upload** in the `[ UPLOAD ]` tab. Files go up in 32 MiB pieces, each checked with SHA-256. A
  dropped connection retries by itself, and after a reload, choosing the same file continues it.
  Nothing is overwritten: a second file with the same name gets `-2`.
- **Copy** anywhere under `recordings/`, which is checked every 30 seconds. A file shows as
  `COPYING...` until its size holds still. `RESCAN` looks straight away.

Keep the SDR++ filename, `baseband_<Hz>Hz_HH-MM-SS_DD-MM-YYYY.wav`, as the center frequency comes
from it. Otherwise, set it above the waterfall. If the radio was retuned during a recording, say
where in `[ MARKERS ]`.

Waterfall tiles and example clips live in `/cache`, which can be deleted at any time and rebuilds
itself. Markers live beside the recordings, as `<recording>.iqscope.json`.

## When something's wrong

The header's archive status and the line under it give the reason in plain words.

| What you see | What's probably wrong | What to do |
| --- | --- | --- |
| `ARCHIVE OFFLINE` | The NAS or the network to it | Fix that. The app recovers by itself, with no restart. |
| `... not on the archive share` | The mount is missing | `sudo mount /mnt/radio`, and read the error. |
| `SERVER NOT RESPONDING` | The container is down | `docker compose up -d`, then `docker logs iq-scope`. |
| Sound stutters, `BUFFERING` | Something's busy, or the network | Each second of audio has to take under 1 s to make. Try `scripts/read_speed.py`. |
| A file says `ERR` | That file can't be read | Click it for the reason. Other files aren't affected. |

To test an outage: a plain `umount` doesn't reach a container that started while the share was up,
as it keeps its own copy of the mount. Stop the file server instead, or unmount and then restart
the container.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest backend/tests -q
cd frontend && npm ci && npm run build && npx playwright test
```

Test recordings are generated by `iqscope.synth`, so tests have known answers and large files stay
out of git. Tests of real captures skip when those aren't present; `IQSCOPE_TEST_CARD` points at a
folder of SDR Console recordings for the ones that use them.

| Path | What it is |
| --- | --- |
| `backend/iqscope/wavfile.py`, `recording.py` | WAV parsing and metadata |
| `backend/iqscope/dsp.py` | Tuning, filtering, and every demodulator |
| `backend/iqscope/waterfall.py` | Waterfall tiles, and the close-up's long FFT |
| `backend/iqscope/library.py` | Watching the folder, sidecars, background ingest |
| `backend/iqscope/uploads.py` | Receiving and checking uploads |
| `backend/iqscope/app.py` | The API |
| `frontend/src/audio.ts`, `transport.ts` | Playback. The audio clock is the master; visuals follow it. |
| `frontend/src/views.ts`, `closeup.ts`, `signalpath.ts`, `waterfall.ts` | The plots |
| `frontend/src/theme.ts`, `style.css` | Colors. A test checks that the two files match. |

The app loads nothing from third parties at runtime.

## License

MIT, as in `LICENSE`. The fonts in `frontend/public/fonts/` are VT323 and Share Tech Mono, under
the SIL Open Font License, with their licenses beside them.
