# handspring

Webcam-based hand, face, and pose tracking that turns your body into a controller. Two entry points:

- **`python -m handspring`** — drive your macOS desktop with your hands. Cursor follows the right index, pinch = click, left-hand radial menu for mode selection, both-hand pinch creates new windows at whatever size you want. Serves a local MJPEG feed for Plash (or any browser) so your camera view can be your desktop background.
- **`handspring-synth`** — an in-process gesture-driven synthesizer. Fist on one hand enters edit mode; the other hand tweaks volume, pitch, cutoff, tremolo, stepping. Also streams OSC.

Both halves run independently, each with their own camera + preview window. Run them in separate terminals.

MediaPipe does the landmark extraction; everything downstream is pure Python. Runtime: macOS (tested on 14+). Tracking + OSC + synth layers are cross-platform, but the OS-control half (cursor, Finder, AX window tiling) is macOS-only.

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Desktop mode: the radial menu](#desktop-mode-the-radial-menu)
4. [Desktop mode: gestures](#desktop-mode-gestures)
5. [Plash setup (camera as desktop background)](#plash-setup)
6. [Synth mode](#synth-mode)
7. [OSC streams](#osc-streams)
8. [Customizing](#customizing)
9. [CLI reference](#cli-reference)
10. [Architecture](#architecture)
11. [Troubleshooting](#troubleshooting)

---

## Install

Requires Python 3.10+. On macOS, grant **Accessibility** (System Settings → Privacy & Security → Accessibility) and **Screen Recording** to your Python interpreter or terminal.

```bash
git clone https://github.com/Teamingzooper/handspring.git
cd handspring
pip install -e '.[dev]'
```

Installs:

- `mediapipe`, `opencv-python`, `python-osc`, `numpy`, `sounddevice`
- On macOS only: `pyobjc-framework-Quartz`, `pyobjc-framework-ApplicationServices` (cursor, mouse events, AppleScript bridge, native overlay window)

## Quick start

Terminal 1:

```bash
python -m handspring
```

You should see:

- A small preview window showing the mirror-flipped camera feed with a neon-green skeleton overlay.
- A translucent grey dot tracking your left hand, drawn on top of every app on the main display. Under the dot is the currently-selected app (default: Finder).
- Your system cursor starts following your right index fingertip.
- Console prints FPS + current left/right gestures + control state.
- The MJPEG feed is live at `http://127.0.0.1:8765/`.

Terminal 2 (optional synth):

```bash
handspring-synth
```

Make a fist with one hand → the synth panel in the preview's lower-left shows `EDIT L` or `EDIT R`. Point or open the other hand to edit parameters.

Hit Ctrl+C in either terminal to quit.

---

## Desktop mode: the radial menu

Pinch your **left** hand (index-tip + thumb-tip touching) and hold. After 400 ms of hold, a wheel appears around your pinch origin. Seven slices, clockwise from top:

| Slice | Has sub-menu? | What it does |
|---|---|---|
| **None** | no | Mode → none. Both-hand-pinch-pull-apart becomes a no-op. Dot goes dim grey, labeled `NONE`. |
| **Create** | `Finder · Safari · Messages · Notes · Terminal · Music` | Mode → create. The selected sub becomes the "app to spawn" for the create gesture. Dot stays grey, labeled with the app name. |
| **Window** | `Close · Minimize · Fullscreen · Left · Right · Center` | One-shot action on the frontmost macOS window. Left/Right tile to halves; Center fits 70% × 75% of usable area. Fullscreen uses macOS native (Ctrl+Cmd+F). |
| **Scroll** | no | Mode → scroll. Dot turns blue. Left-hand Y now scrolls: top third scrolls up, middle idle, bottom third scrolls down. Rate ramps with distance from center. |
| **Mission** | no | Opens Mission Control (Ctrl+Up). |
| **Desktops** | `Left · Right` | Page between Spaces (Ctrl+←/→). |
| **Screenshot** | `Screen · Window · Selection` | Saves to `~/Desktop/handspring_<timestamp>.png`. Screen = instant, Window = click to pick, Selection = drag a rect. |

### Selection physics

Three distance zones from your pinch origin, all in camera-normalized units:

- **`< 0.03`** — center dead zone. Release here = cancel.
- **`0.03 – 0.10`** — root ring. Angle picks which root slice is highlighted. Sub-ring appears as a dim preview around whichever root you're hovering.
- **`≥ 0.10`** — sub ring. Root **locks** to whatever it was at crossover; angle now picks the sub slice. Pull hand back into the root ring to unlock and swap roots.

Release pinch to commit the currently-highlighted selection. Sub wins if non-null; otherwise the root's default action (if any) fires.

### Visual states

- Countdown arc before 400 ms — fills clockwise from the top.
- Root wedges: dim grey by default, brighter grey when hovered.
- Sub wedges: appear low-opacity as soon as you hover a root with children (preview); brighten to "armed" when your hand crosses the sub threshold; fill neon green when that specific sub is hovered.

---

## Desktop mode: gestures

Beyond the radial, these gestures work continuously:

### Right hand — cursor

- **Index + thumb midpoint** → system cursor. The midpoint stays stable when you pinch (index tip alone would dive down as the finger curls, causing click-drift — this is intentional).
- **Pinch** → left mouse button held. Move hand while pinched = drag. Release = click-up. So pinching over a window's title bar and moving drags the window, pinching a scrollbar drags it, pinching a button clicks it.
- Cursor goes through a 35% EMA smoother each frame — responsive but kills per-frame MediaPipe jitter.
- Input inset: camera `x ∈ [0.08, 0.92]` maps to screen `x ∈ [0, 1]`. Same for y. You don't need to stretch to the extreme edges of the camera frame to reach the dock.

### Both hands — create a window

1. Pinch both hands close together (index tips within ~0.08 normalized distance).
2. Pull apart — a grey rectangle tracks your fingertips live on the overlay.
3. Release — new window of the currently-selected app (see Create radial) opens at exactly that bounding box. The rectangle shifts to lighter grey and stays visible for ~2 seconds while the actual window spawns.

Minimum pixel size is 200×150 to avoid degenerate windows.

### Failsafe

Hold **both fists** for 5 seconds → toggles gesture control off (or back on). While off, the cursor freezes where it was, no more clicks, no more radial. A red `GESTURES DISABLED` banner appears on the overlay. Hold both fists another 5s to re-enable.

---

## Plash setup

[Plash](https://github.com/sindresorhus/Plash) lets you set a webpage as your desktop background.

1. Open Plash → add website → paste `http://127.0.0.1:8765/`.
2. Set "browsing mode" to **Off** and disable audio. Plash will now show the live annotated camera feed.
3. Alternatively, point Plash at the `web/` folder in this repo (it contains an `index.html` that loads the same stream).

The feed inset is 4% margin on each side, so the macOS menubar (top) and dock (bottom) still show through the black padding.

---

## Synth mode

`handspring-synth` runs independently. Two edit modes triggered by making a fist:

**Left fist (EDIT L)** — the right hand controls:
- `point` → volume slider at fingertip (Y → volume)
- `open` → Y = pitch (C3..C6 exponential), X = stepping rate (0 = sustained, >0 = retrigger Hz)

**Right fist (EDIT R)** — the left hand controls:
- `point` → cutoff slider at fingertip
- `open` → Y = tremolo depth, X = tremolo rate (0.1..10 Hz)

Default values at startup: vol 0.4, note A4 (440 Hz), stepping 0, cutoff 3000 Hz, mod 0 @ 1 Hz.

Audio engine: saw oscillator → one-pole lowpass → amp envelope → tremolo. 48 kHz, 256 block, mono. Runs on a `sounddevice` OutputStream. `--no-synth` disables audio (OSC still streams).

---

## OSC streams

Both entry points emit OSC to `127.0.0.1:9000` by default (override with `--host` / `--port`). See `src/handspring/osc_out.py` for the full list. Key addresses:

### Per-frame continuous (always)

```
/hand/<side>/x          float 0..1
/hand/<side>/y          float 0..1
/hand/<side>/z          float (relative depth)
/hand/<side>/openness   float 0..1
/hand/<side>/pinch      float 0..1
/hand/<side>/index_x    float 0..1
/hand/<side>/index_y    float 0..1
/face/yaw               float -1..1
/face/pitch             float -1..1
/face/mouth_open        float 0..1
/pose/<joint>/x,y,z,visible  (8 joints)
```

### State-change only

```
/hand/<side>/gesture    string — fist/open/point/peace/thumbs_up/ok/rock/three/none
/face/expression        string — smile/frown/surprise/wink_left/wink_right/neutral
/clap                   one-shot on clap
/app/mode               string — synth/jarvis
```

### Synth-only (from `handspring-synth`)

```
/synth/volume, /synth/note_hz, /synth/stepping_hz,
/synth/cutoff_hz, /synth/mod_depth, /synth/mod_rate
/synth/mode             string
```

### Jarvis-only (from `python -m handspring`, legacy internal-window OSC)

```
/jarvis/window_count    int
/jarvis/window_created  int (id)
/jarvis/window_tap      int (id)
/jarvis/window_destroyed int
/jarvis/window_split    int
```

---

## Customizing

All of the customization points are constants in the source — no config files. Restart after editing. The most common knobs:

### Cursor / control feel

`src/handspring/desktop_controller.py`:

```python
_CURSOR_SMOOTHING = 0.35        # EMA; higher = snappier, lower = smoother
_CURSOR_INSET = 0.08            # camera dead zone at each edge
_RADIAL_HOLD_SECONDS = 0.4      # pinch-and-hold before wheel opens
_RADIAL_INNER = 0.03            # center dead zone (camera space)
_RADIAL_SUB_THRESHOLD = 0.10    # root → sub ring crossover
_FAILSAFE_HOLD_SECONDS = 5.0    # both-fist-to-disable duration
_SCROLL_DEADZONE = 0.12         # middle band that doesn't scroll
_SCROLL_MAX_PIXELS = 30         # per-frame scroll delta at screen edges
```

### Apps in the Create sub-ring

`src/handspring/desktop_controller.py`:

```python
_RADIAL_APPS: tuple[str, ...] = (
    "Finder", "Safari", "Messages", "Notes", "Terminal", "Music",
)
```

Any string that `open -a <Name>` accepts works (e.g. `"Google Chrome"`, `"Visual Studio Code"`, `"Cursor"`).

### Root radial layout

Edit `_ROOT_ITEMS` in `desktop_controller.py` to reorder slices, add new ones, or remove existing items. Order matters — it's clockwise from top.

```python
_ROOT_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("None", ()),
    ("Create", _RADIAL_APPS),
    ("Window", _WINDOW_SUBS),
    ("Scroll", ()),
    ("Mission", ()),
    ("Desktops", _DESKTOP_SUBS),
    ("Screenshot", _SCREENSHOT_SUBS),
)
```

Adding a new root item:

1. Append to `_ROOT_ITEMS` with its name + sub tuple (empty tuple for leaf actions).
2. Add an `elif name == "YourName"` branch to `_commit_radial`.
3. If it needs an OS action, add a helper to `os_control.py`.

### Overlay visuals

`src/handspring/overlay.py`:

```python
r_inner = 40.0       # center dead-zone ring radius
r_root = 220.0       # outer radius of root ring
r_sub_inner = 235.0  # gap before sub ring starts
r_sub = 410.0        # outer radius of sub ring
```

Cursor dot colors per mode are in `drawRect_`: grey (create), blue (scroll), dim grey (none).

### Plash layout

`web/index.html` (and a copy in `src/handspring/web_server.py`):

```css
img { top: 4vh; left: 4vw; width: 92vw; height: 88vh; }
```

Make the margin bigger if you want more OS chrome visible.

### Synth parameters

`src/handspring/synth.py` for audio DSP. `src/handspring/synth_ui.py` for how gestures map to parameter values (Y-axis inversion, stepping ranges, etc.).

---

## CLI reference

```
python -m handspring [options]

  --host HOST              OSC receiver host (default: 127.0.0.1)
  --port PORT              OSC receiver port (default: 9000)
  --camera N               camera index (default: 0)
  --no-preview             hide the OpenCV preview window (web stream still works)
  --no-face                disable face tracking
  --no-pose                disable body/arm pose tracking
  --hands {0,1,2}          max hands to track (default: 2)
  --no-mirror              disable preview mirror
  --no-os-control          disable cursor/click/radial (tracking + OSC only)
  --web-port PORT          MJPEG server port (default: 8765)
  --no-web                 disable the MJPEG web server
  --no-overlay             disable the native always-on-top overlay
  --fps-log-interval S     console FPS readout interval (default: 0.5)
```

```
handspring-synth [options]

  (same options as above, minus --no-os-control / --no-overlay / --web-port / --no-web)
  --no-synth               disable audio output (OSC still streams)
```

---

## Architecture

```
camera (cv2.VideoCapture)
  ↓
tracker.Tracker  — MediaPipe hands/face/pose in one pass
  ↓
features.hand_features() + face_features() — derived floats (openness, pinch, mouth_open, …)
  ↓
gestures.classify_hand() + expressions.classify_face() — discrete labels
  ↓
motion.* + history.* — temporal events (wave, pinch, drag, clap)
  ↓
FrameResult (dataclass, one per frame)
  ↓
 ├─ OscEmitter     → UDP packets to 127.0.0.1:9000
 ├─ DesktopController (python -m handspring)
 │    ↓
 │    os_control.*  → Quartz cursor, CGEventPost clicks, AppleScript, screencapture
 │    overlay.Overlay → native NSWindow float
 ├─ SynthController (handspring-synth)
 │    ↓
 │    Synth (sounddevice OutputStream)
 └─ Preview → OpenCV window + MJPEG to localhost:8765
```

### Key modules

- `tracker.py` — single-pass MediaPipe wrapper. Returns `TrackerOutput` with `FrameResult` + raw landmark lists (for preview drawing).
- `features.py` — pure-function landmark → feature conversion. `is_pinching()` lives here with a raw-distance threshold of 0.05.
- `gestures.py` — thresholded, history-aware gesture classifier.
- `desktop_controller.py` — gesture state machine. Handles cursor, click, radial tree, scroll, create, failsafe.
- `os_control.py` — macOS primitives. Each function is a guarded no-op on non-macOS.
- `overlay.py` — transparent click-through NSWindow rendered from the main loop via event-pump. Level = NSPopUpMenuWindowLevel (above most apps, below menubar).
- `web_server.py` — stdlib `http.server` with MJPEG `multipart/x-mixed-replace` streaming from a shared `LatestFrame`.
- `jarvis.py` — the old internal-windows system. Not wired into the current desktop entry point, but the gesture detection logic is reused as a reference. Still used by the OSC emitter's `/jarvis/*` addresses for external consumers.
- `synth.py`, `synth_params.py`, `synth_ui.py` — audio engine, thread-safe parameter container, gesture → param mapping.

### Testing

```bash
pytest              # 163 tests
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

Tests mock `os_control.*` so they don't actually drive your OS. The tracker/MediaPipe path is bypassed in tests via synthetic `FrameResult` fixtures.

---

## Troubleshooting

**Cursor doesn't move, or clicks do nothing.** You need Accessibility permission. System Settings → Privacy & Security → Accessibility → add the Python binary (or Terminal / iTerm / whatever shell you're running from). Restart handspring.

**Overlay doesn't show.** On newer macOS, transparent overlays above other apps may also need Screen Recording permission. Same path, different row in the list.

**Camera fails to open.** Try `--camera 1` (iPhone Continuity Camera is often index 0 and the built-in is 1, or vice versa). Quit any app holding the camera.

**Finder windows open but aren't sized.** AX `set position/size` can fail silently if the app doesn't have AX permission or if it rejects scripted resizes. Finder is reliable; Chrome and Safari work; some Electron apps don't.

**Feels laggy.** Turn off pose tracking with `--no-pose` (MediaPipe's pose estimator is the heaviest). Or drop hands to 1 with `--hands 1` if you only need one.

**Scroll mode scrolls the wrong direction.** Flip the sign of `direction` in `_handle_scroll` in `desktop_controller.py`, or reverse the constant values at the top.

**Preview window steals focus on start.** OpenCV's `cv2.namedWindow` brings a new window forward. Run with `--no-preview` — the web stream and native overlay are usually enough.

**Both fists seem to toggle randomly.** The failsafe requires 5 continuous seconds of both fists. Brief fist-confusion by the classifier resets the countdown.

---

## Versions

| Tag | What |
|---|---|
| `v0.1.0`–`v0.3.0` | Early tracker + OSC + preview + motion events |
| `v0.4.0` | Gesture-driven synth |
| `v0.5.x` | JARVIS mode: internal transparent-window surface, pinch-to-create, grab-drag, tap, corner resize, split, destroy |
| `v0.6.x` | Split/destroy/tear-apart windows |
| `v0.7.x` | 3D windows (depth + rotation) — later reverted |
| `v0.8.0` | Revert 3D; split synth into `handspring-synth` entry point |
| `v0.9.0` | macOS desktop integration + MJPEG web server for Plash |
| `v0.9.1` | Cursor midpoint + EMA smoothing + live-sized new Finder windows |
| `v0.9.2` | Inset web view + left-hand radial app launcher |
| `v0.9.3` | Decouple radial from create; native always-on-top left-cursor overlay |
| `v0.9.4` | Grey dot + pending/committed create rects in overlay |
| `v0.9.5` | Radial tree (modes + submenus) + scroll mode + screenshot |
| `v0.9.6` | Filled pie wedges + sub ring on hover |
| `v0.9.7` | Radial root-lock when entering sub ring |
| `v0.9.8` | Radial 2× larger |
| `v0.9.9` | Window tiling + Mission Control + Desktops |

---

## License

MIT. See `LICENSE`.
