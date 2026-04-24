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
3. [First-run tutorial](#first-run-tutorial)
4. [Desktop mode: the radial menu](#desktop-mode-the-radial-menu)
5. [Desktop mode: gestures](#desktop-mode-gestures)
6. [Plash setup (camera as desktop background)](#plash-setup)
7. [Synth mode](#synth-mode)
8. [OSC streams](#osc-streams)
9. [Customizing](#customizing)
10. [CLI reference](#cli-reference)
11. [Architecture](#architecture)
12. [Use as a library](#use-as-a-library)
13. [Troubleshooting](#troubleshooting)

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

## First-run tutorial

The first time you run `python -m handspring` (i.e., before `~/.config/handspring/config.toml` exists), an interactive tutorial opens in the preview window. 7 steps, ~60–90 seconds. It walks you through:

1. Showing your right hand.
2. Pointing your right index.
3. Moving the cursor around.
4. Pinching the right hand.
5. Pinching the left hand.
6. Flicking the radial menu.
7. Making a peace sign.

Each step has a 30-second timeout. Press **Space** to skip the current step, **Esc** or **q** to skip everything. The tutorial measures your hand size during step 2 and scales `radial.flick_threshold` + `create.entry_distance` proportionally — bigger hands get roomier thresholds so you don't have to make tiny precise motions.

Re-run anytime:
- `python -m handspring --tutorial` → force-run (even if config exists).
- Click **Replay tutorial** in the Settings UI → queues it for the next launch.
- `python -m handspring --skip-tutorial` → never run (useful in scripts).

---

## Desktop mode: the radial menu

Pinch your **left** hand to open the radial. There's no hold — labels appear instantly around the pinch point. Move your hand a tiny bit toward one of the six commands, then release pinch to fire it. Release at the origin (or very close to it) cancels.

Default commands, clockwise from 12 o'clock:

| Position | Command |
|---|---|
| Top | None (disable gestures) |
| Upper-right | Settings (opens the settings web UI) |
| Lower-right | Mission Control |
| Bottom | New Finder window |
| Lower-left | Scroll mode (left hand y → scroll wheel) |
| Upper-left | Screenshot (whole screen) |

**Why this is fast:**
- No hold-to-open delay.
- Direction is only *committed* on release — intermediate jitter doesn't mis-fire.
- Angular hysteresis prevents flicker between neighboring slices.
- Only one level; no sub-menus.

**Customizing.** Edit `~/.config/handspring/config.toml`'s `radial_tree` section (or use the Settings UI) to reorder, add, or swap commands. Built-in command names are `None`, `Settings`, `Mission`, `Create`, `Scroll`, `Screenshot`. Any other name with a `command = "..."` field runs that shell command when fired, so you can add e.g.:

```toml
[[radial_tree]]
name    = "Slack"
command = "open -a Slack"
```

Tuning (camera-space units):

```toml
[radial]
flick_threshold    = 0.03   # minimum displacement to commit (smaller = twitchier)
angular_hysteresis = 0.15   # larger = stickier slice boundaries
```

The **two-hand create gesture** (pinch both hands near each other, pull apart, release) is unchanged. Its ghost rectangle now smooths with an EMA (`[create] smoothing = 0.35`) so the preview doesn't jitter while you're sizing.

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

### Peace sign — show desktop

Hold a peace sign (index + middle extended, ring + pinky curled) on either hand for **0.3 seconds** → triggers macOS **Show Desktop** (sends F11). Release to rearm — a single hold fires once, you must drop the pose before it can fire again. Respects the failsafe, so a disabled handspring won't trigger.

Customize in `~/.config/handspring/config.toml`:

```toml
[gestures]
peace_hold_seconds = 0.3   # seconds the pose must be held
peace_command      = ""    # empty = built-in show_desktop (F11); set to any
                           # shell command to override (e.g., "osascript -e 'tell app \"Spotify\" to playpause'")
```

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

Handspring reads its settings from a TOML file at `~/.config/handspring/config.toml` (override with `--config PATH`). The file is auto-created with defaults on first run. Two ways to edit:

1. **In the browser:** open the radial with your left hand and select **Settings**. Your default browser opens `http://127.0.0.1:8766/` with sliders for every knob plus a drag-and-drop editor for the radial tree. Saving applies instantly.
2. **In your editor:** edit the TOML directly. A background watcher picks up file changes within ~1 second — no restart needed.

Both paths write through the same config, so you can freely mix them.

### What's in the config

```toml
[cursor]
smoothing     = 0.35   # EMA — higher = snappier, lower = smoother
inset         = 0.08   # camera dead zone at each edge

[radial]
flick_threshold    = 0.03   # minimum displacement to commit (smaller = twitchier)
angular_hysteresis = 0.15   # larger = stickier slice boundaries

[scroll]
deadzone      = 0.12
max_pixels    = 30

[create]
entry_distance = 0.08
min_diagonal   = 0.15

[failsafe]
hold_seconds  = 5.0    # both-fist hold to toggle gestures

[overlay]
enabled       = true
scale         = 1.0

[colors]
radial_highlight = [136, 255, 0]
radial_outline   = [200, 200, 200]
cursor_dot       = [136, 255, 0]

[features]
tiling          = true
spaces          = true
mission_control = true
screenshots     = true

[server]
web_port       = 8765
settings_port  = 8766
```

### Radial tree

The tree is an ordered list of `[[radial_tree]]` entries. Clockwise from top. Each entry has a name and an optional shell `command` for user-defined leaves.

```toml
[[radial_tree]]
name = "None"

[[radial_tree]]
name = "Settings"

# ... built-ins: Mission, Create, Scroll, Screenshot ...

# A user-added leaf with a custom command — fires on release.
[[radial_tree]]
name    = "Slack"
command = "open -a Slack"
```

Built-in command names are `None`, `Settings`, `Mission`, `Create`, `Scroll`, `Screenshot`. Anything else with a `command` field runs that shell command when fired.

### Disabling the settings UI

Pass `--no-settings` to skip the settings server entirely. You can still hand-edit the TOML.

### Overlay visuals

`src/handspring/overlay.py`:

```python
r_inner = 40.0   # center dead-zone ring radius
r_outer = 220.0  # outer radius of the single ring
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
  --settings-port PORT     settings UI port (default: 8766)
  --no-settings            disable the settings web UI (radial → Settings)
  --config PATH            override config.toml location
                           (default: ~/.config/handspring/config.toml)
  --no-overlay             disable the native always-on-top overlay
  --fps-log-interval S     console FPS readout interval (default: 0.5)
  --skip-tutorial          skip the first-run tutorial
  --tutorial               force-run the tutorial even if config exists
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
- `config.py` — typed dataclass `Config`, TOML load/save, thread-safe `ConfigStore`, mtime-poll file watcher. Read-through snapshots; no half-written state.
- `settings_server.py` — separate HTTP server serving the settings SPA (`GET /`, `GET /api/config`, `POST /api/config`, `POST /api/reload`). Opened from the radial via the Settings command.
- `jarvis.py` — the old internal-windows system. Not wired into the current desktop entry point, but the gesture detection logic is reused as a reference. Still used by the OSC emitter's `/jarvis/*` addresses for external consumers.
- `synth.py`, `synth_params.py`, `synth_ui.py` — audio engine, thread-safe parameter container, gesture → param mapping.

### Testing

```bash
pytest              # 183 tests
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

Tests mock `os_control.*` so they don't actually drive your OS. The tracker/MediaPipe path is bypassed in tests via synthetic `FrameResult` fixtures.

---

## Use as a library

Handspring ships with two CLIs (`handspring`, `handspring-synth`), but the package is also importable. Use it to add gesture input to your own Python app — creative coding, accessibility tools, performance art, whatever.

### Minimal event loop

The core of handspring is a `Tracker` that turns camera frames into a typed `FrameResult`. Everything else (OSC, desktop control, overlay, synth) is an optional consumer of that result.

```python
import cv2, time
from handspring.tracker import Tracker, TrackerConfig

cap = cv2.VideoCapture(0)
tracker = Tracker(TrackerConfig(max_hands=2, track_face=False, track_pose=False))

try:
    while True:
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.01); continue
        out = tracker.process(bgr)
        frame = out.frame                    # handspring.types.FrameResult
        if frame.right.present and frame.right.gesture == "pinch":
            print("pinch!", frame.right.features.index_x, frame.right.features.index_y)
finally:
    cap.release(); tracker.close()
```

### Reacting to gestures in your own code

`FrameResult` is a plain dataclass — no callbacks required. Read `left` / `right` / `face` / `pose` fields each frame and drive whatever you like (MIDI, websockets, game input, etc.).

```python
from handspring.features import is_pinching
from handspring.motion import WaveDetector

waves = WaveDetector()
# ...
if is_pinching(frame.right):
    send_midi_note(60)
if waves.update(frame.left, now=time.monotonic()):
    my_app.on_wave()
```

### Reusing the state machine + config

If you want the full desktop gesture vocabulary (radial menu, tiling, etc.) without the CLI wrapper, construct the controller directly and pass callbacks for the "More" actions:

```python
from handspring.config import ConfigStore, start_watcher
from handspring.desktop_controller import DesktopController
from handspring.settings_server import SettingsServer
import webbrowser

store = ConfigStore()                         # ~/.config/handspring/config.toml
watcher = start_watcher(store)                # live reload on file edits
settings = SettingsServer(store, port=8766); settings.start()

desktop = DesktopController(
    mirrored=True,
    store=store,
    on_open_settings=lambda: webbrowser.open(settings.url),
    on_reload_config=store.reload,
    on_quit=my_shutdown,
)

# In your frame loop:
desktop.update(frame, now=time.monotonic())
for event in desktop.pop_events():            # "click_down", "mode:scroll", "run:Slack", ...
    my_app.on_event(event)
```

### Subscribing to config changes

The `ConfigStore` fires a callback whenever the config is swapped (from the UI, from a hand-edit, or from `store.set()`):

```python
store.on_change(lambda cfg: print("new radial tree:", [i.name for i in cfg.radial_tree]))
```

### Threading notes

- The tracker is **not** thread-safe — call `process()` from one thread.
- `ConfigStore`, `WebServer`, `SettingsServer`, and `_MtimeWatcher` all spawn their own daemon threads and are safe to read from any thread.
- The overlay (`handspring.overlay.Overlay`) must be driven from the main thread on macOS because AppKit requires it; pump events each frame via `overlay_inst.pump()`.

### Public surface

Importable names with stable semantics:

```python
from handspring import __version__
from handspring.tracker     import Tracker, TrackerConfig
from handspring.types       import FrameResult, HandResult, HandFeatures
from handspring.features    import is_pinching, hand_features, face_features
from handspring.gestures    import classify_hand
from handspring.motion      import WaveDetector, ClapDetector
from handspring.osc_out     import OscEmitter
from handspring.config      import Config, ConfigStore, RadialItem, load, save, start_watcher
from handspring.desktop_controller import DesktopController
from handspring.settings_server    import SettingsServer
from handspring.web_server         import WebServer, LatestFrame
```

Anything prefixed with `_` is internal. Semver applies from v1.0 onward.

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
