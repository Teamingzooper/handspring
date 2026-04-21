# handspring v1 — Design

**Date:** 2026-04-21
**Status:** Approved
**Project:** handspring

> Note: the host directory is still named `myxr-src` from an abandoned prior project. The Python project inside is `handspring`; the directory name is a cosmetic inconsistency the user can fix (`mv` the parent) whenever convenient.

## Purpose

`handspring` is a small Python script that turns a webcam feed into a live stream of hand/face gesture data, emitted as OSC, ready to drive any creative project — audio synths, visuals, game controls, prompt generators. One person using one webcam, minimum friction, maximum downstream usefulness.

The core deliverable is **a reliable gesture→OSC stream**, not a destination app. A bundled demo receiver (a Python sine synth) proves end-to-end functionality and gives users a template for writing their own.

## In scope (v1)

1. Python 3.10+ script invoked via `python -m handspring` or a `handspring` console entry point.
2. Opens the default webcam; supports overriding index via `--camera N`.
3. MediaPipe hand tracking (up to 2 hands) + MediaPipe face landmarks (1 face).
4. OpenCV preview window with drawn landmarks and current gesture labels. Preview toggleable via `--no-preview`.
5. OSC output over UDP to `127.0.0.1:9000` by default; `--host` and `--port` CLI flags.
6. Continuous feature stream (sent every processed frame, ~30 Hz on modern CPUs):
   - Per hand: `/hand/<side>/x`, `/y`, `/z`, `/openness`, `/pinch`, `/present`
   - Face: `/face/yaw`, `/face/pitch`, `/face/mouth_open`, `/face/present`
   - `<side>` is `left` or `right` (from the user's perspective, not the camera's).
7. Classified discrete gesture events (per hand, sent only on state transitions):
   - `/hand/<side>/gesture` with string payload: `"fist" | "open" | "point" | "peace" | "thumbs_up" | "none"`
8. Terminal status line updates every ~500 ms: current FPS, hands present, active gestures.
9. Graceful shutdown on Ctrl+C: releases camera, stops OSC, closes preview.
10. Bundled example receiver: `examples/tone_synth.py` — plays sine tones modulated by hand position (left Y → pitch, right Y → amplitude), muted by `fist`, unmuted by `open`.
11. Tests (pytest) for feature derivation and gesture classification using synthesized landmark fixtures. No camera required.
12. `pyproject.toml` with PEP 621 metadata, MIT license, dependencies pinned to current majors.
13. CI on GitHub Actions: lint (`ruff`), format check (`ruff format --check`), type-check (`mypy`), pytest. macOS + Linux runners (Windows stretch — see Future work).

## Out of scope (v1; deferred)

- **MIDI output.** OSC only; users bridge to MIDI with existing tools (`osculator`, `loopMIDI`, `QLab`).
- **Body/pose tracking.** MediaPipe has `pose`, but two hands + face already covers the stated use cases.
- **Multi-person.** Two hands and one face per frame, all belonging to the closest subject.
- **Custom gesture training / recording.** Hard-coded classifier in v1.
- **User-editable config file.** CLI flags only.
- **Face gesture classification.** Face features are continuous (`mouth_open`, `yaw`, `pitch`); no discrete face events.
- **GUI configuration app.** Terminal + preview window only.
- **Installer / packaging as standalone binary.** Users run from source (`uv` or `pip install -e .`). PyInstaller / a notarized macOS app is a post-v1 concern.
- **Windows support as a hard requirement.** Should work (pure-Python stack), but not matrixed in CI until someone needs it.
- **Calibration / per-user training.** Default MediaPipe confidence thresholds.

## Architecture

### Module layout

```
handspring/
├── pyproject.toml
├── README.md
├── LICENSE                   # MIT
├── .gitignore
├── .github/workflows/ci.yml
├── src/handspring/
│   ├── __init__.py           # re-exports + __version__
│   ├── __main__.py           # `python -m handspring` entry, CLI parsing, main loop
│   ├── tracker.py            # MediaPipe setup + per-frame inference
│   ├── features.py           # landmarks → normalized continuous features
│   ├── gestures.py           # landmarks → classified discrete gesture
│   ├── osc_out.py            # python-osc client + state-change dedupe
│   ├── preview.py            # OpenCV preview window rendering
│   └── types.py              # dataclasses for FrameResult, HandState, FaceState, etc.
├── tests/
│   ├── test_features.py
│   ├── test_gestures.py
│   ├── test_osc_out.py
│   └── fixtures.py           # hand-written synthetic landmark arrays
└── examples/
    └── tone_synth.py         # OSC receiver → sounddevice sine synth
```

### Dependencies

Pinned in `pyproject.toml`:

- `mediapipe ~= 0.10` — hand + face tracking (Google)
- `opencv-python ~= 4.10` — camera capture, preview rendering
- `python-osc ~= 1.8` — OSC sender
- `numpy ~= 1.26` — landmark math (MediaPipe returns numpy arrays)

Dev:

- `pytest ~= 8.0`
- `ruff ~= 0.6`
- `mypy ~= 1.11`
- `sounddevice ~= 0.4` — only for `examples/tone_synth.py`

Python 3.10+ (MediaPipe's supported range). Prefer `uv` for the project's virtualenv but accept plain `pip` too.

### Data flow

Single main thread, top-of-loop style:

```
cv2.VideoCapture → MediaPipe hand+face → features+gestures → OSC client (UDP:9000)
                                                          → preview.py (OpenCV window)
                                                          → terminal status line
```

Single-threaded is fine at 30 FPS on a modern CPU; no real-time safety constraints (OSC is fire-and-forget UDP, preview is best-effort rendering).

### Core types

```python
# src/handspring/types.py
from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Gesture = Literal["fist", "open", "point", "peace", "thumbs_up", "none"]

@dataclass(frozen=True)
class HandFeatures:
    x: float
    y: float
    z: float
    openness: float
    pinch: float

@dataclass(frozen=True)
class HandState:
    present: bool
    features: HandFeatures | None
    gesture: Gesture

@dataclass(frozen=True)
class FaceFeatures:
    yaw: float
    pitch: float
    mouth_open: float

@dataclass(frozen=True)
class FaceState:
    present: bool
    features: FaceFeatures | None

@dataclass(frozen=True)
class FrameResult:
    left: HandState
    right: HandState
    face: FaceState
    fps: float
```

### Gesture classifier

MediaPipe returns 21 landmarks per hand. A finger is "extended" if its tip is farther from the wrist than its PIP joint. Gestures:

- **fist:** all five fingers curled.
- **open:** all five extended, with spread between fingertips exceeding a threshold.
- **point:** index extended, middle/ring/pinky curled. Thumb state ignored.
- **peace:** index + middle extended, ring + pinky curled.
- **thumbs_up:** thumb extended and pointing upward relative to palm, other four curled.
- **none:** anything else, or no hand present.

Pure function `landmarks: np.ndarray -> Gesture`. No state, no history. Debouncing happens in `osc_out.py` (only emit on change).

### OSC output

Stateful wrapper around `python_osc.udp_client.SimpleUDPClient`:

- Every frame: send all continuous features. No dedupe on continuous values.
- State-change detection per hand for `/hand/<side>/gesture`: only sent when the classified gesture differs from the previous frame.
- Values coerced to OSC-native types: `float` for features, `int` (0/1) for `present`, `string` for `gesture`.

### Preview window

OpenCV window showing live mirrored camera feed with drawn hand landmarks (21 dots + bones per hand), drawn face landmarks (outline + nose + mouth), and text overlay (gestures per hand, FPS, OSC target). Press `q` or close to quit.

### CLI surface

```
usage: python -m handspring [-h] [--host HOST] [--port PORT] [--camera N]
                             [--no-preview] [--no-face] [--hands {0,1,2}]
                             [--mirror/--no-mirror] [--fps-log-interval SEC]
```

## Demo receiver — `examples/tone_synth.py`

Listens on 127.0.0.1:9000. Mapping:

- `/hand/left/y`    → pitch in 200–800 Hz (mapped exponentially: low Y = low pitch)
- `/hand/right/y`   → amplitude in 0..0.3 (linear)
- `/hand/left/gesture == "fist"` → mute
- `/hand/left/gesture == "open"` → unmute

Smooths pitch/amplitude with a one-pole low-pass (coef ~0.9) to avoid zipper noise. Runs until Ctrl+C.

## Tests

`tests/test_features.py`:

- `HandFeatures` values stay in `[0, 1]` (or `[-1, 1]` for face angles)
- `openness` ~0 for fist fixture, ~1 for open fixture
- `pinch` ~1 when thumb + index are coincident, ~0 when spread
- `face.mouth_open` ~0 for closed-mouth fixture, >0.3 for wide-open

`tests/test_gestures.py`:

- Each of the 5 recognized gestures has a fixture landmark array that classifies to itself
- Clearly non-matching fixtures classify as `"none"`
- Deterministic (same input → same output across 100 runs)

`tests/test_osc_out.py`:

- Continuous features emitted every frame
- Gesture event emitted only on state transitions

Fixtures hand-written in `tests/fixtures.py` — factory functions per gesture.

CI runs pytest on macOS-latest and ubuntu-latest.

## Acceptance criteria

1. `uv sync` (or `pip install -e '.[dev]'`) installs all deps on macOS and Linux.
2. `python -m handspring` opens the default webcam, shows a preview window, and prints FPS + gesture states in terminal within 3 seconds.
3. Moving hands changes `/hand/*/x`, `/y`, `/openness` values live.
4. Making a fist emits exactly one `"/hand/<side>/gesture", "fist"` OSC message on the state transition; holding fist does not spam.
5. The five gestures classify correctly ≥95% of the time in natural lighting (subjective user test).
6. Ctrl+C cleanly terminates: camera released, no zombie processes.
7. `python examples/tone_synth.py` plays audible, smooth sine tones modulated by hand position and muted by `fist`.
8. `pytest` passes all tests (offline, no camera).
9. `ruff check` and `ruff format --check` clean.
10. `mypy src/` clean.
11. CI green on macOS-latest + ubuntu-latest.

## Future work (not v1)

- MIDI output via `mido` + virtual device
- Custom gesture recording & training
- More gestures: `ok`, `rock`, `gun`, digit counting (1–5)
- Face gesture events: wink, eyebrow raise
- Body/pose tracking (MediaPipe Pose)
- WebSocket output for browser receivers
- Configurable mapping layer (YAML)
- Tauri "Studio" GUI wrapping the script
- Visual demo receiver (p5.js)
- The "rap freestyle helper" — separate project consuming handspring's OSC stream
