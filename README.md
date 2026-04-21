# handspring

Webcam → hand/face tracking → OSC stream.

Drop-in gesture input for creative-coding projects: audio synths, visuals, game controls, prompt generators. Written in Python, powered by [MediaPipe](https://mediapipe.dev/).

## Quick start

```bash
# 1. Install (Python 3.10+ required)
pip install -e '.[dev]'
# or with uv:
# uv sync

# 2. Run
python -m handspring
```

A preview window opens showing your webcam with hand/face landmarks drawn.
OSC packets fly out to `127.0.0.1:9000` (UDP). Terminal shows FPS + current
gesture state.

## Demo

In one terminal:

```bash
python -m handspring
```

In another:

```bash
python examples/tone_synth.py
```

Move your hands. The synth plays sine tones — left hand Y controls pitch,
right hand Y controls amplitude. Make a fist with your left hand to mute;
open your palm to unmute.

## Built-in synth

handspring includes an in-process synth. The moment you run `python -m handspring`
a sustained tone plays. Put your LEFT hand in a fist to enter the edit mode —
point with the right hand to adjust volume; open right hand to change pitch (Y)
and stepping (X). Put your RIGHT hand in a fist to edit filter cutoff and
modulation with your left hand.

Disable the built-in synth (for OSC-only workflows) with `--no-synth`.

### Synth parameters

```
volume       0..1
note_hz      131..1047 (C3..C6)
stepping_hz  0..16  — envelope retrigger rate for buildup pulses
cutoff_hz    200..8000  — lowpass filter
mod_depth    0..1   — amplitude tremolo depth
mod_rate     0.1..10  — LFO frequency
mode         play | edit_left | edit_right
```

## OSC reference

Continuous features (sent every frame, ~30 Hz):

| Address | Type | Range | Notes |
|---|---|---|---|
| `/hand/<side>/present` | int | 0 or 1 | `<side>` is `left` or `right` (user's perspective) |
| `/hand/<side>/x` | float | 0..1 | palm center, normalized to frame width |
| `/hand/<side>/y` | float | 0..1 | palm center, normalized to frame height |
| `/hand/<side>/z` | float | — | relative depth |
| `/hand/<side>/openness` | float | 0..1 | 0 = fist, 1 = open palm |
| `/hand/<side>/pinch` | float | 0..1 | thumb-index proximity |
| `/face/present` | int | 0 or 1 | |
| `/face/yaw` | float | -1..1 | negative = looking left |
| `/face/pitch` | float | -1..1 | negative = looking down |
| `/face/mouth_open` | float | 0..1 | |

Discrete gesture events (per hand, sent only on state transitions):

| Address | Type | Values |
|---|---|---|
| `/hand/<side>/gesture` | string | `fist` \| `open` \| `point` \| `peace` \| `thumbs_up` \| `ok` \| `rock` \| `three` \| `none` |

Body pose (continuous per frame when present):

| Address | Type | Range | Notes |
|---|---|---|---|
| `/pose/present` | int | 0 or 1 | 1 when a body is detected |
| `/pose/<joint>/visible` | int | 0 or 1 | per-joint visibility; `0` means unreliable |
| `/pose/<joint>/x` | float | 0..1 | sent only when `visible=1` |
| `/pose/<joint>/y` | float | 0..1 | |
| `/pose/<joint>/z` | float | — | relative depth |

`<joint>` is one of: `shoulder_left`, `shoulder_right`, `elbow_left`, `elbow_right`, `wrist_left`, `wrist_right`, `hip_left`, `hip_right`.

Motion state per hand (continuous per frame):

| Address | Type | Notes |
|---|---|---|
| `/hand/<side>/pinching` | int | 0 or 1 |
| `/hand/<side>/dragging` | int | 0 or 1 |
| `/hand/<side>/drag_dx` | float | x offset from drag-start, only when `dragging=1` |
| `/hand/<side>/drag_dy` | float | y offset |

Motion events (one-shot per frame):

| Address | Type | Values |
|---|---|---|
| `/hand/<side>/event` | string | `wave` \| `pinch` \| `expand` \| `drag_start` \| `drag_end` |
| `/motion/clap` | int | `1` on each clap impact |

Face (continuous + state-change):

| Address | Type | Notes |
|---|---|---|
| `/face/eye_left_open` | float | 0..1 |
| `/face/eye_right_open` | float | 0..1 |
| `/face/expression` | string | `smile` \| `frown` \| `surprise` \| `wink_left` \| `wink_right` \| `neutral` — emitted only on change |

Synth state (continuous):

| Address | Type | Notes |
|---|---|---|
| `/synth/volume` | float | 0..1 |
| `/synth/note_hz` | float | |
| `/synth/stepping_hz` | float | 0 = sustained |
| `/synth/cutoff_hz` | float | |
| `/synth/mod_depth` | float | 0..1 |
| `/synth/mod_rate` | float | Hz |
| `/synth/mode` | string | `play` / `edit_left` / `edit_right` — on change |

## CLI flags

```
--host HOST            OSC receiver host (default: 127.0.0.1)
--port PORT            OSC receiver port (default: 9000)
--camera N             camera index (default: 0)
--no-preview           disable the OpenCV preview window
--no-face              disable face tracking (hands only)
--no-pose              disable body/arm pose tracking
--hands {0,1,2}        max hands to track (default: 2)
--no-mirror            do not mirror the preview horizontally
--fps-log-interval SEC print status every N seconds (default: 0.5)
--no-synth             disable the in-process synth
```

## Building your own receiver

The OSC stream is the whole point. Any OSC-speaking tool can be a receiver:

- Max/MSP, Pure Data, TouchDesigner — native OSC support
- SuperCollider — `OSCdef` / `NetAddr`
- Ableton Live — via `Max for Live` OSC bridge
- Unity / Godot — via community OSC libraries
- A Python script — see `examples/tone_synth.py`
- A web app — bridge OSC to WebSocket

For MIDI-only applications, use a bridge tool (`osculator` on macOS, `loopMIDI`
on Windows) to forward OSC to MIDI messages.

## Development

```bash
pip install -e '.[dev]'
pytest              # run unit tests (no camera required)
ruff check .        # lint
ruff format .       # format
mypy src/           # type check
```

CI (GitHub Actions) runs the same checks on Ubuntu and macOS with Python
3.10 and 3.12.

## License

MIT. See [LICENSE](./LICENSE).
