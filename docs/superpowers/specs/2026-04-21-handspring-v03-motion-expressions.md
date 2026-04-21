# handspring v0.3.0 — Motion Gestures, Facial Expressions, More Hand Shapes

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.2.1

## Purpose

Broaden the expressiveness of handspring by adding:

1. **Temporal motion gestures**: wave, pinch, expand, drag, clap — detected over a rolling 1-second history, not from single frames.
2. **Facial expressions**: smile, frown, surprise, wink_left, wink_right — classified from face landmark geometry each frame.
3. **Three more static hand shapes**: `ok`, `rock`, `three`.

Receivers can now react to richer user input without writing their own gesture-recognition logic.

## Architecture additions

- **`history.py`**: per-hand ring buffer of `HandFeatures` + timestamps (30 frames, ~1 s). Plus an inter-hand buffer of hand-to-hand distance for clap detection.
- **`motion.py`**: pure-function motion detectors reading from history buffers. Stateless classifiers + buffer for state.
- **`expressions.py`**: face-expression classifier using geometric rules on FaceMesh landmarks (requires `refine_landmarks=True`).
- **`gestures.py`**: extended with three more shape classifications.

Data flow (per frame):

```
camera → MediaPipe → features → static gesture (hand shape)
                              → expression (face)
history buffer .push(features)
history buffer → motion detectors → motion events
all → OSC emitter + preview
```

## In scope

### Motion events (one-shot, emitted once when detected)

| Event | Trigger | Cooldown |
|---|---|---|
| `/hand/<side>/event "wave"` | `hand.x` oscillates amplitude > 0.05, frequency 1.5–4 Hz, for ≥ 0.8 s, while `hand.y < 0.5` | 1.0 s |
| `/hand/<side>/event "pinch"` | `pinch` feature rising edge past 0.85 | 0.3 s |
| `/hand/<side>/event "expand"` | `pinch` feature falling edge past 0.4 after being above 0.85 | 0.3 s |
| `/hand/<side>/event "drag_start"` | `pinching=1` AND hand velocity > 0.1 units/s for ≥ 0.2 s | — |
| `/hand/<side>/event "drag_end"` | `dragging=1` AND (`pinching=0` OR velocity < 0.03 for ≥ 0.3 s) | — |
| `/motion/clap 1` | hand-to-hand distance falls below 0.08 after being above 0.25 within last 0.3 s | 0.4 s |

### Continuous motion state (per hand, emitted every frame)

```
/hand/<side>/pinching    int 0|1
/hand/<side>/dragging    int 0|1
/hand/<side>/drag_dx     float   (0.0 when not dragging; relative to drag_start position)
/hand/<side>/drag_dy     float
```

### New hand shapes

Extending the existing `Gesture` union:

```
Gesture = "fist" | "open" | "point" | "peace" | "thumbs_up" | "ok" | "rock" | "three" | "none"
```

Classifier rules:

- **`ok`**: thumb + index tips coincident (distance < 0.2 × palm width), middle + ring + pinky extended
- **`rock`**: index + pinky extended; middle + ring curled; thumb ignored
- **`three`**: index + middle + ring extended; pinky curled; thumb ignored

Priority order in classifier: `thumbs_up` → `ok` → `open` → `fist` → `peace` → `rock` → `three` → `point` → `none`. (Priority matters because several rules overlap — `ok` and `open` both have index/middle/ring extended; `ok` must be checked first.)

### Facial expressions

New module `expressions.py`. Requires enabling `refine_landmarks=True` in FaceMesh for better eye and lip landmarks (~10% more CPU, acceptable).

```
Expression = "smile" | "frown" | "surprise" | "wink_left" | "wink_right" | "neutral"
```

Rules (all distances normalized by eye-distance for scale-invariance):

- **`smile`**: both mouth corners above mid-lip baseline by > 0.02 × eye_distance
- **`frown`**: both mouth corners below mid-lip baseline by > 0.015 × eye_distance
- **`surprise`**: `mouth_open > 0.55` AND both eyes open > 0.85
- **`wink_left`**: left eye open < 0.2, right eye open > 0.6
- **`wink_right`**: right eye open < 0.2, left eye open > 0.6
- **`neutral`**: none of the above

Priority: surprise → wink → smile/frown → neutral.

New continuous face features:

```
/face/eye_left_open     float 0..1   vertical lid distance / eye width, normalized
/face/eye_right_open    float 0..1
/face/expression        string       state-change emission, mirrors /hand/*/gesture pattern
```

`eye_open` formula: given upper-lid, lower-lid, left-corner, right-corner landmarks:
- raw = abs(lower.y - upper.y) / abs(right.x - left.x)
- `eye_open` = clamp((raw - 0.02) / 0.3, 0, 1) — ~0 when closed (raw ≈ 0.02), ~1 when wide open (raw ≈ 0.32)

### Updated types (`types.py`)

```python
Expression = Literal["smile", "frown", "surprise", "wink_left", "wink_right", "neutral"]

MotionEvent = Literal["wave", "pinch", "expand", "drag_start", "drag_end"]

@dataclass(frozen=True)
class MotionState:
    pinching: bool
    dragging: bool
    drag_dx: float
    drag_dy: float
    event: MotionEvent | None   # one-shot for this frame; None most frames

# HandState grows:
@dataclass(frozen=True)
class HandState:
    present: bool
    features: HandFeatures | None
    gesture: Gesture
    motion: MotionState         # NEW

# FaceState grows:
@dataclass(frozen=True)
class FaceState:
    present: bool
    features: FaceFeatures | None
    expression: Expression      # NEW — "neutral" when face absent
    eye_left_open: float        # NEW — 0..1, 0 when face absent
    eye_right_open: float       # NEW

# FrameResult grows:
@dataclass(frozen=True)
class FrameResult:
    left: HandState
    right: HandState
    face: FaceState
    pose: PoseState
    fps: float
    clap_event: bool            # NEW — True for exactly one frame when a clap is detected
```

### New OSC messages (additive; all existing messages unchanged)

Per frame (continuous):
- `/hand/<side>/pinching` int
- `/hand/<side>/dragging` int
- `/hand/<side>/drag_dx` float (only when `dragging=1`)
- `/hand/<side>/drag_dy` float (only when `dragging=1`)
- `/face/eye_left_open` float
- `/face/eye_right_open` float

On state transitions (like existing `gesture`):
- `/face/expression` string — emitted only when expression changes

One-shot events (only on the frame they're detected):
- `/hand/<side>/event` string — "wave" | "pinch" | "expand" | "drag_start" | "drag_end"
- `/motion/clap` int 1

## Out of scope (deferred)

- **Pose-based motion** (jumping, arms raised, T-pose, running) — v0.4.0.
- **Custom gesture recording / training** — v0.4.0+.
- **Threshold tuning UI or config file** — thresholds are hard-coded for v0.3.0. Adjust by editing `motion.py` or `expressions.py`. If thresholds need per-user tuning, a future release adds `--gesture-config <path>`.
- **Digit counting (1–5)** — ambiguous with other gestures.
- **Multi-person** — still one face, two hands.
- **Preview-window visualization of motion events** — the status line will show active motion/expression state but not a big pop-up.
- **Clap sub-classification** (single vs double vs triple) — single clap events only.

## Autonomous design decisions

1. **Buffer size = 30 frames** (~1 s at 30 FPS). Big enough for wave periodicity detection, small enough that memory is trivial.
2. **`refine_landmarks=True`** on FaceMesh. Required for accurate eye openness. Accept ~10% FPS cost.
3. **Clap goes under `/motion/clap`**, not `/hand/both/*`, because it's a bi-hand phenomenon that belongs in its own address tree. Future motions like "both_hands_raised" would join `/motion/*`.
4. **`drag_dx`, `drag_dy` are relative to drag-start position**, not frame-over-frame delta. Gives the receiver absolute drag magnitude directly.
5. **Status line in preview** now has 6 lines: FPS / OSC / Left / Right / Pose / Face. Face line shows current expression.
6. **No `drag_dx/dy` emission when not dragging.** Avoids receivers having to filter noise. The `dragging` flag gates the other two.

## Acceptance criteria

1. **Visual:** `python -m handspring` preview still shows the full neon skeleton. Status line includes face-expression state.
2. **Static gestures:** `ok`, `rock`, `three` classify correctly in ≥ 80% of natural lighting cases (subjective).
3. **Wave:** waving an open hand 3–4 times causes exactly one `/hand/<side>/event "wave"` message. Holding the wave continuously re-fires after the 1 s cooldown.
4. **Pinch → expand:** touching thumb to index, then separating, emits `/hand/<side>/event "pinch"` then `"expand"`. `/hand/<side>/pinching` toggles `1` then `0` accordingly.
5. **Drag:** pinch + move hand 10 cm sideways emits `drag_start`, continuous `drag_dx`/`drag_dy`, then `drag_end` on release or stop.
6. **Clap:** clapping emits `/motion/clap 1` once per clap impact.
7. **Expression:** smile → `/face/expression "smile"`; revert to neutral → `/face/expression "neutral"`.
8. **Compatibility:** all v0.2.1 OSC addresses emit unchanged. Receivers written against v0.2.1 continue to work.
9. **Tests:** ≥ 50 total. New tests for history buffer, each motion detector, each expression, each new hand shape.
10. **CI green**: ruff, ruff-format, mypy, pytest all clean on Ubuntu + macOS.

## Known tradeoffs

- **Wave detection is fuzzy.** Short waves (1–2 cycles) won't trigger. False positives from hand-position jitter while holding a peace sign with shaky hand: possible but rare given the amplitude threshold.
- **Pinch/expand events fire whenever the pinch feature crosses the thresholds — including accidentally when the hand rotates and temporarily occludes the finger tips.** Cooldown (0.3 s) mitigates but doesn't eliminate.
- **Clap requires both hands visible.** Clapping with hands partly out of frame won't detect.
- **Wink requires asymmetric eye closure** — a big enough difference between left and right `eye_open` values. Two-eye blinks don't trigger wink; they just drop `surprise` confidence.
- **`refine_landmarks=True`** adds ~10% CPU. On modest hardware with pose enabled, FPS may drop to ~18.
