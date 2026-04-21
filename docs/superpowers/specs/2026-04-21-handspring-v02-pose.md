# handspring v0.2.0 — Body Pose + Unified Skeleton Overlay

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.1.0 (tag `v0.1.0`)

## Purpose

Add body/arm tracking to handspring. Users see a neon-green skeleton over their webcam — arms, hands, face all drawn in one unified style. Pose landmarks are also emitted as OSC so downstream receivers can route arm position to creative parameters.

## In scope

1. MediaPipe Pose integration (`mp.solutions.pose`) alongside the existing Hands + FaceMesh solvers.
2. `PoseState` dataclass; `FrameResult` grows a `pose` field.
3. OSC emission of 8 upper-body joints per frame: `shoulder_left`, `shoulder_right`, `elbow_left`, `elbow_right`, `wrist_left`, `wrist_right`, `hip_left`, `hip_right`. Plus `/pose/present`.
4. Unified neon-green skeleton styling across hand, face, and pose overlays in the preview window. One shared `DrawingSpec` constant.
5. `--no-pose` CLI flag (default: pose ON).
6. Documentation updates: README OSC reference + CLI flags, new protocol types in `@myxr/protocol`-style docs (well, there's no TS here — just the README tables).
7. Tests: `PoseState` dataclass contract; OSC emitter sends pose addresses when present; suppresses them when absent. Classifier and features unchanged.
8. All 31 existing tests still pass.

## Out of scope

- Leg/foot pose landmarks (MediaPipe Pose gives them; we skip them because webcam framing usually cuts them off). Add later on request.
- Pose-based classified gestures ("waving", "hands up", etc.) — continuous positions only for v0.2.0.
- Holistic (a single MediaPipe model combining all three): MediaPipe Holistic exists but has worse per-part accuracy than running the three separate solvers. Defer unless perf demands it.
- Depth disambiguation between the pose model's wrist and the hands model's wrist — both are emitted; consumers pick whichever they prefer. The `/hand/*/*` addresses remain unchanged.

## Autonomous design decisions

1. **Color: `#00ff88` bright green for bones, `#66cc99` desaturated for joint dots.** Classic mocap aesthetic; reads over any webcam background.
2. **Line thickness:** bones 2 px, joints 3 px circle radius. One shared `DrawingSpec` across hand/face/pose.
3. **Pose model complexity:** MediaPipe Pose has three `model_complexity` tiers (0 light, 1 full, 2 heavy). Use complexity 1 (full). Light is too jittery; heavy is too slow.
4. **Face overlay styling change:** currently uses FACEMESH_CONTOURS grey — switch to neon-green to match the skeleton.
5. **Hand overlay styling change:** currently uses MediaPipe's default (white dots + white lines) — switch to neon-green.
6. **OSC addresses:** `/pose/<joint>/x`, `/y`, `/z` for each of the 8 joints, plus `/pose/present`. Values 0..1 normalized for x,y, relative units for z.
7. **Pose landmarks skipped when low visibility:** MediaPipe returns per-landmark `visibility`. If below 0.5, we consider that joint absent and omit its x/y/z messages (but still emit the others and `present=1` for the pose as a whole). `/pose/<joint>/visible` (0/1) tells receivers which joints are trustworthy.
8. **Pose ON by default.** Disabled via `--no-pose` for users who want the performance back or don't need it.

## Architecture

### New MediaPipe landmark indices used (from `mp.solutions.pose.PoseLandmark`)

| Index | Name | Our joint name |
|---|---|---|
| 11 | LEFT_SHOULDER | `shoulder_left` |
| 12 | RIGHT_SHOULDER | `shoulder_right` |
| 13 | LEFT_ELBOW | `elbow_left` |
| 14 | RIGHT_ELBOW | `elbow_right` |
| 15 | LEFT_WRIST | `wrist_left` |
| 16 | RIGHT_WRIST | `wrist_right` |
| 23 | LEFT_HIP | `hip_left` |
| 24 | RIGHT_HIP | `hip_right` |

Note: MediaPipe's "Left"/"Right" is camera-perspective. We invert to user-perspective to match the hand convention established in v0.1.0.

### New types

```python
# src/handspring/types.py (additions)

from typing import Literal

Joint = Literal[
    "shoulder_left", "shoulder_right",
    "elbow_left", "elbow_right",
    "wrist_left", "wrist_right",
    "hip_left", "hip_right",
]

@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visible: bool   # visibility score >= 0.5

@dataclass(frozen=True)
class PoseState:
    present: bool
    joints: dict[Joint, PoseLandmark] | None

# FrameResult grows a `pose: PoseState` field.
```

### OSC wire additions

```
/pose/present                       int  0|1
/pose/<joint>/visible               int  0|1   (per-joint; absent if pose not present)
/pose/<joint>/x                     float 0..1    (normalized; only sent if visible)
/pose/<joint>/y                     float 0..1
/pose/<joint>/z                     float          (relative depth)
```

`<joint>` is one of the 8 strings above.

### Preview styling (unified)

Single module-level constant in `preview.py`:

```python
_SKELETON_DOT = DrawingSpec(color=(153, 204, 102), thickness=-1, circle_radius=3)
_SKELETON_LINE = DrawingSpec(color=(136, 255, 0), thickness=2)
```

(Colors in BGR to match OpenCV — note MediaPipe `DrawingSpec` color is BGR.)

Used for hand (replacing default white), face (replacing grey contours), and pose.

### Tracker changes

`TrackerConfig.track_pose: bool = True`. Tracker constructs a `mp.solutions.pose.Pose` instance alongside Hands and FaceMesh. `process()` runs all three, builds `PoseState` from the result.

### CLI change

```
--no-pose             disable body/arm pose tracking
```

Default: pose on.

### Performance note

MediaPipe Pose (complexity 1) adds ~15-25% CPU overhead in our experience. Typical FPS drop: 30 → 22. Still usable. For users on slower machines or who want the 30 FPS back, `--no-pose` restores the v0.1.0 behavior exactly.

## Tests

- `test_types.py`: `PoseState` + `PoseLandmark` frozen; `FrameResult.pose` accessible.
- `test_osc_out.py`: new tests
  - When pose present with all joints visible, all 8 joints' addresses fire.
  - When pose absent, only `/pose/present 0` fires.
  - Joints with visibility < 0.5 skip their x/y/z messages but emit `visible=0`.
- `test_features.py`, `test_gestures.py`: unchanged.
- No tracker unit tests (no camera); rely on downstream checks.

## Acceptance criteria

1. `python -m handspring` with default flags shows a single unified green skeleton overlay: face contours + hand bones + arm bones, all the same color.
2. Moving arms: OSC receivers see `/pose/shoulder_left/x` etc. change live.
3. Moving out of camera's view: `/pose/*/visible` flips to 0 for that joint; x/y/z messages stop for it.
4. `--no-pose` brings back exactly the v0.1.0 behavior (no pose overlay, no pose OSC).
5. All 31 existing tests + ~6 new pose tests pass.
6. `ruff check`, `ruff format --check`, `mypy src/`, `pytest` all clean.

## Migration / backward compatibility

- `.myxr`-style persisted project files: N/A — handspring has no persistence.
- OSC receivers written against v0.1.0: fully compatible. All existing addresses emit unchanged. New addresses under `/pose/*` are purely additive.
- Users who don't want the new feature: `--no-pose` restores v0.1.0 behavior pixel-for-pixel.
