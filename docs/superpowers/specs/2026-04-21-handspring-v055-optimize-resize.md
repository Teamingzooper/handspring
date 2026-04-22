# handspring v0.5.5 — Optimization + Window Resize

## Part A: eliminate double MediaPipe inference

Currently `__main__.main()` calls `tracker.process(bgr)` (runs hands+face+pose) AND `_extract_landmark_lists(tracker, bgr)` (runs them AGAIN) to get raw landmark lists for the preview.

Fix: `Tracker.process()` now returns `(FrameResult, TrackerRawLandmarks)` where the latter carries the raw landmark lists the preview needs. `_extract_landmark_lists` is deleted.

Expected FPS gain: ~20-30% on modest hardware.

## Part B: window resize via top-right corner pinch

Each window gets a small cyan corner handle at its top-right. When a single hand pinches with its index tip within 0.04 of a window's top-right corner, resize mode activates for that window. While the pinch is held, the top-right corner tracks the index tip (bottom-left stays fixed). On pinch release, resize commits.

New constants:
- `_RESIZE_CORNER_RADIUS = 0.04`
- `_MIN_RESIZE_SIZE = 0.05`

New state: `_ResizeState(window_id, side, anchor_x, anchor_bottom_y)`. Handler priority: resize > create > grab > tap. Can't enter resize during create.

### Requirements for entry
1. Exactly one hand is pinching (`is_pinching` True).
2. That hand's index tip is within `_RESIZE_CORNER_RADIUS` of some window's top-right corner.
3. `_create` is None (not mid-create).

If the other hand is also pinching (both pinching), create takes priority if index tips are close; otherwise resize still active (only triggers when explicitly near a corner).

### Visual
- Always-visible cyan square handle (~10px) at top-right corner of each window.
- When a window is being resized: handle pulses green + border flashes green (same pattern as grab/tap highlight).

## Acceptance
1. FPS increases with optimization (subjective — see-for-yourself).
2. Cyan square visible at top-right of every window.
3. Pinch near top-right corner → window resizes live.
4. Release pinch → resize commits; handle returns to normal.
5. Minimum size 0.05 on both axes.
6. All existing tests pass. New tests for resize state machine.
