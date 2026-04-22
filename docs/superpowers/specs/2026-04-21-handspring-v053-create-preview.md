# handspring v0.5.3 — Live Window Creation Preview

Make the in-progress window visible while creating. While both hands are pinching and the inter-hand line is green (create state active), render a dashed "ghost" rectangle whose corners are the current index tips. The ghost resizes in real time as you move your hands. On pinch release, the ghost commits to a real window (existing behavior unchanged).

## Changes
1. `_CreateState` tracks current positions (not just start positions).
2. `JarvisController.pending_rect() -> (x, y, w, h) | None` exposes the in-progress bounds.
3. Preview renders the pending rect as a semi-transparent ghost with dashed border.

## Acceptance
- Enter create (both pinching, index tips close): dashed rectangle appears spanning the two index tips.
- Move hands apart: rectangle grows/shrinks in real time.
- Release pinch: rectangle commits as a solid window (existing behavior).
- Cancel (e.g., pull hands back together past entry threshold): rectangle still shows, but tiny-diagonal fumble on release is discarded (existing behavior).
