# handspring v0.5.4 — Align Pinch Detection With Visual Threshold

Window creation in Jarvis wasn't triggering even when the viz shows green pinch lines. Root cause: two inconsistent thresholds for "pinching".

- **Viz threshold** (preview.py): raw thumb↔index distance < 0.05 in normalized frame coords
- **Logical threshold** (features.is_pinching): pinch feature ≥ 0.85, i.e. distance < ~0.018 × hand_span

Raw and span-normalized thresholds disagree — the UX hint (green line) fired well before the logic registered.

Extra complication: when physically pinching, the index finger curls inward, the classifier calls it "fist", and `is_pinching` was excluding fists — blocking what was functionally a real pinch.

## Fix

1. Add `thumb_x`, `thumb_y` to `HandFeatures` (same pattern as `index_x`/`index_y`).
2. Redefine `is_pinching(state)`:
   ```
   is_pinching(state) := state.present AND state.features is not None
                        AND hypot(thumb_x - index_x, thumb_y - index_y) < 0.05
   ```
   Same formula the viz uses. What you see IS what the logic sees.
3. Drop the `gesture != "fist"` exclusion. Raw 0.05 is tight enough that real balled-up fists (thumb tucked aside) fail it on their own.
4. Emit `/hand/<side>/thumb_x` and `/hand/<side>/thumb_y` via OSC.

## Acceptance
- Window creation works when both hands approach, pinch, index tips green inter-hand line, then pull apart.
- Real fists (thumb off to the side) do not register as pinches.
- No regression in volume/cutoff/create edit modes.

## Scope
- No change to the `pinch` feature itself — downstream OSC consumers keep their span-normalized value.
- No change to motion detector pinch hysteresis constants.
