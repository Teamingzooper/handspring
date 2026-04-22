# handspring v0.5.1 — Refinements

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.5.0

Three refinements to real-world usability problems found after shipping v0.5.0:

## 1. Fist ⇄ thumbs_up classifier confusion

**Problem:** Making a natural fist often classifies as `thumbs_up` because `_thumb_extended` is too permissive (0.7× palm width threshold) and `_thumb_up` only checks that the thumb is slightly above the wrist — both conditions easily met by a relaxed fist where the thumb naturally rests off to the side.

**Fix:**
- `_thumb_extended`: bump threshold from `palm_width × 0.7` → `palm_width × 0.95`. A thumb resting against the fist side won't exceed this; an intentionally extended thumb will.
- `_thumb_up`: add a stricter geometric constraint — the thumb tip must be higher (smaller Y) than every other fingertip (index, middle, ring, pinky tips). A closed fist that happens to have its thumb sticking slightly out to the side still won't satisfy "tip above all other tips".
- Reinforce by requiring the thumb-above-wrist delta to be ≥ 0.15 of frame height (up from 0.10).

## 2. Synth slider anchors in space

**Problem:** In slider edit mode (e.g., left fist + right pointing), the vertical slider widget follows the pointing fingertip. Moving the hand up/down moves the slider widget AND changes the value — the widget drifts away from the point where it originally spawned.

**Fix:** When entering slider mode (first frame the non-fist hand is in `point` gesture during an edit mode), record the fingertip's (x, y) as the anchor. Subsequent frames:
- Display the slider at the anchor position (not the current fingertip).
- Use the current Y to compute the value, not the anchor's Y.

Exit / reset:
- The anchor is discarded when the hand leaves `point` gesture OR when the edit mode ends (fist released). Entering slider mode again later creates a new anchor at the then-current fingertip.

**Scope:** Applies to both slider modes — left-fist+right-point → volume, and right-fist+left-point → cutoff. The XY crosshair mode stays position-following (XY is a spatial control by its nature).

## 3. Jarvis window creation via fingertip pinch

**Problem (implicit):** Current window-create entry condition is "both hands pinching" without a proximity check — gesture is hard to enter deliberately. Window corners come from palm centers, which feels imprecise.

**Fix:**
- **Entry:** both hands pinching (pinch feature ≥ 0.85) AND the distance between the two index fingertips < 0.08 in normalized screen space. This is the "bring the pinch points together" trigger.
- **While in creation:** window is the bounding box of the two current index fingertips (not palm centers). Pull hands apart → window grows live. Aspect-ratio clamp and min-diagonal check remain.
- **Exit/commit:** either hand's pinch feature drops below 0.85. Current rectangle is committed IF it passes the min-diagonal check; otherwise discarded (fumble).

**Existing constants stay:** `_MIN_WINDOW_DIAGONAL = 0.1`, `_PINCH_ON_THRESHOLD = 0.85`. New constant: `_CREATE_ENTRY_DISTANCE = 0.08`.

## Out of scope

- No threshold sliders / user tuning UI. These constants stay hardcoded.
- No changes to grab/tap gestures — those work fine.
- No changes to the synth XY crosshair mode.
- No changes to `/synth/*` or `/jarvis/*` OSC protocol. This is a behavior refinement, not a protocol change.

## Acceptance criteria

1. Clenched fist with thumb tucked inward: classifies as `fist` ≥ 95% of natural attempts (subjective).
2. Intentional thumbs-up (thumb pointed up, other fingers curled): still classifies as `thumbs_up`.
3. In synth edit mode, entering slider mode: the slider draws at the finger's current position and stays pinned there; moving the hand up/down changes only the fill.
4. In Jarvis mode, pinching with hands apart (> 0.08 normalized distance between index tips): does NOT create a window. Bringing pinch points together while both pinching: enters creation mode. Index tips define the corners.
5. All existing tests pass. New tests cover the new behaviors.
6. No OSC changes. Receivers written against v0.5.0 work unchanged.

## Autonomous design decisions

1. **Fist threshold bump:** `0.7 → 0.95`. Picked empirically — 0.85 was my first instinct, but the user reported frequent misclassification, so stricter.
2. **Thumb-tip-above-all-other-tips** rule replaces the current "thumb-above-wrist 0.1" check. Simpler, more robust geometry.
3. **Slider anchor** is reset on gesture change (point → open, or point → none), not only on mode exit. Cleaner UX — a brief non-point frame re-anchors next time.
4. **Index-tip entry distance 0.08** — hands need to be close enough that the pinch points are "together" but loose enough to tolerate hand tremor. Tunable if too strict.
5. **Corners from index fingertips**, not palms. Matches the user's mental model ("the corners are where my index fingers are").
6. **Aspect ratio + min-diagonal checks** still apply post-gesture to discard fumbles.
