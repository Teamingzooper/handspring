# handspring v0.5.2 — Pinch Visualization + Big Sliders

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.5.1

## 1. Giant slider (1:1 fingertip = level)

**Problem:** The v0.5.1 slider widget is a 140-px vertical bar anchored where pointing began. Value still derives from `_y_to_norm(finger_y)` but the widget is small so the 1:1 feel is lost.

**Fix:** Make the slider widget span ~80% of the frame height (top at 10% of height, bottom at 90%). The fill level reaches exactly up to the fingertip's current Y coordinate. The user's fingertip IS the level indicator — moving the finger up raises the fill to meet it.

- Slider horizontal width: bumped from 18 → 36 px for readability on bigger canvases
- Label / value text now draws to the RIGHT of the slider's top
- Anchor behavior preserved: X anchors on first point-frame; Y is always the live fingertip Y

## 2. Per-hand pinch visualization

**Problem:** User can't see whether a pinch is being detected. The raw feature exists but it's invisible.

**Fix:** For every present hand, draw a **faint dotted line** from thumb tip (landmark 4) to index tip (landmark 8). Near the line's midpoint, overlay the normalized Euclidean distance (e.g., `0.032`). When the hand is "actively pinching" the line turns **solid neon green** with the same label, and a small pinch indicator circle draws at the pinch point.

**Definition of "actively pinching"** (new — replaces the implicit `pinch ≥ 0.85` checks sprinkled through motion.py and jarvis.py):

```
is_pinching(hand) := hand.present
                    AND hand.gesture != "fist"
                    AND hand.features.pinch >= 0.85
```

The fist exclusion addresses a real-world confusion: when making a fist, thumb tip is often geometrically close to index tip, giving a high pinch value. A fist is not a pinch.

A new module-level helper `is_pinching(state: HandState) -> bool` lives in `src/handspring/types.py` (or `features.py`; either works — choose `features.py` to keep types pure-data). Motion and Jarvis use it instead of raw threshold checks.

## 3. Inter-hand pinch line

**Problem:** In Jarvis create mode, the user can't tell how close the pinch points are — entry is opaque.

**Fix:** When **both hands are actively pinching**, draw a dotted line between the two index fingertips with the normalized distance at the midpoint. Line turns solid neon green when distance drops below the `_CREATE_ENTRY_DISTANCE` threshold, confirming "you are about to enter create mode".

This replaces guessing — you can see the distance shrink and the line transition as your hands approach.

## 4. Git remote

Add `origin` pointing to `https://github.com/Teamingzooper/handspring.git`. Verify fetch works (indicates the repo exists on GitHub). Do not push in the automated process; the user may want to review before pushing.

## Scope boundaries

- Not changing the `0.85` pinch threshold itself. Just adding fist exclusion + visibility.
- Not changing the `0.08` create-entry distance threshold.
- Not changing the Jarvis create logic beyond plumbing `is_pinching` through.
- Not touching the XY crosshair in synth mode (still frame-spanning, still positional).
- No OSC protocol changes.

## Acceptance criteria

1. In any mode, raise either hand into the frame: a faint dotted line appears from thumb tip to index tip with a small distance number at its midpoint.
2. Bring thumb and index close together (hand gesture not fist): line turns solid neon green, distance number drops below ~0.05.
3. Make a fist: pinch line stays dim no matter how close thumb and index get.
4. In synth `EDIT L` + right pointing: a big vertical volume slider fills the center of the frame. Fingertip Y exactly matches the fill top.
5. In Jarvis with both hands pinching: a dotted line draws between the two index fingertips with distance. When distance drops below 0.08, line turns solid green — window creation is about to trigger.
6. All 123 existing tests pass. New tests for `is_pinching` exclusion.
7. `git remote -v` shows the origin URL.
