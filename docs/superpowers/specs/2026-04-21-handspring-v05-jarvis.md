# handspring v0.5.0 — JARVIS Mode + App-Level Mode Toggle

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.4.0

## Purpose

Turn handspring into a **shell** that hosts multiple interaction modes. Opening your mouth toggles between modes. The new **JARVIS mode** renders semi-transparent windows you can create, grab, and tap with hand gestures.

## In scope

### Mode toggle

- **Trigger:** `face.features.mouth_open > 0.7` sustained for 15 consecutive frames (~500 ms).
- **Cooldown:** 1.5 s after each toggle to prevent rapid flipping.
- **Modes:** `synth` (default) and `jarvis`.
- **In Jarvis:** synth is muted (volume forced to 0) but the audio stream stays open; switching back restores the last user-set volume.

### JARVIS gestures

1. **Create window** — both hands pinching (`pinch ≥ 0.85`) + hand centers move apart.
   - While both are pinching: a preview window draws between the hands in real time, sized by the distance.
   - Committed when either hand releases pinch, IF the final diagonal is at least 0.1 of frame size; otherwise discarded.
   - Window aspect ratio constrained to [0.5, 2.0] — prevents pathological slivers from hand twists.
2. **Grab + drag** — single hand open → fist while palm is over a window.
   - On `gesture` transition `open → fist` (any hand): if the palm center is inside any window's bounds, that window is "grabbed". If over multiple windows, the topmost (highest z-order) wins.
   - While grabbed: window follows the palm position (frame-over-frame delta added to window position).
   - On `fist → open`: released. Released window remains at new position, promoted to top of z-order.
3. **Tap** — single hand pointing.
   - When `gesture == "point"`: if the index fingertip is inside a window's bounds, start a hover timer.
   - Timer must accumulate 150 ms (5 frames at 30 FPS) continuously inside the window.
   - On threshold crossed: fire a tap event on the topmost window under the fingertip. 400 ms per-window cooldown prevents re-tapping immediately.
   - Tap effect: window cycles through 3 colors (blue → green → purple → blue).

### Windows

- Data: rectangular regions in normalized (0..1) coordinates with a z-order index.
- Max 8 windows simultaneously; creating a 9th evicts the oldest (first-in-first-out).
- Default color: semi-transparent blue `#4DB4FF` @ 35% alpha (OpenCV overlay + `addWeighted`).
- Border: cyan 2 px.
- Title bar: "Window N" at top-center of each window.
- Highlighted (grabbed or mid-tap): border flashes neon green and thickens to 3 px.

### OSC additions

```
/app/mode                string    "synth" | "jarvis"      on change only
/jarvis/window_count     int                               on change only
/jarvis/window_created   int       (new window id)         one-shot
/jarvis/window_tap       int       (window id)             one-shot
```

Windows are identified by stable `int` ids assigned at creation time via a monotonic counter.

### Preview overlay additions

- **Mode badge** — top-center of the preview, compact pill-style label: "SYNTH" (white) or "JARVIS" (neon green).
- **Hint strip** (Jarvis mode only) — thin bar under the mode badge: `pinch-open to spawn · grab to drag · point to tap`
- Synth panel in lower-left is **hidden** when in Jarvis mode.
- Skeleton overlay (hands, face, pose) stays visible in both modes.

## New required feature: index tip in HandFeatures

For `tap` to work, we need each hand's index fingertip position. We already compute palm center (MIDDLE_MCP) in `features.hand_features`; now we also extract INDEX_TIP (landmark 8).

**`HandFeatures` grows two required fields:** `index_x`, `index_y` (normalized 0..1). Every call site that constructs `HandFeatures` needs updating.

New OSC addresses (additive):
```
/hand/<side>/index_x    float 0..1
/hand/<side>/index_y    float 0..1
```

## Out of scope (deferred to later releases)

- Actual window content (buttons, forms, text rendering). They're geometric rectangles for v1.
- Resize after creation — fixed size once spawned.
- Close gesture (throw-away, swipe-down, etc.).
- Minimize/maximize.
- Multi-hand simultaneous drag (one grab at a time).
- Menu / window spawner (all windows are generic).
- Real app launching (the "interaction" in v1 is just a color cycle).
- Window persistence across runs.
- Two-finger window manipulation (rotate/zoom).

## Autonomous design decisions

1. **Mouth threshold 0.7.** Feels robust against talking but catches "open wide" intent. If users trigger it accidentally while yawning, we can bump to 0.85 later.
2. **500 ms debounce, 1.5 s cooldown.** Separate durations: debounce prevents false-triggers from micro-expressions; cooldown prevents rapid flipping. Both tunable as constants.
3. **Grab origin = palm center.** Whole-hand grab feels more natural than fingertip-grab.
4. **Tap origin = index tip.** Matches pointing semantics.
5. **Tap cooldown per window.** Global cooldown would block taps on different windows; per-window is more intuitive.
6. **FIFO window eviction.** Oldest-first is simple; users can refine with a close gesture later.
7. **Aspect ratio clamp [0.5, 2.0].** Discards pathologically-narrow creations without arbitrary minimum dimensions.
8. **Z-order** maintained as a list; tapped and grabbed windows promote to top.
9. **Palm "inside window" uses hand.x, hand.y (palm center).** Tap uses `index_x, index_y`. Different reference points intentional.
10. **Synth mute in Jarvis**, not disable. Audio stream stays alive; fast toggle back.
11. **Drag delta** is computed from the palm's frame-over-frame movement, not absolute position. This lets you pick up a window from any offset inside its bounds — it doesn't teleport to your palm.
12. **Color cycle is hardcoded 3 colors.** Tap effect is primarily for visual feedback; the OSC `/jarvis/window_tap` event is where real receivers hook in.

## Acceptance criteria

1. **Toggle:** open your mouth wide for ~0.5 s. Mode badge flips between `SYNTH` and `JARVIS`. Audio goes silent entering Jarvis, resumes returning to synth.
2. **Create:** In Jarvis, pinch both thumb-index pairs so they start close together. Pull hands apart. A semi-transparent blue rectangle appears and grows in real time. Release either pinch → window commits. If you pull apart only a little, nothing spawns (fumble).
3. **Grab:** Move an open hand over a window. Close to a fist. Window's border flashes green; window follows your hand. Open your hand → window stays at the drop point, promoted above other windows.
4. **Tap:** Point with an index finger. Move the fingertip into a window. After ~150 ms, the window's color cycles. OSC emits `/jarvis/window_tap <id>`.
5. **Cap:** Creating a 9th window removes the first-created one.
6. **Compat:** all v0.4.0 behaviors work when in Synth mode.
7. **OSC:** `/app/mode`, `/jarvis/window_count`, `/jarvis/window_created`, `/jarvis/window_tap`, and new `/hand/<side>/index_x|y` all stream as specified.
8. **Test count:** ≥ 110 tests pass. `ruff check`, `ruff format --check`, `mypy src/`, CI all green.
