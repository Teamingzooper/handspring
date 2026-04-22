"""JARVIS mode: transparent-window creation, grab/drag, tap."""

from __future__ import annotations

from dataclasses import dataclass, replace

from handspring.features import is_pinching
from handspring.types import FrameResult, HandState, Side

_MAX_WINDOWS_DEFAULT = 8
_NUM_COLORS = 3
# Palette (BGR tuples for OpenCV).
WINDOW_COLORS = [
    (255, 180, 77),  # blue  #4DB4FF
    (100, 230, 120),  # green
    (220, 110, 210),  # purple
]


@dataclass(frozen=True)
class Window:
    """Rectangle in normalized (0..1) coordinates."""

    id: int
    x: float
    y: float
    width: float
    height: float
    z: int  # z-order index (higher = on top)
    color_idx: int  # index into WINDOW_COLORS

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


class WindowManager:
    def __init__(self, *, max_windows: int = _MAX_WINDOWS_DEFAULT) -> None:
        self._max = max_windows
        self._next_id = 1
        self._next_z = 1
        self._windows: list[Window] = []  # creation order

    def create(self, *, x: float, y: float, width: float, height: float) -> Window:
        w = Window(
            id=self._next_id,
            x=x,
            y=y,
            width=width,
            height=height,
            z=self._next_z,
            color_idx=0,
        )
        self._next_id += 1
        self._next_z += 1
        self._windows.append(w)
        # Evict FIFO when exceeding the cap.
        while len(self._windows) > self._max:
            self._windows.pop(0)
        return w

    def windows(self) -> list[Window]:
        """Return windows in z-order (bottom → top)."""
        return sorted(self._windows, key=lambda w: w.z)

    def get(self, window_id: int) -> Window | None:
        for w in self._windows:
            if w.id == window_id:
                return w
        return None

    def topmost_at(self, px: float, py: float) -> Window | None:
        hit = [w for w in self._windows if w.contains(px, py)]
        if not hit:
            return None
        return max(hit, key=lambda w: w.z)

    def promote(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        new_w = replace(w, z=self._next_z)
        self._next_z += 1
        self._replace(new_w)

    def move(self, window_id: int, *, dx: float, dy: float) -> None:
        w = self.get(window_id)
        if w is None:
            return
        nx = max(0.0, min(1.0 - w.width, w.x + dx))
        ny = max(0.0, min(1.0 - w.height, w.y + dy))
        self._replace(replace(w, x=nx, y=ny))

    def cycle_color(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        self._replace(replace(w, color_idx=(w.color_idx + 1) % _NUM_COLORS))

    def destroy(self, window_id: int) -> None:
        self._windows = [w for w in self._windows if w.id != window_id]

    def _replace(self, new_w: Window) -> None:
        for i, w in enumerate(self._windows):
            if w.id == new_w.id:
                self._windows[i] = new_w
                return


# ---------------------------------------------------------------------------
# Gesture state machines
# ---------------------------------------------------------------------------

_TAP_HOVER_FRAMES = 5  # 5 frames @ 30fps ≈ 150 ms
_TAP_COOLDOWN_SECONDS = 0.4
_MIN_WINDOW_DIAGONAL = 0.1
_PINCH_ON_THRESHOLD = 0.85
_CREATE_ENTRY_DISTANCE = 0.08
_RESIZE_CORNER_RADIUS = 0.08
_MIN_RESIZE_SIZE = 0.05
_DESTROY_Y_THRESHOLD = 0.88  # palm y below which a released grab deletes the window


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


_SPLIT_MIN_PULL = 0.15  # hands must pull apart by this much (on the dominant axis) to split
_MIN_SPLIT_CHILD_SIZE = 0.04  # don't split windows too small to halve cleanly


@dataclass
class _GrabState:
    window_id: int
    side: Side
    last_palm_x: float
    last_palm_y: float


@dataclass
class _CreateState:
    start_left: tuple[float, float]
    start_right: tuple[float, float]
    current_left: tuple[float, float]
    current_right: tuple[float, float]


@dataclass
class _SplitState:
    window_id: int
    start_left: tuple[float, float]
    start_right: tuple[float, float]
    current_left: tuple[float, float]
    current_right: tuple[float, float]


@dataclass
class _ResizeState:
    window_id: int
    side: Side
    # Bottom-left corner of the window at the moment resize started — kept fixed.
    anchor_x: float
    anchor_bottom_y: float  # y + height (image coords; larger y = lower on screen)


class JarvisController:
    def __init__(self, *, mirrored: bool = False) -> None:
        self.manager = WindowManager()
        self._mirrored = mirrored
        self._grab: _GrabState | None = None
        self._create: _CreateState | None = None
        self._resize: _ResizeState | None = None
        self._split: _SplitState | None = None
        self._hover_frames: dict[Side, int] = {"left": 0, "right": 0}
        self._last_hover_window: dict[Side, int | None] = {"left": None, "right": None}
        self._tap_cooldown_by_window: dict[int, float] = {}
        self._last_tap_id: int | None = None
        self._tap_set_at_frame: int = -1
        self._frame_count: int = 0
        self._events_out: list[tuple[str, int]] = []

    # ---- Events consumed by __main__ to emit OSC ----

    def pop_events(self) -> list[tuple[str, int]]:
        """Return events accumulated during update() and clear the buffer.

        Event tuples: (kind, window_id). Kinds:
        "created" | "tap" | "destroyed" | "split".
        """
        out = self._events_out
        self._events_out = []
        return out

    def last_tap_id(self) -> int | None:
        return self._last_tap_id

    # ---- Main update loop ----

    def update(self, frame: FrameResult, now: float | None = None) -> None:
        # Use a synthetic monotonic timestamp if caller didn't pass one.
        if now is None:
            now = 0.0

        self._frame_count += 1
        # Clear last_tap_id if it was set in a prior update (transient: one frame).
        if self._tap_set_at_frame != self._frame_count - 1:
            self._last_tap_id = None

        if self._handle_resize(frame, now):
            return  # resize takes the frame
        if self._handle_split(frame, now):
            return  # split takes the frame (both-hand fist pull-apart)

        self._handle_create(frame, now)
        self._handle_grab(frame, now)
        self._handle_tap(frame, now)

    def _handle_resize(self, frame: FrameResult, _now: float) -> bool:
        # Continuing resize?
        if self._resize is not None:
            side_state = getattr(frame, self._resize.side)
            if not is_pinching(side_state) or side_state.features is None:
                # Released — commit is already live (we've been updating every frame).
                self._resize = None
                return False
            f = side_state.features
            new_top = f.index_y
            max_top = self._resize.anchor_bottom_y - _MIN_RESIZE_SIZE
            if new_top > max_top:
                new_top = max_top
            new_height = max(self._resize.anchor_bottom_y - new_top, _MIN_RESIZE_SIZE)
            if self._mirrored:
                # Dragged corner is the raw TOP-LEFT; anchor is raw top-right.
                new_left = f.index_x
                max_left = self._resize.anchor_x - _MIN_RESIZE_SIZE
                if new_left > max_left:
                    new_left = max_left
                new_x = new_left
                new_width = max(self._resize.anchor_x - new_left, _MIN_RESIZE_SIZE)
            else:
                # Dragged corner is raw TOP-RIGHT; anchor is raw top-left.
                new_right = f.index_x
                min_right = self._resize.anchor_x + _MIN_RESIZE_SIZE
                if new_right < min_right:
                    new_right = min_right
                new_x = self._resize.anchor_x
                new_width = max(new_right - self._resize.anchor_x, _MIN_RESIZE_SIZE)
            w = self.manager.get(self._resize.window_id)
            if w is not None:
                # Direct replacement (can't use move — that's translation-only).
                self.manager._replace(
                    replace(w, x=new_x, y=new_top, width=new_width, height=new_height)
                )
            return True

        # Not resizing — check for entry trigger. Skip entirely if creating.
        if self._create is not None:
            return False

        for side in ("left", "right"):
            state = getattr(frame, side)
            if not is_pinching(state) or state.features is None:
                continue
            fx = state.features.index_x
            fy = state.features.index_y
            # Check topmost window whose displayed top-right corner is within
            # radius. When mirrored, the visible top-right corresponds to the
            # raw top-LEFT of the window (x) because features are in raw coords.
            best: tuple[int, float] | None = None  # (z, window_id)
            target_window = None
            for w in self.manager.windows():
                corner_x = w.x if self._mirrored else w.x + w.width
                corner_y = w.y
                dx = fx - corner_x
                dy = fy - corner_y
                if dx * dx + dy * dy < _RESIZE_CORNER_RADIUS**2 and (best is None or w.z > best[0]):
                    best = (w.z, w.id)
                    target_window = w
            if target_window is not None:
                self.manager.promote(target_window.id)
                # Anchor the OPPOSITE corner from the one being dragged.
                # Mirrored: user drags raw top-left, so the raw bottom-RIGHT stays put.
                # Non-mirrored: user drags raw top-right, raw bottom-LEFT stays put.
                if self._mirrored:
                    anchor_x = target_window.x + target_window.width
                else:
                    anchor_x = target_window.x
                self._resize = _ResizeState(
                    window_id=target_window.id,
                    side=side,  # type: ignore[arg-type]
                    anchor_x=anchor_x,
                    anchor_bottom_y=target_window.y + target_window.height,
                )
                return True

        return False

    def resizing_window_id(self) -> int | None:
        return self._resize.window_id if self._resize is not None else None

    # ---- 1. Create: both-hand pinch + pull apart ----

    def _handle_create(self, frame: FrameResult, _now: float) -> None:
        left = frame.left
        right = frame.right
        both_pinching = is_pinching(left) and is_pinching(right)

        if self._create is None:
            if both_pinching and left.features is not None and right.features is not None:
                dx = left.features.index_x - right.features.index_x
                dy = left.features.index_y - right.features.index_y
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < _CREATE_ENTRY_DISTANCE:
                    lp = (left.features.index_x, left.features.index_y)
                    rp = (right.features.index_x, right.features.index_y)
                    self._create = _CreateState(
                        start_left=lp,
                        start_right=rp,
                        current_left=lp,
                        current_right=rp,
                    )
            return

        # Already creating.
        if both_pinching and left.features is not None and right.features is not None:
            self._create = replace(
                self._create,
                current_left=(left.features.index_x, left.features.index_y),
                current_right=(right.features.index_x, right.features.index_y),
            )
            return

        # Released — commit with the last-tracked positions.
        x1, y1 = self._create.current_left
        x2, y2 = self._create.current_right
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        w = x_max - x_min
        h = y_max - y_min
        diag = (w * w + h * h) ** 0.5
        if diag >= _MIN_WINDOW_DIAGONAL:
            if w > 0 and h > 0:
                ar = w / h
                if ar < 0.5:
                    h = w / 0.5
                elif ar > 2.0:
                    h = w / 2.0
            created = self.manager.create(x=x_min, y=y_min, width=w, height=h)
            self._events_out.append(("created", created.id))
        self._create = None

    def pending_rect(self) -> tuple[float, float, float, float] | None:
        """Return (x, y, width, height) of the in-progress window preview, or None."""
        if self._create is None:
            return None
        x1, y1 = self._create.current_left
        x2, y2 = self._create.current_right
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return x_min, y_min, x_max - x_min, y_max - y_min

    # ---- 2. Grab/drag: open → fist while palm is over a window ----

    def _handle_grab(self, frame: FrameResult, _now: float) -> None:
        if self._grab is None:
            # Look for either hand transitioning to fist while over a window.
            for side in ("left", "right"):
                state: HandState = getattr(frame, side)
                if not state.present or state.features is None:
                    continue
                if state.gesture != "fist":
                    continue
                target = self.manager.topmost_at(state.features.x, state.features.y)
                if target is None:
                    continue
                self.manager.promote(target.id)
                self._grab = _GrabState(
                    window_id=target.id,
                    side=side,  # type: ignore[arg-type]
                    last_palm_x=state.features.x,
                    last_palm_y=state.features.y,
                )
                return
            return

        # Already grabbing: track motion, release on open.
        grab = self._grab
        state = getattr(frame, grab.side)
        if not state.present or state.features is None or state.gesture != "fist":
            # Released — if the window is near the bottom, destroy it (trash zone).
            if grab.last_palm_y > _DESTROY_Y_THRESHOLD:
                self.manager.destroy(grab.window_id)
                self._events_out.append(("destroyed", grab.window_id))
            self._grab = None
            return
        dx = state.features.x - grab.last_palm_x
        dy = state.features.y - grab.last_palm_y
        self.manager.move(grab.window_id, dx=dx, dy=dy)
        self._grab = replace(grab, last_palm_x=state.features.x, last_palm_y=state.features.y)

    def grabbed_window_id(self) -> int | None:
        return self._grab.window_id if self._grab is not None else None

    def grab_in_destroy_zone(self) -> bool:
        """True when the active grab's palm is in the bottom destroy strip."""
        return self._grab is not None and self._grab.last_palm_y > _DESTROY_Y_THRESHOLD

    # ---- Split: both hands fist on the same window, pull apart ----

    def _handle_split(self, frame: FrameResult, _now: float) -> bool:
        left = frame.left
        right = frame.right
        both_fist = (
            left.present
            and right.present
            and left.gesture == "fist"
            and right.gesture == "fist"
            and left.features is not None
            and right.features is not None
        )

        if self._split is not None:
            if not both_fist:
                # Released — try to commit.
                self._commit_split()
                self._split = None
                return True  # consumed the frame
            # Safe: both_fist already ensured features are not None.
            assert left.features is not None and right.features is not None
            self._split = replace(
                self._split,
                current_left=(left.features.x, left.features.y),
                current_right=(right.features.x, right.features.y),
            )
            return True

        # Not splitting — try to enter. Requires both hands fist over the SAME window.
        if not both_fist:
            return False
        assert left.features is not None and right.features is not None
        lw = self.manager.topmost_at(left.features.x, left.features.y)
        rw = self.manager.topmost_at(right.features.x, right.features.y)
        if lw is None or rw is None or lw.id != rw.id:
            return False
        # Cancel any lingering single-hand grab — split takes over.
        self._grab = None
        self.manager.promote(lw.id)
        lp = (left.features.x, left.features.y)
        rp = (right.features.x, right.features.y)
        self._split = _SplitState(
            window_id=lw.id,
            start_left=lp,
            start_right=rp,
            current_left=lp,
            current_right=rp,
        )
        return True

    def _commit_split(self) -> None:
        s = self._split
        if s is None:
            return
        w = self.manager.get(s.window_id)
        if w is None:
            return
        dx_start = abs(s.start_left[0] - s.start_right[0])
        dy_start = abs(s.start_left[1] - s.start_right[1])
        dx_end = abs(s.current_left[0] - s.current_right[0])
        dy_end = abs(s.current_left[1] - s.current_right[1])
        dx_growth = dx_end - dx_start
        dy_growth = dy_end - dy_start
        if max(dx_growth, dy_growth) < _SPLIT_MIN_PULL:
            return  # fumble — no split
        # Each half gets teleported so it's centered on the hand that "tore"
        # it away. The hand with smaller coordinate on the split axis owns the
        # "low" half (left for vertical, top for horizontal); the other hand
        # owns the "high" half.
        lx, ly = s.current_left
        rx, ry = s.current_right
        if dx_growth >= dy_growth:
            # Vertical split line → left + right halves.
            half_w = w.width / 2.0
            if half_w < _MIN_SPLIT_CHILD_SIZE:
                return
            if lx <= rx:
                low_hand, high_hand = (lx, ly), (rx, ry)
            else:
                low_hand, high_hand = (rx, ry), (lx, ly)
            a_x = _clamp(low_hand[0] - half_w / 2.0, 0.0, 1.0 - half_w)
            a_y = _clamp(low_hand[1] - w.height / 2.0, 0.0, 1.0 - w.height)
            b_x = _clamp(high_hand[0] - half_w / 2.0, 0.0, 1.0 - half_w)
            b_y = _clamp(high_hand[1] - w.height / 2.0, 0.0, 1.0 - w.height)
            a = self.manager.create(x=a_x, y=a_y, width=half_w, height=w.height)
            b = self.manager.create(x=b_x, y=b_y, width=half_w, height=w.height)
        else:
            # Horizontal split line → top + bottom halves.
            half_h = w.height / 2.0
            if half_h < _MIN_SPLIT_CHILD_SIZE:
                return
            if ly <= ry:
                low_hand, high_hand = (lx, ly), (rx, ry)
            else:
                low_hand, high_hand = (rx, ry), (lx, ly)
            a_x = _clamp(low_hand[0] - w.width / 2.0, 0.0, 1.0 - w.width)
            a_y = _clamp(low_hand[1] - half_h / 2.0, 0.0, 1.0 - half_h)
            b_x = _clamp(high_hand[0] - w.width / 2.0, 0.0, 1.0 - w.width)
            b_y = _clamp(high_hand[1] - half_h / 2.0, 0.0, 1.0 - half_h)
            a = self.manager.create(x=a_x, y=a_y, width=w.width, height=half_h)
            b = self.manager.create(x=b_x, y=b_y, width=w.width, height=half_h)
        self.manager.destroy(w.id)
        self._events_out.append(("split", w.id))
        self._events_out.append(("created", a.id))
        self._events_out.append(("created", b.id))

    def split_preview(self) -> tuple[Window, str] | None:
        """Return (window, axis) where axis is 'vertical' or 'horizontal' for the
        prospective split line, or None if not splitting."""
        s = self._split
        if s is None:
            return None
        w = self.manager.get(s.window_id)
        if w is None:
            return None
        dx_growth = abs(s.current_left[0] - s.current_right[0]) - abs(
            s.start_left[0] - s.start_right[0]
        )
        dy_growth = abs(s.current_left[1] - s.current_right[1]) - abs(
            s.start_left[1] - s.start_right[1]
        )
        axis = "vertical" if dx_growth >= dy_growth else "horizontal"
        return w, axis

    # ---- 3. Tap: point + hover on window for _TAP_HOVER_FRAMES frames ----

    def _handle_tap(self, frame: FrameResult, now: float) -> None:
        for side in ("left", "right"):
            state: HandState = getattr(frame, side)
            if not state.present or state.features is None or state.gesture != "point":
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            target = self.manager.topmost_at(state.features.index_x, state.features.index_y)
            if target is None:
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            if self._last_hover_window[side] == target.id:  # type: ignore[index]
                self._hover_frames[side] += 1  # type: ignore[index]
            else:
                self._hover_frames[side] = 1  # type: ignore[index]
                self._last_hover_window[side] = target.id  # type: ignore[index]

            if self._hover_frames[side] >= _TAP_HOVER_FRAMES:  # type: ignore[index]
                last_tap = self._tap_cooldown_by_window.get(target.id, -1.0)
                if now - last_tap >= _TAP_COOLDOWN_SECONDS:
                    self.manager.cycle_color(target.id)
                    self.manager.promote(target.id)
                    self._tap_cooldown_by_window[target.id] = now
                    self._events_out.append(("tap", target.id))
                    self._last_tap_id = target.id
                    self._tap_set_at_frame = self._frame_count
                # Reset hover counter so a continuous hover doesn't retap until cooldown.
                self._hover_frames[side] = 0  # type: ignore[index]
