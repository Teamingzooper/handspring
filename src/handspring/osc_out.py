"""Emit per-frame tracking results as OSC messages over UDP."""

from __future__ import annotations

from typing import Any, Protocol

from handspring.types import FaceState, FrameResult, Gesture, HandState, Side


class _SendsOsc(Protocol):
    def send_message(self, address: str, value: Any) -> None: ...


def _make_client(host: str, port: int) -> _SendsOsc:
    # Import inside the function so tests can construct OscEmitter with a fake
    # client without importing python-osc.
    from pythonosc.udp_client import SimpleUDPClient

    return SimpleUDPClient(host, port)


class OscEmitter:
    """Thin stateful wrapper that emits continuous features every frame and
    discrete gesture events only on change."""

    def __init__(
        self,
        *,
        client: _SendsOsc | None = None,
        host: str = "127.0.0.1",
        port: int = 9000,
    ) -> None:
        self._client: _SendsOsc = client if client is not None else _make_client(host, port)
        self._last_gesture: dict[Side, Gesture | None] = {"left": None, "right": None}

    def emit(self, frame: FrameResult) -> None:
        self._emit_hand("left", frame.left)
        self._emit_hand("right", frame.right)
        self._emit_face(frame.face)

    def _emit_hand(self, side: Side, state: HandState) -> None:
        self._client.send_message(f"/hand/{side}/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message(f"/hand/{side}/x", float(f.x))
            self._client.send_message(f"/hand/{side}/y", float(f.y))
            self._client.send_message(f"/hand/{side}/z", float(f.z))
            self._client.send_message(f"/hand/{side}/openness", float(f.openness))
            self._client.send_message(f"/hand/{side}/pinch", float(f.pinch))

        # Gesture events: emit only on state change.
        current: Gesture = state.gesture
        if current != self._last_gesture[side]:
            self._client.send_message(f"/hand/{side}/gesture", current)
            self._last_gesture[side] = current

    def _emit_face(self, state: FaceState) -> None:
        self._client.send_message("/face/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message("/face/yaw", float(f.yaw))
            self._client.send_message("/face/pitch", float(f.pitch))
            self._client.send_message("/face/mouth_open", float(f.mouth_open))
