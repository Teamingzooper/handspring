"""Emit per-frame tracking results as OSC messages over UDP."""

from __future__ import annotations

from typing import Any, Protocol

from handspring.synth_params import SynthSnapshot
from handspring.types import (
    Expression,
    FaceState,
    FrameResult,
    Gesture,
    HandState,
    PoseState,
    Side,
)


class _SendsOsc(Protocol):
    def send_message(self, address: str, value: Any) -> None: ...


def _make_client(host: str, port: int) -> _SendsOsc:
    # Import inside the function so tests can construct OscEmitter with a fake
    # client without importing python-osc.
    from pythonosc.udp_client import SimpleUDPClient

    return SimpleUDPClient(host, port)


class OscEmitter:
    """Thin stateful wrapper that emits continuous features every frame and
    discrete gesture/expression events only on change."""

    def __init__(
        self,
        *,
        client: _SendsOsc | None = None,
        host: str = "127.0.0.1",
        port: int = 9000,
    ) -> None:
        self._client: _SendsOsc = client if client is not None else _make_client(host, port)
        self._last_gesture: dict[Side, Gesture] = {"left": "none", "right": "none"}
        self._last_expression: Expression = "neutral"
        self._last_synth_mode: str | None = None

    def emit(self, frame: FrameResult) -> None:
        self._emit_hand("left", frame.left)
        self._emit_hand("right", frame.right)
        self._emit_face(frame.face)
        self._emit_pose(frame.pose)
        if frame.clap_event:
            self._client.send_message("/motion/clap", 1)

    def _emit_hand(self, side: Side, state: HandState) -> None:
        self._client.send_message(f"/hand/{side}/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message(f"/hand/{side}/x", float(f.x))
            self._client.send_message(f"/hand/{side}/y", float(f.y))
            self._client.send_message(f"/hand/{side}/z", float(f.z))
            self._client.send_message(f"/hand/{side}/openness", float(f.openness))
            self._client.send_message(f"/hand/{side}/pinch", float(f.pinch))

        # Motion continuous state (always emitted).
        m = state.motion
        self._client.send_message(f"/hand/{side}/pinching", 1 if m.pinching else 0)
        self._client.send_message(f"/hand/{side}/dragging", 1 if m.dragging else 0)
        if m.dragging:
            self._client.send_message(f"/hand/{side}/drag_dx", float(m.drag_dx))
            self._client.send_message(f"/hand/{side}/drag_dy", float(m.drag_dy))
        if m.event is not None:
            self._client.send_message(f"/hand/{side}/event", m.event)

        # Static gesture (state-change only).
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

        # Eye openness (continuous; emits 0.0 when face absent, which matches default).
        self._client.send_message("/face/eye_left_open", float(state.eye_left_open))
        self._client.send_message("/face/eye_right_open", float(state.eye_right_open))

        # Expression (state-change only).
        if state.expression != self._last_expression:
            self._client.send_message("/face/expression", state.expression)
            self._last_expression = state.expression

    def _emit_pose(self, state: PoseState) -> None:
        self._client.send_message("/pose/present", 1 if state.present else 0)
        if not state.present or state.joints is None:
            return
        for joint_name, lm in state.joints.items():
            self._client.send_message(f"/pose/{joint_name}/visible", 1 if lm.visible else 0)
            if not lm.visible:
                continue
            self._client.send_message(f"/pose/{joint_name}/x", float(lm.x))
            self._client.send_message(f"/pose/{joint_name}/y", float(lm.y))
            self._client.send_message(f"/pose/{joint_name}/z", float(lm.z))

    def emit_synth(self, snap: SynthSnapshot) -> None:
        self._client.send_message("/synth/volume", float(snap.volume))
        self._client.send_message("/synth/note_hz", float(snap.note_hz))
        self._client.send_message("/synth/stepping_hz", float(snap.stepping_hz))
        self._client.send_message("/synth/cutoff_hz", float(snap.cutoff_hz))
        self._client.send_message("/synth/mod_depth", float(snap.mod_depth))
        self._client.send_message("/synth/mod_rate", float(snap.mod_rate))
        if snap.mode != self._last_synth_mode:
            self._client.send_message("/synth/mode", snap.mode)
            self._last_synth_mode = snap.mode
