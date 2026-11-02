from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_SIZES = {
    "shoulder": 3,
    "upper_arm": 1,
    "forearm": 1,
    "hand": 1,
    "chin": 2,
    "chest": 3,
    "waist": 3,
    "hip": 2,
}
_MAX_M = 3.0


@dataclass(frozen=True)
class Body:
    shoulder: Tuple[float, float, float] = (0.208, 0.529, -0.028)
    upper_arm: float = 0.282
    forearm: float = 0.267
    hand: float = 0.18
    chin: Tuple[float, float] = (0.711, 0.126)
    chest: Tuple[float, float, float] = (0.382, 0.134, 0.184)
    waist: Tuple[float, float, float] = (0.173, 0.14, 0.165)
    hip: Tuple[float, float] = (0.074, 0.15)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "Body":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError(f"body must be a mapping, got {type(values).__name__}")
        unknown = sorted(set(values) - set(_SIZES))
        if unknown:
            raise ValueError(f"body has no {', '.join(unknown)} (has {list(_SIZES)})")
        clean: Dict[str, Any] = {}
        for name, value in values.items():
            size = _SIZES[name]
            numbers = [value] if size == 1 else value
            if (
                isinstance(numbers, (str, bytes))
                or not isinstance(numbers, Sequence)
                or len(numbers) != size
            ):
                raise ValueError(f"body.{name} must be {size} number(s), got {value!r}")
            try:
                floats = [float(x) for x in numbers]
            except (TypeError, ValueError):
                raise ValueError(f"body.{name} must be numbers, got {value!r}")
            if not all(np.isfinite(x) and abs(x) <= _MAX_M for x in floats):
                raise ValueError(f"body.{name} must be metres, got {value!r}")
            if size == 1 and floats[0] <= 0:
                raise ValueError(f"body.{name} must be > 0, got {value!r}")
            clean[name] = floats[0] if size == 1 else tuple(floats)
        return cls(**clean)

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def radius(self) -> float:
        return 0.16 * self.upper_arm

    def shoulder_at(self, side: float) -> np.ndarray:
        x, y, z = self.shoulder
        return np.array([side * x, y, z])

    def front(self, y: float) -> Tuple[float, float]:
        def lerp(y0: float, y1: float, v0: float, v1: float) -> float:
            f = float(np.clip((y - y0) / (y1 - y0), 0.0, 1.0)) if y1 != y0 else 1.0
            return v0 + (v1 - v0) * f

        (hip_y, hip_w), (y0, z0, w0), (y1, z1, w1) = self.hip, self.waist, self.chest
        if y < y0:
            return z0, lerp(hip_y, y0, hip_w, w0)
        if y < y1:
            return lerp(y0, y1, z0, z1), lerp(y0, y1, w0, w1)
        return lerp(y1, self.shoulder[1], z1, self.shoulder[2]), w1

    def depth(self, point: np.ndarray, margin: float) -> float:
        x, y, z = point
        if not self.hip[0] - margin < y < self.shoulder[1]:
            return 0.0
        front, half = self.front(y)
        if abs(x) >= half + margin:
            return 0.0
        side_z = self.shoulder[2]
        bulge = (front - side_z) * np.sqrt(max(0.0, 1.0 - (x / (half + margin)) ** 2))
        return max(0.0, min(side_z + bulge + margin - z, half + margin - abs(x)))

    def inside(self, point: np.ndarray, margin: float) -> bool:
        return self.depth(point, margin) > 0.0


# --- Rotations (quaternions are w, x, y, z) ---------------------------------


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * v / angle])


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    q = q if q[0] >= 0 else -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(s, q[0]) * q[1:] / s


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), q * [1, -1, -1, -1])[1:]


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + (b - a) * t
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# --- Arms --------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPose:
    shoulder: np.ndarray
    elbow: np.ndarray
    elbow_at: np.ndarray
    wrist_at: np.ndarray


def arm_points(
    body: Body, side: float, shoulder: np.ndarray, elbow: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    rest = np.array([side, 0.0, 0.0])
    q_shoulder = quat_from_rotvec(shoulder)
    s = body.shoulder_at(side)
    e = s + body.upper_arm * quat_rotate(q_shoulder, rest)
    q_elbow = quat_mul(q_shoulder, quat_from_rotvec(elbow))
    w = e + body.forearm * quat_rotate(q_elbow, rest)
    return e, w


def _elbow(
    body: Body, s: np.ndarray, wrist: np.ndarray, pole: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    a, b = body.upper_arm, body.forearm
    to_wrist = wrist - s
    d = float(np.clip(np.linalg.norm(to_wrist), abs(a - b) + 1e-3, (a + b) * 0.999))
    u = _unit(to_wrist)
    cos_a = (a * a + d * d - b * b) / (2 * a * d)
    sin_a = np.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    v0 = pole - np.dot(pole, u) * u
    if np.linalg.norm(v0) < 1e-6:
        v0 = np.cross(u, [0.0, 0.0, 1.0])
    v0 = _unit(v0)
    w_axis = np.cross(u, v0)
    w = s + d * u
    best, least = s, np.inf
    for deg in [0] + [k * sign for k in range(10, 190, 10) for sign in (1, -1)]:
        turn = np.radians(deg)
        e = s + a * (cos_a * u + sin_a * (np.cos(turn) * v0 + np.sin(turn) * w_axis))
        points = [s + (e - s) * 0.5, e, e + (w - e) * 0.5, w]
        depth = sum(body.depth(p, 0.8 * body.radius) for p in points)
        if depth < least:
            best, least = e, depth
        if depth == 0:
            break
    return best, w, least


def reach(
    body: Body, side: float, wrist: Sequence[float], pole: Sequence[float]
) -> ArmPose:
    s = body.shoulder_at(side)
    target = np.asarray(wrist, dtype=float)
    pole = np.asarray(pole, dtype=float)
    for step in range(8):
        elbow, w, depth = _elbow(body, s, target + [0.0, 0.0, step * body.radius], pole)
        if depth == 0:
            break

    e1 = _unit(elbow - s)
    e2 = _unit(w - elbow)
    bend = float(np.arccos(np.clip(np.dot(e1, e2), -1.0, 1.0)))
    p = e2 - np.dot(e2, e1) * e1
    p = _unit(p) if np.linalg.norm(p) > 1e-6 else _unit(np.cross(e1, [0.0, 1.0, 0.0]))
    x_axis = side * e1
    z_axis = p
    y_axis = np.cross(z_axis, x_axis)
    q = quat_from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    return ArmPose(
        shoulder=rotvec_from_quat(q),
        elbow=np.array([0.0, -side * bend, 0.0]),
        elbow_at=elbow,
        wrist_at=w,
    )


def thinking_arms(body: Body, active: float) -> Dict[str, Dict[float, ArmPose]]:
    h, r = body.hand, body.radius
    a, o = active, -active
    chin_y, chin_z = body.chin

    chin_wrist = (a * 0.1 * h, chin_y - 0.5 * h, chin_z + 0.3 * h)
    chin = reach(body, a, chin_wrist, (o * 0.25, -1.0, 0.6))

    e = chin.elbow_at
    under_y = e[1] - 1.5 * r
    under = (e[0] + o * 0.5 * h, under_y, max(e[2], body.front(under_y)[0] + r))
    support = reach(body, o, under, (o * 0.6, -1.0, 0.1))

    hip_y = body.hip[0] + 0.4 * h
    on_hip = reach(
        body,
        o,
        (o * (body.front(hip_y)[1] + 0.5 * r), hip_y, -0.2 * r),
        (o, -0.2, -0.7),
    )

    cross_y = (body.chest[0] + body.waist[0]) / 2
    front, half = body.front(cross_y)
    grip = half + r - 0.6 * h
    bottom = reach(
        body, o, (a * grip, cross_y - 0.6 * r, front + r), (o * 0.3, -1.0, 0.4)
    )
    top = reach(
        body, a, (o * grip, cross_y + 0.6 * r, front + 2.4 * r), (a * 0.3, -1.0, 0.4)
    )

    return {
        "chin": {a: chin, o: support},
        "chin_hip": {a: chin, o: on_hip},
        "crossed": {a: top, o: bottom},
    }
