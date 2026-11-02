from typing import Dict, Optional, Tuple

import numpy as np

from huri_think.body import (
    Body,
    quat_from_rotvec,
    rotvec_from_quat,
    slerp,
    thinking_arms,
)

N_JOINTS = 55
POSE_DIM = N_JOINTS * 3

SPINE1, SPINE2, SPINE3, NECK = 3, 6, 9, 12
L_COLLAR, R_COLLAR, HEAD = 13, 14, 15
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 16, 17, 18, 19
L_WRIST, R_WRIST, L_EYE, R_EYE = 20, 21, 23, 24
L_INDEX1, R_INDEX1 = 25, 40
L_FINGERS, R_FINGERS = tuple(range(25, 37)), tuple(range(40, 52))

REST_DEG = np.zeros((N_JOINTS, 3))
REST_DEG[L_SHOULDER, 2] = -80.0
REST_DEG[R_SHOULDER, 2] = 80.0
REST_POSE = np.radians(REST_DEG).reshape(POSE_DIM)

EYE_JOINTS = (L_EYE, R_EYE)

_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19)]
_PAIRS += [(20, 21), (23, 24)] + [(25 + k, 40 + k) for k in range(15)]
MIRROR = np.arange(N_JOINTS)
for _l, _r in _PAIRS:
    MIRROR[_l], MIRROR[_r] = _r, _l


def mirrored(deg: np.ndarray) -> np.ndarray:
    out = deg[MIRROR].copy()
    out[:, 1:] *= -1.0
    return out


POSES = ("chin", "chin_hip", "crossed")
GESTURE_NAMES = POSES + tuple(name + "_left" for name in POSES)
_GESTURE_SIDE = np.array([-1.0 if n.endswith("_left") else 1.0 for n in GESTURE_NAMES])
ARM_JOINTS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
_CURL = {"chin": 55.0, "chin_hip": 55.0, "crossed": 35.0}
_WRIST = {"chin": 15.0, "chin_hip": 15.0, "crossed": 0.0}
_RELAXED = 20.0


def solve_poses(body: Body) -> Tuple[np.ndarray, np.ndarray]:
    quats = np.zeros((len(GESTURE_NAMES), len(ARM_JOINTS), 4))
    extra = np.zeros((len(GESTURE_NAMES), N_JOINTS, 3))
    for g, name in enumerate(GESTURE_NAMES):
        pose = name.removesuffix("_left")
        active = 1.0 if name.endswith("_left") else -1.0  # SMPL-X: +1 is left
        arms = thinking_arms(body, active)[pose]
        wrist = np.zeros((2, 3))
        wrist[0 if active > 0 else 1, 2] = -active * _WRIST[pose]
        rotvecs = [
            arms[1.0].shoulder,
            arms[-1.0].shoulder,
            arms[1.0].elbow,
            arms[-1.0].elbow,
            np.radians(wrist[0]),
            np.radians(wrist[1]),
        ]
        quats[g] = [quat_from_rotvec(v) for v in rotvecs]
        own, other = (L_FINGERS, R_FINGERS) if active > 0 else (R_FINGERS, L_FINGERS)
        extra[g, list(own), 2] = -active * _CURL[pose]
        extra[g, list(other), 2] = active * _RELAXED
    return quats, extra


_REST_QUATS = np.array([quat_from_rotvec(np.radians(REST_DEG[j])) for j in ARM_JOINTS])
_SOLVED: Dict[Body, Tuple[np.ndarray, np.ndarray]] = {}


def _solved(body: Body) -> Tuple[np.ndarray, np.ndarray]:
    if body not in _SOLVED:
        _SOLVED[body] = solve_poses(body)
    return _SOLVED[body]


ONSET_S = 0.5
ARMS_S = (1.2, 1.6)
SACCADE_S = 0.08
HEAD_SPEED = 60.0
MAX_EYE_DEG = 20.0
FIDGETS = ("nod", "rub", "tap")


def min_jerk(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


class _Channel:
    def __init__(self, size: int):
        self.src = np.zeros(size)
        self.dst = np.zeros(size)
        self.t0 = 0.0
        self.dur = 0.0

    def value(self, t: float) -> np.ndarray:
        if self.dur <= 0.0 or t >= self.t0 + self.dur:
            return self.dst
        return self.src + (self.dst - self.src) * min_jerk((t - self.t0) / self.dur)

    def go(self, t: float, dst, dur: float) -> None:
        self.src = self.value(t).copy()
        self.dst = np.asarray(dst, dtype=float)
        self.t0 = t
        self.dur = dur

    def arrived(self, t: float) -> bool:
        return t >= self.t0 + self.dur


class _Wobble:
    def __init__(self, rng: np.random.Generator, size: int, amp: float):
        self.freq = rng.uniform(0.15, 0.8, (3, size))
        self.phase = rng.uniform(0.0, 2 * np.pi, (3, size))
        self.amp = amp / np.sqrt(3)

    def value(self, t: float) -> np.ndarray:
        return self.amp * np.sin(2 * np.pi * self.freq * t + self.phase).sum(axis=0)


class ThinkingMotion:
    def __init__(
        self,
        fps: int = 30,
        intensity: float = 1.0,
        arms: bool = True,
        rng: Optional[np.random.Generator] = None,
        body: Optional[Body] = None,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.intensity = float(np.clip(intensity, 0.0, 2.0))
        self.arms_on = arms
        self.rng = rng if rng is not None else np.random.default_rng()
        self.body = body if body is not None else Body()
        self._quats, self._extra = _solved(self.body)

        self.head = _Channel(3)
        self.gaze = _Channel(2)
        self.arms = _Channel(len(GESTURE_NAMES))
        self.wobble = _Wobble(self.rng, 3, 1.6 * self.intensity)
        self.sway = _Wobble(self.rng, 2, 1.2 * self.intensity)
        self.breath_hz = self.rng.uniform(0.2, 0.28)
        self.breath_deg = 2.0 * self.intensity

        self._i = 0
        self._glances = 0
        self._side = self.rng.choice((-1.0, 1.0))
        self._next_glance = self.rng.uniform(0.0, 0.15)
        self._gesture: Optional[int] = None
        self._next_arms = self.rng.uniform(0.4, 1.2)
        self._fidget: Optional[Tuple[str, float, float]] = None
        self._next_fidget = self.rng.uniform(1.5, 3.0)
        self._settle_t0: Optional[float] = None
        self._settle_dur = 0.0

    @property
    def t(self) -> float:
        return self._i / self.fps

    def next(self, n: int) -> np.ndarray:
        return self._frames(n, plan=True)

    def settle(self, seconds: float = 0.6) -> np.ndarray:
        raised = float(self.arms.value(self.t).sum())
        seconds = max(seconds, ARMS_S[1] * raised, 1 / self.fps)
        n = max(1, int(round(seconds * self.fps)))
        dur = n / self.fps
        last = (self._i - 1) / self.fps
        for channel in (self.head, self.gaze, self.arms):
            channel.go(last, np.zeros_like(channel.dst), dur)
        self._settle_t0, self._settle_dur = last, dur
        return self._frames(n, plan=False)

    def _frames(self, n: int, plan: bool) -> np.ndarray:
        out = np.empty((n, POSE_DIM), dtype=np.float32)
        for k in range(n):
            t = self.t
            if plan:
                self._plan(t)
            out[k] = self._pose(t)
            self._i += 1
        return out

    def _plan(self, t: float) -> None:
        if self.arms_on and t >= self._next_arms:
            self._plan_arms(t)
        if t >= self._next_fidget:
            self._plan_fidget(t)
        if t >= self._next_glance:
            self._plan_glance(t)

    def _plan_arms(self, t: float) -> None:
        rng = self.rng
        if self._gesture is not None and rng.random() < 0.5:
            self._gesture = None
            self.arms.go(t, np.zeros(len(GESTURE_NAMES)), rng.uniform(*ARMS_S))
            self._next_arms = t + rng.uniform(*ARMS_S) + rng.uniform(1.0, 3.0)
            return
        others = [k for k in range(len(GESTURE_NAMES)) if k != self._gesture]
        if self._gesture is None and rng.random() < 0.15:
            self._next_arms = t + rng.uniform(1.0, 2.5)
            return
        self._gesture = int(rng.choice(others))
        self.arms.go(t, np.eye(len(GESTURE_NAMES))[self._gesture], rng.uniform(*ARMS_S))
        self._next_arms = t + rng.uniform(*ARMS_S) + rng.uniform(2.5, 5.0)

    def _plan_fidget(self, t: float) -> None:
        rng = self.rng
        posed = self.arms.value(t).sum() > 0.8 and self.arms.arrived(t)
        kind = str(rng.choice(FIDGETS if posed else FIDGETS[:1]))
        dur = rng.uniform(1.0, 1.8)
        self._fidget = (kind, t, dur)
        self._next_fidget = t + dur + rng.uniform(1.0, 3.0)

    def _plan_glance(self, t: float) -> None:
        rng, first = self.rng, self._glances == 0
        roll = rng.random()
        if not first and roll < 0.6:
            self._side = -self._side
        if roll < 0.55:
            pitch = -rng.uniform(10.0, 20.0)
        elif roll < 0.8:
            pitch = rng.uniform(5.0, 12.0)
        else:
            pitch = rng.uniform(-3.0, 3.0)
        yaw = self._side * rng.uniform(15.0, 35.0)
        tilt = self._side * rng.uniform(3.0, 9.0)
        if not first and rng.random() < 0.15:
            pitch, yaw, tilt = 0.3 * pitch, 0.2 * yaw, 0.3 * tilt
        if self._gesture is not None:
            pitch += 4.0
        target = self.intensity * np.array([pitch, yaw, tilt])
        distance = np.abs(target - self.head.value(t)).max()
        move = max(rng.uniform(0.35, 0.7), distance / HEAD_SPEED)
        hold = rng.uniform(0.7, 2.0)

        self.head.go(t, target, move)
        self.gaze.go(t, self.intensity * 1.4 * np.array([pitch, yaw]), SACCADE_S)

        self._glances += 1
        self._next_glance = t + move + hold

    def _envelope(self, t: float) -> float:
        env = float(min_jerk(t / ONSET_S))
        if self._settle_t0 is not None:
            env *= 1.0 - float(min_jerk((t - self._settle_t0) / self._settle_dur))
        return env

    def _add_fidget(
        self, deg: np.ndarray, t: float, env: float, weights: np.ndarray
    ) -> None:
        if self._fidget is None:
            return
        kind, t0, dur = self._fidget
        if not t0 <= t < t0 + dur:
            return
        u = (t - t0) / dur
        amp = env * self.intensity * np.sin(np.pi * u)
        if kind == "nod":
            deg[HEAD, 0] += 5.0 * amp * np.sin(2 * np.pi * 1.8 * (t - t0))
            return
        right = float(weights[_GESTURE_SIDE > 0].sum())
        left = float(weights[_GESTURE_SIDE < 0].sum())
        if kind == "rub":
            wave = 6.0 * amp * np.sin(2 * np.pi * 1.6 * (t - t0))
            deg[R_ELBOW, 1] += right * wave
            deg[L_ELBOW, 1] -= left * wave
        else:
            wave = 15.0 * amp * (0.5 - 0.5 * np.cos(2 * np.pi * 2.5 * (t - t0)))
            deg[R_INDEX1, 2] += right * wave
            deg[L_INDEX1, 2] -= left * wave

    def _pose(self, t: float) -> np.ndarray:
        env = self._envelope(t)
        deg = REST_DEG.copy()

        pitch, yaw, tilt = self.head.value(t) + env * self.wobble.value(t)
        deg[NECK] += 0.4 * np.array([pitch, yaw, tilt])
        deg[HEAD] += 0.6 * np.array([pitch, yaw, tilt])
        deg[SPINE3, 1] += 0.25 * yaw
        deg[SPINE2, 1] += 0.15 * yaw
        deg[SPINE3, 0] += 0.3 * pitch
        sway = env * self.sway.value(t)
        deg[SPINE1, 1] += 0.1 * yaw + sway[0]
        deg[SPINE1, 2] += sway[1]

        breath = env * self.breath_deg * np.sin(2 * np.pi * self.breath_hz * t)
        deg[SPINE3, 0] += breath
        deg[L_COLLAR, 2] -= 0.5 * breath
        deg[R_COLLAR, 2] += 0.5 * breath

        eyes = np.clip(
            self.gaze.value(t) - np.array([pitch, yaw]), -MAX_EYE_DEG, MAX_EYE_DEG
        )
        for eye in EYE_JOINTS:
            deg[eye, :2] += eyes

        weights = self.arms.value(t)
        total = float(weights.sum())
        if total > 1e-6:
            # Each arm joint turns from rest towards the poses it is going to
            # (slerp: the hand travels in front of the body, not through it).
            mix = np.tensordot(weights, self._quats, axes=1)
            mix /= np.linalg.norm(mix, axis=1, keepdims=True)
            for k, joint in enumerate(ARM_JOINTS):
                q = slerp(_REST_QUATS[k], mix[k], min(1.0, total))
                deg[joint] = np.degrees(rotvec_from_quat(q))
            deg += np.tensordot(weights, self._extra, axes=1)
        self._add_fidget(deg, t, env, weights)

        return np.radians(deg).reshape(POSE_DIM)
