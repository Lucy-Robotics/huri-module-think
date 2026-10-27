"""Thinking body language, generated frame by frame (no recorded clip).

People who think look away from whoever they talk to: up and to a side,
sometimes down, in glances of one to two seconds, the eyes leading the head.
Most also take a thinking pose: a hand at the chin, the other arm across the
belly or on the hip, or the arms crossed, and fidget in it: they rub their
chin, tap a finger, nod to themselves. They keep breathing and never hold
perfectly still. `ThinkingMotion` produces that as SMPL-X poses (55 joints x
3 axis-angle, radians: the layout of `huri_events.motion.Motion`), from a
random generator: every thinking episode is new.

SMPL-X axes (zero pose, the T-pose): y up, x towards the body's left, z
forward. About x is a pitch (positive: looking down), about y a yaw (positive:
towards the body's left), about z a roll. The neutral pose has the arms
lowered along the body (shoulder z -/+80 degrees), the pose HuRI-inmoov maps
to the robot's rest.
"""

from typing import Dict, Optional, Tuple

import numpy as np

N_JOINTS = 55
POSE_DIM = N_JOINTS * 3

# SMPL-X joint indices.
SPINE1, SPINE2, SPINE3, NECK = 3, 6, 9, 12
L_COLLAR, R_COLLAR, HEAD = 13, 14, 15
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 16, 17, 18, 19
L_WRIST, R_WRIST, L_EYE, R_EYE = 20, 21, 23, 24
L_INDEX1, R_INDEX1 = 25, 40
# Index, middle, pinky and ring, three joints each (the thumbs follow them).
L_FINGERS, R_FINGERS = tuple(range(25, 37)), tuple(range(40, 52))

# Degrees. Arms lowered along the body; everything else at zero.
REST_DEG = np.zeros((N_JOINTS, 3))
REST_DEG[L_SHOULDER, 2] = -80.0
REST_DEG[R_SHOULDER, 2] = 80.0
REST_POSE = np.radians(REST_DEG).reshape(POSE_DIM)

# Joints that only move the eyes: they saccade, much faster than the body.
EYE_JOINTS = (L_EYE, R_EYE)

# Each joint and its mirror image across the body's mid-plane.
_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19)]
_PAIRS += [(20, 21), (23, 24)] + [(25 + k, 40 + k) for k in range(15)]
MIRROR = np.arange(N_JOINTS)
for _l, _r in _PAIRS:
    MIRROR[_l], MIRROR[_r] = _r, _l


def mirrored(deg: np.ndarray) -> np.ndarray:
    """The same pose, left and right swapped (a reflection: y and z flip)."""

    out = deg[MIRROR].copy()
    out[:, 1:] *= -1.0
    return out


def _pose_deg(angles: Dict[int, Tuple[float, float, float]], curl: float) -> np.ndarray:
    """Degrees added to the rest pose; `curl` bends the right hand's fingers."""

    deg = np.zeros((N_JOINTS, 3))
    for joint, xyz in angles.items():
        deg[joint] += xyz
    deg[list(R_FINGERS), 2] += curl  # right fingers curl about +z
    return deg


# The thinking poses, degrees added to the rest pose, the right hand active
# (at the chin, or on top). Solved by forward kinematics on the neutral SMPL-X
# skeleton, so the hands land where they should and pass in front of the body
# on the way. Each also exists mirrored, the left hand active.
_CHIN = {R_SHOULDER: (14, 51, -7), R_ELBOW: (0, 130, 0), R_WRIST: (0, 0, 15)}
GESTURES = {
    # Right hand at the chin, left forearm across the belly under its elbow.
    "chin": _pose_deg({**_CHIN, L_SHOULDER: (40, -71, 6), L_ELBOW: (0, -73, 0)}, 55),
    # Right hand at the chin, left hand on the hip.
    "chin_hip": _pose_deg(
        {**_CHIN, L_SHOULDER: (54, -19, 17), L_ELBOW: (0, -68, 0)}, 55
    ),
    # Arms crossed, the right forearm on top.
    "crossed": _pose_deg(
        {
            R_SHOULDER: (32, 72, -15),
            R_ELBOW: (0, 92, 0),
            L_SHOULDER: (30, -70, 14),
            L_ELBOW: (0, -84, 0),
        },
        35,
    ),
}
# The left hand of the right-handed poses relaxes too.
for _deg in GESTURES.values():
    _deg[list(L_FINGERS), 2] -= 20.0
GESTURES.update({name + "_left": mirrored(deg) for name, deg in list(GESTURES.items())})
GESTURE_NAMES = tuple(GESTURES)
_GESTURE_DEG = np.stack([GESTURES[name] for name in GESTURE_NAMES])
# +1 when the right hand is the active one, -1 the left.
_GESTURE_SIDE = np.array([-1.0 if n.endswith("_left") else 1.0 for n in GESTURE_NAMES])

ONSET_S = 0.5  # micro-motion and breathing fade in over this
ARMS_S = (1.2, 1.6)  # time to bring the arms into a pose, or back down
SACCADE_S = 0.08
HEAD_SPEED = 60.0  # deg/s, average over a glance (its peak is 1.9 times that)
MAX_EYE_DEG = 20.0
FIDGETS = ("nod", "rub", "tap")


def min_jerk(x: np.ndarray | float) -> np.ndarray | float:
    """0 -> 1 with zero velocity and acceleration at both ends."""

    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


class _Channel:
    """A value moving to its targets along minimum-jerk transitions."""

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
    """Smooth aperiodic noise: a few slow sines of random frequency and phase."""

    def __init__(self, rng: np.random.Generator, size: int, amp: float):
        self.freq = rng.uniform(0.15, 0.8, (3, size))
        self.phase = rng.uniform(0.0, 2 * np.pi, (3, size))
        self.amp = amp / np.sqrt(3)

    def value(self, t: float) -> np.ndarray:
        return self.amp * np.sin(2 * np.pi * self.freq * t + self.phase).sum(axis=0)


class ThinkingMotion:
    """Streams thinking frames: `next(n)` while thinking, `settle(s)` to end.

    :fps: frames per second of the poses.
    :intensity: scales every angle (0: still, 1: natural, 2: theatrical).
    :arms: allow the thinking poses (hand at the chin, arms crossed...).
    :rng: the source of every choice (seed it for a reproducible episode).

    Frame 0 is the neutral pose and `settle()` ends on it exactly, so the
    motion starts and ends where speech gestures and the robot's rest are.
    """

    def __init__(
        self,
        fps: int = 30,
        intensity: float = 1.0,
        arms: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.intensity = float(np.clip(intensity, 0.0, 2.0))
        self.arms_on = arms
        self.rng = rng if rng is not None else np.random.default_rng()

        self.head = _Channel(3)  # pitch, yaw, roll of the head (degrees)
        self.gaze = _Channel(2)  # pitch, yaw where the eyes look (degrees)
        self.arms = _Channel(len(GESTURES))  # how far into each thinking pose
        self.wobble = _Wobble(self.rng, 3, 1.6 * self.intensity)
        self.sway = _Wobble(self.rng, 2, 1.2 * self.intensity)  # weight shifts
        self.breath_hz = self.rng.uniform(0.2, 0.28)
        self.breath_deg = 2.0 * self.intensity

        self._i = 0  # index of the next frame
        self._glances = 0
        self._side = self.rng.choice((-1.0, 1.0))
        self._next_glance = self.rng.uniform(0.0, 0.15)
        self._gesture: Optional[int] = None  # the pose the arms go to or hold
        self._next_arms = self.rng.uniform(0.4, 1.2)
        self._fidget: Optional[Tuple[str, float, float]] = None  # kind, t0, dur
        self._next_fidget = self.rng.uniform(1.5, 3.0)
        self._settle_t0: Optional[float] = None
        self._settle_dur = 0.0

    @property
    def t(self) -> float:
        """Time of the next frame, seconds from the start of the episode."""

        return self._i / self.fps

    def next(self, n: int) -> np.ndarray:
        """The next `n` thinking frames, (n, 165) float32 radians."""

        return self._frames(n, plan=True)

    def settle(self, seconds: float = 0.6) -> np.ndarray:
        """Frames easing back to the neutral pose; the last one is exactly it.

        Raised arms take longer to come down: servos have speed limits.
        """

        raised = float(self.arms.value(self.t).sum())
        seconds = max(seconds, ARMS_S[1] * raised, 1 / self.fps)
        n = max(1, int(round(seconds * self.fps)))
        dur = n / self.fps
        last = (self._i - 1) / self.fps  # frames continue from the last one sent
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
        """Decide the next glance, pose and fidget when it is time."""

        if self.arms_on and t >= self._next_arms:
            self._plan_arms(t)
        if t >= self._next_fidget:
            self._plan_fidget(t)
        if t >= self._next_glance:
            self._plan_glance(t)

    def _plan_arms(self, t: float) -> None:
        rng = self.rng
        if self._gesture is not None and rng.random() < 0.5:
            self._gesture = None  # back down for a while
            self.arms.go(t, np.zeros(len(GESTURES)), rng.uniform(*ARMS_S))
            self._next_arms = t + rng.uniform(*ARMS_S) + rng.uniform(1.0, 3.0)
            return
        others = [k for k in range(len(GESTURES)) if k != self._gesture]
        if self._gesture is None and rng.random() < 0.15:
            self._next_arms = t + rng.uniform(1.0, 2.5)  # not yet
            return
        self._gesture = int(rng.choice(others))
        self.arms.go(t, np.eye(len(GESTURES))[self._gesture], rng.uniform(*ARMS_S))
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
            pitch = -rng.uniform(10.0, 20.0)  # up: recalling, constructing
        elif roll < 0.8:
            pitch = rng.uniform(5.0, 12.0)  # down: weighing
        else:
            pitch = rng.uniform(-3.0, 3.0)  # level, to a side
        yaw = self._side * rng.uniform(15.0, 35.0)
        tilt = self._side * rng.uniform(3.0, 9.0)
        if not first and rng.random() < 0.15:  # a glance back, almost centred
            pitch, yaw, tilt = 0.3 * pitch, 0.2 * yaw, 0.3 * tilt
        if self._gesture is not None:
            pitch += 4.0  # looking a bit down over the arms
        target = self.intensity * np.array([pitch, yaw, tilt])
        # A longer way takes longer: at most HEAD_SPEED on average.
        distance = np.abs(target - self.head.value(t)).max()
        move = max(rng.uniform(0.35, 0.7), distance / HEAD_SPEED)
        hold = rng.uniform(0.7, 2.0)

        self.head.go(t, target, move)
        # The eyes get there first and further: the head only follows.
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
        # Up and down in a sine envelope, a few times over its duration.
        u = (t - t0) / dur
        amp = env * self.intensity * np.sin(np.pi * u)
        if kind == "nod":  # "hmm, hmm"
            deg[HEAD, 0] += 5.0 * amp * np.sin(2 * np.pi * 1.8 * (t - t0))
            return
        right = float(weights[_GESTURE_SIDE > 0].sum())
        left = float(weights[_GESTURE_SIDE < 0].sum())
        if kind == "rub":  # the active hand rubs the chin or the arm
            wave = 6.0 * amp * np.sin(2 * np.pi * 1.6 * (t - t0))
            deg[R_ELBOW, 1] += right * wave
            deg[L_ELBOW, 1] -= left * wave
        else:  # its index finger taps
            wave = 15.0 * amp * (0.5 - 0.5 * np.cos(2 * np.pi * 2.5 * (t - t0)))
            deg[R_INDEX1, 2] += right * wave
            deg[L_INDEX1, 2] -= left * wave

    def _pose(self, t: float) -> np.ndarray:
        env = self._envelope(t)
        deg = REST_DEG.copy()

        pitch, yaw, tilt = self.head.value(t) + env * self.wobble.value(t)
        deg[NECK] += 0.4 * np.array([pitch, yaw, tilt])
        deg[HEAD] += 0.6 * np.array([pitch, yaw, tilt])
        # The chest turns with the head, and leans with it.
        deg[SPINE3, 1] += 0.25 * yaw
        deg[SPINE2, 1] += 0.15 * yaw
        deg[SPINE3, 0] += 0.3 * pitch
        sway = env * self.sway.value(t)
        deg[SPINE1, 1] += 0.1 * yaw + sway[0]  # from the waist: the feet stay
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
        deg += np.tensordot(weights, _GESTURE_DEG, axes=1)
        self._add_fidget(deg, t, env, weights)

        return np.radians(deg).reshape(POSE_DIM)
