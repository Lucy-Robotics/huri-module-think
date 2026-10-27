"""The motion generator alone: thinking poses."""

import numpy as np
import pytest

from huri_think.motion import (
    EYE_JOINTS,
    GESTURE_NAMES,
    GESTURES,
    HEAD,
    L_ELBOW,
    POSE_DIM,
    R_ELBOW,
    REST_POSE,
    ThinkingMotion,
    mirrored,
)


def _episode(seed, seconds=10.0, fps=30, **kw):
    motion = ThinkingMotion(fps, rng=np.random.default_rng(seed), **kw)
    chunks = [motion.next(6) for _ in range(int(seconds * fps / 6))]
    return np.concatenate(chunks + [motion.settle(0.6)])


def _degrees(poses):
    return np.degrees(poses.reshape(poses.shape[0], 55, 3))


@pytest.mark.parametrize("seed", range(5))
def test_motion_starts_and_ends_at_rest(seed):
    poses = _episode(seed)
    assert poses.shape[1] == POSE_DIM and poses.dtype == np.float32
    assert np.isfinite(poses).all()
    np.testing.assert_allclose(poses[0], REST_POSE, atol=1e-6)
    np.testing.assert_allclose(poses[-1], REST_POSE, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_the_body_moves_smoothly(seed):
    """At most 270 deg/s at 30 fps (an arm swinging up), but the eyes, which
    saccade."""

    body = np.delete(_degrees(_episode(seed, arms=True)), EYE_JOINTS, axis=1)
    assert np.abs(np.diff(body, axis=0)).max() < 9.0


def test_the_head_looks_away():
    head = _degrees(_episode(0))[:, HEAD]
    assert np.abs(head[:, 1]).max() > 4.0  # yaw: to a side
    assert np.abs(head[:, 0]).max() > 2.0  # pitch: up or down


def test_every_episode_is_new():
    assert not np.allclose(_episode(1)[:300], _episode(2)[:300])
    np.testing.assert_array_equal(_episode(3), _episode(3))


def test_intensity_zero_keeps_still():
    poses = _episode(0, intensity=0.0, arms=False)
    np.testing.assert_allclose(poses, np.tile(REST_POSE, (len(poses), 1)), atol=1e-6)


def test_the_arms_take_thinking_poses_only_when_allowed():
    for seed in range(8):
        elbows = _degrees(_episode(seed, 10.0, arms=True))[:, [L_ELBOW, R_ELBOW], 1]
        assert np.abs(elbows).max() > 60.0
        elbows = _degrees(_episode(seed, 10.0, arms=False))[:, [L_ELBOW, R_ELBOW], 1]
        assert np.abs(elbows).max() < 1e-4


def test_episodes_differ_in_their_poses():
    """Several poses, left- and right-handed, not one gesture on a loop."""

    used = set()
    for seed in range(12):
        motion = ThinkingMotion(rng=np.random.default_rng(seed))
        for _ in range(50):
            motion.next(6)
            if motion._gesture is not None:
                used.add(GESTURE_NAMES[motion._gesture])
    assert len(used) >= 5


def test_left_poses_mirror_right_ones():
    for name in ("chin", "chin_hip", "crossed"):
        right, left = GESTURES[name], GESTURES[name + "_left"]
        np.testing.assert_allclose(left[L_ELBOW], right[R_ELBOW] * [1, -1, -1])
        np.testing.assert_allclose(mirrored(left), right)


def test_settle_lowers_a_raised_hand_slowly():
    for seed in range(20):
        motion = ThinkingMotion(rng=np.random.default_rng(seed), arms=True)
        for _ in range(100):
            motion.next(6)
            if motion.arms.value(motion.t).sum() > 0.9:
                assert motion.settle(0.6).shape[0] >= 1.4 * 30
                return
    pytest.fail("no seed raised the hand")
