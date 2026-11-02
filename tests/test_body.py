import numpy as np
import pytest

from huri_think.body import Body, arm_points, thinking_arms
from huri_think.module import Think

BODIES = {
    "adult": Body(),
    "child": Body(
        shoulder=(0.12, 0.33, -0.02),
        upper_arm=0.18,
        forearm=0.16,
        hand=0.12,
        chin=(0.45, 0.08),
        chest=(0.24, 0.09, 0.12),
        waist=(0.11, 0.1, 0.11),
        hip=(0.05, 0.1),
    ),
    "tall": Body.from_mapping(
        {k: (np.array(v) * 1.15).tolist() for k, v in Body().as_dict().items()}
    ),
    "cartoon": Body(
        shoulder=(0.175, 0.398, 0.008),
        upper_arm=0.2,
        forearm=0.22,
        hand=0.2,
        chin=(0.5, 0.22),
        chest=(0.284, 0.25, 0.182),
        waist=(0.128, 0.256, 0.233),
        hip=(0.055, 0.247),
    ),
}


@pytest.mark.parametrize("name", BODIES)
@pytest.mark.parametrize("active", [1.0, -1.0])
def test_the_arms_reach_their_targets_outside_the_body(name, active):
    body = BODIES[name]
    for pose, arms in thinking_arms(body, active).items():
        for side, arm in arms.items():
            elbow, wrist = arm_points(body, side, arm.shoulder, arm.elbow)
            np.testing.assert_allclose(elbow, arm.elbow_at, atol=1e-6)
            np.testing.assert_allclose(wrist, arm.wrist_at, atol=1e-6)
            for point in (elbow, (elbow + wrist) / 2, wrist):
                assert not body.inside(point, 0.4 * body.radius), (pose, side, point)


@pytest.mark.parametrize("name", BODIES)
def test_the_hand_is_at_the_chin_on_any_body(name):
    body = BODIES[name]
    arm = thinking_arms(body, -1.0)["chin"][-1.0]
    _, wrist = arm_points(body, -1.0, arm.shoulder, arm.elbow)
    chin = np.array([0.0, *body.chin])
    assert np.linalg.norm(wrist - chin) < 0.7 * body.hand


def test_the_poses_follow_the_body():
    adult = thinking_arms(BODIES["adult"], -1.0)["chin"][-1.0]
    child = thinking_arms(BODIES["child"], -1.0)["chin"][-1.0]
    assert not np.allclose(adult.shoulder, child.shoulder, atol=1e-3)
    tall = thinking_arms(BODIES["tall"], -1.0)["chin"][-1.0]
    np.testing.assert_allclose(adult.shoulder, tall.shoulder, atol=1e-6)
    np.testing.assert_allclose(adult.elbow, tall.elbow, atol=1e-6)


def test_a_partial_body_keeps_the_other_defaults():
    body = Body.from_mapping({"upper_arm": 0.3, "chin": [0.7, 0.1]})
    assert body.upper_arm == 0.3 and body.chin == (0.7, 0.1)
    assert body.forearm == Body().forearm


@pytest.mark.parametrize(
    "values",
    [
        {"elbow": 0.3},
        {"upper_arm": -0.1},
        {"upper_arm": "long"},
        {"chin": [0.7]},
        {"chest": [0.4, 0.1, float("nan")]},
        {"shoulder": [0.2, 50.0, 0.0]},
        [0.2],
    ],
)
def test_a_bad_body_is_refused(values):
    with pytest.raises(ValueError):
        Body.from_mapping(values)


def test_the_module_takes_a_body_from_its_args():
    think = Think("user", body={"upper_arm": 0.2, "forearm": 0.2})
    assert think.body.upper_arm == 0.2
    with pytest.raises(ValueError):
        Think("user", body={"knee": 1.0})
