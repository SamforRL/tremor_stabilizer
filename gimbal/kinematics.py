"""World-from-tool orientation: R_handle @ Rx(roll) @ Ry(pitch)."""

import math

import numpy as np


WORLD_UP = np.array([0.0, 0.0, 1.0])


def tool_axis(handle_rotation, joints):
    """Return the tool frame's +Z unit axis in world coordinates."""
    roll, pitch = joints
    local_axis = np.array([
        math.sin(pitch),
        -math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    ])
    return np.asarray(handle_rotation).reshape(3, 3) @ local_axis


def axis_error(actual, target=WORLD_UP):
    """Unsigned angle between unit axes, in radians."""
    return math.atan2(float(np.linalg.norm(np.cross(actual, target))),
                      float(np.dot(actual, target)))


def axis_joint_reference(handle_rotation, target_axis, joint_limits):
    """Map a world-frame target axis to roll/pitch; clip at joint limits.

    This is the principal branch of the analytic axis-alignment IK, valid
    around the upright configuration. Clipping is a bounded fallback, not an
    optimal projection; the returned residual exposes any unreachable target.
    """
    target = np.asarray(target_axis, dtype=float)
    target = target / np.linalg.norm(target)
    local_target = np.asarray(handle_rotation).reshape(3, 3).T @ target
    pitch = math.asin(float(np.clip(local_target[0], -1.0, 1.0)))
    roll = math.atan2(-local_target[1], local_target[2])
    raw = np.array([roll, pitch])
    limits = np.asarray(joint_limits)
    desired = np.clip(raw, limits[:, 0], limits[:, 1])
    residual = axis_error(tool_axis(handle_rotation, desired), target)
    reachable = bool(np.allclose(raw, desired, atol=1e-12) and residual < 1e-8)
    return desired, residual, reachable


def upright_joint_reference(handle_rotation, joint_limits):
    """Special case of axis IK with fixed world +Z target."""
    return axis_joint_reference(handle_rotation, WORLD_UP, joint_limits)
