"""Deterministic prescribed handle poses; world-from-handle uses Rx @ Ry."""

import math

import numpy as np


def roll_pitch_quaternion(roll, pitch):
    """MuJoCo wxyz quaternion for world-from-handle Rx(roll) Ry(pitch)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return cr * cp, sr * cp, cr * sp, sr * sp


def roll_pitch_rotation(roll, pitch):
    """World-from-body matrix for Rx(roll) Ry(pitch)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cp, 0.0, sp],
        [sr * sp, cr, -sr * cp],
        [-cr * sp, sr, cr * cp],
    ])


def measured_roll_pitch(handle_rotation):
    """Extract RxRy angles near upright; this first model has no handle yaw."""
    R = np.asarray(handle_rotation).reshape(3, 3)
    return np.array([math.atan2(R[2, 1], R[1, 1]),
                     math.atan2(R[0, 2], R[0, 0])])


def handle_pose(t, scenario):
    """Return roll, pitch, and MuJoCo wxyz quaternion at time t (SI units)."""
    if scenario == "stationary":
        roll = pitch = 0.0
    elif scenario == "oscillatory":
        roll = 0.10 * math.sin(2 * math.pi * 2.0 * t)
        pitch = 0.05 * math.sin(2 * math.pi * 1.5 * t)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return roll, pitch, roll_pitch_quaternion(roll, pitch)


def follow_motion(t, scenario):
    """Return total, voluntary and tremor roll/pitch for controlled tests.

    The two components are available to this trajectory generator and to
    evaluation, never to the online estimator.
    """
    if scenario not in ("mixed", "voluntary_only", "tremor_only"):
        raise ValueError(f"Unknown follow scenario: {scenario}")
    voluntary = np.array([
        0.12 * math.sin(2 * math.pi * 0.25 * t),
        0.08 * math.sin(2 * math.pi * 0.18 * t),
    ]) if scenario != "tremor_only" else np.zeros(2)
    tremor_phase = 2 * math.pi * 5.2 * t
    ramp = min(1.0, t / 0.5)
    tremor = ramp * np.array([
        0.06 * math.sin(tremor_phase + 0.5),
        0.04 * math.sin(tremor_phase + 1.1),
    ]) if scenario != "voluntary_only" else np.zeros(2)
    return voluntary + tremor, voluntary, tremor


def position_stabilization_motion(t, scenario):
    """Prescribed handle translation and RxRy rotation for weighted tasks."""
    if scenario == "stationary":
        position = np.zeros(3)
        roll = pitch = 0.0
    elif scenario == "moving":
        position = np.array([
            0.006 * math.sin(2 * math.pi * 0.7 * t),
            0.004 * math.sin(2 * math.pi * 0.9 * t),
            0.008 * math.sin(2 * math.pi * 1.1 * t),
        ])
        roll = 0.08 * math.sin(2 * math.pi * 1.7 * t)
        pitch = 0.05 * math.sin(2 * math.pi * 1.3 * t)
    else:
        raise ValueError(f"Unknown position scenario: {scenario}")
    return position, roll, pitch, roll_pitch_quaternion(roll, pitch)
