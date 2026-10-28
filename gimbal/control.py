"""Ideal torque joint servo with explicit saturation."""

import numpy as np


def pd_torque(q_des, q, qvel, bias, torque_limits,
              kp=(0.6, 0.6), kd=(0.002, 0.002)):
    """PD plus MuJoCo bias compensation; zero desired joint velocity.

    `bias` is data.qfrc_bias from the current mj_step1 call. It includes
    gravity and velocity-dependent bias forces for this idealized plant.
    """
    raw = np.asarray(kp) * (q_des - q) - np.asarray(kd) * qvel + bias
    limits = np.asarray(torque_limits)
    clipped = np.clip(raw, limits[:, 0], limits[:, 1])
    return clipped, bool(np.any(np.abs(raw - clipped) > 1e-12))
