"""Damped BMFLC from the 2026 Autonomous Robots paper.

The paper's logistic coefficient-wise step size (Eq. 8) is implemented
literally. The project-facing interface accepts/returns deg/s, while the
internal coefficient state uses rad/s so the reported ``x_dmp`` is applied in
SI angular-rate units. The paper's model has no constant bias term.
"""

from __future__ import annotations

import math

import numpy as np


class DampedBMFLC:
    """Causal per-axis Fourier estimator with logistic-damped LMS updates."""

    PAPER_REFERENCE_FS_HZ = 1000.0
    PAPER_LAMBDA = 0.99999

    def __init__(self, sample_rate_hz: float, low_hz: float = 3.0,
                 high_hz: float = 12.0, spacing_hz: float = 0.1,
                 eta: float = 0.020, lambda_ref: float = PAPER_LAMBDA,
                 reference_sample_rate_hz: float = PAPER_REFERENCE_FS_HZ,
                 kdmp: float = 350.0, xdmp_rad_s: float = 0.009):
        params = (sample_rate_hz, low_hz, high_hz, spacing_hz, eta,
                  lambda_ref, reference_sample_rate_hz, kdmp, xdmp_rad_s)
        if not all(math.isfinite(float(x)) for x in params):
            raise ValueError("Damped BMFLC parameters must be finite")
        if not (0 < low_hz < high_hz < sample_rate_hz / 2
                and spacing_hz > 0 and eta > 0
                and 0 < lambda_ref <= 1 and reference_sample_rate_hz > 0
                and kdmp > 0 and xdmp_rad_s >= 0):
            raise ValueError("Invalid Damped BMFLC configuration")
        intervals = (high_hz - low_hz) / spacing_hz
        if abs(intervals - round(intervals)) > 1e-8:
            raise ValueError("Frequency spacing must divide the band")
        self.sample_rate_hz = float(sample_rate_hz)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.spacing_hz = float(spacing_hz)
        self.frequencies_hz = np.linspace(low_hz, high_hz, round(intervals) + 1)
        self.eta = float(eta)
        self.lambda_ref = float(lambda_ref)
        self.reference_sample_rate_hz = float(reference_sample_rate_hz)
        # Preserve the forgetting time constant when porting the paper's
        # per-sample lambda from 1 kHz to the project's native sensor rate.
        self.forgetting_factor = self.lambda_ref ** (
            self.reference_sample_rate_hz / self.sample_rate_hz
        )
        self.kdmp = float(kdmp)
        self.xdmp_rad_s = float(xdmp_rad_s)
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._weights = None

    def step(self, sample_deg_s: np.ndarray) -> tuple[np.ndarray, dict]:
        observation_deg_s = np.asarray(sample_deg_s, dtype=float)
        if (observation_deg_s.ndim != 1 or not len(observation_deg_s)
                or not np.all(np.isfinite(observation_deg_s))):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        axes = len(observation_deg_s)
        n_freq = len(self.frequencies_hz)
        n_weights = 2 * n_freq
        if self._weights is None:
            self._weights = np.zeros((axes, n_weights), dtype=float)
        elif len(self._weights) != axes:
            raise ValueError("axis count changed; reset before changing dimensions")

        observation = np.deg2rad(observation_deg_s)
        phase = 2 * np.pi * self.frequencies_hz * self._sample_index / self.sample_rate_hz
        basis = np.r_[np.sin(phase), np.cos(phase)]
        estimate_rad_s = self._weights @ basis
        error_rad_s = observation - estimate_rad_s

        # Eq. (8): mu_dmp,k = eta*g_k / (1 + exp(-kdmp*(w_k-xdmp))).
        # Keep the printed signed-coefficient dependence; no absolute value is
        # introduced. Clipping only avoids numerical overflow in exp().
        logistic_argument = np.clip(
            -self.kdmp * (self._weights - self.xdmp_rad_s), -60.0, 60.0
        )
        damping_gain = 1.0 / (1.0 + np.exp(logistic_argument))
        self._weights = (
            self.forgetting_factor * self._weights
            + self.eta * error_rad_s[:, None] * basis[None, :] * damping_gain
        )
        self._sample_index += 1
        return np.rad2deg(estimate_rad_s), {
            "internal_error": np.rad2deg(error_rad_s),
            "adaptive_weights": self._weights.copy(),
            "damping_gain_mean": np.mean(damping_gain, axis=1),
            "damping_gain_min": np.min(damping_gain, axis=1),
            "damping_gain_max": np.max(damping_gain, axis=1),
            "forgetting_factor_per_sample": self.forgetting_factor,
            "initialization_state": "zero_weights",
        }
