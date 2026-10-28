"""BMFLC Fourier/LMS recursion from Veluvolu & Ang (2010), eqs. (1)-(5).

The optional constant-basis bias follows the same authors' Sensors 2011,
section 3.2, eqs. (17)-(19). The Fourier component alone is tremor output.
"""

import math

import numpy as np


class BMFLCFourierCore:
    """Causal, per-axis fixed frequency Fourier basis with ordinary LMS."""

    def __init__(self, sample_rate_hz: float, low_hz: float = 6.0,
                 high_hz: float = 14.0, spacing_hz: float = 0.1,
                 lms_gain: float = 0.01, with_bias: bool = False):
        if not all(math.isfinite(v) for v in
                   (sample_rate_hz, low_hz, high_hz, spacing_hz, lms_gain)):
            raise ValueError("Parameters must be finite")
        if not (0 < low_hz < high_hz < sample_rate_hz / 2 and
                spacing_hz > 0 and lms_gain > 0):
            raise ValueError("Invalid frequency band, spacing, sample rate, or gain")
        intervals = (high_hz - low_hz) / spacing_hz
        if abs(intervals - round(intervals)) > 1e-8:
            raise ValueError("The band must contain an integer number of intervals")
        self.sample_rate_hz = float(sample_rate_hz)
        self.frequencies_hz = np.linspace(low_hz, high_hz, round(intervals) + 1)
        self.lms_gain = float(lms_gain)
        self.with_bias = bool(with_bias)
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._weights = None

    def step(self, sample: np.ndarray) -> tuple[np.ndarray, dict]:
        observation = np.asarray(sample, dtype=float)
        if observation.ndim != 1 or not len(observation) or not np.all(np.isfinite(observation)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        count = len(self.frequencies_hz)
        if self._weights is None:
            self._weights = np.zeros((len(observation), 2 * count + int(self.with_bias)), dtype=float)
        elif len(observation) != len(self._weights):
            raise ValueError("axis count changed; reset before changing dimensions")
        phase = 2 * np.pi * self.frequencies_hz * self._sample_index / self.sample_rate_hz
        fourier_basis = np.r_[np.sin(phase), np.cos(phase)]
        basis = np.r_[fourier_basis, 1.0] if self.with_bias else fourier_basis
        # Paper eqs. (3)-(5): output uses w_k; w_{k+1} uses current error.
        tremor_estimate = self._weights[:, :2 * count] @ fourier_basis
        voluntary_bias = self._weights[:, -1].copy() if self.with_bias else np.zeros(len(observation))
        error = observation - tremor_estimate - voluntary_bias
        self._weights += 2 * self.lms_gain * error[:, None] * basis[None, :]
        self._sample_index += 1
        return tremor_estimate, {"internal_error": error.copy(),
                                 "intended_motion_bias": voluntary_bias,
                                 "adaptive_weights": self._weights.copy()}


class BMFLC(BMFLCFourierCore):
    """Full 2010 LMS BMFLC with 2011 clarified intended-motion bias."""

    def __init__(self, sample_rate_hz: float, low_hz: float = 6.0,
                 high_hz: float = 14.0, spacing_hz: float = 0.1,
                 lms_gain: float = 0.01):
        super().__init__(sample_rate_hz, low_hz, high_hz, spacing_hz,
                         lms_gain, with_bias=True)
