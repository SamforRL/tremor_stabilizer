"""Round-3 experimental variants; original ABPF and BMFLC are untouched."""

import math

import numpy as np

from .abpf import tustin_abpf_biquad


class FixedCenterABPF:
    """The ABPF Tustin biquad with constant independently calibrated center."""

    def __init__(self, sample_rate_hz: float, modal_frequency_hz: float | list[float]):
        self.sample_rate_hz = float(sample_rate_hz)
        self.modal_frequency_hz = np.atleast_1d(modal_frequency_hz).astype(float)
        if (not np.all(np.isfinite(self.modal_frequency_hz))
                or np.any(self.modal_frequency_hz <= 0)
                or np.any(self.modal_frequency_hz >= sample_rate_hz / 2)):
            raise ValueError("Invalid fixed ABPF frequency")
        self.reset()

    def reset(self):
        self._center = None
        self._biquad = None
        self._state = None

    def step(self, sample: np.ndarray):
        x = np.asarray(sample, dtype=float)
        if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        if self._center is None:
            self._center = np.repeat(self.modal_frequency_hz, len(x)) if len(self.modal_frequency_hz) == 1 else self.modal_frequency_hz.copy()
            if len(self._center) != len(x):
                raise ValueError("axis count and modal count differ")
            self._biquad = np.stack([tustin_abpf_biquad(f, self.sample_rate_hz) for f in self._center])
            self._state = np.zeros((len(x), 2))
        elif len(x) != len(self._center):
            raise ValueError("axis count changed")
        output = np.empty_like(x)
        for axis, value in enumerate(x):
            b0, b1, b2, _, a1, a2 = self._biquad[axis]
            d1, d2 = self._state[axis]
            y = b0 * value + d1
            self._state[axis, 0] = b1 * value - a1 * y + d2
            self._state[axis, 1] = b2 * value - a2 * y
            output[axis] = y
        return output, {"estimated_frequency_hz": float(self._center[0]) if len(x) == 1 else self._center.copy(),
                        "center_frequency_hz": self._center.copy(), "initialization_state": "modal_fixed"}


class RobustBMFLC:
    """Unchanged Fourier basis/bias with Huber or leaky LMS coefficient update.

    The project baseline uses 2*mu*error*basis, matching the original paper's
    squared-error LMS convention. Both variants retain that effective gain.
    """

    def __init__(self, sample_rate_hz: float, low_hz: float = 3.0,
                 high_hz: float = 12.0, spacing_hz: float = 0.1,
                 lms_gain: float = 0.01, variant: str = "huber",
                 huber_delta: float | None = None, leak_lambda: float = 0.0):
        if variant not in {"huber", "leaky"}:
            raise ValueError("variant must be huber or leaky")
        if not all(math.isfinite(v) for v in (sample_rate_hz, low_hz, high_hz, spacing_hz, lms_gain, leak_lambda)):
            raise ValueError("parameters must be finite")
        if not (0 < low_hz < high_hz < sample_rate_hz / 2 and spacing_hz > 0
                and lms_gain > 0 and 0 <= leak_lambda < 1):
            raise ValueError("invalid frequency band or gain")
        intervals = (high_hz - low_hz) / spacing_hz
        if abs(intervals - round(intervals)) > 1e-8:
            raise ValueError("frequency spacing must divide band")
        if variant == "huber" and (huber_delta is None or not math.isfinite(huber_delta) or huber_delta <= 0):
            raise ValueError("Huber requires positive finite delta")
        self.sample_rate_hz = float(sample_rate_hz)
        self.frequencies_hz = np.linspace(low_hz, high_hz, round(intervals) + 1)
        self.lms_gain = float(lms_gain)
        self.variant = variant
        self.huber_delta = float(huber_delta) if huber_delta is not None else None
        self.leak_lambda = float(leak_lambda)
        self.reset()

    def reset(self):
        self._index = 0
        self._weights = None

    def step(self, sample: np.ndarray):
        observation = np.asarray(sample, dtype=float)
        if observation.ndim != 1 or not len(observation) or not np.all(np.isfinite(observation)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        count = len(self.frequencies_hz)
        if self._weights is None:
            self._weights = np.zeros((len(observation), 2 * count + 1))
        elif len(observation) != len(self._weights):
            raise ValueError("axis count changed")
        phase = 2 * np.pi * self.frequencies_hz * self._index / self.sample_rate_hz
        fourier_basis = np.r_[np.sin(phase), np.cos(phase)]
        basis = np.r_[fourier_basis, 1.0]
        tremor = self._weights[:, :2 * count] @ fourier_basis
        bias = self._weights[:, -1].copy()
        error = observation - tremor - bias
        robust_error = (np.clip(error, -self.huber_delta, self.huber_delta)
                        if self.variant == "huber" else error.copy())
        if self.variant == "leaky":
            self._weights *= 1 - self.leak_lambda
        self._weights += 2 * self.lms_gain * robust_error[:, None] * basis[None, :]
        self._index += 1
        return tremor, {"internal_error": error.copy(),
                        "robust_update_error": robust_error.copy(),
                        "intended_motion_bias": bias,
                        "adaptive_weights": self._weights.copy()}
