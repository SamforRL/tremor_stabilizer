"""Causal Popović--Šekara--Popović 2010 ABPF, with documented discretization.

Paper eq. (1) and Figs. 4--5 define a continuous second-order Butterworth
band-pass loop. The printed p.302 damping code supplies the two step rules.
Tustin discretization and interpolated sample-level zero crossings implement
that loop at a finite sample rate; see REPRODUCTION_NOTES.md for deviations.
"""

import math

import numpy as np


def tustin_abpf_biquad(center_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Bilinear transform of beta*omega*s/(s²+beta*omega*s+omega²)."""
    if not (math.isfinite(center_hz) and math.isfinite(sample_rate_hz)
            and 0 < center_hz < sample_rate_hz / 2):
        raise ValueError("ABPF center must be between zero and Nyquist")
    omega = 2 * math.pi * center_hz
    beta = math.sqrt(2)
    k = 2 * sample_rate_hz
    a0 = k * k + beta * omega * k + omega * omega
    return np.array([beta * omega * k / a0, 0.0, -beta * omega * k / a0,
                     1.0, 2 * (omega * omega - k * k) / a0,
                     (k * k - beta * omega * k + omega * omega) / a0])


class ABPF:
    """Independent per-axis ABPF states, using independent modal calibration."""

    def __init__(self, sample_rate_hz: float, modal_frequency_hz: float | list[float],
                 frequency_step_hz: float = 0.1, limit_offset_hz: float = 1.5):
        self.sample_rate_hz = float(sample_rate_hz)
        self.modal_frequency_hz = np.atleast_1d(modal_frequency_hz).astype(float)
        self.frequency_step_hz = float(frequency_step_hz)
        self.limit_offset_hz = float(limit_offset_hz)
        if (not np.all(np.isfinite(self.modal_frequency_hz))
                or np.any(self.modal_frequency_hz <= limit_offset_hz)
                or np.any(self.modal_frequency_hz + limit_offset_hz >= sample_rate_hz / 2)
                or not math.isfinite(frequency_step_hz) or frequency_step_hz <= 0
                or not math.isfinite(limit_offset_hz) or limit_offset_hz <= 0):
            raise ValueError("Invalid ABPF modal frequency, range, or step")
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._center = None
        self._biquad = None
        self._state = None
        self._previous_output = None
        self._last_crossing_s = None

    def _initialize(self, axes: int) -> None:
        modal = self.modal_frequency_hz
        if len(modal) == 1:
            modal = np.repeat(modal, axes)
        if len(modal) != axes:
            raise ValueError("modal frequency must be scalar or match axis count")
        self._modal = modal.copy()
        self._center = modal.copy()
        self._biquad = np.stack([tustin_abpf_biquad(f, self.sample_rate_hz)
                                 for f in modal])
        self._state = np.zeros((axes, 2))
        self._previous_output = np.zeros(axes)
        self._last_crossing_s = np.full(axes, np.nan)

    def step(self, sample: np.ndarray) -> tuple[np.ndarray, dict]:
        x = np.asarray(sample, dtype=float)
        if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        if self._center is None:
            self._initialize(len(x))
        elif len(x) != len(self._center):
            raise ValueError("axis count changed; reset before changing dimensions")
        output = np.empty_like(x)
        updates = np.zeros(len(x))
        measured_h = np.full(len(x), np.nan)
        for axis, value in enumerate(x):
            b0, b1, b2, _, a1, a2 = self._biquad[axis]
            d1, d2 = self._state[axis]
            y = b0 * value + d1
            self._state[axis, 0] = b1 * value - a1 * y + d2
            self._state[axis, 1] = b2 * value - a2 * y
            output[axis] = y
            old_y = self._previous_output[axis]
            if self._index > 0 and old_y * y < 0:
                fraction = -old_y / (y - old_y)
                crossing_s = (self._index - 1 + fraction) / self.sample_rate_hz
                if np.isfinite(self._last_crossing_s[axis]):
                    h = crossing_s - self._last_crossing_s[axis]
                    measured_h[axis] = h
                    if h > 0:
                        fin = 1 / (2 * h)
                        old = self._center[axis]
                        modal = self._modal[axis]
                        delta = self.frequency_step_hz
                        # The exact ordered conditions in the paper's p.302
                        # damping fcn: second condition overrides the first.
                        updated = old
                        if fin >= old + delta:
                            updated = old + delta
                        if fin >= old + 2 * delta and old <= modal - 2 * delta:
                            updated = old + 2 * delta
                        if fin <= old - delta:
                            updated = old - delta
                        if fin <= old - 2 * delta and old >= modal + 2 * delta:
                            updated = old - 2 * delta
                        # The limiter passes in-range input and clips outside.
                        updated = float(np.clip(updated, modal - self.limit_offset_hz,
                                                modal + self.limit_offset_hz))
                        if updated != old:
                            self._center[axis] = updated
                            self._biquad[axis] = tustin_abpf_biquad(updated,
                                                                     self.sample_rate_hz)
                            updates[axis] = updated - old
                self._last_crossing_s[axis] = crossing_s
            self._previous_output[axis] = y
        self._index += 1
        frequency_diag = (float(self._center[0]) if len(x) == 1
                          else self._center.copy())
        return output, {"estimated_frequency_hz": frequency_diag,
                        "center_frequency_hz": self._center.copy(),
                        "frequency_update_hz": updates,
                        "half_period_s": measured_h,
                        "initialization_state": "modal_calibrated"}
