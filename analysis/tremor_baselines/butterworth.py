"""Causal Butterworth band-pass filter in second-order-section form.

The analog Butterworth prototype is transformed to a band-pass prototype,
prewarped, and mapped to the z plane by the bilinear transform. This is the
standard Butterworth construction, not an adaptive tremor estimator. Design
and streaming response are checked against scipy.signal in the tests, while
runtime needs only NumPy (the local me5418 environment lacks SciPy).
"""

import math

import numpy as np


def butterworth_bandpass_sos(order: int, low_hz: float, high_hz: float,
                             sample_rate_hz: float) -> np.ndarray:
    """Design an order-N prototype, resulting in N biquad band-pass sections.

As in scipy.signal.butter(N, [low, high], btype='bandpass'), the final
digital transfer function has order 2N and -3 dB critical frequencies at
low_hz and high_hz.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 12:
        raise ValueError("order must be an integer in 1..12")
    if not all(math.isfinite(x) for x in (low_hz, high_hz, sample_rate_hz)):
        raise ValueError("frequencies must be finite")
    if not 0 < low_hz < high_hz < sample_rate_hz / 2:
        raise ValueError("require 0 < low_hz < high_hz < Nyquist")

    k = np.arange(order)
    prototype_poles = np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order))
    warped = 2 * sample_rate_hz * np.tan(
        np.pi * np.array([low_hz, high_hz]) / sample_rate_hz
    )
    bandwidth = warped[1] - warped[0]
    analog_center = math.sqrt(warped[0] * warped[1])
    root = np.sqrt((bandwidth * prototype_poles) ** 2 - 4 * analog_center ** 2)
    analog_poles = np.concatenate(((bandwidth * prototype_poles + root) / 2,
                                   (bandwidth * prototype_poles - root) / 2))
    digital_poles = (2 * sample_rate_hz + analog_poles) / (2 * sample_rate_hz - analog_poles)
    if np.max(np.abs(digital_poles)) >= 1:
        raise ArithmeticError("Unstable Butterworth design")

    unused = list(digital_poles)
    sections = []
    while unused:
        pole = unused.pop(0)
        partner_index = int(np.argmin(np.abs(np.asarray(unused) - pole.conjugate())))
        partner = unused.pop(partner_index)
        if abs(partner - pole.conjugate()) > 1e-8:
            raise ArithmeticError("Could not pair conjugate poles")
        # A band-pass transformation contributes one zero at z=+1 and one
        # at z=-1 to each biquad section.
        denominator = np.poly([pole, partner]).real
        sections.append(np.r_[1.0, 0.0, -1.0, denominator])
    sos = np.asarray(sections, dtype=float)
    center_hz = sample_rate_hz / np.pi * math.atan(analog_center / (2 * sample_rate_hz))
    center_gain = abs(sos_response(sos, np.array([center_hz]), sample_rate_hz)[0])
    sos[0, :3] /= center_gain
    return sos


def sos_response(sos: np.ndarray, frequencies_hz: np.ndarray,
                 sample_rate_hz: float) -> np.ndarray:
    """Complex frequency response of a cascade of normalized biquads."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    z1 = np.exp(-2j * np.pi * frequencies / sample_rate_hz)
    z2 = z1 * z1
    response = np.ones_like(z1, dtype=complex)
    for b0, b1, b2, a0, a1, a2 in np.asarray(sos):
        response *= (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return response


class FixedButterworthBPF:
    """One identical causal scalar filter per axis, with independent state."""

    def __init__(self, sample_rate_hz: float, order: int = 4,
                 low_hz: float = 3.0, high_hz: float = 12.0):
        self.sample_rate_hz = float(sample_rate_hz)
        self.order = order
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.sos = butterworth_bandpass_sos(order, low_hz, high_hz, sample_rate_hz)
        self._state = None

    def reset(self) -> None:
        self._state = None

    def step(self, sample: np.ndarray) -> tuple[np.ndarray, dict]:
        x = np.asarray(sample, dtype=float)
        if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        if self._state is None:
            self._state = np.zeros((len(self.sos), len(x), 2), dtype=float)
        elif self._state.shape[1] != len(x):
            raise ValueError("axis count changed; reset before changing dimensions")
        y = x.copy()
        # Transposed direct-form II: each output uses current and past samples.
        for section, (b0, b1, b2, _, a1, a2) in enumerate(self.sos):
            d1 = self._state[section, :, 0].copy()
            d2 = self._state[section, :, 1].copy()
            next_y = b0 * y + d1
            self._state[section, :, 0] = b1 * y - a1 * next_y + d2
            self._state[section, :, 1] = b2 * y - a2 * next_y
            y = next_y
        return y, {}
