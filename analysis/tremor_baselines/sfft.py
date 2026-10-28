"""Causal interpretation of Wang et al. (2016) adaptive sliding FFT.

Window: paper eqs. (2)-(7); dual-bin ratio: eqs. (8)-(14); limiter with a
separate Hz bandwidth: eq. (15); symmetric IFFT reconstruction: eqs. (16)-(18).
The inverse-ratio polynomial is calibrated deterministically offline from
the same window, following the user's resolution of the missing fit settings.
"""

import math

import numpy as np


def fourth_order_blackman_window(length: int) -> np.ndarray:
    """Four length-N Blackman sequences convolved, then three trailing zeros."""
    if length < 16 or length % 4:
        raise ValueError("SFFT window length must be a multiple of four, >=16")
    n = length // 4
    index = np.arange(n)
    base = 0.42 - 0.5 * np.cos(2 * np.pi * index / n) + 0.08 * np.cos(4 * np.pi * index / n)
    window = base.copy()
    for _ in range(3):
        window = np.convolve(window, base)
    return np.pad(window, (0, 3))


def _bin_ratio(window: np.ndarray, fractional_offset: np.ndarray) -> np.ndarray:
    """Paper eq. (10) using the actual order-four window's DTFT."""
    offset = np.asarray(fractional_offset, dtype=float)
    positions = np.arange(len(window))
    response_lower = abs(np.exp(2j * np.pi * offset[:, None] * positions / len(window)) @ window)
    response_upper = abs(np.exp(2j * np.pi * (offset[:, None] - 1) * positions / len(window)) @ window)
    return (response_upper - response_lower) / (response_upper + response_lower)


def calibrate_inverse_ratio(window: np.ndarray, max_error_bin: float = 0.01) -> dict:
    """Fit odd alpha(beta) polynomials of degree 3, 5, 7; choose lowest passing."""
    training_alpha = np.linspace(-0.5, 0.5, 1001)
    validation_alpha = np.linspace(-0.4995, 0.4995, 1000)
    training_beta = _bin_ratio(window, training_alpha + 0.5)
    validation_beta = _bin_ratio(window, validation_alpha + 0.5)
    results = {}
    choice = None
    for degree in (3, 5, 7):
        powers = tuple(range(1, degree + 1, 2))
        coefficients = np.linalg.lstsq(
            np.column_stack([training_beta ** p for p in powers]), training_alpha, rcond=None
        )[0]
        predicted = sum(c * validation_beta ** p for c, p in zip(coefficients, powers))
        maximum = float(np.max(abs(predicted - validation_alpha)))
        results[degree] = {"powers": powers, "coefficients": coefficients,
                           "validation_max_error_bin": maximum}
        if choice is None and maximum <= max_error_bin:
            choice = degree
    if choice is None:
        raise ArithmeticError("No requested odd polynomial order meets calibration tolerance")
    return {"selected_order": choice, "selected": results[choice],
            "candidates": results, "max_error_bin": max_error_bin}


class AdaptiveSFFT:
    """One causal sliding SFFT state per axis; common fixed window mapping."""

    def __init__(self, sample_rate_hz: float, min_frequency_hz: float,
                 max_frequency_hz: float, extraction_bandwidth_hz: float = 3.0,
                 window_duration_s: float = 1.0,
                 interpolation_max_error_bin: float = 0.01):
        if not all(math.isfinite(v) for v in
                   (sample_rate_hz, min_frequency_hz, max_frequency_hz,
                    extraction_bandwidth_hz, window_duration_s)):
            raise ValueError("SFFT configuration must be finite")
        if not (0 < min_frequency_hz < max_frequency_hz < sample_rate_hz / 2
                and 0 < extraction_bandwidth_hz < max_frequency_hz - min_frequency_hz
                and window_duration_s > 0):
            raise ValueError("Invalid SFFT frequency/window configuration")
        self.sample_rate_hz = float(sample_rate_hz)
        self.min_frequency_hz = float(min_frequency_hz)
        self.max_frequency_hz = float(max_frequency_hz)
        self.extraction_bandwidth_hz = float(extraction_bandwidth_hz)
        # Four-convolution requirement: preserve the paper's ~1 s duration.
        self.window_samples = 4 * round(sample_rate_hz * window_duration_s / 4)
        self.window = fourth_order_blackman_window(self.window_samples)
        self.interpolation = calibrate_inverse_ratio(self.window,
                                                       interpolation_max_error_bin)
        self.frequencies_hz = np.fft.rfftfreq(self.window_samples, 1 / self.sample_rate_hz)
        self.reset()

    def reset(self) -> None:
        self._buffer = None
        self._seen = 0

    def step(self, sample: np.ndarray) -> tuple[np.ndarray, dict]:
        observed = np.asarray(sample, dtype=float)
        if observed.ndim != 1 or not len(observed) or not np.all(np.isfinite(observed)):
            raise ValueError("sample must be a finite nonempty one-dimensional array")
        if self._buffer is None:
            self._buffer = np.zeros((self.window_samples, len(observed)))
        elif self._buffer.shape[1] != len(observed):
            raise ValueError("axis count changed; reset before changing dimensions")
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = observed
        self._seen += 1
        weighted_spectrum = np.fft.rfft(self._buffer * self.window[:, None], axis=0)
        raw_spectrum = np.fft.rfft(self._buffer, axis=0)
        output = np.zeros(len(observed))
        estimated_frequency = np.full(len(observed), np.nan)
        raw_peak = np.full(len(observed), np.nan)
        lower = self.min_frequency_hz + self.extraction_bandwidth_hz / 2
        upper = self.max_frequency_hz - self.extraction_bandwidth_hz / 2
        permitted = np.flatnonzero((self.frequencies_hz >= self.min_frequency_hz)
                               & (self.frequencies_hz <= self.max_frequency_hz))
        if len(permitted) == 0:
            raise ValueError("SFFT search band contains no FFT bin")
        powers = self.interpolation["selected"]["powers"]
        coefficients = self.interpolation["selected"]["coefficients"]
        for axis in range(len(observed)):
            magnitude = abs(weighted_spectrum[:, axis])
            peak = int(permitted[np.argmax(magnitude[permitted])])
            if magnitude[peak] <= 1e-12:
                continue
            adjacent = peak + 1 if magnitude[min(peak + 1, len(magnitude) - 1)] >= magnitude[peak - 1] else peak - 1
            first = min(peak, adjacent)
            second = first + 1
            y1, y2 = magnitude[first], magnitude[second]
            ratio = (y2 - y1) / (y2 + y1)
            alpha = float(sum(c * ratio ** p for c, p in zip(coefficients, powers)))
            alpha = float(np.clip(alpha, -0.5, 0.5))
            fpeak = (first + alpha + 0.5) * self.sample_rate_hz / self.window_samples
            fdom = float(np.clip(fpeak, lower, upper))
            mask = (self.frequencies_hz >= fdom - self.extraction_bandwidth_hz / 2) & (
                    self.frequencies_hz <= fdom + self.extraction_bandwidth_hz / 2)
            masked = np.where(mask, raw_spectrum[:, axis], 0)
            # rFFT/irFFT keep conjugate bins together. The newest reconstructed
            # sample belongs to the current time; no center sample is relabelled.
            output[axis] = np.fft.irfft(masked, n=self.window_samples)[-1]
            estimated_frequency[axis] = fdom
            raw_peak[axis] = fpeak
        frequency_diag = (float(estimated_frequency[0]) if len(observed) == 1
                          else estimated_frequency.copy())
        return output, {"estimated_frequency_hz": frequency_diag,
                        "raw_peak_frequency_hz": raw_peak.copy(),
                        "initialization_state": ("zero_padded" if self._seen < self.window_samples
                                                 else "full_window"),
                        "window_filled_samples": min(self._seen, self.window_samples),
                        "extraction_bandwidth_hz": self.extraction_bandwidth_hz}
