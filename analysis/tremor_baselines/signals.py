"""Deterministic angular-velocity benchmark signals (deg/s throughout)."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BenchmarkCaseConfig:
    sample_rate_hz: float = 148.14814814814815
    duration_s: float = 12.0
    tremor_scale: float = 1.0
    noise_std_deg_s: float = 0.0
    random_seed: int = 20261026
    smooth_frequency_hz: float = 0.25
    smooth_amplitude_deg_s: float = 2.0
    point_start_s: float = 3.0
    point_duration_s: float = 2.5
    point_displacement_deg: float = 10.0
    fast_start_s: float = 3.0
    fast_duration_s: float = 0.35
    fast_displacement_deg: float = 10.0
    overlap_frequency_hz: float = 3.2
    overlap_amplitude_deg_s: float = 0.8


@dataclass(frozen=True)
class SignalComponents:
    time_s: np.ndarray
    voluntary_true_deg_s: np.ndarray
    tremor_true_deg_s: np.ndarray
    noise_true_deg_s: np.ndarray
    mixed_observation_deg_s: np.ndarray
    case_name: str
    config: BenchmarkCaseConfig


def _minimum_jerk_velocity(time: np.ndarray, start: float, duration: float,
                           displacement: float) -> np.ndarray:
    phase = (time - start) / duration
    inside = (phase >= 0) & (phase <= 1)
    velocity = np.zeros_like(time)
    u = phase[inside]
    velocity[inside] = displacement / duration * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4)
    return velocity


def generate_case(case_name: str, tremor_deg_s: np.ndarray,
                  config: BenchmarkCaseConfig = BenchmarkCaseConfig()) -> SignalComponents:
    """Add a designed voluntary velocity to a supplied tremor waveform.

    ``tremor_deg_s`` can be controlled synthetic ground truth or a separately
    documented recorded-tremor proxy. This function does not infer tremor from
    raw patient motion, and never passes ground truth to an estimator.
    """
    if case_name not in {"slow_smooth", "point_to_point", "fast_transition", "spectral_overlap"}:
        raise ValueError(f"Unknown benchmark case: {case_name}")
    if not math.isfinite(config.sample_rate_hz) or config.sample_rate_hz <= 2:
        raise ValueError("sample_rate_hz must be > 2")
    if not math.isfinite(config.duration_s) or config.duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not math.isfinite(config.tremor_scale) or config.tremor_scale < 0:
        raise ValueError("tremor_scale must be nonnegative")
    if not math.isfinite(config.noise_std_deg_s) or config.noise_std_deg_s < 0:
        raise ValueError("noise_std_deg_s must be nonnegative")
    tremor = np.asarray(tremor_deg_s, dtype=float)
    count = round(config.duration_s * config.sample_rate_hz)
    if tremor.ndim != 2 or tremor.shape[0] != count or tremor.shape[1] < 1:
        raise ValueError(f"tremor must have shape ({count}, n_axes>=1)")
    if not np.all(np.isfinite(tremor)):
        raise ValueError("tremor must be finite")
    time = np.arange(count) / config.sample_rate_hz
    if case_name == "slow_smooth":
        scalar = config.smooth_amplitude_deg_s * np.sin(2 * np.pi * config.smooth_frequency_hz * time)
    elif case_name == "point_to_point":
        scalar = _minimum_jerk_velocity(time, config.point_start_s,
                                        config.point_duration_s,
                                        config.point_displacement_deg)
    elif case_name == "fast_transition":
        scalar = _minimum_jerk_velocity(time, config.fast_start_s,
                                        config.fast_duration_s,
                                        config.fast_displacement_deg)
    else:
        scalar = (config.smooth_amplitude_deg_s * np.sin(2 * np.pi * config.smooth_frequency_hz * time)
                  + config.overlap_amplitude_deg_s * np.sin(2 * np.pi * config.overlap_frequency_hz * time))
    voluntary = np.repeat(scalar[:, None], tremor.shape[1], axis=1)
    rng = np.random.default_rng(config.random_seed)
    noise = rng.normal(0, config.noise_std_deg_s, size=tremor.shape)
    mixed = voluntary + config.tremor_scale * tremor + noise
    return SignalComponents(time, voluntary, config.tremor_scale * tremor,
                            noise, mixed, case_name, config)


def controlled_tremor(case_name: str, sample_rate_hz: float, duration_s: float,
                      axes: int = 1, amplitude_deg_s: float = 5.0):
    """Return controlled tremor and known frequency for algorithm checks.

    The multifrequency case has no single true instantaneous frequency.
    """
    if case_name not in {"constant", "drift", "step", "amplitude_modulated", "multifrequency"}:
        raise ValueError(f"Unknown verification case: {case_name}")
    if axes < 1 or sample_rate_hz <= 24 or duration_s <= 0 or amplitude_deg_s < 0:
        raise ValueError("Invalid controlled-signal configuration")
    time = np.arange(round(sample_rate_hz * duration_s)) / sample_rate_hz
    if case_name == "constant":
        frequency = np.full(len(time), 6.0)
    elif case_name == "drift":
        frequency = 4.0 + 4.0 * time / duration_s
    elif case_name == "step":
        frequency = np.where(time < duration_s / 2, 5.0, 8.0)
    elif case_name == "amplitude_modulated":
        frequency = np.full(len(time), 6.0)
    else:
        frequency = None
    if frequency is None:
        scalar = amplitude_deg_s * (np.sin(2 * np.pi * 5.0 * time)
                                    + 0.7 * np.sin(2 * np.pi * 7.0 * time))
    else:
        phase = 2 * np.pi * np.cumsum(frequency) / sample_rate_hz
        amplitude = (amplitude_deg_s * (1 + 0.4 * np.sin(2 * np.pi * 0.3 * time))
                     if case_name == "amplitude_modulated" else amplitude_deg_s)
        scalar = amplitude * np.sin(phase)
    return time, np.repeat(scalar[:, None], axes, axis=1), frequency
