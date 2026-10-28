"""Causal gyro-only kinematic/BMFLC decompositions for Round 4.

All state variables use deg/s and deg/s^2.  A jerk value is assumed constant
over each sample interval, independently across intervals.  For
G = [T^2/2, T]^T, its scalar variance sigma_j^2 gives Q_v = G sigma_j^2 G^T.
This is a discrete piecewise-constant white-jerk convention, not the
continuous white-noise spectral-density convention.
"""

from __future__ import annotations

import numpy as np

from .bmflc_core import BMFLC


def jerk_covariance(sample_rate_hz: float, sigma_j_deg_s3: float) -> np.ndarray:
    t = 1.0 / sample_rate_hz
    g = np.array([0.5 * t * t, t])
    return sigma_j_deg_s3 ** 2 * np.outer(g, g)


class _VoluntaryKF:
    def __init__(self, fs: float, sigma_j: float, r: float, p_v: float, p_a: float):
        self.fs, self.r = fs, r
        self.q = jerk_covariance(fs, sigma_j)
        self.initial_p = np.diag([p_v, p_a])
        self.reset()

    def reset(self):
        self.x = np.zeros(2)
        self.p = self.initial_p.copy()

    def step(self, y: float):
        t = 1 / self.fs
        self.x[0] += t * self.x[1]
        a = np.array([[1.0, t], [0.0, 1.0]])
        self.p = a @ self.p @ a.T + self.q
        innovation = y - self.x[0]
        gain = self.p[:, 0] / (self.p[0, 0] + self.r)
        self.x += gain * innovation
        self.p -= np.outer(gain, self.p[0].copy())
        return self.x.copy(), innovation, gain.copy(), self.p.diagonal().copy()


class KinematicKFThenBMFLC:
    """First estimate voluntary gyro, then pass residual to original LMS BMFLC."""

    def __init__(self, sample_rate_hz: float, sigma_j_deg_s3: float,
                 voluntary_measurement_variance: float,
                 initial_voluntary_variance: float,
                 initial_acceleration_variance: float,
                 low_hz: float = 3, high_hz: float = 12,
                 spacing_hz: float = 0.1, lms_gain: float = 0.01):
        self.sample_rate_hz = float(sample_rate_hz)
        self.args = (sigma_j_deg_s3, voluntary_measurement_variance,
                     initial_voluntary_variance, initial_acceleration_variance)
        self.bmflc = BMFLC(sample_rate_hz, low_hz, high_hz, spacing_hz, lms_gain)
        self.reset()

    def reset(self):
        self.kfs = None
        self.bmflc.reset()

    def step(self, sample: np.ndarray):
        y = np.asarray(sample, dtype=float)
        if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
            raise ValueError("Expected finite one-dimensional gyro sample")
        if self.kfs is None:
            self.kfs = [_VoluntaryKF(self.sample_rate_hz, *self.args) for _ in y]
        if len(self.kfs) != len(y):
            raise ValueError("Axis count changed")
        states = [kf.step(float(value)) for kf, value in zip(self.kfs, y)]
        voluntary = np.array([state[0][0] for state in states])
        tremor, info = self.bmflc.step(y - voluntary)
        return tremor, {**info, "voluntary_velocity_deg_s": voluntary,
                        "voluntary_acceleration_deg_s2": np.array([s[0][1] for s in states]),
                        "innovation_deg_s": np.array([s[1] for s in states]),
                        "kalman_gain_voluntary": np.array([s[2] for s in states]),
                        "voluntary_variance": np.array([s[3] for s in states])}


class JointKinematicBMFLCKF:
    """Linear time-varying Kalman filter, with a two-state or random-walk v prior.

    Basis order exactly matches existing BMFLC: all sine coefficients followed
    by all cosine coefficients.  Posterior estimates at sample k are output.
    No future sample, smoothing pass, robust loss, or voluntary detector is used.
    """

    def __init__(self, sample_rate_hz: float, sigma_j_deg_s3: float,
                 weight_process_variance: float, measurement_variance: float,
                 initial_voluntary_variance: float,
                 initial_acceleration_variance: float,
                 initial_weight_variance: float,
                 low_hz: float = 3, high_hz: float = 12,
                 spacing_hz: float = 0.1, voluntary_model: str = "constant_acceleration"):
        values = np.array([sample_rate_hz, sigma_j_deg_s3, weight_process_variance,
                           measurement_variance, initial_voluntary_variance,
                           initial_acceleration_variance, initial_weight_variance,
                           low_hz, high_hz, spacing_hz], dtype=float)
        if not np.all(np.isfinite(values)) or sample_rate_hz <= 2 * high_hz or not (0 < low_hz < high_hz):
            raise ValueError("Invalid sample rate, band, or nonfinite configuration")
        if min(weight_process_variance, sigma_j_deg_s3, initial_voluntary_variance,
               initial_acceleration_variance, initial_weight_variance) < 0 or measurement_variance <= 0:
            raise ValueError("Variances must be nonnegative and measurement variance positive")
        if voluntary_model not in {"constant_acceleration", "random_walk"}:
            raise ValueError("Unknown voluntary model")
        intervals = (high_hz - low_hz) / spacing_hz
        if abs(intervals - round(intervals)) > 1e-8:
            raise ValueError("Frequency spacing must divide band")
        self.fs = float(sample_rate_hz)
        self.frequencies_hz = np.linspace(low_hz, high_hz, round(intervals) + 1)
        self.model = voluntary_model
        self.nv = 2 if voluntary_model == "constant_acceleration" else 1
        self.nw = 2 * len(self.frequencies_hz)
        self.n = self.nv + self.nw
        self.r = float(measurement_variance)
        self.qw = float(weight_process_variance)
        self.qv = jerk_covariance(self.fs, sigma_j_deg_s3) if self.nv == 2 else np.array([[sigma_j_deg_s3 ** 2]])
        self.initial_p = np.asarray(np.r_[initial_voluntary_variance,
                               [initial_acceleration_variance] if self.nv == 2 else [],
                               np.full(self.nw, initial_weight_variance)], dtype=float)
        self.reset()

    def reset(self):
        self.index = 0
        self.states = None
        self.covariances = None

    def _basis(self):
        phase = 2 * np.pi * self.frequencies_hz * self.index / self.fs
        return np.r_[np.sin(phase), np.cos(phase)]

    def step(self, sample: np.ndarray):
        y = np.asarray(sample, dtype=float)
        if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
            raise ValueError("Expected finite one-dimensional gyro sample")
        if self.states is None:
            self.states = np.zeros((len(y), self.n))
            self.covariances = np.array([np.diag(self.initial_p) for _ in y])
        if len(self.states) != len(y):
            raise ValueError("Axis count changed")
        phi = self._basis()
        h = np.r_[1.0, np.zeros(self.nv - 1), phi]
        tremor = np.empty(len(y))
        voluntary = np.empty(len(y))
        innovation = np.empty(len(y))
        gain = np.empty((len(y), self.n))
        diag = np.empty((len(y), self.n))
        for axis, observation in enumerate(y):
            x = self.states[axis]
            p = self.covariances[axis]
            if self.nv == 2:
                t = 1 / self.fs
                x[0] += t * x[1]
                # In-place F P F^T for F[0,1]=T; retained symmetry.
                old_row = p[1].copy()
                p[0] += t * old_row
                p[:, 0] += t * p[:, 1]
                p[:2, :2] += self.qv
            else:
                p[0, 0] += self.qv[0, 0]
            p[self.nv:, self.nv:].flat[::self.nw + 1] += self.qw
            ph = p @ h
            s = float(h @ ph + self.r)
            k = ph / s
            e = float(observation - h @ x)
            x += k * e
            p -= np.outer(ph, k)
            tremor[axis] = phi @ x[self.nv:]
            voluntary[axis] = x[0]
            innovation[axis] = e
            gain[axis] = k
            diag[axis] = np.diag(p)
        self.index += 1
        return tremor, {"voluntary_velocity_deg_s": voluntary.copy(),
                        "voluntary_acceleration_deg_s2": self.states[:, 1].copy() if self.nv == 2 else np.zeros(len(y)),
                        "innovation_deg_s": innovation.copy(),
                        "kalman_gain_voluntary": gain[:, :self.nv].copy(),
                        "kalman_gain_tremor_norm": np.linalg.norm(gain[:, self.nv:], axis=1),
                        "posterior_voluntary_variance": diag[:, :self.nv].copy(),
                        "posterior_weight_variance_sum": np.sum(diag[:, self.nv:], axis=1),
                        "adaptive_weights": self.states[:, self.nv:].copy(),
                        "fourier_basis": phi.copy()}
