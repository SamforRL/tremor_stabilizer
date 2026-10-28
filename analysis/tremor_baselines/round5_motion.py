"""Causal single-axis motion KFs and exact Singer discretization for Round 5."""

from __future__ import annotations

import math
import numpy as np

from .round4_estimators import jerk_covariance


def motion_matrices(model: str, fs: float, process_scale: float,
                    correlation_time_s: float | None = None):
    """Return F,Q for physical motion model parameters.

    RW: process_scale is stepwise angular-velocity SD (deg/s).
    CA: process_scale is discrete piecewise-constant jerk SD (deg/s^3).
    Singer: process_scale is stationary acceleration SD (deg/s^2), and
    alpha follows a continuous Ornstein-Uhlenbeck process with time constant
    tau. F,Q are its exact sampled moments.
    """
    if fs <= 0 or process_scale < 0 or not math.isfinite(process_scale):
        raise ValueError("Invalid process model parameters")
    t = 1 / fs
    if model == "random_walk":
        return np.array([[1.0]]), np.array([[process_scale**2]])
    if model == "constant_acceleration":
        return np.array([[1.0,t],[0,1.0]]), jerk_covariance(fs,process_scale)
    if model != "singer" or correlation_time_s is None or correlation_time_s <= 0:
        raise ValueError("Unknown model or invalid Singer correlation time")
    beta = 1 / correlation_time_s
    a = math.exp(-beta*t)
    a2 = a*a
    f = np.array([[1.0,(1-a)/beta],[0,a]])
    # da=-beta*a dt+sqrt(qc)dW, qc=2*beta*sigma_a^2.
    qc = 2*beta*process_scale**2
    q11 = qc/beta**2*(t-2*(1-a)/beta+(1-a2)/(2*beta))
    q12 = qc/(2*beta**2)*(1-2*a+a2)
    q22 = process_scale**2*(1-a2)
    q = np.array([[q11,q12],[q12,q22]])
    return f,q


class MotionKF:
    """Kalman filter with unfiltered gyro observation y=[1,0]x+noise."""

    def __init__(self, model: str, fs: float, process_scale: float,
                 measurement_sd_deg_s: float, correlation_time_s: float | None = None,
                 initial_velocity_sd_deg_s: float = 50.0,
                 initial_acceleration_sd_deg_s2: float = 500.0):
        if measurement_sd_deg_s <= 0:
            raise ValueError("Measurement SD must be positive")
        self.model=model;self.fs=fs
        self.f,self.q=motion_matrices(model,fs,process_scale,correlation_time_s)
        self.r=measurement_sd_deg_s**2
        self.initial_p=np.diag([initial_velocity_sd_deg_s**2]
                               +([initial_acceleration_sd_deg_s2**2] if len(self.f)==2 else []))
        self.reset()

    def reset(self):
        self.x=np.zeros(len(self.f));self.p=self.initial_p.copy()

    def step(self, observation: float):
        self.x=self.f@self.x
        self.p=self.f@self.p@self.f.T+self.q
        innovation=float(observation-self.x[0])
        s=float(self.p[0,0]+self.r)
        gain=self.p[:,0]/s
        self.x+=gain*innovation
        self.p-=np.outer(gain,self.p[0].copy())
        return float(self.x[0]),innovation,gain.copy(),self.p.diagonal().copy()

    def forecast(self,horizon_samples: int):
        if horizon_samples < 1:
            raise ValueError("Horizon must be >=1")
        return float((np.linalg.matrix_power(self.f,horizon_samples)@self.x)[0])


def prediction_trace(values: np.ndarray, method: MotionKF,
                     horizons: tuple[int,...] = (1,5,12)):
    """At k use posterior x_k to forecast y_{k+h}; never read future y."""
    y=np.asarray(values,dtype=float)
    if y.ndim!=1 or len(y)<max(horizons)+2 or not np.all(np.isfinite(y)):
        raise ValueError("Expected finite one-axis sequence")
    method.reset()
    forecasts={h:np.full(len(y),np.nan) for h in horizons}
    forecast_rows={h:np.linalg.matrix_power(method.f,h)[0] for h in horizons}
    innovations=np.empty(len(y))
    filtered=np.empty(len(y))
    acceleration=np.empty(len(y))
    for k,value in enumerate(y):
        filtered[k],innovations[k],_,_=method.step(float(value))
        acceleration[k]=method.x[1] if len(method.x)>1 else 0.0
        for h in horizons:
            if k+h<len(y):forecasts[h][k+h]=forecast_rows[h]@method.x
    return filtered,innovations,forecasts,acceleration
