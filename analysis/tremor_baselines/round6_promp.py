"""Single-axis population ProMP with oracle activity phase (Round 6 only)."""

from __future__ import annotations

import math
import numpy as np


def phase_basis(phase, count: int):
    """Normalized Gaussian RBF; width equals adjacent-center spacing."""
    if count not in (4, 6, 8):
        raise ValueError("Round-6 basis count must be 4, 6, or 8")
    s=np.asarray(phase,dtype=float)
    if np.any(~np.isfinite(s)) or np.any((s<0)|(s>1)):
        raise ValueError("phase must be in [0,1]")
    centers=np.linspace(0.0,1.0,count)
    width=1.0/(count-1)
    raw=np.exp(-0.5*((s[...,None]-centers)/width)**2)
    return raw/raw.sum(axis=-1,keepdims=True)


def ridge_weights(y, basis, ridge: float):
    y=np.asarray(y,dtype=float)
    phi=np.asarray(basis,dtype=float)
    if ridge<=0 or phi.ndim!=2 or len(y)!=len(phi):
        raise ValueError("invalid ridge fit")
    return np.linalg.solve(phi.T@phi+ridge*np.eye(phi.shape[1]),phi.T@y)


def fit_population(weights, residuals, shrinkage: float=0.5):
    """Diagonal shrinkage with fixed strength, plus tiny numerical jitter."""
    w=np.asarray(weights,dtype=float)
    if w.ndim!=2 or len(w)<3 or not 0<=shrinkage<=1:
        raise ValueError("invalid population weights")
    mean=w.mean(axis=0)
    sample=np.cov(w,rowvar=False,ddof=1)
    cov=(1-shrinkage)*sample+shrinkage*np.diag(np.diag(sample))
    floor=max(float(np.trace(sample)/w.shape[1])*1e-8,1e-8)
    cov+=floor*np.eye(w.shape[1])
    residual=np.concatenate([np.asarray(r,dtype=float) for r in residuals])
    obs_var=max(float(np.mean(residual**2)),1e-6)
    return mean,cov,obs_var


def distance_squared(weight,mean,cov):
    delta=np.asarray(weight)-mean
    return float(delta@np.linalg.solve(cov,delta))


class CausalProMP:
    """Prior prediction, then posterior conditioning on the current sample."""

    def __init__(self,mean,cov,observation_variance):
        self.prior_mean=np.asarray(mean,dtype=float).copy()
        self.prior_cov=np.asarray(cov,dtype=float).copy()
        self.r=float(observation_variance)
        if self.r<=0 or self.prior_cov.shape!=(len(self.prior_mean),len(self.prior_mean)):
            raise ValueError("invalid ProMP distribution")
        self.reset()

    def reset(self):
        self.mean=self.prior_mean.copy()
        self.cov=self.prior_cov.copy()

    def predict(self,basis_row):
        phi=np.asarray(basis_row,dtype=float)
        return float(phi@self.mean),float(phi@self.cov@phi+self.r)

    def update(self,basis_row,observation):
        phi=np.asarray(basis_row,dtype=float)
        predicted,variance=self.predict(phi)
        gain=self.cov@phi/variance
        self.mean+=gain*(observation-predicted)
        self.cov-=np.outer(gain,phi@self.cov)
        self.cov=(self.cov+self.cov.T)/2
        innovation=(observation-predicted)/math.sqrt(variance)
        nll=0.5*(math.log(2*math.pi*variance)+innovation**2)
        return predicted,variance,float(innovation),float(nll)
