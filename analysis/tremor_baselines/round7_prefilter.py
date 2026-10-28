"""Causal second-order HPF or fixed BPF before unchanged BMFLC recursion."""

from __future__ import annotations

import math
import numpy as np

from .butterworth import butterworth_bandpass_sos,sos_response
from .interface import create_estimator
from .round3_variants import RobustBMFLC


def highpass2_sos(cutoff_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Bilinear/Tustin transform of analog 2-pole Butterworth high-pass."""
    if not (math.isfinite(cutoff_hz) and math.isfinite(sample_rate_hz)
            and 0<cutoff_hz<sample_rate_hz/2):
        raise ValueError("invalid high-pass design")
    k=math.tan(math.pi*cutoff_hz/sample_rate_hz)
    norm=1/(1+math.sqrt(2)*k+k*k)
    a1=2*(k*k-1)*norm
    a2=(1-math.sqrt(2)*k+k*k)*norm
    return np.array([[norm,-2*norm,norm,1.,a1,a2]])


class CausalSOS:
    def __init__(self,sos):
        self.sos=np.asarray(sos,dtype=float)
        if self.sos.ndim!=2 or self.sos.shape[1]!=6:
            raise ValueError("invalid SOS")
        self.reset()

    def reset(self):self._state=None

    def step(self,sample):
        x=np.asarray(sample,dtype=float)
        if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("finite one-dimensional sample required")
        if self._state is None:self._state=np.zeros((len(self.sos),len(x),2))
        if self._state.shape[1]!=len(x):raise ValueError("axis count changed")
        y=x.copy()
        for i,(b0,b1,b2,_,a1,a2) in enumerate(self.sos):
            d1,d2=self._state[i,:,0].copy(),self._state[i,:,1].copy()
            z=b0*y+d1
            self._state[i,:,0]=b1*y-a1*z+d2
            self._state[i,:,1]=b2*y-a2*z
            y=z
        return y


def design_prefilter(kind,fs,cutoff_hz=2.0):
    if kind=="none":return None
    if kind=="hpf":return highpass2_sos(cutoff_hz,fs)
    if kind=="bpf":return butterworth_bandpass_sos(2,3.,12.,fs)
    raise ValueError("unknown prefilter")


class PrefilteredBMFLC:
    """BMFLC reads only the causally filtered value z[k]."""

    def __init__(self,fs,kind="none",cutoff_hz=2.,update="lms",huber_delta=None):
        self.fs=float(fs);self.kind=kind;self.cutoff_hz=float(cutoff_hz)
        self.sos=design_prefilter(kind,fs,cutoff_hz)
        self.prefilter=CausalSOS(self.sos) if self.sos is not None else None
        bm=dict(sample_rate_hz=fs,low_hz=3.,high_hz=12.,spacing_hz=.1,lms_gain=.01)
        if update=="lms":self.bmflc=create_estimator("bmflc",bm)
        elif update=="huber":self.bmflc=RobustBMFLC(**bm,variant="huber",huber_delta=huber_delta)
        else:raise ValueError("update must be lms/huber")
        self.update=update

    def reset(self):
        if self.prefilter is not None:self.prefilter.reset()
        self.bmflc.reset()

    def step(self,sample):
        x=np.asarray(sample,dtype=float)
        z=x if self.prefilter is None else self.prefilter.step(x)
        estimate,diagnostics=self.bmflc.step(z)
        return estimate,{**diagnostics,"raw_input":x.copy(),"prefiltered_input":z.copy()}


def filter_trace(values,sos):
    x=np.asarray(values,dtype=float)
    if sos is None:return x.copy()
    filt=CausalSOS(sos)
    return np.array([filt.step(np.atleast_1d(value))[0] for value in x])


def response_metrics(sos,fs):
    if sos is None:
        return dict(mean_band_magnitude_error=0.,sample_frequencies_hz=[4,6,8,10],
                    magnitude=[1.]*4,phase_rad=[0.]*4,phase_delay_ms=[0.]*4,
                    group_delay_ms=[0.]*4)
    f=np.linspace(.05,fs/2-.05,5000)
    h=sos_response(sos,f,fs)
    mag=abs(h);phase=np.unwrap(np.angle(h))
    gd=-np.gradient(phase,2*np.pi*f/fs)/fs*1000
    sample=np.array([4.,6.,8.,10.])
    hs=sos_response(sos,sample,fs)
    phases=np.angle(hs)
    return dict(mean_band_magnitude_error=float(np.mean(abs(mag[(f>=3)&(f<=12)]-1))),
                sample_frequencies_hz=sample.tolist(),magnitude=abs(hs).tolist(),
                phase_rad=phases.tolist(),phase_delay_ms=(-phases/(2*np.pi*sample)*1000).tolist(),
                group_delay_ms=np.interp(sample,f,gd).tolist())
