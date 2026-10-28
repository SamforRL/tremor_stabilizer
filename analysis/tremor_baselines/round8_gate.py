"""Causal update-only gates for the unchanged raw BMFLC Fourier output."""

from __future__ import annotations

from collections import deque
import math
import numpy as np

from .round7_prefilter import CausalSOS,design_prefilter


class GatedHuberBMFLC:
    """Raw Huber-BMFLC recursion multiplied by an independent update gate.

The Fourier output is computed from the raw input before any state update.
The diagnostic BPF acts only on the residual and never enters that output.
    """

    def __init__(self,sample_rate_hz,scale_deg_s,mode="band",window_s=.5,
                 persistence_windows=3,frequency_sigma_hz=1.,minimum_gate=.05,
                 analysis_band_threshold=.2,frequency_update_s=.1):
        fs=float(sample_rate_hz)
        if mode not in ("band","persistence") or not (fs>24 and scale_deg_s>0):
            raise ValueError("invalid gated estimator")
        if not (0<window_s<=1 and 2<=persistence_windows<=4 and
                frequency_sigma_hz>0 and 0<=minimum_gate<1 and
                0<=analysis_band_threshold<=1 and frequency_update_s>0):
            raise ValueError("invalid gate parameters")
        self.fs=fs;self.scale=float(scale_deg_s);self.mode=mode
        self.window_n=max(4,round(window_s*fs))
        self.m_p=int(persistence_windows);self.sigma_f=float(frequency_sigma_hz)
        self.gmin=float(minimum_gate);self.band_threshold=float(analysis_band_threshold)
        self.freq_stride=max(1,round(frequency_update_s*fs))
        self.frequencies=np.linspace(3.,12.,91)
        self.mu=.01;self.delta=.5*self.scale
        self.reset()

    def reset(self):
        self._index=0;self._weights=None;self._filters=None
        self._raw_window=None;self._band_window=None;self._frequency_history=None
        self._last_frequency=None;self._last_band_energy=0.;self._persistence=0.

    def _init_axes(self,n):
        self._weights=np.zeros((n,2*len(self.frequencies)+1))
        self._filters=[CausalSOS(design_prefilter("bpf",self.fs)) for _ in range(n)]
        self._raw_window=[deque(maxlen=self.window_n) for _ in range(n)]
        self._band_window=[deque(maxlen=self.window_n) for _ in range(n)]
        self._frequency_history=[deque(maxlen=self.m_p) for _ in range(n)]
        self._last_frequency=np.full(n,np.nan)
        self._last_band_energy=np.zeros(n)
        self._persistence=np.zeros(n)

    def _frequency(self,values):
        x=np.asarray(values,dtype=float)
        nfft=max(256,2**math.ceil(math.log2(4*self.fs)))
        spectrum=np.fft.rfft((x-x.mean())*np.hanning(len(x)),n=nfft)
        f=np.fft.rfftfreq(nfft,1/self.fs)
        band=(f>=3)&(f<=12)
        power=np.abs(spectrum[band])**2
        return float(f[band][np.argmax(power)]),float(np.sum(power))

    def step(self,sample):
        x=np.asarray(sample,dtype=float)
        if x.ndim!=1 or len(x)==0 or not np.all(np.isfinite(x)):
            raise ValueError("finite one-dimensional sample required")
        if self._weights is None:self._init_axes(len(x))
        elif len(x)!=len(self._weights):raise ValueError("axis count changed")
        phase=2*np.pi*self.frequencies*self._index/self.fs
        fourier=np.r_[np.sin(phase),np.cos(phase)]
        basis=np.r_[fourier,1.]
        output=self._weights[:,:-1]@fourier
        bias=self._weights[:,-1].copy()
        residual=x-output-bias
        robust=np.clip(residual,-self.delta,self.delta)
        gates=np.ones(len(x));band_ratios=np.ones(len(x));persistence=np.ones(len(x))
        dominant=np.full(len(x),np.nan);band_energy=np.zeros(len(x))
        for axis,value in enumerate(residual):
            filtered=self._filters[axis].step(np.array([value]))[0]
            self._raw_window[axis].append(float(value))
            self._band_window[axis].append(float(filtered))
            if len(self._raw_window[axis])<self.window_n:
                continue  # explicit causal warm-up: full Huber update
            total=float(np.mean(np.square(self._raw_window[axis])))
            inband=float(np.mean(np.square(self._band_window[axis])))
            ratio=float(np.clip(inband/(total+(0.01*self.scale)**2),0.,1.))
            band_ratios[axis]=ratio
            if self.mode=="persistence":
                if ratio<self.band_threshold:
                    self._frequency_history[axis].clear()
                    self._last_frequency[axis]=np.nan
                    self._last_band_energy[axis]=0.
                    self._persistence[axis]=0.
                elif self._index%self.freq_stride==0:
                    freq,energy=self._frequency(self._raw_window[axis])
                    self._frequency_history[axis].append(freq)
                    self._last_frequency[axis]=freq
                    self._last_band_energy[axis]=energy
                    if len(self._frequency_history[axis])==self.m_p:
                        std=float(np.std(self._frequency_history[axis]))
                        self._persistence[axis]=math.exp(-std*std/(2*self.sigma_f**2))
                    else:self._persistence[axis]=0.
                persistence[axis]=self._persistence[axis]
                dominant[axis]=self._last_frequency[axis]
                band_energy[axis]=self._last_band_energy[axis]
            gates[axis]=self.gmin+(1-self.gmin)*ratio*persistence[axis]
        self._weights+=2*self.mu*(gates*robust)[:,None]*basis[None,:]
        self._index+=1
        return output,{"internal_error":residual.copy(),"robust_update_error":robust.copy(),
                       "intended_motion_bias":bias,"adaptive_weights":self._weights.copy(),
                       "gate":gates,"band_ratio":band_ratios,"persistence_score":persistence,
                       "dominant_frequency_hz":dominant,"dominant_band_energy":band_energy,
                       "warmup":np.array([len(w)<self.window_n for w in self._raw_window])}
