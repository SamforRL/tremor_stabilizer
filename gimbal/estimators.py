"""Public causal estimator API. No MuJoCo or datasets required for streaming.

Native literature implementations remain in analysis.tremor_baselines.
Outputs use the pre-update prediction. Fourier diagnostics exclude bias and
use the selected signal unit, with [all sin weights, all cos weights].
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from analysis.tremor_baselines.abpf import ABPF
from analysis.tremor_baselines.bmflc_core import BMFLC
from analysis.tremor_baselines.butterworth import FixedButterworthBPF
from analysis.tremor_baselines.damped_bmflc import DampedBMFLC
from analysis.tremor_baselines.round3_variants import RobustBMFLC

METHODS = ('FixedBPF', 'ABPF', 'OriginalBMFLC', 'HuberBMFLC', 'DampedBMFLC')


@dataclass(frozen=True)
class EstimatorConfig:
    sample_rate_hz: float
    axes: int = 2
    unit: str = 'deg/s'
    low_hz: float = 3.
    high_hz: float = 12.
    spacing_hz: float = .1
    bpf_order: int = 4
    lms_gain: float = .01
    modal_frequency_hz: tuple = ()
    frequency_step_hz: float = .1
    limit_offset_hz: float = 1.5
    huber_delta_deg_s: tuple = ()
    damped_eta: float = .02
    damped_lambda_ref: float = .99999
    damped_reference_fs_hz: float = 1000.
    damped_kdmp: float = 350.
    damped_xdmp_rad_s: float = .009

    def __post_init__(self):
        if self.unit not in ('deg/s', 'rad/s'):
            raise ValueError('unit must be deg/s or rad/s')
        if not isinstance(self.axes, int) or isinstance(self.axes, bool) or self.axes < 1:
            raise ValueError('axes must be a positive integer')
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 2*self.high_hz:
            raise ValueError('sample rate must exceed twice the upper cutoff')
        for name in ('modal_frequency_hz', 'huber_delta_deg_s'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @classmethod
    def reference(cls, unit='deg/s'):
        """Frozen two-axis demonstration calibration; not a patient default."""
        p=Path(__file__).resolve().parents[1]/'delivery/config.json'
        d=json.loads(p.read_text(encoding='utf-8'))
        return cls(**{k:(unit if k=='unit' else d[k]) for k in cls.__dataclass_fields__})


class TremorEstimator:
    def __init__(self, method, config):
        if method not in METHODS:raise ValueError(f'Unknown method: {method}; choose {METHODS}')
        self.method=method;self.config=config;c=config
        if method=='FixedBPF':
            self._core=FixedButterworthBPF(c.sample_rate_hz,c.bpf_order,c.low_hz,c.high_hz)
        elif method=='ABPF':
            if len(c.modal_frequency_hz)!=c.axes:
                raise ValueError('ABPF requires one independently calibrated modal frequency per axis')
            self._core=ABPF(c.sample_rate_hz,c.modal_frequency_hz,c.frequency_step_hz,c.limit_offset_hz)
        elif method=='OriginalBMFLC':
            self._core=BMFLC(c.sample_rate_hz,c.low_hz,c.high_hz,c.spacing_hz,c.lms_gain)
        elif method=='HuberBMFLC':
            if len(c.huber_delta_deg_s)!=c.axes:
                raise ValueError('HuberBMFLC requires one positive calibration threshold per axis in deg/s')
            self._core=[RobustBMFLC(c.sample_rate_hz,c.low_hz,c.high_hz,c.spacing_hz,c.lms_gain,
                                    variant='huber',huber_delta=d) for d in c.huber_delta_deg_s]
        else:
            self._core=DampedBMFLC(c.sample_rate_hz,c.low_hz,c.high_hz,c.spacing_hz,
                                  eta=c.damped_eta,lambda_ref=c.damped_lambda_ref,
                                  reference_sample_rate_hz=c.damped_reference_fs_hz,
                                  kdmp=c.damped_kdmp,xdmp_rad_s=c.damped_xdmp_rad_s)
        self.reset()

    def reset(self):
        for core in (self._core if isinstance(self._core,list) else [self._core]):core.reset()
        self._index=0

    @property
    def configuration(self):
        return {'method':self.method,**asdict(self.config)}

    def step(self, sample):
        x=np.asarray(sample,dtype=float)
        if x.shape!=(self.config.axes,) or not np.isfinite(x).all():
            raise ValueError(f'Expected {self.config.axes} finite axis samples, shape ({self.config.axes},)')
        deg=np.rad2deg(x) if self.config.unit=='rad/s' else x
        if isinstance(self._core,list):
            pairs=[core.step(deg[i:i+1]) for i,core in enumerate(self._core)]
            y=np.concatenate([p[0] for p in pairs])
            diag={k:np.concatenate([p[1][k] for p in pairs],axis=0) for k in pairs[0][1]}
        else:y,diag=self._core.step(deg)
        diag={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in diag.items()}
        diag.setdefault('internal_error',deg-y)
        if 'adaptive_weights' in diag:
            if self.method in ('OriginalBMFLC','HuberBMFLC'):
                diag['adaptive_weights']=diag['adaptive_weights'][:,:-1]
            if self.method=='DampedBMFLC':
                diag['adaptive_weights']=np.rad2deg(diag['adaptive_weights'])
            diag['adaptive_weights_unit']=self.config.unit
        if self.config.unit=='rad/s':
            y=np.deg2rad(y)
            for k in ('internal_error','robust_update_error','intended_motion_bias','adaptive_weights'):
                if k in diag:diag[k]=np.deg2rad(diag[k])
        diag['signal_unit']=self.config.unit;diag['step_index']=self._index
        self._index+=1
        return np.asarray(y).copy(),diag


def create_estimator(method='ABPF', config=None):
    """Use EstimatorConfig for your own Fs, axes, units and calibration."""
    return TremorEstimator(method,EstimatorConfig.reference() if config is None else config)
