"""Common sample-by-sample interface for causal tremor estimators."""

from typing import Protocol

import numpy as np


class TremorEstimator(Protocol):
    """Input and output are one angular-velocity sample, in deg/s per axis."""

    def reset(self) -> None:
        ...

    def step(self, sample: np.ndarray) -> tuple[np.ndarray, dict]:
        ...


def create_estimator(method: str, config: dict) -> TremorEstimator:
    """Create only implementations whose equations have been verified."""
    if method == "fixed_bpf":
        from .butterworth import FixedButterworthBPF

        return FixedButterworthBPF(**config)
    if method == "bmflc":
        from .bmflc_core import BMFLC

        return BMFLC(**config)
    if method == "abpf":
        from .abpf import ABPF

        return ABPF(**config)
    if method == "sfft":
        from .sfft import AdaptiveSFFT

        return AdaptiveSFFT(**config)
    raise ValueError(f"Unknown estimator method: {method}")
