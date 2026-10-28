"""Pinheiro gyroscope loader and explicitly offline tremor-band proxy.

This source-construction step may inspect context on both sides of a window.
It never belongs inside a causal estimator. The result is a measured-motion
frequency-band proxy, not an independently labelled physiological tremor.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RecordedWindow:
    time_s: np.ndarray
    tremor_band_proxy_deg_s: np.ndarray
    sample_rate_hz: float
    source_file: str
    source_sha256: str
    table: str
    channels: tuple[str, ...]
    start_s: float
    duration_s: float
    band_hz: tuple[float, float]
    context_s: float


def load_pinheiro_window(table: str = "p02_2", start_s: float = 2.0,
                         duration_s: float = 12.0,
                         band_hz: tuple[float, float] = (3.0, 12.0),
                         context_s: float = 1.0,
                         channel_indices: tuple[int, ...] = (0,)) -> RecordedWindow:
    """Extract an offline gyro-band proxy, then sample it on a uniform grid.

    The 3–12 Hz choice follows the existing project stress-test band and the
    user's explicit approval for this proxy. It is not a biological label.
    """
    if not re.fullmatch(r"p0[1-5]_[1-6]", table):
        raise ValueError("Invalid Pinheiro participant table name")
    path = ROOT / f"data/processed/pinheiro_{table}_lumbrical_gyro.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Extract Pinheiro table before benchmarking: {path}")
    with np.load(path) as file:
        time = np.asarray(file["time_s"], dtype=float)
        gyro = np.asarray(file["gyro_deg_s"], dtype=float)
        channels = tuple(str(x) for x in file["channels"])
        declared_table = str(file["table"])
    if declared_table != table or gyro.shape != (len(time), 3):
        raise ValueError("Unexpected processed gyro layout")
    if not np.all(np.isfinite(gyro)) or not np.all(np.diff(time) > 0):
        raise ValueError("Gyroscope data or timestamps are invalid")
    if not channel_indices or any(index not in (0, 1, 2) for index in channel_indices):
        raise ValueError("channel_indices must select one or more of X/Y/Z")
    sample_rate_hz = 1 / float(np.median(np.diff(time)))
    if not 0 < band_hz[0] < band_hz[1] < sample_rate_hz / 2:
        raise ValueError("Invalid tremor proxy band")
    if context_s <= 0 or start_s - context_s < 0 or start_s + duration_s + context_s > time[-1]:
        raise ValueError("Requested window lacks full offline context")
    context = (time >= start_s - context_s) & (time < start_s + duration_s + context_s)
    context_time = time[context]
    values = gyro[context][:, channel_indices]
    frequencies = np.fft.rfftfreq(len(context_time), 1 / sample_rate_hz)
    spectrum = np.fft.rfft(values, axis=0)
    spectrum[(frequencies < band_hz[0]) | (frequencies > band_hz[1])] = 0
    filtered = np.fft.irfft(spectrum, n=len(context_time), axis=0)
    output_time = np.arange(round(duration_s * sample_rate_hz)) / sample_rate_hz
    absolute_time = start_s + output_time
    proxy = np.column_stack([np.interp(absolute_time, context_time, filtered[:, axis])
                             for axis in range(len(channel_indices))])
    return RecordedWindow(
        output_time, proxy, sample_rate_hz, str(path.relative_to(ROOT)),
        hashlib.sha256(path.read_bytes()).hexdigest(), table,
        tuple(channels[index] for index in channel_indices),
        start_s, duration_s, band_hz, context_s,
    )
