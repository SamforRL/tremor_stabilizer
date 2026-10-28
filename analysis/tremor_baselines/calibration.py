"""Independent modal-frequency calibration for paper-like ABPF initialization."""

from dataclasses import dataclass
import re

import numpy as np

from .recorded import ROOT


@dataclass(frozen=True)
class ModalCalibration:
    modal_frequency_hz: np.ndarray
    start_s: float
    duration_s: float
    band_hz: tuple[float, float]
    method: str


def modal_frequency_from_segment(signal: np.ndarray, sample_rate_hz: float,
                                 band_hz: tuple[float, float] = (3.0, 12.0)) -> np.ndarray:
    """Hann-windowed full-segment periodogram peak, one frequency per axis."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2 or len(x) < 4 or not np.all(np.isfinite(x)):
        raise ValueError("calibration signal must be a finite (samples, axes) array")
    if not 0 < band_hz[0] < band_hz[1] < sample_rate_hz / 2:
        raise ValueError("Invalid modal-frequency search band")
    frequencies = np.fft.rfftfreq(len(x), 1 / sample_rate_hz)
    power = abs(np.fft.rfft((x - np.mean(x, axis=0))
                              * np.hanning(len(x))[:, None], axis=0)) ** 2
    selected = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    return frequencies[selected][np.argmax(power[selected], axis=0)]


def calibrate_pinheiro_modal(table: str, channel_indices: tuple[int, ...],
                             evaluation_start_s: float, evaluation_duration_s: float,
                             calibration_start_s: float = 0.0,
                             preferred_duration_s: float = 15.0,
                             band_hz: tuple[float, float] = (3.0, 12.0)) -> ModalCalibration:
    """Read a strictly earlier, nonoverlapping raw-gyro segment, up to 15 s."""
    if not re.fullmatch(r"p0[1-5]_[1-6]", table):
        raise ValueError("Invalid Pinheiro participant table name")
    if not channel_indices or any(axis not in (0, 1, 2) for axis in channel_indices):
        raise ValueError("Invalid gyro axis")
    if calibration_start_s < 0 or preferred_duration_s <= 0 or evaluation_duration_s <= 0:
        raise ValueError("Invalid calibration/evaluation timing")
    path = ROOT / f"data/processed/pinheiro_{table}_lumbrical_gyro.npz"
    with np.load(path) as recording:
        time = np.asarray(recording["time_s"], dtype=float)
        gyro = np.asarray(recording["gyro_deg_s"], dtype=float)
    if gyro.shape != (len(time), 3) or not np.all(np.diff(time) > 0):
        raise ValueError("Invalid processed gyro recording")
    duration = min(preferred_duration_s, evaluation_start_s - calibration_start_s,
                   time[-1] - calibration_start_s)
    if duration < 1.0 or calibration_start_s + duration > evaluation_start_s:
        raise ValueError("No independent preceding calibration segment")
    fs = 1 / np.median(np.diff(time))
    samples = round(duration * fs)
    uniform_time = calibration_start_s + np.arange(samples) / fs
    selected = np.column_stack([np.interp(uniform_time, time, gyro[:, axis])
                                for axis in channel_indices])
    return ModalCalibration(modal_frequency_from_segment(selected, fs, band_hz),
                            calibration_start_s, samples / fs, band_hz,
                            "Hann full-segment periodogram, raw gyro deg/s")
