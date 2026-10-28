"""Shared signal-domain metrics; no estimator-specific scoring branches."""

import math
import time

import numpy as np

from .signals import SignalComponents


def run_stream(estimator, observations: np.ndarray):
    """Run one causal estimator with deterministic reset and per-step timing."""
    observed = np.asarray(observations, dtype=float)
    if observed.ndim != 2 or not np.all(np.isfinite(observed)):
        raise ValueError("observations must be a finite (samples, axes) array")
    estimator.reset()
    output = np.empty_like(observed)
    diagnostics = []
    elapsed_ns = np.empty(len(observed), dtype=np.int64)
    for index, sample in enumerate(observed):
        before = time.perf_counter_ns()
        estimate, info = estimator.step(sample.copy())
        elapsed_ns[index] = time.perf_counter_ns() - before
        output[index] = estimate
        if not np.all(np.isfinite(output[index])):
            raise ValueError(f"Estimator produced nonfinite output at sample {index}")
        diagnostics.append(info)
    return output, diagnostics, elapsed_ns


def welch_psd(signal: np.ndarray, sample_rate_hz: float, segment_samples: int):
    """One-sided Hann Welch PSD, 50% overlap, constant detrend per segment."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2 or segment_samples < 4 or len(x) < segment_samples:
        raise ValueError("signal must be (samples, axes) with at least one segment")
    step = segment_samples // 2
    window = np.hanning(segment_samples)
    scale = sample_rate_hz * np.sum(window ** 2)
    spectra = []
    for start in range(0, len(x) - segment_samples + 1, step):
        segment = x[start:start + segment_samples]
        transform = np.fft.rfft((segment - np.mean(segment, axis=0)) * window[:, None], axis=0)
        power = np.abs(transform) ** 2 / scale
        if segment_samples % 2 == 0:
            power[1:-1] *= 2
        else:
            power[1:] *= 2
        spectra.append(power)
    return np.fft.rfftfreq(segment_samples, 1 / sample_rate_hz), np.mean(spectra, axis=0)


def _band_power(frequencies: np.ndarray, psd: np.ndarray, band_hz: tuple[float, float]):
    selected = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if np.count_nonzero(selected) < 2:
        raise ValueError("Tremor band contains fewer than two PSD bins")
    spectrum = np.mean(psd[selected], axis=1)
    bins = frequencies[selected]
    return float(np.sum((spectrum[1:] + spectrum[:-1]) * np.diff(bins) / 2))


def diagnostic_lag_ms(true: np.ndarray, estimated: np.ndarray,
                      sample_rate_hz: float, max_lag_s: float = 0.5):
    """Peak-correlation lag for diagnostics; never realign causal RMSE."""
    actual = np.asarray(true, dtype=float)
    predicted = np.asarray(estimated, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("true and estimated signals must have equal 2D shapes")
    limit = min(round(max_lag_s * sample_rate_hz), len(actual) - 1)
    scores = []
    for lag in range(-limit, limit + 1):
        if lag >= 0:
            a, b = actual[:-lag or None], predicted[lag:]
        else:
            a, b = actual[-lag:], predicted[:lag]
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
        scores.append(float(np.sum(a * b) / denominator) if denominator > 0 else -math.inf)
    # Periodic tremor has several equally plausible correlation peaks a cycle
    # apart. Report the nearest lag among peaks within 2% of the maximum;
    # retaining the unrestricted maximum would often return the search edge.
    scores = np.asarray(scores)
    candidates = np.flatnonzero(scores >= np.max(scores) - 0.02)
    nearest = candidates[np.argmin(np.abs(candidates - limit))]
    return float((nearest - limit) * 1000 / sample_rate_hz)


def evaluate(components: SignalComponents, tremor_estimate: np.ndarray,
             elapsed_ns: np.ndarray, tremor_band_hz: tuple[float, float] = (3.0, 12.0),
             psd_segment_s: float = 2.0) -> dict:
    """Evaluate in deg/s. Causal scores use unshifted, uncompensated output."""
    estimate = np.asarray(tremor_estimate, dtype=float)
    true = components.tremor_true_deg_s
    mixed = components.mixed_observation_deg_s
    if estimate.shape != true.shape or not np.all(np.isfinite(estimate)):
        raise ValueError("Invalid tremor estimate")
    if len(elapsed_ns) != len(estimate):
        raise ValueError("Timing vector length mismatch")
    reconstructed_voluntary = mixed - estimate
    tremor_error = true - estimate
    voluntary_error = components.voluntary_true_deg_s - reconstructed_voluntary
    fs = components.config.sample_rate_hz
    segment_samples = round(psd_segment_s * fs)
    frequencies, mixed_psd = welch_psd(mixed, fs, segment_samples)
    _, output_psd = welch_psd(reconstructed_voluntary, fs, segment_samples)
    _, true_psd = welch_psd(true, fs, segment_samples)
    _, error_psd = welch_psd(tremor_error, fs, segment_samples)
    input_power = _band_power(frequencies, mixed_psd, tremor_band_hz)
    output_power = _band_power(frequencies, output_psd, tremor_band_hz)
    true_power = _band_power(frequencies, true_psd, tremor_band_hz)
    estimation_error_power = _band_power(frequencies, error_psd, tremor_band_hz)
    true_rms = float(np.sqrt(np.mean(true ** 2)))
    tremor_rmse = float(np.sqrt(np.mean(tremor_error ** 2)))
    return {
        "case": components.case_name,
        "sample_rate_hz": fs,
        "samples": len(estimate),
        "axes": estimate.shape[1],
        "tremor_rmse_deg_s": tremor_rmse,
        "tremor_nrmse_by_true_rms": tremor_rmse / true_rms if true_rms > 0 else None,
        "voluntary_reconstruction_rmse_deg_s": float(np.sqrt(np.mean(voluntary_error ** 2))),
        "observation_band_power_deg2_s2": input_power,
        "reconstructed_voluntary_band_power_deg2_s2": output_power,
        "band_power_reduction_db": (10 * math.log10(input_power / output_power)
                                    if input_power > 0 and output_power > 0 else None),
        "true_tremor_band_power_deg2_s2": true_power,
        "tremor_estimation_error_band_power_deg2_s2": estimation_error_power,
        "tremor_error_band_power_reduction_db": (
            10 * math.log10(true_power / estimation_error_power)
            if true_power > 0 and estimation_error_power > 0 else None),
        "diagnostic_lag_ms": diagnostic_lag_ms(true, estimate, fs),
        "mean_update_us": float(np.mean(elapsed_ns) / 1000),
        "max_update_us": float(np.max(elapsed_ns) / 1000),
        "effective_update_rate_hz": float(1e9 / np.mean(elapsed_ns)),
        "notes": "Band power is mean across axes; voluntary RMSE includes measurement noise when noise is enabled; diagnostic lag does not adjust causal RMSE.",
    }


def evaluate_frequency_tracking(diagnostics: list[dict], true_frequency_hz: np.ndarray,
                                sample_rate_hz: float, transition_s: float | None = None,
                                tolerance_hz: float = 0.5,
                                hold_s: float = 0.5) -> dict:
    """Score a method that naturally emits ``estimated_frequency_hz``.

    Undefined/uninitialized estimates reduce coverage; they are never filled
    with a future value. Re-lock requires a continuous post-transition hold.
    """
    truth = np.asarray(true_frequency_hz, dtype=float)
    if len(diagnostics) != len(truth) or not np.all(np.isfinite(truth)):
        raise ValueError("Frequency truth and diagnostics must align")
    estimate = np.array([item.get("estimated_frequency_hz", np.nan)
                         for item in diagnostics], dtype=float)
    valid = np.isfinite(estimate)
    if not np.any(valid):
        return {"frequency_coverage_fraction": 0.0,
                "frequency_mae_hz": None, "transition_relock_s": None}
    result = {
        "frequency_coverage_fraction": float(np.mean(valid)),
        "frequency_mae_hz": float(np.mean(np.abs(estimate[valid] - truth[valid]))),
        "transition_relock_s": None,
    }
    if transition_s is not None:
        first = round(transition_s * sample_rate_hz)
        hold = max(1, round(hold_s * sample_rate_hz))
        within = valid & (np.abs(estimate - truth) <= tolerance_hz)
        for index in range(first, len(truth) - hold + 1):
            if np.all(within[index:index + hold]):
                result["transition_relock_s"] = float(index / sample_rate_hz - transition_s)
                break
    return result
