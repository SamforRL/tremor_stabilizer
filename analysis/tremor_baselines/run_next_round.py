"""Read-only diagnostics and replication of the unchanged four baselines.

Run from the repository root:
  python -m analysis.tremor_baselines.run_next_round --part all
Outputs are separate from the original baseline benchmark.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import calibrate_pinheiro_modal
from .abpf import tustin_abpf_biquad
from .interface import create_estimator
from .metrics import evaluate, run_stream, welch_psd
from .recorded import ROOT, load_pinheiro_window
from .signals import BenchmarkCaseConfig, generate_case


HERE = Path(__file__).resolve().parent
MAIN_CONFIG = json.loads((HERE / "benchmark_config.json").read_text())["project_benchmark"]
RESULT = ROOT / "experiments/results/tremor_baselines/next_round"
REPORT = ROOT / "reports/tremor_baselines/next_round"
METHODS = ("fixed_bpf", "abpf", "sfft", "bmflc")
CASES = ("slow_smooth", "point_to_point", "fast_transition", "spectral_overlap")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def components(table: str, axis: int, start_s: float, case: str):
    window = load_pinheiro_window(table, start_s, 12.0, (3.0, 12.0), 1.0, (axis,))
    cfg = BenchmarkCaseConfig(sample_rate_hz=window.sample_rate_hz,
                              duration_s=12.0, random_seed=20261026)
    return generate_case(case, window.tremor_band_proxy_deg_s, cfg), window


def _sfft_snapshot(estimator, output: float, diag: dict) -> dict:
    """Recompute the unchanged step's spectra from its current trailing buffer."""
    buffer = estimator._buffer[:, 0].copy()
    n = estimator.window_samples
    fs = estimator.sample_rate_hz
    weighted = np.fft.rfft(buffer * estimator.window)
    raw = np.fft.rfft(buffer)
    center = float(diag["estimated_frequency_hz"])
    band = estimator.extraction_bandwidth_hz
    mask = ((estimator.frequencies_hz >= center - band / 2)
            & (estimator.frequencies_hz <= center + band / 2))
    masked = np.where(mask, raw, 0)
    reconstructed = np.fft.irfft(masked, n=n)
    # Explicit full complex spectrum: positive bins plus their conjugates.
    full = np.r_[masked, np.conj(masked[-2:0:-1])] if n % 2 == 0 else np.r_[masked, np.conj(masked[-1:0:-1])]
    symmetry_error = float(np.max(np.abs(full[1:] - np.conj(full[:0:-1]))))
    full_ifft = np.fft.ifft(full)
    return {
        "buffer_deg_s": buffer, "window": estimator.window.copy(),
        "frequency_bins_hz": estimator.frequencies_hz.copy(),
        "weighted_spectrum": weighted, "raw_spectrum": raw,
        "selected_bin_indices": np.flatnonzero(mask), "masked_spectrum": masked,
        "reconstructed_window_deg_s": reconstructed,
        "dominant_frequency_hz": center,
        "raw_peak_frequency_hz": float(diag["raw_peak_frequency_hz"][0]),
        "selected_low_hz": float(center - band / 2),
        "selected_high_hz": float(center + band / 2),
        "symmetry_error": symmetry_error,
        "imaginary_ifft_max": float(np.max(abs(full_ifft.imag))),
        "irfft_vs_ifft_max": float(np.max(abs(reconstructed - full_ifft.real))),
        "newest_sample_error": float(abs(output - reconstructed[-1])),
        "raw_fft_error": float(np.max(abs(raw - np.fft.rfft(buffer)))),
        "windowed_peak_fft_error": float(np.max(abs(weighted - np.fft.rfft(buffer * estimator.window)))),
        "fft_ifft_roundtrip_max": float(np.max(abs(np.fft.irfft(raw, n=n) - buffer))),
    }


def _plot_sfft_trace(case: str, t, true, output, centers, lows, highs, lowbins, highbins, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, true, label="injected disturbance", color="black", lw=1)
    axes[0].plot(t, output, label="causal SFFT output", lw=1)
    axes[0].set_ylabel("deg/s"); axes[0].legend()
    axes[1].plot(t, centers, label="dominant frequency")
    axes[1].plot(t, lows, ":", label="band lower")
    axes[1].plot(t, highs, ":", label="band upper")
    axes[1].set_ylabel("Hz"); axes[1].legend()
    axes[2].step(t, lowbins, where="post", label="lowest selected bin")
    axes[2].step(t, highbins, where="post", label="highest selected bin")
    axes[2].set_ylabel("FFT bin index"); axes[2].set_xlabel("time (s)"); axes[2].legend()
    fig.suptitle(f"SFFT reconstruction audit: {case}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_sfft_snapshot(case: str, label: str, snapshot: dict, path: Path):
    freq = snapshot["frequency_bins_hz"]
    raw = snapshot["raw_spectrum"]
    masked = snapshot["masked_spectrum"]
    samples = np.arange(len(snapshot["buffer_deg_s"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(freq, abs(raw), label="raw window spectrum")
    axes[0].plot(freq, abs(masked), label="masked spectrum")
    axes[0].axvspan(snapshot["selected_low_hz"], snapshot["selected_high_hz"],
                    color="orange", alpha=0.12, label="extraction band")
    axes[0].set_xlim(0, 15); axes[0].set_ylabel("FFT magnitude (deg/s × samples)")
    axes[0].legend()
    axes[1].plot(samples, snapshot["buffer_deg_s"], label="current trailing window")
    axes[1].plot(samples, snapshot["reconstructed_window_deg_s"], label="IFFT reconstructed window")
    axes[1].scatter([samples[-1]], [snapshot["reconstructed_window_deg_s"][-1]],
                    color="red", zorder=3, label="reported causal sample")
    axes[1].set(xlabel="sample index in trailing window", ylabel="deg/s")
    axes[1].legend()
    fig.suptitle(f"{case} {label}: SFFT spectrum and latest-sample reconstruction")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fit_tone(signal: np.ndarray, fs: float, frequency: float):
    t = np.arange(len(signal)) / fs
    basis = np.column_stack([np.sin(2 * np.pi * frequency * t),
                             np.cos(2 * np.pi * frequency * t), np.ones(len(t))])
    sine, cosine, _ = np.linalg.lstsq(basis, signal, rcond=None)[0]
    return float(np.hypot(sine, cosine)), float(np.arctan2(cosine, sine))


def run_sfft() -> dict:
    out = RESULT / "sfft_reconstruction"; figdir = REPORT / "sfft_reconstruction"
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    rows = []
    max_checks = {key: 0.0 for key in ("symmetry_error", "imaginary_ifft_max",
                                           "irfft_vs_ifft_max", "newest_sample_error",
                                           "raw_fft_error", "windowed_peak_fft_error",
                                           "fft_ifft_roundtrip_max")}
    for case in ("slow_smooth", "fast_transition"):
        signal, source = components("p02_2", 0, 20.0, case)
        estimator = create_estimator("sfft", {"sample_rate_hz": source.sample_rate_hz, **MAIN_CONFIG["sfft"]})
        estimator.reset()
        n = len(signal.time_s)
        output = np.empty(n); centers = np.empty(n); peaks = np.empty(n)
        lows = np.empty(n); highs = np.empty(n); lowbins = np.empty(n, dtype=int); highbins = np.empty(n, dtype=int)
        oracle_centers = np.full(n, np.nan)
        oracle_mixture = np.zeros(n); oracle_proxy = np.zeros(n)
        true_buffer = np.zeros(estimator.window_samples)
        snapshots = {}
        sample_targets = {round(s * source.sample_rate_hz) for s in (1.0, 3.1, 3.3, 4.0, 6.0, 9.0)}
        for k, sample in enumerate(signal.mixed_observation_deg_s):
            value, diag = estimator.step(sample)
            true_buffer[:-1] = true_buffer[1:]
            true_buffer[-1] = signal.tremor_true_deg_s[k, 0]
            output[k] = value[0]
            center = float(diag["estimated_frequency_hz"])
            centers[k] = center
            peaks[k] = float(diag["raw_peak_frequency_hz"][0])
            lows[k] = center - estimator.extraction_bandwidth_hz / 2
            highs[k] = center + estimator.extraction_bandwidth_hz / 2
            selected = np.flatnonzero((estimator.frequencies_hz >= lows[k]) & (estimator.frequencies_hz <= highs[k]))
            lowbins[k], highbins[k] = ((selected[0], selected[-1]) if len(selected) else (-1, -1))
            # Offline diagnostic only: use the known injected proxy to choose
            # the center, then apply the *same* mask and latest-sample IFFT.
            # This oracle must never be interpreted as an online estimator.
            if k >= estimator.window_samples - 1:
                permitted = np.flatnonzero((estimator.frequencies_hz >= estimator.min_frequency_hz)
                                           & (estimator.frequencies_hz <= estimator.max_frequency_hz))
                true_weighted = abs(np.fft.rfft(true_buffer * estimator.window))
                oracle_peak = estimator.frequencies_hz[permitted[np.argmax(true_weighted[permitted])]]
                oracle_center = float(np.clip(oracle_peak,
                    estimator.min_frequency_hz + estimator.extraction_bandwidth_hz / 2,
                    estimator.max_frequency_hz - estimator.extraction_bandwidth_hz / 2))
                oracle_centers[k] = oracle_center
                oracle_mask = ((estimator.frequencies_hz >= oracle_center - estimator.extraction_bandwidth_hz / 2)
                               & (estimator.frequencies_hz <= oracle_center + estimator.extraction_bandwidth_hz / 2))
                oracle_mixture[k] = np.fft.irfft(np.where(oracle_mask, np.fft.rfft(estimator._buffer[:, 0]), 0), n=estimator.window_samples)[-1]
                oracle_proxy[k] = np.fft.irfft(np.where(oracle_mask, np.fft.rfft(true_buffer), 0), n=estimator.window_samples)[-1]
            if (k in sample_targets or k == n - 1) and np.isfinite(center):
                snapshot = _sfft_snapshot(estimator, value[0], diag)
                snapshots[f"sample_{k}"] = snapshot
                for key in max_checks:
                    max_checks[key] = max(max_checks[key], snapshot[key])
        np.savez_compressed(out / f"{case}_trace.npz", time_s=signal.time_s,
                            disturbance_true_deg_s=signal.tremor_true_deg_s[:, 0],
                            observation_deg_s=signal.mixed_observation_deg_s[:, 0],
                            output_deg_s=output, dominant_frequency_hz=centers,
                            offline_oracle_proxy_peak_hz=oracle_centers,
                            offline_oracle_mixture_output_deg_s=oracle_mixture,
                            offline_oracle_proxy_only_output_deg_s=oracle_proxy,
                            raw_peak_frequency_hz=peaks, band_lower_hz=lows,
                            band_upper_hz=highs, lowest_bin=lowbins, highest_bin=highbins)
        for label, snapshot in snapshots.items():
            np.savez_compressed(out / f"{case}_{label}.npz", **snapshot)
            if label in {f"sample_{round(s * source.sample_rate_hz)}" for s in (3.3, 6.0)}:
                _plot_sfft_snapshot(case, label, snapshot, figdir / f"{case}_{label}_snapshot.png")
        _plot_sfft_trace(case, signal.time_s, signal.tremor_true_deg_s[:, 0], output,
                         centers, lows, highs, lowbins, highbins, figdir / f"{case}_trace.png")
        rmse = float(np.sqrt(np.mean((output - signal.tremor_true_deg_s[:, 0]) ** 2)))
        stable = np.isfinite(oracle_centers)
        true_wave = signal.tremor_true_deg_s[:, 0]
        rows.append({"case": case, "rmse_deg_s": rmse,
                     "post_window_actual_rmse_deg_s": float(np.sqrt(np.mean((output[stable] - true_wave[stable]) ** 2))),
                     "post_window_oracle_center_mixture_rmse_deg_s": float(np.sqrt(np.mean((oracle_mixture[stable] - true_wave[stable]) ** 2))),
                     "post_window_oracle_center_proxy_only_rmse_deg_s": float(np.sqrt(np.mean((oracle_proxy[stable] - true_wave[stable]) ** 2))),
                     "mean_abs_actual_vs_proxy_peak_hz": float(np.mean(abs(centers[stable] - oracle_centers[stable]))),
                     "selected_bin_min": int(lowbins.min()), "selected_bin_max": int(highbins.max()),
                     "no_detected_peak_samples": int(np.count_nonzero(~np.isfinite(centers))),
                     "snapshot_count": len(snapshots), "window_samples": estimator.window_samples})
    # Controlled tone uses the same project sample rate and SFFT configuration,
    # zero voluntary motion, and a 6.0 Hz sinusoid with known amplitude/phase.
    fs = source.sample_rate_hz; t = np.arange(round(12 * fs)) / fs
    truth = 5 * np.sin(2 * np.pi * 6.0 * t + 0.3)
    estimator = create_estimator("sfft", {"sample_rate_hz": fs, **MAIN_CONFIG["sfft"]})
    output, diag, _ = run_stream(estimator, truth[:, None])
    output = output[:, 0]
    centers = np.array([d["estimated_frequency_hz"] for d in diag])
    steady = t >= 2.0
    amp_true, phase_true = _fit_tone(truth[steady], fs, 6.0)
    amp_est, phase_est = _fit_tone(output[steady], fs, 6.0)
    phase_error = float(np.angle(np.exp(1j * (phase_est - phase_true))))
    tone = {"frequency_true_hz": 6.0, "input_amplitude_deg_s": 5.0,
            "steady_start_s": 2.0, "amplitude_ratio": amp_est / amp_true,
            "phase_error_deg": math.degrees(phase_error),
            "rmse_deg_s": float(np.sqrt(np.mean((truth[steady] - output[steady]) ** 2))),
            "frequency_mae_hz": float(np.mean(abs(centers[steady] - 6.0))),
            "frequency_bias_hz": float(np.mean(centers[steady] - 6.0))}
    np.savez_compressed(out / "known_sinusoid.npz", time_s=t, true_deg_s=truth,
                        output_deg_s=output, dominant_frequency_hz=centers)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    crop = (t >= 3) & (t <= 4)
    ax.plot(t[crop], truth[crop], color="black", label="true 6 Hz sinusoid")
    ax.plot(t[crop], output[crop], label="SFFT causal reconstructed")
    ax.set(xlabel="time (s)", ylabel="deg/s", title="Known tone: causal waveform after startup")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "known_sinusoid.png", dpi=150); plt.close(fig)
    write_csv(figdir / "cases.csv", rows)
    write_json(out / "diagnostic.json", {"snapshot_checks_max": max_checks,
                                         "known_sinusoid": tone,
                                         "reconstruction_scaling": "weighted FFT for peak only; raw FFT masked, no amplitude normalization; numpy irfft 1/N inverse scaling",
                                         "snapshot_times_s": [1.0, 3.1, 3.3, 4.0, 6.0, 9.0, 12.0]})
    if any(value > 1e-10 for value in max_checks.values()):
        raise AssertionError(f"SFFT reconstruction invariant failed: {max_checks}")
    return {"checks": max_checks, "tone": tone, "cases": rows}


def run_bmflc() -> dict:
    out = RESULT / "bmflc_transient"; figdir = REPORT / "bmflc_transient"
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    signal, source = components("p02_2", 0, 20.0, "fast_transition")
    estimator = create_estimator("bmflc", {"sample_rate_hz": source.sample_rate_hz,
                                            **MAIN_CONFIG["bmflc"]})
    estimate, diagnostics, timing = run_stream(estimator, signal.mixed_observation_deg_s)
    count = len(estimator.frequencies_hz)
    weights = np.stack([d["adaptive_weights"][0] for d in diagnostics])
    # Diagnostic weights are post-update; outputs use pre-update weights.
    fourier = weights[:, :2 * count]
    magnitude = np.hypot(fourier[:, :count], fourier[:, count:])
    norm = np.linalg.norm(fourier, axis=1)
    bias = np.array([d["intended_motion_bias"][0] for d in diagnostics])
    error = np.array([d["internal_error"][0] for d in diagnostics])
    t = signal.time_s
    np.savez_compressed(out / "trace.npz", time_s=t, frequencies_hz=estimator.frequencies_hz,
                        fourier_weights_post_update=fourier, weight_magnitude_post_update=magnitude,
                        fourier_norm_post_update=norm, bias_pre_update_deg_s=bias,
                        lms_error_deg_s=error, tremor_estimate_deg_s=estimate[:, 0],
                        tremor_true_deg_s=signal.tremor_true_deg_s[:, 0],
                        voluntary_true_deg_s=signal.voluntary_true_deg_s[:, 0],
                        observation_deg_s=signal.mixed_observation_deg_s[:, 0],
                        processing_time_us=timing / 1000)
    rows = []
    for label, a, b in (("before", 0, 3), ("transition", 3, 5), ("recovery", 5, 12)):
        mask = (t >= a) & (t < b)
        residual = estimate[mask, 0] - signal.tremor_true_deg_s[mask, 0]
        rows.append({"period": label, "start_s": a, "end_s": b,
                     "mean_fourier_norm": float(np.mean(norm[mask])),
                     "peak_fourier_norm": float(np.max(norm[mask])),
                     "rmse_deg_s": float(np.sqrt(np.mean(residual ** 2))),
                     "mean_absolute_lms_error_deg_s": float(np.mean(abs(error[mask]))),
                     "peak_absolute_lms_error_deg_s": float(np.max(abs(error[mask]))),
                     "mean_bias_deg_s": float(np.mean(bias[mask]))})
    write_csv(figdir / "periods.csv", rows)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, signal.tremor_true_deg_s[:, 0], label="injected disturbance", color="black")
    axes[0].plot(t, estimate[:, 0], label="BMFLC tremor estimate")
    axes[0].legend(); axes[0].set_ylabel("deg/s")
    axes[1].plot(t, norm); axes[1].set_ylabel("Fourier weight L2 norm")
    axes[2].plot(t, signal.voluntary_true_deg_s[:, 0], color="black", label="true voluntary")
    axes[2].plot(t, bias, label="bias estimate")
    axes[2].legend(); axes[2].set_ylabel("deg/s")
    axes[3].plot(t, error); axes[3].set_ylabel("LMS error (deg/s)"); axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.axvspan(3, 5, color="orange", alpha=0.1)
    fig.tight_layout(); fig.savefig(figdir / "state_trace.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(magnitude.T, origin="lower", aspect="auto", interpolation="nearest",
                      extent=[t[0], t[-1], estimator.frequencies_hz[0], estimator.frequencies_hz[-1]])
    ax.axvspan(3, 5, color="white", alpha=0.12)
    ax.set(xlabel="time (s)", ylabel="Fourier basis frequency (Hz)",
           title="BMFLC post-update sine/cosine weight magnitude")
    fig.colorbar(image, ax=ax, label="weight magnitude (deg/s)")
    fig.tight_layout(); fig.savefig(figdir / "weight_heatmap.png", dpi=150); plt.close(fig)
    return {"periods": rows, "peak_norm_at_s": float(t[np.argmax(norm)])}


def _relock_after(t, center, reference_hz, after_s=3.35, tolerance_hz=0.5, hold_s=0.5):
    """Diagnostic: sustained proximity to whole-window proxy PSD peak."""
    first = int(np.searchsorted(t, after_s)); hold = round(hold_s / (t[1] - t[0]))
    close = abs(center - reference_hz) <= tolerance_hz
    for k in range(first, len(t) - hold + 1):
        if np.all(close[k:k + hold]):
            return float(t[k] - after_s)
    return None


def run_abpf() -> dict:
    out = RESULT / "abpf_calibration"; figdir = REPORT / "abpf_calibration"
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    offsets = (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
    rows = []; traces = {}; crossing_rows = []
    for case in ("slow_smooth", "fast_transition", "spectral_overlap"):
        signal, source = components("p02_2", 0, 20.0, case)
        calibration = calibrate_pinheiro_modal("p02_2", (0,), 20.0, 12.0)
        modal = float(calibration.modal_frequency_hz[0])
        freqs, psd = welch_psd(signal.tremor_true_deg_s, source.sample_rate_hz,
                               round(2 * source.sample_rate_hz))
        band = (freqs >= 3) & (freqs <= 12)
        peak = float(freqs[band][np.argmax(psd[band, 0])])
        traces[case] = {}
        for offset in offsets:
            test_modal = modal + offset
            estimator = create_estimator("abpf", {"sample_rate_hz": source.sample_rate_hz,
                                                     "modal_frequency_hz": test_modal,
                                                     **MAIN_CONFIG["abpf"]})
            estimate, diag, timing = run_stream(estimator, signal.mixed_observation_deg_s)
            center = np.array([d["estimated_frequency_hz"] for d in diag])
            if offset == 0:
                half_period = np.array([d["half_period_s"][0] for d in diag])
                crossing_frequency = 1 / (2 * half_period)
                valid_crossing = np.isfinite(crossing_frequency)
                updates = np.array([d["frequency_update_hz"][0] for d in diag])
                percentiles = np.percentile(crossing_frequency[valid_crossing], [10, 50, 90])
                center_percentiles = np.percentile(center, [10, 50, 90])
                crossing_rows.append({"case": case,
                                      "half_period_measurements": int(np.count_nonzero(valid_crossing)),
                                      "nonzero_center_updates": int(np.count_nonzero(updates)),
                                      "crossing_frequency_p10_hz": float(percentiles[0]),
                                      "crossing_frequency_median_hz": float(percentiles[1]),
                                      "crossing_frequency_p90_hz": float(percentiles[2]),
                                      "center_p10_hz": float(center_percentiles[0]),
                                      "center_median_hz": float(center_percentiles[1]),
                                      "center_p90_hz": float(center_percentiles[2])})
            metrics = evaluate(signal, estimate, timing)
            lower = test_modal - MAIN_CONFIG["abpf"]["limit_offset_hz"]
            upper = test_modal + MAIN_CONFIG["abpf"]["limit_offset_hz"]
            # Bound occupancy uses half a frequency step to cover finite-step states.
            margin = MAIN_CONFIG["abpf"]["frequency_step_hz"] / 2 + 1e-10
            at_lower = center <= lower + margin
            at_upper = center >= upper - margin
            windows = {}
            for label, a, b in (("pre", 0, 3), ("transition", 3, 5), ("post", 5, 12)):
                mask = (signal.time_s >= a) & (signal.time_s < b)
                windows[f"{label}_rmse_deg_s"] = float(np.sqrt(np.mean(
                    (estimate[mask, 0] - signal.tremor_true_deg_s[mask, 0]) ** 2)))
            row = {"case": case, "offset_hz": offset, "modal_test_hz": test_modal,
                   "proxy_psd_peak_hz": peak,
                   "proxy_peak_inside_allowed_range": lower <= peak <= upper,
                   "rmse_deg_s": metrics["tremor_rmse_deg_s"],
                   "nrmse": metrics["tremor_nrmse_by_true_rms"],
                   "center_within_0p5hz_proxy_peak_fraction": float(np.mean(abs(center - peak) <= 0.5)),
                   "lower_bound_fraction": float(np.mean(at_lower)),
                   "upper_bound_fraction": float(np.mean(at_upper)),
                   "either_bound_fraction": float(np.mean(at_lower | at_upper)),
                   "relock_after_3p35s_s": _relock_after(signal.time_s, center, peak),
                   **windows}
            rows.append(row)
            traces[case][str(offset)] = center
        np.savez_compressed(out / f"{case}_center_trajectories.npz", time_s=signal.time_s,
                            offsets_hz=np.array(offsets), centers_hz=np.array([traces[case][str(o)] for o in offsets]),
                            proxy_psd_peak_hz=peak)
        fig, ax = plt.subplots(figsize=(12, 5))
        for offset in offsets:
            ax.plot(signal.time_s, traces[case][str(offset)], lw=0.9, label=f"{offset:+g} Hz")
        ax.axhline(peak, color="black", ls="--", label="whole-window proxy PSD peak")
        ax.set(xlabel="time (s)", ylabel="ABPF center (Hz)", title=f"ABPF initialization sensitivity: {case}")
        ax.legend(ncol=5, fontsize=8); fig.tight_layout()
        fig.savefig(figdir / f"{case}_centers.png", dpi=150); plt.close(fig)
    write_csv(figdir / "offset_metrics.csv", rows)
    write_csv(figdir / "crossing_statistics.csv", crossing_rows)
    response_rows = []
    for frequency in (modal - MAIN_CONFIG["abpf"]["limit_offset_hz"], modal,
                      modal + MAIN_CONFIG["abpf"]["limit_offset_hz"]):
        b0, b1, b2, _, a1, a2 = tustin_abpf_biquad(frequency, source.sample_rate_hz)
        grid = np.linspace(0, source.sample_rate_hz / 2, 250001)
        z = np.exp(-2j * np.pi * grid / source.sample_rate_hz)
        response = abs((b0 + b1*z + b2*z*z) / (1 + a1*z + a2*z*z)) ** 2
        indices = np.flatnonzero(response >= response.max() / 2)
        response_rows.append({"configured_center_hz": frequency,
                              "actual_peak_hz": float(grid[np.argmax(response)]),
                              "minus_3db_lower_hz": float(grid[indices[0]]),
                              "minus_3db_upper_hz": float(grid[indices[-1]]),
                              "minus_3db_width_hz": float(grid[indices[-1]] - grid[indices[0]])})
    write_csv(figdir / "filter_response_summary.csv", response_rows)
    write_json(out / "protocol.json", {"offsets_hz": offsets,
                                        "calibrated_modal_hz": modal,
                                        "proxy_peak_definition": "maximum 2 s Hann Welch PSD bin in 3-12 Hz of injected proxy; not independently known instantaneous frequency",
                                        "relock_definition": "after 3.35 s, first 0.5 s continuous within 0.5 Hz of proxy whole-window PSD peak; blank if never",
                                        "bound_definition": "center within half frequency step of allowed endpoint"})
    return {"rows": rows, "calibrated_modal_hz": modal}


def run_replication() -> dict:
    out = RESULT / "cross_window_axis"; figdir = REPORT / "cross_window_axis"
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    # All starts have independent 0-15 s modal calibration and 1 s proxy context.
    # Within each table 20-32, 33-45, 46-58 s windows are non-overlapping.
    rows = []; window_provenance = []
    for table in ("p02_1", "p02_2"):
        for start in (20.0, 33.0, 46.0):
            for axis in (0, 1, 2):
                exemplar, source = components(table, axis, start, "slow_smooth")
                cal = calibrate_pinheiro_modal(table, (axis,), start, 12.0)
                fs = source.sample_rate_hz
                configs = {
                    "fixed_bpf": {"sample_rate_hz": fs, **MAIN_CONFIG["fixed_bpf"]},
                    "abpf": {"sample_rate_hz": fs, "modal_frequency_hz": cal.modal_frequency_hz.tolist(), **MAIN_CONFIG["abpf"]},
                    "sfft": {"sample_rate_hz": fs, **MAIN_CONFIG["sfft"]},
                    "bmflc": {"sample_rate_hz": fs, **MAIN_CONFIG["bmflc"]},
                }
                estimators = {name: create_estimator(name, configs[name]) for name in METHODS}
                window_provenance.append({"table": table, "axis": "XYZ"[axis],
                                          "start_s": start, "end_s": start + 12,
                                          "recording_sha256": source.source_sha256,
                                          "fs_hz": fs, "modal_hz": float(cal.modal_frequency_hz[0]),
                                          "proxy_rms_deg_s": float(np.sqrt(np.mean(exemplar.tremor_true_deg_s ** 2)))})
                for case in CASES:
                    signal, _ = components(table, axis, start, case)
                    for name in METHODS:
                        estimate, _, timing = run_stream(estimators[name], signal.mixed_observation_deg_s)
                        metrics = evaluate(signal, estimate, timing)
                        row = {"subject": "p02", "table": table, "axis": "XYZ"[axis],
                               "start_s": start, "case": case, "method": name,
                               "rmse_deg_s": metrics["tremor_rmse_deg_s"],
                               "nrmse": metrics["tremor_nrmse_by_true_rms"],
                               "band_power_reduction_db": metrics["band_power_reduction_db"],
                               "mean_update_us": metrics["mean_update_us"],
                               "max_update_us": metrics["max_update_us"]}
                        for label, a, b in (("pre", 0, 3), ("transition", 3, 5), ("post", 5, 12)):
                            mask = (signal.time_s >= a) & (signal.time_s < b)
                            row[f"{label}_rmse_deg_s"] = float(np.sqrt(np.mean(
                                (estimate[mask] - signal.tremor_true_deg_s[mask]) ** 2)))
                        rows.append(row)
    write_csv(figdir / "all_trials.csv", rows)
    write_csv(figdir / "windows.csv", window_provenance)
    # Keep each scenario's distribution separate; no pooled pseudoreplication
    # or cross-subject conclusion from 2 tables of the same person.
    summary = []
    for case in CASES:
        for name in METHODS:
            subset = [r for r in rows if r["case"] == case and r["method"] == name]
            record = {"case": case, "method": name, "n_axis_windows": len(subset),
                      "n_subjects": 1}
            for key in ("rmse_deg_s", "nrmse", "pre_rmse_deg_s",
                        "transition_rmse_deg_s", "post_rmse_deg_s",
                        "band_power_reduction_db", "mean_update_us", "max_update_us"):
                values = np.array([r[key] for r in subset])
                for label, value in (("min", np.min(values)), ("p25", np.percentile(values, 25)),
                                     ("median", np.median(values)), ("p75", np.percentile(values, 75)),
                                     ("max", np.max(values))):
                    record[f"{key}_{label}"] = float(value)
            summary.append(record)
    write_csv(figdir / "distribution_summary.csv", summary)
    compact = []
    for record in summary:
        compact.append({key: record[key] for key in (
            "case", "method", "n_axis_windows", "n_subjects",
            "rmse_deg_s_p25", "rmse_deg_s_median", "rmse_deg_s_p75", "rmse_deg_s_max",
            "pre_rmse_deg_s_median", "transition_rmse_deg_s_median",
            "post_rmse_deg_s_median", "band_power_reduction_db_median",
            "mean_update_us_median")})
    write_csv(figdir / "summary_core.csv", compact)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, case in zip(axes.flat, CASES):
        data = [np.array([r["rmse_deg_s"] for r in rows if r["case"] == case and r["method"] == name])
                for name in METHODS]
        ax.boxplot(data, tick_labels=METHODS, showfliers=True)
        ax.set_title(case); ax.set_ylabel("full-window RMSE (deg/s)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figdir / "rmse_distributions.png", dpi=150); plt.close(fig)
    write_json(out / "protocol.json", {"subjects": ["p02"], "tables": ["p02_1", "p02_2"],
                                        "axes": ["X", "Y", "Z"], "window_starts_s": [20, 33, 46],
                                        "cases": CASES, "methods": METHODS,
                                        "calibration": "0-15 s raw gyro independently per table/axis; no per-window tuning",
                                        "note": "18 axis-windows are correlated within one subject; no across-subject replication"})
    return {"summary": summary, "all_trials": rows, "windows": window_provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", "sfft", "bmflc", "abpf", "replication"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    if args.part in ("all", "sfft"):
        value = run_sfft(); print("sfft", value["tone"], flush=True)
    if args.part in ("all", "bmflc"):
        value = run_bmflc(); print("bmflc", value["periods"], flush=True)
    if args.part in ("all", "abpf"):
        value = run_abpf(); print("abpf", len(value["rows"]), "offset trials", flush=True)
    if args.part in ("all", "replication"):
        value = run_replication(); print("replication", len(value["all_trials"]), "method trials", flush=True)
    print(f"completed in {time.monotonic() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
