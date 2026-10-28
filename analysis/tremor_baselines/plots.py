"""Standardized comparison plots for all available estimators."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tremor_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .butterworth import FixedButterworthBPF, sos_response
from .metrics import welch_psd
from .signals import SignalComponents


def plot_filter_response(estimator: FixedButterworthBPF, output_dir: Path):
    """Save magnitude and phase/group-delay data for the fixed BPF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fs = estimator.sample_rate_hz
    frequencies = np.linspace(0, fs / 2, 4097)
    response = sos_response(estimator.sos, frequencies, fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    phase = np.unwrap(np.angle(response))
    group_delay_s = -np.gradient(phase, 2 * np.pi * frequencies)
    visible = ((frequencies >= max(0.1, estimator.low_hz / 2))
               & (frequencies <= min(fs / 2, estimator.high_hz * 1.5))
               & (magnitude_db > -40))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             constrained_layout=True)
    axes[0].plot(frequencies[visible], magnitude_db[visible])
    axes[0].axhline(-3.0103, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("magnitude (dB)")
    axes[1].plot(frequencies[visible], 1000 * group_delay_s[visible])
    axes[1].set_ylabel("group delay (ms)")
    axes[1].set_xlabel("frequency (Hz)")
    for axis in axes:
        axis.axvline(estimator.low_hz, color="gray", linestyle=":")
        axis.axvline(estimator.high_hz, color="gray", linestyle=":")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Causal Butterworth BPF: prototype order {estimator.order}, "
                 f"{estimator.low_hz:g}–{estimator.high_hz:g} Hz, fs={fs:.2f} Hz")
    fig.savefig(output_dir / "fixed_bpf_response.png", dpi=150)
    plt.close(fig)
    np.savetxt(output_dir / "fixed_bpf_response.csv",
               np.column_stack((frequencies, magnitude_db, phase, group_delay_s)),
               delimiter=",", comments="",
               header="frequency_hz,magnitude_db,phase_rad,group_delay_s")


def plot_comparison(components: SignalComponents, estimates: dict[str, np.ndarray],
                    diagnostics: dict[str, list[dict]], output_dir: Path,
                    axis: int = 0, known_frequency_hz: np.ndarray | None = None,
                    psd_segment_s: float = 2.0):
    """Use identical axes, time samples, and PSD settings for every method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = components.time_s
    mixed = components.mixed_observation_deg_s[:, axis]
    voluntary = components.voluntary_true_deg_s[:, axis]
    tremor = components.tremor_true_deg_s[:, axis]
    for name, value in estimates.items():
        if np.asarray(value).shape != components.mixed_observation_deg_s.shape:
            raise ValueError(f"Shape mismatch for {name}")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, mixed, label="mixed observation")
    axes[0].plot(t, voluntary, label="true voluntary")
    axes[1].plot(t, tremor, color="black", label="known injected disturbance")
    axes[2].plot(t, voluntary, color="black", label="true voluntary")
    for name, estimate in estimates.items():
        axes[1].plot(t, estimate[:, axis], label=name)
        axes[2].plot(t, mixed - estimate[:, axis], label=name)
        axes[3].plot(t, tremor - estimate[:, axis], label=name)
    for plot_axis, title in zip(axes, ("input", "tremor estimate", "voluntary reconstruction",
                                        "tremor estimation error")):
        plot_axis.set_ylabel(f"{title} (deg/s)")
        plot_axis.legend(ncol=min(4, len(estimates) + 1))
        plot_axis.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{components.case_name}: axis {axis}")
    fig.savefig(output_dir / f"{components.case_name}_axis{axis}_time.png", dpi=150)
    plt.close(fig)

    fs = components.config.sample_rate_hz
    nperseg = round(psd_segment_s * fs)
    frequencies, true_psd = welch_psd(components.tremor_true_deg_s, fs, nperseg)
    _, mixed_psd = welch_psd(components.mixed_observation_deg_s, fs, nperseg)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.semilogy(frequencies, np.maximum(true_psd[:, axis], 1e-15),
                color="black", label="known injected disturbance")
    ax.semilogy(frequencies, np.maximum(mixed_psd[:, axis], 1e-15),
                label="mixed observation")
    for name, estimate in estimates.items():
        _, estimate_psd = welch_psd(estimate, fs, nperseg)
        ax.semilogy(frequencies, np.maximum(estimate_psd[:, axis], 1e-15), label=name)
    ax.set_xlim(0, min(30, fs / 2))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD ((deg/s)^2 / Hz)")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(f"{components.case_name}: identical Hann Welch PSD settings")
    fig.savefig(output_dir / f"{components.case_name}_axis{axis}_psd.png", dpi=150)
    plt.close(fig)

    frequency_lines = {}
    for name, infos in diagnostics.items():
        if len(infos) != len(t):
            raise ValueError(f"Diagnostic length mismatch for {name}")
        if any("estimated_frequency_hz" in info for info in infos):
            frequency_lines[name] = np.array([
                info.get("estimated_frequency_hz", np.nan) for info in infos
            ])
    if frequency_lines or known_frequency_hz is not None:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        if known_frequency_hz is not None:
            if len(known_frequency_hz) != len(t):
                raise ValueError("Known frequency trajectory length mismatch")
            ax.plot(t, known_frequency_hz, color="black", label="known frequency")
        for name, values in frequency_lines.items():
            ax.plot(t, values, label=name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / f"{components.case_name}_frequency.png", dpi=150)
        plt.close(fig)
