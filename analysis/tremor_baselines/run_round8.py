"""Round 8: update-only causal gates on the unchanged raw Huber-BMFLC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .recorded import ROOT
from .round3_variants import RobustBMFLC
from .round8_gate import GatedHuberBMFLC
from .run_round7 import (FS, RHO, SCENARIOS, SUBJECTS, WINDOWS,
                         calibration_band_rms, recorded, rms, synthetic)
from .signals import BenchmarkCaseConfig, SignalComponents, controlled_tremor

OUT = ROOT / "experiments/results/tremor_baselines/round8"
REPORT = ROOT / "reports/tremor_baselines/round8"
METHODS = ("Raw_Huber", "BandGate_Huber", "BandPersistenceGate_Huber")
DEFAULT = dict(window_s=.5, persistence_windows=3, frequency_sigma_hz=1.,
               minimum_gate=.05, analysis_band_threshold=.2,
               frequency_update_s=.1)


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def build(name, fs, scale, params):
    if name == "Raw_Huber":
        return RobustBMFLC(fs, variant="huber", huber_delta=.5 * scale)
    mode = "band" if name == "BandGate_Huber" else "persistence"
    return GatedHuberBMFLC(fs, scale, mode=mode, **params)


def trace(signal, name, scale, params, save=None):
    fs = signal.config.sample_rate_hz
    method = build(name, fs, scale, params)
    y = signal.mixed_observation_deg_s[:, 0]
    d = signal.tremor_true_deg_s[:, 0]
    t = signal.time_s
    estimate = np.empty_like(y)
    gate = np.ones_like(y)
    ratio = np.ones_like(y)
    persistence = np.ones_like(y)
    residual = np.empty_like(y)
    weight_norm = np.empty_like(y)
    weights = np.empty((len(y), 182)) if save else None
    import time
    elapsed = np.empty_like(y)
    for k, sample in enumerate(y):
        begin = time.perf_counter_ns()
        output, diag = method.step(np.array([sample]))
        elapsed[k] = (time.perf_counter_ns() - begin) / 1e3
        estimate[k] = output[0]
        residual[k] = diag["internal_error"][0]
        weight = diag["adaptive_weights"][0, :-1]
        weight_norm[k] = np.linalg.norm(weight)
        if weights is not None:
            weights[k] = weight
        if name != "Raw_Huber":
            gate[k] = diag["gate"][0]
            ratio[k] = diag["band_ratio"][0]
            persistence[k] = diag["persistence_score"][0]
    if not np.all(np.isfinite(estimate)):
        raise AssertionError("nonfinite estimator output")
    active = (t >= 3) & (t < 5)
    steady = (t >= 1) & (t < 3)
    post = t >= 5
    after_warmup = t >= params["window_s"] + .2
    seek = np.flatnonzero((t >= 3) & (gate < .5))
    delay = float((t[seek[0]] - 3) * 1000) if len(seek) else None
    row = dict(method=name, tremor_rmse_deg_s=rms(estimate-d),
               tremor_nrmse=rms(estimate-d)/max(rms(d), 1e-9),
               pre_nrmse=rms((estimate-d)[steady])/max(rms(d[steady]), 1e-9),
               transition_nrmse=rms((estimate-d)[active])/max(rms(d[active]), 1e-9),
               post_nrmse=rms((estimate-d)[post])/max(rms(d[post]), 1e-9),
               gate_tremor_mean=float(np.mean(gate[steady])),
               gate_fast_mean=float(np.mean(gate[active])),
               gate_post_mean=float(np.mean(gate[post])),
               gate_full_mean=float(np.mean(gate[after_warmup])),
               band_ratio_fast_mean=float(np.mean(ratio[active])),
               persistence_fast_mean=float(np.mean(persistence[active])),
               gate_below_half_delay_ms=delay,
               residual_fast_rms_deg_s=rms(residual[active]),
               weight_fast_peak=float(np.max(weight_norm[active])),
               weight_post_mean=float(np.mean(weight_norm[post])),
               update_mean_us=float(np.mean(elapsed)),
               update_max_us=float(np.max(elapsed)))
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, time_s=t, mixed=y, tremor=d,
                            voluntary=signal.voluntary_true_deg_s[:, 0],
                            estimate=estimate, gate=gate, band_ratio=ratio,
                            persistence=persistence, residual=residual,
                            weight_norm=weight_norm, weights=weights)
    return row


def contamination(signal, name, scale, params, save=None):
    fs = signal.config.sample_rate_hz
    t = signal.time_s
    a, b = build(name, fs, scale, params), build(name, fs, scale, params)
    diff = np.empty(len(t))
    for k, (mix, pure) in enumerate(zip(signal.mixed_observation_deg_s[:, 0],
                                        signal.tremor_true_deg_s[:, 0])):
        _, da = a.step(np.array([mix]))
        _, db = b.step(np.array([pure]))
        diff[k] = np.linalg.norm(da["adaptive_weights"][0, :-1]
                                 - db["adaptive_weights"][0, :-1])
    pre, active, post = diff[t < 3], diff[(t >= 3) & (t < 5)], diff[t >= 5]
    baseline = float(np.mean(pre))
    peak = float(np.max(active))
    threshold = baseline + .2 * max(peak-baseline, 0)
    width = max(1, round(.5 * fs))
    recovery = None
    for k in range(np.searchsorted(t, 5), len(diff)-width+1):
        if np.all(diff[k:k+width] <= threshold):
            recovery = float(t[k] - 5)
            break
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, time_s=t, weight_difference=diff)
    return dict(weight_difference_pre_mean=baseline,
                weight_difference_fast_peak=peak,
                weight_difference_peak_excess=max(peak-baseline, 0),
                weight_difference_post_mean=float(np.mean(post)),
                weight_difference_recovery_s=recovery)


def synthetic_development():
    """Parameter check on a separate smooth 0.6 s maneuver, never recorded trials."""
    rows = []
    candidates = [dict(DEFAULT, window_s=x) for x in (.25, .5, .75)]
    candidates += [dict(DEFAULT, persistence_windows=x) for x in (2, 4)]
    candidates += [dict(DEFAULT, frequency_sigma_hz=x) for x in (.5, 1.5)]
    candidates += [dict(DEFAULT, minimum_gate=x) for x in (0., .1)]
    for case in ("constant", "drift", "step", "amplitude_modulated", "multifrequency"):
        t, d, _ = controlled_tremor(case, FS, 12., amplitude_deg_s=5.)
        v = np.zeros(len(t))
        if case in ("constant", "multifrequency"):
            tau = np.clip((t-3.2)/.6, 0, 1)
            # Derivative of a minimum-jerk 8-degree rotation.
            mask = (t >= 3.2) & (t < 3.8)
            v[mask] = 8 * (30*tau[mask]**2 - 60*tau[mask]**3 + 30*tau[mask]**4)/.6
        config = BenchmarkCaseConfig(sample_rate_hz=FS, duration_s=12., noise_std_deg_s=0)
        signal = SignalComponents(t, v[:, None], d, np.zeros_like(d), d+v[:, None], case, config)
        for index, params in enumerate(candidates):
            for name in METHODS[1:]:
                score = trace(signal, name, rms(d), params)
                rows.append(dict(candidate=index, case=case,
                                 **{k: params[k] for k in params}, **score))
    csv_write(OUT / "development/synthetic_parameter_check.csv", rows)
    # Predeclared conservative middle values; no evaluation-data optimization.
    protocol = dict(gate=DEFAULT, selection="predeclared middle settings, checked on synthetic development only",
                    synthetic_development="0.6 s, 8 degree minimum-jerk maneuver plus five controlled tremors",
                    subjects=list(SUBJECTS), windows_s=list(WINDOWS), scenarios=list(SCENARIOS),
                    ratio_levels=list(RHO), scale="0.5 × independent 0–15 s subject/axis 3–12 Hz RMS",
                    bmflc=dict(frequency_hz=[3, 12], spacing_hz=.1, mu=.01),
                    gate_bpf="causal Butterworth bandpass 3–12 Hz, order 2 prototype",
                    window_energy="mean-square of raw residual vs causal filtered residual",
                    persistence="trailing residual Hann periodogram, 0.1 s updates; 3 consecutive dominant frequencies",
                    warmup="gate=1 until full window", no_recorded_tuning=True)
    path = OUT / "frozen_gate_parameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n")


def frozen():
    path = OUT / "frozen_gate_parameters.json"
    if not path.exists():
        raise RuntimeError("run synthetic development first to freeze parameters")
    return json.loads(path.read_text())["gate"]


def run_synthetic():
    params = frozen()
    rows, mechanisms, ratios = [], [], []
    for tremor in ("constant", "drift", "step", "amplitude_modulated", "multifrequency"):
        for scenario in SCENARIOS:
            signal = synthetic(tremor, scenario)
            scale = rms(signal.tremor_true_deg_s[:, 0])
            for name in METHODS:
                path = OUT / f"traces/synthetic_{tremor}_{scenario}_{name}.npz" if (tremor == "step" and scenario == "fast_transition") else None
                rows.append(dict(tremor=tremor, scenario=scenario,
                                 **trace(signal, name, scale, params, path)))
                if scenario in ("fast_transition", "spectral_overlap"):
                    mechanisms.append(dict(tremor=tremor, scenario=scenario, method=name,
                                           **contamination(signal, name, scale, params)))
        print("synthetic", tremor, flush=True)
    for scenario in ("fast_transition", "spectral_overlap"):
        for rho in RHO:
            signal = synthetic("multifrequency", scenario, rho)
            scale = rms(signal.tremor_true_deg_s[:, 0])
            for name in METHODS:
                ratios.append(dict(scenario=scenario, rho=rho,
                                   **trace(signal, name, scale, params),
                                   **contamination(signal, name, scale, params)))
    csv_write(OUT / "synthetic/cases.csv", rows)
    csv_write(OUT / "synthetic/weight_contamination.csv", mechanisms)
    csv_write(OUT / "synthetic/ratio_sweep.csv", ratios)
    run_step(params)


def run_step(params):
    rows = []
    t, d, freq = controlled_tremor("step", FS, 12., amplitude_deg_s=5.)
    for name in METHODS:
        model = build(name, FS, rms(d), params)
        estimates, gates, weights = [], [], []
        for value in d[:, 0]:
            out, diagnostic = model.step(np.array([value]))
            estimates.append(out[0])
            gates.append(diagnostic.get("gate", np.array([1.]))[0])
            weights.append(diagnostic["adaptive_weights"][0, :-1])
        estimates = np.asarray(estimates)
        weights = np.asarray(weights)
        energy5 = np.hypot(weights[:, 20], weights[:, 20+91])
        energy8 = np.hypot(weights[:, 50], weights[:, 50+91])
        half_second = max(1, round(.5*FS))
        lock = None
        for k in range(np.searchsorted(t, 6), len(t)-half_second+1):
            if np.all(energy8[k:k+half_second] > energy5[k:k+half_second]):
                lock = float(t[k]-6)
                break
        rows.append(dict(method=name, pre_rmse=rms((estimates-d[:, 0])[t<6]),
                         early_step_rmse=rms((estimates-d[:, 0])[(t>=6)&(t<8)]),
                         settled_rmse=rms((estimates-d[:, 0])[t>=8]),
                         lock_to_8hz_s=lock, mean_gate_before=float(np.mean(np.asarray(gates)[(t>=3)&(t<6)])),
                         mean_gate_after=float(np.mean(np.asarray(gates)[(t>=6)&(t<9)]))))
        path = OUT / f"traces/frequency_step_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time_s=t, true_tremor=d[:, 0], estimated=estimates,
                            true_frequency=freq, gate=gates, energy_5hz=energy5, energy_8hz=energy8)
    csv_write(OUT / "synthetic/frequency_step.csv", rows)


def run_recorded():
    params = frozen()
    rows, mechanism, ratios = [], [], []
    for subject in SUBJECTS:
        for axis in range(3):
            scale = calibration_band_rms(f"{subject}_1", axis, FS)
            for start in WINDOWS:
                for scenario in SCENARIOS:
                    signal, window = recorded(subject, start, axis, scenario)
                    for name in METHODS:
                        save = OUT / f"traces/{subject}_X_20_fast_{name}.npz" if (subject == "p02" and axis == 0 and start == 20 and scenario == "fast_transition") else None
                        rows.append(dict(subject=subject, axis="XYZ"[axis], start_s=start,
                                         scenario=scenario, source_sha256=window.source_sha256,
                                         scale_deg_s=scale, **trace(signal, name, scale, params, save)))
                        if start == 20 and scenario in ("fast_transition", "spectral_overlap"):
                            save_mech = OUT / f"traces/{subject}_X_{scenario}_{name}_weights.npz" if (subject == "p02" and axis == 0) else None
                            mechanism.append(dict(subject=subject, axis="XYZ"[axis], scenario=scenario,
                                                  method=name, **contamination(signal, name, scale, params, save_mech)))
            print("recorded", subject, "XYZ"[axis], flush=True)
            for scenario in ("fast_transition", "spectral_overlap"):
                for rho in RHO:
                    signal, _ = recorded(subject, 20., axis, scenario, rho)
                    for name in METHODS:
                        ratios.append(dict(subject=subject, axis="XYZ"[axis], scenario=scenario,
                                           rho=rho, **trace(signal, name, scale, params)))
    csv_write(OUT / "recorded/trials.csv", rows)
    csv_write(OUT / "recorded/weight_contamination.csv", mechanism)
    csv_write(OUT / "recorded/ratio_sweep.csv", ratios)


def summarize():
    rows = list(csv.DictReader((OUT / "recorded/trials.csv").open()))
    mechanisms = list(csv.DictReader((OUT / "recorded/weight_contamination.csv").open()))
    summary = []
    for scenario in SCENARIOS:
        for name in METHODS:
            per_subject = []
            for subject in SUBJECTS:
                subset = [r for r in rows if r["scenario"] == scenario and r["method"] == name and r["subject"] == subject]
                m = [r for r in mechanisms if r["scenario"] == scenario and r["method"] == name and r["subject"] == subject]
                metric = dict(subject=subject, scenario=scenario, method=name, n_trials=len(subset))
                for key in ("tremor_nrmse", "pre_nrmse", "transition_nrmse", "post_nrmse",
                            "gate_fast_mean", "gate_tremor_mean", "gate_post_mean", "update_mean_us"):
                    metric[key] = float(np.median([float(r[key]) for r in subset]))
                metric["weight_difference_fast_peak"] = float(np.median([float(r["weight_difference_fast_peak"]) for r in m])) if m else None
                per_subject.append(metric)
                summary.append(metric)
            keys = ("tremor_nrmse", "pre_nrmse", "transition_nrmse", "post_nrmse",
                    "gate_fast_mean", "gate_tremor_mean", "gate_post_mean", "update_mean_us",
                    "weight_difference_fast_peak")
            summary.append(dict(subject="MEDIAN_5_SUBJECTS", scenario=scenario, method=name,
                                **{key: (float(np.median(values)) if values else None)
                                   for key in keys
                                   for values in [[r[key] for r in per_subject if r[key] is not None]]}))
    csv_write(OUT / "recorded/subject_summary.csv", summary)


def plots():
    REPORT.mkdir(parents=True, exist_ok=True)
    for subject, scenario in (("p02", "fast_transition"),):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for name in METHODS:
            with np.load(OUT / f"traces/{subject}_X_20_fast_{name}.npz") as data:
                t = data["time_s"]
                axes[0].plot(t, data["estimate"], label=name, lw=1)
                axes[1].plot(t, data["gate"], label=name, lw=1)
                axes[2].plot(t, data["weight_norm"], label=name, lw=1)
                if name == "Raw_Huber":
                    axes[0].plot(t, data["tremor"], color="black", lw=.8, label="injected proxy")
        for ax in axes:
            ax.axvspan(3, 5, color="gray", alpha=.13)
        axes[0].set_ylabel("deg/s")
        axes[1].set_ylabel("update gate")
        axes[2].set_ylabel("Fourier weight norm")
        axes[2].set_xlabel("s")
        axes[0].legend(fontsize=7)
        fig.savefig(REPORT / "representative_fast_transition.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for name in METHODS:
        with np.load(OUT / f"traces/frequency_step_{name}.npz") as data:
            t = data["time_s"]
            axes[0].plot(t, data["estimated"], lw=1, label=name)
            axes[1].plot(t, data["gate"], lw=1, label=name)
            if name == "Raw_Huber":
                axes[0].plot(t, data["true_tremor"], color="black", lw=.8, label="true")
    axes[0].set_ylabel("deg/s")
    axes[1].set_ylabel("update gate")
    axes[1].set_xlabel("s")
    axes[0].legend(fontsize=7)
    fig.savefig(REPORT / "frequency_step.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for name in METHODS:
        with np.load(OUT / f"traces/p02_X_fast_transition_{name}_weights.npz") as data:
            ax.plot(data["time_s"], data["weight_difference"], label=name)
    ax.axvspan(3, 5, color="gray", alpha=.13)
    ax.set(xlabel="s", ylabel="||w(mixture) − w(proxy-only)||", title="p02 X, fast transition")
    ax.legend(fontsize=7)
    fig.savefig(REPORT / "weight_contamination.png", dpi=150)
    plt.close(fig)
    ratios = list(csv.DictReader((OUT / "recorded/ratio_sweep.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ("fast_transition", "spectral_overlap")):
        for name in METHODS:
            values = []
            for rho in RHO:
                subject_medians = [np.median([float(r["tremor_nrmse"]) for r in ratios
                                              if r["subject"] == subject and r["scenario"] == scenario
                                              and r["method"] == name and float(r["rho"]) == rho])
                                   for subject in SUBJECTS]
                values.append(float(np.median(subject_medians)))
            ax.plot(RHO, values, marker="o", label=name)
        ax.set(xscale="log", xlabel="ρ", ylabel="5-subject median tremor nRMSE", title=scenario)
    axes[0].legend(fontsize=7)
    fig.savefig(REPORT / "ratio_sweep.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("development", "synthetic", "recorded", "summary", "plots", "all"), default="all")
    args = parser.parse_args()
    if args.part in ("development", "all"):
        synthetic_development()
    if args.part in ("synthetic", "all"):
        run_synthetic()
    if args.part in ("recorded", "all"):
        run_recorded()
    if args.part in ("summary", "all"):
        summarize()
    if args.part in ("plots", "all"):
        plots()


if __name__ == "__main__":
    main()
