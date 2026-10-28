"""Round-3 cross-subject evaluation and isolated ABPF/BMFLC experiments.

No baseline algorithm files are modified. Development selection uses synthetic
signals only; final Pinheiro evaluation uses frozen choices and identical
subject/window/axis/scenario inputs for every method.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .abpf import tustin_abpf_biquad
from .calibration import calibrate_pinheiro_modal
from .interface import create_estimator
from .metrics import evaluate, run_stream
from .recorded import ROOT, load_pinheiro_window
from .round3_variants import FixedCenterABPF, RobustBMFLC
from .signals import BenchmarkCaseConfig, SignalComponents, controlled_tremor, generate_case


ROOT_CONFIG = json.loads((Path(__file__).with_name("benchmark_config.json")).read_text())["project_benchmark"]
RESULT = ROOT / "experiments/results/tremor_baselines/round3"
REPORT = ROOT / "reports/tremor_baselines/round3"
SUBJECTS = tuple(f"p{i:02d}" for i in range(1, 6))
WINDOWS = (20.0, 33.0, 46.0)
CASES = ("slow_smooth", "point_to_point", "fast_transition", "spectral_overlap")
BASELINES = ("fixed_bpf", "sfft", "bmflc_lms", "abpf_adaptive")
PROPOSED = ("bmflc_huber", "bmflc_leaky", "abpf_fixed")
ALL_METHODS = BASELINES + PROPOSED


def csv_write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def first_table(subject: str) -> str:
    """Predeclared choice independent of tremor strength or method results."""
    return f"{subject}_1"


def run_data_audit():
    rows = []
    for subject in SUBJECTS:
        for path in sorted((ROOT / "data/processed").glob(f"pinheiro_{subject}_*_lumbrical_gyro.npz")):
            with np.load(path) as data:
                t = np.asarray(data["time_s"])
                gyro = np.asarray(data["gyro_deg_s"])
                table = str(data["table"])
                channels = [str(value) for value in data["channels"]]
            fs = 1 / float(np.median(np.diff(t)))
            rows.append({"subject": subject, "table": table, "samples": len(t),
                         "start_s": float(t[0]), "end_s": float(t[-1]),
                         "duration_s": float(t[-1] - t[0]), "fs_hz": fs,
                         "three_gyro_axes_deg_s": gyro.shape == (len(t), 3)
                         and all("Gyro" in name and "degsec" in name for name in channels),
                         "monotonic_timestamps": bool(np.all(np.diff(t) > 0)),
                         "finite_values": bool(np.all(np.isfinite(gyro))),
                         "sufficient_for_0_15_calibration_and_46_58_eval": float(t[-1]) >= 59,
                         "selected_primary_table": table == first_table(subject)})
    csv_write(REPORT / "data_audit/tables.csv", rows)
    return rows


def calibration_band_rms(table: str, axis: int, fs: float, duration_s: float = 15.0) -> float:
    """Scale for Huber threshold from independent 0-15 s raw gyro only."""
    path = ROOT / f"data/processed/pinheiro_{table}_lumbrical_gyro.npz"
    with np.load(path) as recording:
        t = recording["time_s"]
        gyro = recording["gyro_deg_s"][:, axis]
    uniform = np.arange(round(duration_s * fs)) / fs
    sample = np.interp(uniform, t, gyro)
    freq = np.fft.rfftfreq(len(sample), 1 / fs)
    spectrum = np.fft.rfft(sample)
    spectrum[(freq < 3) | (freq > 12)] = 0
    return max(rms(np.fft.irfft(spectrum, n=len(sample))), 1e-6)


def all_configs(fs: float, modal_hz: float, huber_delta: float, leak_lambda: float):
    fixed_band = {"sample_rate_hz": fs, **ROOT_CONFIG["fixed_bpf"]}
    sfft = {"sample_rate_hz": fs, **ROOT_CONFIG["sfft"]}
    bmflc = {"sample_rate_hz": fs, **ROOT_CONFIG["bmflc"]}
    abpf = {"sample_rate_hz": fs, "modal_frequency_hz": modal_hz,
            **ROOT_CONFIG["abpf"]}
    return {"fixed_bpf": fixed_band, "sfft": sfft,
            "bmflc_lms": bmflc,
            "bmflc_huber": {**bmflc, "variant": "huber", "huber_delta": huber_delta},
            "bmflc_leaky": {**bmflc, "variant": "leaky", "leak_lambda": leak_lambda},
            "abpf_adaptive": abpf,
            "abpf_fixed": {"sample_rate_hz": fs, "modal_frequency_hz": modal_hz}}


def create_method(name: str, config: dict):
    if name == "bmflc_lms":
        return create_estimator("bmflc", config)
    if name == "abpf_adaptive":
        return create_estimator("abpf", config)
    if name == "abpf_fixed":
        return FixedCenterABPF(**config)
    if name in ("bmflc_huber", "bmflc_leaky"):
        return RobustBMFLC(**config)
    return create_estimator(name, config)


def synthetic_development(fs: float, scale: float, transient: bool):
    """Independent synthetic tones and a different 0.6 s/8 deg task pulse."""
    t = np.arange(round(12 * fs)) / fs
    tremor = scale * (np.sin(2 * np.pi * (6.35 * t + 0.12 * t*t / 12))
                      + 0.4 * np.sin(2 * np.pi * 8.15 * t + 0.4))
    voluntary = 0.4 * scale * np.sin(2 * np.pi * 0.22 * t)
    if transient:
        phase = (t - 3.2) / 0.6
        mask = (phase >= 0) & (phase <= 1)
        u = phase[mask]
        voluntary[mask] += (8.0 / 0.6) * (30*u*u - 60*u**3 + 30*u**4)
    return t, tremor, voluntary, tremor + voluntary


def run_development() -> dict:
    """Freeze δ multiplier and leakage using synthetic data, not final windows."""
    out = RESULT / "bmflc_development"; report = REPORT / "bmflc_development"
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    fs = 148.14814814814815
    scales = (1.0, 5.0, 25.0, 100.0)
    huber_factors = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    # Cap at 1e-3 per sample to preserve the requested *small* leak regime.
    leaks = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)
    rows = []
    for scale in scales:
        data = {label: synthetic_development(fs, scale, label == "transient")
                for label in ("steady", "transient")}
        cal_scale = rms(data["steady"][1])  # synthetic independent known scale
        candidates = [("bmflc_lms", None)]
        candidates += [("bmflc_huber", factor) for factor in huber_factors]
        candidates += [("bmflc_leaky", leak) for leak in leaks]
        for method, value in candidates:
            config = {"sample_rate_hz": fs, **ROOT_CONFIG["bmflc"]}
            if method == "bmflc_huber":
                config.update(variant="huber", huber_delta=value * cal_scale)
            elif method == "bmflc_leaky":
                config.update(variant="leaky", leak_lambda=value)
            estimator = create_method(method, config)
            for scenario, (t, tremor, voluntary, observed) in data.items():
                output, _, _ = run_stream(estimator, observed[:, None])
                output = output[:, 0]
                mask_steady = (t >= 1) & (t < 3)
                mask_transition = (t >= 3) & (t < 5)
                mask_recovery = (t >= 5) & (t < 12)
                rows.append({"scale": scale, "calibration_scale": cal_scale,
                             "method": method, "parameter": value, "scenario": scenario,
                             "steady_rmse": rms(output[mask_steady] - tremor[mask_steady]),
                             "transition_rmse": rms(output[mask_transition] - tremor[mask_transition]),
                             "recovery_rmse": rms(output[mask_recovery] - tremor[mask_recovery]),
                             "full_rmse": rms(output - tremor)})
    csv_write(report / "candidate_trials.csv", rows)
    selection = {}
    score_rows = []
    for method, candidates in (("bmflc_huber", huber_factors), ("bmflc_leaky", leaks)):
        for value in candidates:
            ratios = []; steady_ratios = []
            for scale in scales:
                baseline = next(r for r in rows if r["scale"] == scale and r["method"] == "bmflc_lms" and r["scenario"] == "transient")
                trial = next(r for r in rows if r["scale"] == scale and r["method"] == method and r["parameter"] == value and r["scenario"] == "transient")
                ratios.append((trial["transition_rmse"] + trial["recovery_rmse"])
                              / (baseline["transition_rmse"] + baseline["recovery_rmse"]))
                steady_ratios.append(trial["steady_rmse"] / baseline["steady_rmse"])
            # Predeclared aggregate penalty: reward transient/recovery gains,
            # penalize >15% steady deterioration across the synthetic scales.
            score = float(np.mean(ratios) + 3 * np.mean(np.maximum(0, np.array(steady_ratios) - 1.15)))
            score_rows.append({"method": method, "parameter": value, "score": score,
                               "mean_transient_recovery_ratio": float(np.mean(ratios)),
                               "mean_steady_ratio": float(np.mean(steady_ratios))})
        chosen = min((r for r in score_rows if r["method"] == method), key=lambda r: r["score"])
        selection[method] = chosen["parameter"]
    csv_write(report / "selection_scores.csv", score_rows)
    json_write(out / "frozen_selection.json", {
        "huber_factor_of_independent_calibration_band_rms": selection["bmflc_huber"],
        "leak_lambda": selection["bmflc_leaky"],
        "synthetic_scales_deg_s": scales,
        "huber_candidates": huber_factors, "leak_candidates": leaks,
        "objective": "mean transition+recovery RMSE ratio to LMS + 3*mean(max(steady RMSE ratio-1.15,0))",
        "no_final_window_used_for_selection": True,
        "original_effective_lms_gain": "2*0.01=0.02; retained in both variants"})
    return selection


def load_frozen_selection():
    path = RESULT / "bmflc_development/frozen_selection.json"
    if not path.is_file():
        raise FileNotFoundError("Run --part development before final evaluation")
    config = json.loads(path.read_text())
    return config["huber_factor_of_independent_calibration_band_rms"], config["leak_lambda"]


def _period_rows(signal, output, diagnostics, method, subject, table, axis, start):
    if method not in ("bmflc_lms", "bmflc_huber", "bmflc_leaky"):
        return []
    nfreq = (len(diagnostics[0]["adaptive_weights"][0]) - 1) // 2
    weights = np.stack([d["adaptive_weights"][0, :2*nfreq] for d in diagnostics])
    norm = np.linalg.norm(weights, axis=1)
    error = np.array([d["internal_error"][0] for d in diagnostics])
    robust = np.array([d.get("robust_update_error", d["internal_error"])[0] for d in diagnostics])
    rows = []
    for label, a, b in (("pre", 0, 3), ("transition", 3, 5), ("recovery", 5, 12)):
        mask = (signal.time_s >= a) & (signal.time_s < b)
        rows.append({"subject": subject, "table": table, "axis": axis, "start_s": start,
                     "method": method, "period": label,
                     "rmse_deg_s": rms(output[mask, 0] - signal.tremor_true_deg_s[mask, 0]),
                     "mean_weight_norm": float(np.mean(norm[mask])),
                     "peak_weight_norm": float(np.max(norm[mask])),
                     "mean_abs_prediction_error_deg_s": float(np.mean(abs(error[mask]))),
                     "mean_abs_update_error_deg_s": float(np.mean(abs(robust[mask])))})
    return rows


def _recovery_time(signal, output, threshold, after_s=3.35):
    """First 0.5 s whose rolling RMSE stays below development threshold."""
    fs = signal.config.sample_rate_hz
    error = (output[:, 0] - signal.tremor_true_deg_s[:, 0]) ** 2
    width = round(0.5 * fs)
    rolling = np.sqrt(np.convolve(error, np.ones(width) / width, mode="valid"))
    start = int(np.searchsorted(signal.time_s, after_s))
    for i in range(start, len(rolling)):
        if rolling[i] <= threshold:
            return float(signal.time_s[i] - after_s)
    return None


def _save_weight_trace(subject, method, signal, estimator, output, diagnostics):
    """Representative first-window X-axis trace for each subject/variant."""
    raw_dir = RESULT / "bmflc_final_traces" / subject
    plot_dir = REPORT / "bmflc_final_traces" / subject
    raw_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)
    nfreq = len(estimator.frequencies_hz)
    weights = np.stack([d["adaptive_weights"][0, :2*nfreq] for d in diagnostics])
    magnitude = np.hypot(weights[:, :nfreq], weights[:, nfreq:])
    norm = np.linalg.norm(weights, axis=1)
    bias = np.array([d["intended_motion_bias"][0] for d in diagnostics])
    residual = np.array([d["internal_error"][0] for d in diagnostics])
    update_error = np.array([d.get("robust_update_error", d["internal_error"])[0] for d in diagnostics])
    np.savez_compressed(raw_dir / f"{method}.npz", time_s=signal.time_s,
                        frequencies_hz=estimator.frequencies_hz,
                        weight_magnitude=magnitude, weight_norm=norm,
                        bias_deg_s=bias, residual_deg_s=residual,
                        update_error_deg_s=update_error,
                        tremor_estimate_deg_s=output[:, 0],
                        tremor_proxy_deg_s=signal.tremor_true_deg_s[:, 0])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(signal.time_s, signal.tremor_true_deg_s[:, 0], color="black", label="injected proxy")
    axes[0].plot(signal.time_s, output[:, 0], label=method)
    axes[0].legend(); axes[0].set_ylabel("deg/s")
    axes[1].plot(signal.time_s, norm, label="Fourier weight norm")
    axes[1].plot(signal.time_s, bias, label="bias estimate")
    axes[1].legend(); axes[1].set_ylabel("state")
    axes[2].plot(signal.time_s, residual, label="prediction residual")
    axes[2].plot(signal.time_s, update_error, label="update residual")
    axes[2].legend(); axes[2].set_ylabel("deg/s"); axes[2].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(plot_dir / f"{method}_trace.png", dpi=135); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(magnitude.T, origin="lower", aspect="auto",
                      extent=[0, signal.time_s[-1], estimator.frequencies_hz[0], estimator.frequencies_hz[-1]])
    ax.set(xlabel="time (s)", ylabel="Fourier basis frequency (Hz)", title=f"{subject}: {method} weights")
    fig.colorbar(image, ax=ax, label="weight magnitude (deg/s)")
    fig.tight_layout(); fig.savefig(plot_dir / f"{method}_heatmap.png", dpi=135); plt.close(fig)


def _distribution(values):
    v = np.asarray(values, dtype=float)
    return {"min": float(np.min(v)), "q25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q75": float(np.percentile(v, 75)),
            "max": float(np.max(v))}


def _plot_final_comparison(by_subject, pairs):
    folder = REPORT / "final_comparison"; folder.mkdir(parents=True, exist_ok=True)
    compact = [{"case": r["case"], "method": r["method"],
                "across_subject_median_nrmse": float(np.median([
                    x["nrmse_median"] for x in by_subject
                    if x["case"] == r["case"] and x["method"] == r["method"]]))}
               for r in by_subject if r["subject"] == SUBJECTS[0]]
    csv_write(folder / "core_nrmse.csv", compact)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, case in zip(axes.flat, CASES):
        matrix = np.array([[next(r["nrmse_median"] for r in by_subject
                                 if r["subject"] == subject and r["case"] == case and r["method"] == method)
                            for method in ALL_METHODS] for subject in SUBJECTS])
        # A log color scale retains legibility across very different proxy RMS.
        im = ax.imshow(np.log10(np.maximum(matrix, 1e-2)), vmin=-1, vmax=1.2, cmap="viridis", aspect="auto")
        ax.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS)
        ax.set_xticks(np.arange(len(ALL_METHODS)), ALL_METHODS, rotation=35, ha="right", fontsize=7)
        ax.set_title(case)
        for i in range(len(SUBJECTS)):
            for j in range(len(ALL_METHODS)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i,j] < 0.4 or matrix[i,j] > 3 else "black", fontsize=6)
    fig.colorbar(im, ax=axes, label="log10(normalized RMSE)", shrink=0.8)
    fig.savefig(folder / "subject_scenario_nrmse.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for subject in SUBJECTS:
        subset = [r for r in pairs if r["subject"] == subject]
        ax.scatter([r["fixed_rmse_deg_s"] for r in subset],
                   [r["adaptive_rmse_deg_s"] for r in subset], s=15, label=subject, alpha=0.7)
    upper = max(max(r["fixed_rmse_deg_s"], r["adaptive_rmse_deg_s"]) for r in pairs)
    ax.plot([0, upper], [0, upper], color="black", ls="--", lw=0.8)
    ax.set(xlabel="fixed-center ABPF RMSE (deg/s)", ylabel="adaptive ABPF RMSE (deg/s)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORT / "abpf_ablation/recorded_pairs.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ("bmflc_lms", "bmflc_huber", "bmflc_leaky"):
        xs=[];ys=[]
        for subject in SUBJECTS:
            xs.append(next(r["nrmse_median"] for r in by_subject if r["subject"]==subject and r["case"]=="slow_smooth" and r["method"]==name))
            ys.append(next(r["nrmse_median"] for r in by_subject if r["subject"]==subject and r["case"]=="fast_transition" and r["method"]==name))
        ax.scatter(xs,ys,label=name,s=45)
        for x,y,subject in zip(xs,ys,SUBJECTS):
            ax.annotate(subject,(x,y),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.set(xlabel="slow-smooth normalized RMSE",ylabel="fast-transition normalized RMSE",
           title="BMFLC steady quality vs transient robustness")
    ax.set_yscale("log"); ax.legend(); fig.tight_layout()
    fig.savefig(folder / "bmflc_tradeoff.png", dpi=150); plt.close(fig)


def run_final() -> dict:
    factor, leak = load_frozen_selection()
    result = RESULT / "cross_subject_final"; report = REPORT / "cross_subject_final"
    result.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    baseline_rows = []; proposed_rows = []; period_rows = []; window_rows = []
    for subject in SUBJECTS:
        table = first_table(subject)
        for start in WINDOWS:
            for axis_index, axis in enumerate("XYZ"):
                window = load_pinheiro_window(table, start, 12, (3, 12), 1, (axis_index,))
                fs = window.sample_rate_hz
                cal = calibrate_pinheiro_modal(table, (axis_index,), start, 12)
                modal = float(cal.modal_frequency_hz[0])
                scale = calibration_band_rms(table, axis_index, fs)
                configs = all_configs(fs, modal, factor * scale, leak)
                methods = {name: create_method(name, configs[name]) for name in ALL_METHODS}
                window_rows.append({"subject": subject, "table": table, "axis": axis,
                                    "start_s": start, "end_s": start + 12,
                                    "modal_hz": modal, "calibration_band_rms_deg_s": scale,
                                    "huber_delta_deg_s": factor * scale,
                                    "recording_sha256": window.source_sha256,
                                    "sample_rate_hz": fs})
                cfg = BenchmarkCaseConfig(sample_rate_hz=fs, duration_s=12,
                                          tremor_scale=1, noise_std_deg_s=0,
                                          random_seed=20261026)
                for case in CASES:
                    signal = generate_case(case, window.tremor_band_proxy_deg_s, cfg)
                    for name in ALL_METHODS:
                        estimator = methods[name]
                        output, diagnostics, elapsed = run_stream(estimator, signal.mixed_observation_deg_s)
                        metric = evaluate(signal, output, elapsed, (3, 12))
                        row = {"subject": subject, "table": table, "axis": axis,
                               "start_s": start, "case": case, "method": name,
                               "injected_proxy_rms_deg_s": rms(signal.tremor_true_deg_s),
                               "rmse_deg_s": metric["tremor_rmse_deg_s"],
                               "nrmse": metric["tremor_nrmse_by_true_rms"],
                               "band_power_reduction_db": metric["band_power_reduction_db"],
                               "diagnostic_lag_ms": metric["diagnostic_lag_ms"],
                               "mean_update_us": metric["mean_update_us"],
                               "max_update_us": metric["max_update_us"]}
                        for label, a, b in (("pre", 0, 3), ("transition", 3, 5), ("post", 5, 12)):
                            mask = (signal.time_s >= a) & (signal.time_s < b)
                            row[f"{label}_rmse_deg_s"] = rms(output[mask] - signal.tremor_true_deg_s[mask])
                        if name in ("bmflc_lms", "bmflc_huber", "bmflc_leaky"):
                            # Recovery is first rolling 0.5 s below 1.5x the
                            # method's own pre-transition error; diagnostic
                            # only, no parameter/gating dependence.
                            threshold = 1.5 * row["pre_rmse_deg_s"]
                            row["recovery_time_s"] = _recovery_time(signal, output, threshold)
                            if case == "fast_transition":
                                period_rows.extend(_period_rows(signal, output, diagnostics,
                                                                name, subject, table, axis, start))
                                if start == 20 and axis == "X":
                                    _save_weight_trace(subject, name, signal, estimator, output, diagnostics)
                        else:
                            row["recovery_time_s"] = None
                        (baseline_rows if name in BASELINES else proposed_rows).append(row)
        print("completed", subject, flush=True)
    csv_write(report / "baseline_trials.csv", baseline_rows)
    csv_write(report / "proposed_trials.csv", proposed_rows)
    csv_write(report / "windows_and_calibration.csv", window_rows)
    csv_write(REPORT / "bmflc_final_traces/period_diagnostics.csv", period_rows)
    all_rows = baseline_rows + proposed_rows
    by_subject = []
    for subject in SUBJECTS:
        for case in CASES:
            for name in ALL_METHODS:
                subset = [r for r in all_rows if r["subject"] == subject and r["case"] == case and r["method"] == name]
                record = {"subject": subject, "case": case, "method": name,
                          "n_axis_windows": len(subset)}
                for key in ("rmse_deg_s", "nrmse", "pre_rmse_deg_s", "transition_rmse_deg_s",
                            "post_rmse_deg_s", "band_power_reduction_db", "mean_update_us"):
                    record.update({f"{key}_{stat}": value for stat, value in
                                   _distribution([r[key] for r in subset]).items()})
                by_subject.append(record)
        csv_write(report / "by_subject" / f"{subject}.csv",
                  [r for r in by_subject if r["subject"] == subject])
    csv_write(report / "by_subject_medians.csv", by_subject)
    across = []
    for case in CASES:
        for name in ALL_METHODS:
            subset = [r for r in by_subject if r["case"] == case and r["method"] == name]
            record = {"case": case, "method": name, "n_subjects": len(subset)}
            for key in ("rmse_deg_s_median", "nrmse_median", "pre_rmse_deg_s_median",
                        "transition_rmse_deg_s_median", "post_rmse_deg_s_median",
                        "band_power_reduction_db_median", "mean_update_us_median"):
                record.update({f"{key}_{stat}": value for stat, value in
                               _distribution([r[key] for r in subset]).items()})
            across.append(record)
    csv_write(report / "across_subject_distribution.csv", across)
    # Paired ABPF comparisons use the exact same subject/table/axis/window/case.
    pairs = []
    for fixed in proposed_rows:
        if fixed["method"] != "abpf_fixed":
            continue
        adaptive = next(r for r in baseline_rows if r["method"] == "abpf_adaptive"
                        and all(r[k] == fixed[k] for k in ("subject", "table", "axis", "start_s", "case")))
        pairs.append({"subject": fixed["subject"], "axis": fixed["axis"],
                      "start_s": fixed["start_s"], "case": fixed["case"],
                      "fixed_rmse_deg_s": fixed["rmse_deg_s"],
                      "adaptive_rmse_deg_s": adaptive["rmse_deg_s"],
                      "fixed_minus_adaptive_deg_s": fixed["rmse_deg_s"] - adaptive["rmse_deg_s"],
                      "relative_improvement_pct": 100 * (fixed["rmse_deg_s"] - adaptive["rmse_deg_s"]) / fixed["rmse_deg_s"],
                      "fixed_lag_ms": fixed["diagnostic_lag_ms"],
                      "adaptive_lag_ms": adaptive["diagnostic_lag_ms"]})
    csv_write(REPORT / "abpf_ablation/recorded_pairs.csv", pairs)
    _plot_final_comparison(by_subject, pairs)
    json_write(result / "protocol.json", {"subjects": SUBJECTS,
                 "table_rule": "first complete table _1 for each subject, predeclared before results",
                 "windows_s": WINDOWS, "axes": "XYZ", "cases": CASES,
                 "calibration_s": [0, 15], "proxy_band_hz": [3, 12],
                 "noise_std_deg_s": 0, "random_seed": 20261026,
                 "baseline_methods": BASELINES, "proposed_methods": PROPOSED,
                 "huber_factor": factor, "leak_lambda": leak,
                 "subject_count": len(SUBJECTS), "axis_windows_per_subject": 9,
                 "source_tables_available": "17 complete tables; primary uses one table per person to balance subjects",
                 "excluded_short_table": "p01_6 has 3.99 s only"})
    return {"baseline_rows": baseline_rows, "proposed_rows": proposed_rows,
            "by_subject": by_subject, "across": across, "pairs": pairs,
            "periods": period_rows}


def run_controlled_abpf():
    report = REPORT / "abpf_ablation"; out = RESULT / "abpf_ablation"
    report.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    fs = 100.0; duration = 10.0; rows = []
    for kind in ("constant", "drift", "step", "amplitude_modulated", "multifrequency"):
        _, tremor, freq = controlled_tremor(kind, fs, duration)
        if kind == "multifrequency":
            continue  # no unique frequency target; not requested here
        signal = generate_case("slow_smooth", tremor,
                               BenchmarkCaseConfig(sample_rate_hz=fs, duration_s=duration))
        trajectories = {}
        for name, estimator in (("abpf_adaptive", create_estimator("abpf", {
                                 "sample_rate_hz": fs, "modal_frequency_hz": 6.0,
                                 "frequency_step_hz": 0.25, "limit_offset_hz": 1.5})),
                                ("abpf_fixed", FixedCenterABPF(fs, 6.0))):
            output, diag, timing = run_stream(estimator, signal.mixed_observation_deg_s)
            metrics = evaluate(signal, output, timing)
            center = np.array([d["estimated_frequency_hz"] for d in diag])
            trajectories[name] = center
            phase_error = None
            if kind in ("constant", "amplitude_modulated"):
                mask = signal.time_s >= 2
                t = signal.time_s[mask]
                basis = np.column_stack([np.sin(2*np.pi*6*t), np.cos(2*np.pi*6*t), np.ones(len(t))])
                true_coef = np.linalg.lstsq(basis, signal.tremor_true_deg_s[mask, 0], rcond=None)[0]
                estimated_coef = np.linalg.lstsq(basis, output[mask, 0], rcond=None)[0]
                true_phase = np.arctan2(true_coef[1], true_coef[0])
                estimated_phase = np.arctan2(estimated_coef[1], estimated_coef[0])
                phase_error = float(np.degrees(np.angle(np.exp(1j*(estimated_phase-true_phase)))))
            rows.append({"case": kind, "method": name,
                         "rmse_deg_s": metrics["tremor_rmse_deg_s"],
                         "nrmse": metrics["tremor_nrmse_by_true_rms"],
                         "diagnostic_lag_ms": metrics["diagnostic_lag_ms"],
                         "steady_6hz_phase_error_deg": phase_error,
                         "frequency_mae_hz": float(np.mean(abs(center - freq))),
                         "center_start_hz": float(center[0]),
                         "center_end_hz": float(center[-1])})
            np.savez_compressed(out / f"{kind}_{name}.npz", time_s=signal.time_s,
                                true_tremor_deg_s=signal.tremor_true_deg_s[:, 0],
                                estimate_deg_s=output[:, 0],
                                true_frequency_hz=freq, center_frequency_hz=center)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(signal.time_s, freq, color="black", ls="--", label="known synthetic frequency")
        for name, center in trajectories.items():
            ax.plot(signal.time_s, center, label=name)
        ax.set(xlabel="time (s)", ylabel="frequency (Hz)", title=f"ABPF center ablation: {kind}")
        ax.legend(); fig.tight_layout(); fig.savefig(report / f"{kind}_centers.png", dpi=145); plt.close(fig)
    csv_write(report / "controlled_cases.csv", rows)
    # Recorded-proxy trajectory example uses the exact final protocol, with
    # independent preceding calibration and no instantaneous frequency truth.
    example_table = "p02_1"
    window = load_pinheiro_window(example_table, 20, 12, (3, 12), 1, (0,))
    modal = float(calibrate_pinheiro_modal(example_table, (0,), 20, 12).modal_frequency_hz[0])
    example = generate_case("fast_transition", window.tremor_band_proxy_deg_s,
                            BenchmarkCaseConfig(sample_rate_hz=window.sample_rate_hz,
                                                duration_s=12))
    trajectories = {}
    for name, estimator in (("abpf_adaptive", create_estimator("abpf", {
                                "sample_rate_hz": window.sample_rate_hz,
                                "modal_frequency_hz": modal, **ROOT_CONFIG["abpf"]})),
                            ("abpf_fixed", FixedCenterABPF(window.sample_rate_hz, modal))):
        output, diag, _ = run_stream(estimator, example.mixed_observation_deg_s)
        trajectories[name] = np.array([d["estimated_frequency_hz"] for d in diag])
        np.savez_compressed(out / f"recorded_p02_x_fast_{name}.npz",
                            time_s=example.time_s, center_frequency_hz=trajectories[name],
                            estimate_deg_s=output[:, 0],
                            injected_proxy_deg_s=example.tremor_true_deg_s[:, 0])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, center in trajectories.items():
        ax.plot(example.time_s, center, label=name)
    ax.axvspan(3, 3.35, color="orange", alpha=0.15, label="designed active turn")
    ax.set(xlabel="time (s)", ylabel="center frequency (Hz)",
           title="Recorded proxy p02 X: no instantaneous frequency ground truth")
    ax.legend(); fig.tight_layout(); fig.savefig(report / "recorded_p02_x_fast_centers.png", dpi=145); plt.close(fig)
    return rows


def run_response():
    out = REPORT / "abpf_response"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in SUBJECTS:
        table = first_table(subject)
        fs = load_pinheiro_window(table, 20, 12, (3, 12), 1, (0,)).sample_rate_hz
        for axis_idx, axis in enumerate("XYZ"):
            modal = float(calibrate_pinheiro_modal(table, (axis_idx,), 20, 12).modal_frequency_hz[0])
            for position, center in (("lower", modal - 1.5), ("modal", modal), ("upper", modal + 1.5)):
                b0, b1, b2, _, a1, a2 = tustin_abpf_biquad(center, fs)
                frequency = np.linspace(0, fs/2, 100001)
                z = np.exp(-2j * np.pi * frequency / fs)
                transfer = (b0+b1*z+b2*z*z)/(1+a1*z+a2*z*z)
                mag2 = abs(transfer)**2
                idx = np.flatnonzero(mag2 >= np.max(mag2)/2)
                phase = np.unwrap(np.angle(transfer))
                delay_ms = -1000 * np.gradient(phase, frequency) / (2*np.pi)
                lookup = {f"group_delay_at_modal_{offset:+g}hz_ms": float(np.interp(modal + offset, frequency, delay_ms))
                          for offset in (-0.5, 0, 0.5)}
                rows.append({"subject": subject, "axis": axis, "center_position": position,
                             "configured_center_hz": center,
                             "actual_peak_hz": float(frequency[np.argmax(mag2)]),
                             "minus_3db_lower_hz": float(frequency[idx[0]]),
                             "minus_3db_upper_hz": float(frequency[idx[-1]]),
                             "minus_3db_width_hz": float(frequency[idx[-1]]-frequency[idx[0]]),
                             **lookup})
    csv_write(out / "frequency_response.csv", rows)
    p02 = [r for r in rows if r["subject"] == "p02" and r["axis"] == "X"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fs = load_pinheiro_window("p02_1", 20, 12, (3,12), 1, (0,)).sample_rate_hz
    f = np.linspace(0.1, 20, 2500)
    for r in p02:
        b0,b1,b2,_,a1,a2=tustin_abpf_biquad(r["configured_center_hz"],fs)
        z=np.exp(-2j*np.pi*f/fs);h=(b0+b1*z+b2*z*z)/(1+a1*z+a2*z*z)
        axes[0].plot(f,20*np.log10(np.maximum(abs(h),1e-10)),label=r["center_position"])
        axes[1].plot(f,-1000*np.gradient(np.unwrap(np.angle(h)),f)/(2*np.pi),label=r["center_position"])
    axes[0].axhline(-3.0103,color="black",ls="--",lw=0.8)
    axes[0].set(ylabel="magnitude (dB)",ylim=(-30,2)); axes[0].legend()
    axes[1].set(xlabel="frequency (Hz)",ylabel="group delay (ms)",ylim=(0,300))
    fig.tight_layout();fig.savefig(out/"p02_x_response.png",dpi=150);plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", "audit", "development", "final", "abpf_controlled", "response"), default="all")
    args = parser.parse_args(); started = time.monotonic()
    if args.part in ("all", "audit"):
        print("audited tables", len(run_data_audit()), flush=True)
    if args.part in ("all", "development"):
        print("development selection", run_development(), flush=True)
    if args.part in ("all", "response"):
        print("response rows", len(run_response()), flush=True)
    if args.part in ("all", "abpf_controlled"):
        print("controlled ABPF rows", len(run_controlled_abpf()), flush=True)
    if args.part in ("all", "final"):
        outcome = run_final()
        print("final trials", len(outcome["baseline_rows"])+len(outcome["proposed_rows"]), flush=True)
    print(f"completed in {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
