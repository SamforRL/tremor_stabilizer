"""Run separate paper-style verification and four-method project benchmark.

Run from repository root:
  python -m analysis.tremor_baselines.run_baseline_benchmark --mode all
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .calibration import calibrate_pinheiro_modal
from .interface import create_estimator
from .metrics import evaluate, evaluate_frequency_tracking, run_stream
from .plots import plot_comparison, plot_filter_response
from .recorded import ROOT, load_pinheiro_window
from .signals import BenchmarkCaseConfig, SignalComponents, controlled_tremor, generate_case


DEFAULT_CONFIG = Path(__file__).with_name("benchmark_config.json")
RESULT_BASE = ROOT / "experiments/results/tremor_baselines"
REPORT_BASE = ROOT / "reports/tremor_baselines"
METHODS = ("fixed_bpf", "abpf", "sfft", "bmflc")
CASES = ("slow_smooth", "point_to_point", "fast_transition", "spectral_overlap")
CONTROLLED = ("constant", "drift", "step", "amplitude_modulated", "multifrequency")


def _make_paper_case(method: str, fs: float) -> tuple[SignalComponents, np.ndarray | None]:
    """Deterministic paper-style forms; unavailable noise spectra are omitted."""
    duration = 10.0 if method != "bmflc" else 8.0
    time = np.arange(round(fs * duration)) / fs
    voluntary = np.zeros_like(time)
    truth_frequency = None
    if method == "abpf":
        truth_frequency = np.where(time < 2, 3.0,
                                   np.where(time < 3, 3 + 3 * (time - 2),
                                            np.where(time < 7, 6.0,
                                                     np.where(time < 8, 6 - 3 * (time - 7), 3.0))))
        tremor = 2 * np.sin(2 * np.pi * np.cumsum(truth_frequency) / fs)
        voluntary = np.sin(2 * np.pi * 0.5 * time)
    elif method == "sfft":
        levels = np.array([4, 5, 6, 7, 8, 9, 10, 4], dtype=float)
        boundaries = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
        indices = np.clip(np.searchsorted(boundaries[1:], time, side="right"), 0, 7)
        truth_frequency = levels[indices]
        tremor = 2 * np.sin(2 * np.pi * np.cumsum(truth_frequency) / fs)
        voluntary = np.sin(2 * np.pi * 0.5 * time)
    elif method == "bmflc":
        tremor = 3.5 * np.sin(2 * np.pi * 8 * time) + 2.5 * np.cos(2 * np.pi * 10 * time)
    else:
        tremor = 2 * np.sin(2 * np.pi * 6 * time)
        voluntary = np.sin(2 * np.pi * 0.5 * time)
    config = BenchmarkCaseConfig(sample_rate_hz=fs, duration_s=duration)
    return SignalComponents(time, voluntary[:, None], tremor[:, None],
                            np.zeros((len(time), 1)), (voluntary + tremor)[:, None],
                            f"paper_{method}", config), truth_frequency


def _save_sfft_calibration(estimator, path: Path) -> None:
    calibration = estimator.interpolation
    path.write_text(json.dumps({
        "window_samples": estimator.window_samples,
        "sample_rate_hz": estimator.sample_rate_hz,
        "window_definition": "four convolutions of periodic N-point Blackman; 3 trailing zeros",
        "selected_order": calibration["selected_order"],
        "maximum_allowed_validation_error_bin": calibration["max_error_bin"],
        "candidates": {str(order): {"powers": list(data["powers"]),
                                    "coefficients": data["coefficients"].tolist(),
                                    "validation_max_error_bin": data["validation_max_error_bin"]}
                       for order, data in calibration["candidates"].items()},
    }, indent=2) + "\n")


def _write_metrics(rows: list[dict], result_dir: Path, report_dir: Path,
                   provenance: dict, config: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps({
        "provenance": provenance, "config": config, "metrics": rows,
    }, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (report_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_paper_validation(config: dict) -> list[dict]:
    report_dir = REPORT_BASE / "paper_validation"
    result_dir = RESULT_BASE / "paper_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        method_config = (config[method] if method != "fixed_bpf" else
                         {"sample_rate_hz": 100.0, "order": 4,
                          "low_hz": 3.0, "high_hz": 12.0})
        estimator = create_estimator(method, method_config)
        signal, known_frequency = _make_paper_case(method, method_config["sample_rate_hz"])
        output, diagnostics, timing = run_stream(estimator, signal.mixed_observation_deg_s)
        band = ((6.0, 14.0) if method == "bmflc" else
                (3.0, 6.0) if method == "abpf" else
                (2.0, 12.0) if method == "sfft" else (3.0, 12.0))
        row = evaluate(signal, output, timing, band)
        row.update({"mode": "paper_validation", "method": method})
        if known_frequency is not None:
            row.update(evaluate_frequency_tracking(diagnostics, known_frequency,
                                                    method_config["sample_rate_hz"]))
        rows.append(row)
        method_report = report_dir / method
        method_report.mkdir(exist_ok=True)
        plot_comparison(signal, {method: output}, {method: diagnostics}, method_report,
                        known_frequency_hz=known_frequency)
        if method == "fixed_bpf":
            plot_filter_response(estimator, method_report)
        if method == "sfft":
            _save_sfft_calibration(estimator, result_dir / "sfft_interpolation.json")
        np.savez_compressed(result_dir / f"{method}.npz", time_s=signal.time_s,
                            observation_deg_s=signal.mixed_observation_deg_s,
                            tremor_injected_deg_s=signal.tremor_true_deg_s,
                            voluntary_designed_deg_s=signal.voluntary_true_deg_s,
                            tremor_estimate_deg_s=output,
                            known_frequency_hz=(known_frequency if known_frequency is not None
                                                else np.array([])),
                            estimated_frequency_hz=np.array([
                                item.get("estimated_frequency_hz", np.nan) for item in diagnostics]))
        print(f"paper_validation {method}: tremor RMSE {row['tremor_rmse_deg_s']:.3f} deg/s")
    _write_metrics(rows, result_dir, report_dir, {
        "purpose": "paper-style algorithm checks, not original paper numerical reproduction",
        "synthetic_noise": "omitted because original noise passbands/realizations are unspecified",
        "source": "three primary PDFs under reference_essays/",
    }, config)
    return rows


def run_project_benchmark(config: dict) -> list[dict]:
    fs_window = load_pinheiro_window(
        table=config["recording_table"], start_s=config["evaluation_start_s"],
        duration_s=config["evaluation_duration_s"],
        band_hz=tuple(config["proxy_band_hz"]), context_s=config["proxy_context_s"],
        channel_indices=tuple(config["axis_indices"]),
    )
    modal = calibrate_pinheiro_modal(
        config["recording_table"], tuple(config["axis_indices"]),
        config["evaluation_start_s"], config["evaluation_duration_s"],
        config["calibration_start_s"], config["calibration_preferred_duration_s"],
        tuple(config["proxy_band_hz"]),
    )
    report_dir = REPORT_BASE / "project_benchmark"
    result_dir = RESULT_BASE / "project_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    fs = fs_window.sample_rate_hz
    method_config = {
        "fixed_bpf": {"sample_rate_hz": fs, **config["fixed_bpf"]},
        "abpf": {"sample_rate_hz": fs,
                 "modal_frequency_hz": modal.modal_frequency_hz.tolist(), **config["abpf"]},
        "sfft": {"sample_rate_hz": fs, **config["sfft"]},
        "bmflc": {"sample_rate_hz": fs, **config["bmflc"]},
    }
    estimators = {name: create_estimator(name, method_config[name]) for name in METHODS}
    plot_filter_response(estimators["fixed_bpf"], report_dir)
    _save_sfft_calibration(estimators["sfft"], result_dir / "sfft_interpolation.json")
    case_config = replace(BenchmarkCaseConfig(), sample_rate_hz=fs,
                          duration_s=config["evaluation_duration_s"],
                          tremor_scale=config["tremor_scale"],
                          noise_std_deg_s=config["noise_std_deg_s"],
                          random_seed=config["random_seed"])
    rows = []
    for case_name in CASES:
        signal = generate_case(case_name, fs_window.tremor_band_proxy_deg_s, case_config)
        estimates = {}
        diagnostics = {}
        saved = {"time_s": signal.time_s,
                 "voluntary_designed_deg_s": signal.voluntary_true_deg_s,
                 "tremor_injected_deg_s": signal.tremor_true_deg_s,
                 "noise_injected_deg_s": signal.noise_true_deg_s,
                 "observation_deg_s": signal.mixed_observation_deg_s}
        for name in METHODS:
            estimate, info, elapsed = run_stream(estimators[name],
                                                  signal.mixed_observation_deg_s)
            estimates[name] = estimate
            diagnostics[name] = info
            saved[f"{name}_estimate_deg_s"] = estimate
            saved[f"{name}_frequency_hz"] = np.array([
                item.get("estimated_frequency_hz", np.nan) for item in info])
            row = evaluate(signal, estimate, elapsed, tuple(config["proxy_band_hz"]))
            row.update({"mode": "project_benchmark", "method": name})
            rows.append(row)
        np.savez_compressed(result_dir / f"{case_name}.npz", **saved)
        plot_comparison(signal, estimates, diagnostics, report_dir)
        print(f"project_benchmark {case_name}: " + ", ".join(
            f"{name} {next(r for r in rows if r['case'] == case_name and r['method'] == name)['tremor_rmse_deg_s']:.2f}"
            for name in METHODS) + " deg/s tremor RMSE")
    provenance = {"source": "Pinheiro ET gyro; offline band-limited recorded tremor proxy",
                  "physiological_ground_truth": False,
                  "injected_disturbance_ground_truth": True,
                  "recording_file": fs_window.source_file,
                  "recording_sha256": fs_window.source_sha256,
                  "channels": fs_window.channels,
                  "evaluation_start_s": fs_window.start_s,
                  "evaluation_duration_s": fs_window.duration_s,
                  "proxy_band_hz": fs_window.band_hz,
                  "calibration_start_s": modal.start_s,
                  "calibration_duration_s": modal.duration_s,
                  "calibration_method": modal.method,
                  "modal_frequency_hz": modal.modal_frequency_hz.tolist(),
                  "calibration_and_evaluation_overlap": False,
                  "method_config": method_config,
                  "case_config": asdict(case_config)}
    _write_metrics(rows, result_dir, report_dir, provenance, config)
    sensitivity_rows = []
    for bandwidth in config["sfft_bandwidth_sensitivity_hz"]:
        candidate_config = {**method_config["sfft"],
                            "extraction_bandwidth_hz": float(bandwidth)}
        candidate = create_estimator("sfft", candidate_config)
        for case_name in CASES:
            signal = generate_case(case_name, fs_window.tremor_band_proxy_deg_s,
                                   case_config)
            estimate, _, timing = run_stream(candidate, signal.mixed_observation_deg_s)
            row = evaluate(signal, estimate, timing, tuple(config["proxy_band_hz"]))
            sensitivity_rows.append({"method": "sfft", "case": case_name,
                                     "extraction_bandwidth_hz": bandwidth,
                                     "tremor_rmse_deg_s": row["tremor_rmse_deg_s"],
                                     "voluntary_reconstruction_rmse_deg_s": row[
                                         "voluntary_reconstruction_rmse_deg_s"],
                                     "band_power_reduction_db": row["band_power_reduction_db"],
                                     "diagnostic_lag_ms": row["diagnostic_lag_ms"]})
    (result_dir / "sfft_bandwidth_sensitivity.json").write_text(json.dumps({
        "status": "separate sensitivity analysis; no per-case parameter selection",
        "rows": sensitivity_rows,
    }, indent=2) + "\n")
    with (report_dir / "sfft_bandwidth_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    return rows


def run_controlled_verification() -> list[dict]:
    """All four methods see each of the five declared synthetic disturbances."""
    fs = 100.0
    duration = 10.0
    configs = {
        "fixed_bpf": {"sample_rate_hz": fs, "order": 4, "low_hz": 3.0, "high_hz": 12.0},
        "abpf": {"sample_rate_hz": fs, "modal_frequency_hz": 6.0,
                 "frequency_step_hz": 0.25},
        "sfft": {"sample_rate_hz": fs, "min_frequency_hz": 3.0,
                 "max_frequency_hz": 12.0, "extraction_bandwidth_hz": 3.0},
        "bmflc": {"sample_rate_hz": fs, "low_hz": 3.0, "high_hz": 12.0,
                  "spacing_hz": 0.1, "lms_gain": 0.01},
    }
    estimators = {name: create_estimator(name, configs[name]) for name in METHODS}
    report_dir = REPORT_BASE / "controlled_verification"
    result_dir = RESULT_BASE / "controlled_verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    case_config = BenchmarkCaseConfig(sample_rate_hz=fs, duration_s=duration)
    rows = []
    for kind in CONTROLLED:
        _, tremor, true_frequency = controlled_tremor(kind, fs, duration)
        signal = generate_case("slow_smooth", tremor, case_config)
        signal = SignalComponents(signal.time_s, signal.voluntary_true_deg_s,
                                  signal.tremor_true_deg_s, signal.noise_true_deg_s,
                                  signal.mixed_observation_deg_s, f"controlled_{kind}",
                                  signal.config)
        estimates = {}
        diagnostics = {}
        saved = {"time_s": signal.time_s,
                 "observation_deg_s": signal.mixed_observation_deg_s,
                 "voluntary_designed_deg_s": signal.voluntary_true_deg_s,
                 "tremor_injected_deg_s": signal.tremor_true_deg_s,
                 "frequency_true_hz": (true_frequency if true_frequency is not None
                                       else np.array([]))}
        for name in METHODS:
            estimate, info, timing = run_stream(estimators[name], signal.mixed_observation_deg_s)
            estimates[name] = estimate
            diagnostics[name] = info
            saved[f"{name}_estimate_deg_s"] = estimate
            row = evaluate(signal, estimate, timing, (3.0, 12.0))
            row.update({"method": name, "mode": "controlled_verification"})
            if true_frequency is not None and name in {"abpf", "sfft"}:
                row.update(evaluate_frequency_tracking(info, true_frequency, fs,
                                                        transition_s=(duration / 2 if kind == "step"
                                                                      else None)))
            rows.append(row)
        np.savez_compressed(result_dir / f"{kind}.npz", **saved)
        plot_comparison(signal, estimates, diagnostics, report_dir,
                        known_frequency_hz=true_frequency)
        print(f"controlled_verification {kind}: complete")
    _write_metrics(rows, result_dir, report_dir, {
        "purpose": "controlled known-frequency diagnostics, not patient-data substitute",
        "methods_share_same_input_per_case": True,
        "synthetic_frequency_range": "4-8 Hz; ABPF modal +/-1.5 Hz limits tracking by design",
    }, configs)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("paper_validation", "controlled_verification",
                                           "project_benchmark", "all"),
                        default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode in ("paper_validation", "all"):
        run_paper_validation(config["paper_validation"])
    if args.mode in ("controlled_verification", "all"):
        run_controlled_verification()
    if args.mode in ("project_benchmark", "all"):
        run_project_benchmark(config["project_benchmark"])


if __name__ == "__main__":
    main()
