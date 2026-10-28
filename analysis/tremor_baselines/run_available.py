"""Retained BPF-only preliminary runner from before the four-method benchmark.

python -m analysis.tremor_baselines.run_available
python -m analysis.tremor_baselines.run_available --source controlled_synthetic
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .interface import create_estimator
from .metrics import evaluate, run_stream
from .plots import plot_comparison, plot_filter_response
from .recorded import ROOT, load_pinheiro_window
from .signals import BenchmarkCaseConfig, controlled_tremor, generate_case


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
REPORT_DIR = ROOT / "reports/tremor_baselines_preliminary"
RESULT_DIR = ROOT / "experiments/results/tremor_baselines_preliminary"
CASES = ("slow_smooth", "point_to_point", "fast_transition", "spectral_overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("pinheiro_band_proxy", "controlled_synthetic"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = args.source or config["source"]
    duration = float(config["duration_s"])
    provenance = {"source_type": source}
    if source == "pinheiro_band_proxy":
        recording = load_pinheiro_window(
            table=config["table"], start_s=float(config["start_s"]),
            duration_s=duration, band_hz=tuple(config["proxy_band_hz"]),
            context_s=float(config["context_s"]),
            channel_indices=tuple(config["channel_indices"]),
        )
        tremor = recording.tremor_band_proxy_deg_s
        fs = recording.sample_rate_hz
        known_frequency = None
        provenance.update({
            "source_file": recording.source_file,
            "source_sha256": recording.source_sha256,
            "table": recording.table,
            "channels": recording.channels,
            "start_s": recording.start_s,
            "duration_s": recording.duration_s,
            "proxy_band_hz": recording.band_hz,
            "offline_context_s": recording.context_s,
            "source_label": "offline recorded tremor-band proxy; NOT pure tremor ground truth",
        })
    else:
        fs = 148.14814814814815
        _, tremor, known_frequency = controlled_tremor(
            "constant", fs, duration, axes=len(config["channel_indices"]),
        )
        provenance.update({"source_label": "controlled synthetic constant 6 Hz tremor",
                           "frequency_true_hz": 6.0})
    case_config = replace(BenchmarkCaseConfig(), sample_rate_hz=fs,
                          duration_s=duration, tremor_scale=float(config["tremor_scale"]),
                          noise_std_deg_s=float(config["noise_std_deg_s"]),
                          random_seed=int(config["random_seed"]))
    report_dir = REPORT_DIR / source
    result_dir = RESULT_DIR / source
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    filter_config = {"sample_rate_hz": fs, **config["fixed_bpf"]}
    bpf = create_estimator("fixed_bpf", filter_config)
    plot_filter_response(bpf, report_dir)
    metrics = []
    for case_name in CASES:
        case = generate_case(case_name, tremor, case_config)
        estimate, diagnostics, elapsed = run_stream(bpf, case.mixed_observation_deg_s)
        row = evaluate(case, estimate, elapsed,
                       tremor_band_hz=tuple(config["proxy_band_hz"]))
        row["method"] = "fixed_bpf"
        metrics.append(row)
        np.savez_compressed(result_dir / f"{case_name}.npz",
                            time_s=case.time_s,
                            voluntary_true_deg_s=case.voluntary_true_deg_s,
                            tremor_true_deg_s=case.tremor_true_deg_s,
                            noise_true_deg_s=case.noise_true_deg_s,
                            mixed_observation_deg_s=case.mixed_observation_deg_s,
                            fixed_bpf_estimate_deg_s=estimate)
        plot_comparison(case, {"fixed_bpf": estimate}, {"fixed_bpf": diagnostics},
                        report_dir, known_frequency_hz=known_frequency)
        print(f"{case_name}: tremor RMSE={row['tremor_rmse_deg_s']:.3f} deg/s; "
              f"voluntary RMSE={row['voluntary_reconstruction_rmse_deg_s']:.3f} deg/s")
    (result_dir / "metrics.json").write_text(json.dumps({
        "status": "preliminary fixed-BPF-only verification; no four-method ranking",
        "config": config, "effective_case_config": asdict(case_config),
        "provenance": provenance, "metrics": metrics,
    }, indent=2) + "\n")
    (report_dir / "README.md").write_text(
        "# Preliminary fixed-BPF check\n\n"
        "This older preliminary directory contains only the fixed BPF. "
        "For the implemented four-method benchmark use "
        "`python -m analysis.tremor_baselines.run_baseline_benchmark --mode all`; "
        "see `analysis/tremor_baselines/REPRODUCTION_NOTES.md`.\n\n"
        f"Source: {provenance['source_label']}.\n\n"
        f"Filter: prototype order {bpf.order}, {bpf.low_hz:g}–{bpf.high_hz:g} Hz "
        f"at {fs:.5f} Hz; digital order {2*bpf.order}. "
        "See `fixed_bpf_response.png` and `.csv` for magnitude, phase and group delay.\n\n"
        "Plots use the same signal and PSD conventions intended for every "
        "eventual method. Raw signals and numeric metrics are stored in "
        f"`experiments/results/tremor_baselines_preliminary/{source}/`.\n"
    )


if __name__ == "__main__":
    main()
