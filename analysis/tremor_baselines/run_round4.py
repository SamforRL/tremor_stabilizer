"""Round 4: separate kinematic voluntary motion and BMFLC tremor priors.

Run development before any final evaluation.  Previous algorithms/results are
read but never modified.  All parameter selection uses controlled synthetic
signals; five Pinheiro subjects remain final evaluation only.
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

from .interface import create_estimator
from .metrics import evaluate, run_stream
from .recorded import ROOT, load_pinheiro_window
from .round3_variants import RobustBMFLC
from .round4_estimators import JointKinematicBMFLCKF, KinematicKFThenBMFLC
from .signals import BenchmarkCaseConfig, SignalComponents, controlled_tremor, generate_case, _minimum_jerk_velocity


RESULT = ROOT / "experiments/results/tremor_baselines/round4"
REPORT = ROOT / "reports/tremor_baselines/round4"
CONFIG = json.loads((Path(__file__).with_name("benchmark_config.json")).read_text())["project_benchmark"]
SUBJECTS = tuple(f"p{i:02d}" for i in range(1, 6))
WINDOWS = (20.0, 33.0, 46.0)
CASES = ("slow_smooth", "point_to_point", "fast_transition", "spectral_overlap")
METHODS = ("bmflc_lms", "bmflc_huber", "abpf_adaptive", "fixed_bpf", "sfft",
           "kf_then_bmflc", "joint_kinematic_bmflc_kf", "joint_random_walk_bmflc_kf")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def calibration_band_rms(table, axis, fs):
    with np.load(ROOT / f"data/processed/pinheiro_{table}_lumbrical_gyro.npz") as data:
        t = np.asarray(data["time_s"])
        x = np.asarray(data["gyro_deg_s"])[:, axis]
    z = np.interp(np.arange(round(15 * fs)) / fs, t, x)
    f = np.fft.rfftfreq(len(z), 1/fs)
    spectrum = np.fft.rfft(z)
    spectrum[(f < 3) | (f > 12)] = 0
    return max(rms(np.fft.irfft(spectrum, n=len(z))), 1e-6)


def synthetic_case(tremor_kind: str, voluntary_kind: str, scale: float,
                   ratio: float, fs: float = 148.14814814814815,
                   duration: float = 10.0):
    """Controlled ground truth; ratio refers to 3-5 s RMS, not peak."""
    t, d, _ = controlled_tremor(tremor_kind, fs, duration, amplitude_deg_s=scale)
    d = d[:, 0]
    if voluntary_kind == "slow":
        v = np.sin(2*np.pi*0.27*t)
    elif voluntary_kind == "point":
        v = _minimum_jerk_velocity(t, 3, 2.2, 10)
    elif voluntary_kind == "fast_smooth":
        v = _minimum_jerk_velocity(t, 3.1, 0.55, 8)
    elif voluntary_kind == "overlap":
        v = 0.3*np.sin(2*np.pi*0.27*t) + 0.7*np.sin(2*np.pi*3.5*t)
    else:
        raise ValueError(voluntary_kind)
    mask = (t >= 3) & (t < 5)
    v *= ratio * rms(d[mask]) / rms(v[mask])
    cfg = BenchmarkCaseConfig(sample_rate_hz=fs, duration_s=duration,
                              noise_std_deg_s=0, random_seed=20261027)
    return SignalComponents(t, v[:, None], d[:, None], np.zeros((len(t), 1)),
                            (v+d)[:, None], f"{tremor_kind}_{voluntary_kind}_rho{ratio}", cfg)


def make_method(name: str, fs: float, scale: float, frozen: dict,
                modal_hz: float = 6.0):
    if name == "bmflc_lms":
        return create_estimator("bmflc", {"sample_rate_hz": fs, **CONFIG["bmflc"]})
    if name == "bmflc_huber":
        return RobustBMFLC(sample_rate_hz=fs, **CONFIG["bmflc"], variant="huber",
                           huber_delta=frozen["huber_factor"] * scale)
    if name == "abpf_adaptive":
        return create_estimator("abpf", {"sample_rate_hz": fs,
                                        "modal_frequency_hz": modal_hz, **CONFIG["abpf"]})
    if name in ("fixed_bpf", "sfft"):
        return create_estimator(name, {"sample_rate_hz": fs, **CONFIG[name]})
    p_v = frozen["initial_voluntary_variance_factor"] * scale**2
    p_a = frozen["initial_acceleration_variance_factor"] * scale**2
    p_w = frozen["initial_weight_variance_factor"] * scale**2 / 91
    if name == "kf_then_bmflc":
        return KinematicKFThenBMFLC(fs, frozen["two_stage_sigma_j_factor"] * scale,
                                    frozen["two_stage_r_factor"] * scale**2,
                                    p_v, p_a, **CONFIG["bmflc"])
    sigma = frozen["sigma_j_factor"] * scale
    common = dict(sample_rate_hz=fs, sigma_j_deg_s3=sigma,
                  weight_process_variance=frozen["qw_factor"] * scale**2 / 91,
                  measurement_variance=frozen["r_factor"] * scale**2,
                  initial_voluntary_variance=p_v,
                  initial_acceleration_variance=p_a,
                  initial_weight_variance=p_w,
                  low_hz=CONFIG["bmflc"]["low_hz"],
                  high_hz=CONFIG["bmflc"]["high_hz"],
                  spacing_hz=CONFIG["bmflc"]["spacing_hz"])
    if name == "joint_random_walk_bmflc_kf":
        # Same one-step voluntary velocity variance as the main model.
        common["sigma_j_deg_s3"] = sigma * (0.5/fs**2)
        common["voluntary_model"] = "random_walk"
    elif name != "joint_kinematic_bmflc_kf":
        raise ValueError(name)
    return JointKinematicBMFLCKF(**common)


def development_cases():
    # Different fast waveform (0.55 s, 8 deg) from final (0.35 s, 10 deg).
    designs = (("constant", "slow"), ("drift", "point"),
               ("step", "fast_smooth"), ("amplitude_modulated", "slow"),
               ("multifrequency", "overlap"), ("constant", "overlap"),
               ("multifrequency", "fast_smooth"), ("drift", "slow"))
    for scale in (1.0, 20.0):
        for ratio in (0.5, 1.0, 4.0):
            for tremor_kind, voluntary_kind in designs:
                yield scale, ratio, tremor_kind, voluntary_kind, synthetic_case(
                    tremor_kind, voluntary_kind, scale, ratio)


def run_development():
    report = REPORT / "development"
    result = RESULT / "development"
    # Factor values define physical process variances after multiplication by
    # independent calibration RMS. They never inspect a final evaluation window.
    base = dict(huber_factor=0.5, initial_voluntary_variance_factor=1.0,
                initial_acceleration_variance_factor=10000.0,
                initial_weight_variance_factor=1.0)
    candidates = [(j, q, r) for j in (1000.0, 10000.0, 100000.0)
                  for q in (1e-5, 1e-3) for r in (0.01, 0.2)]
    rows = []
    cases = list(development_cases())
    for j, q, r in candidates:
        cfg = {**base, "sigma_j_factor": j, "qw_factor": q, "r_factor": r}
        for scale, ratio, dk, vk, signal in cases:
            estimator = make_method("joint_kinematic_bmflc_kf", signal.config.sample_rate_hz, scale, cfg)
            output, diag, _ = run_stream(estimator, signal.mixed_observation_deg_s)
            vstate = np.array([item["voluntary_velocity_deg_s"][0] for item in diag])
            mask = (signal.time_s >= 3) & (signal.time_s < 5)
            rows.append(dict(sigma_j_factor=j, qw_factor=q, r_factor=r,
                             tremor_kind=dk, voluntary_kind=vk, scale_deg_s=scale, rho=ratio,
                             tremor_nrmse=rms(output[:, 0]-signal.tremor_true_deg_s[:, 0])/rms(signal.tremor_true_deg_s),
                             voluntary_state_nrmse=rms(vstate-signal.voluntary_true_deg_s[:, 0])/rms(signal.tremor_true_deg_s),
                             transient_nrmse=rms(output[mask, 0]-signal.tremor_true_deg_s[mask, 0])/rms(signal.tremor_true_deg_s[mask])))
        print("development joint", j, q, r, flush=True)
    write_csv(report / "joint_grid.csv", rows)
    scores = []
    for j, q, r in candidates:
        subset = [row for row in rows if (row["sigma_j_factor"], row["qw_factor"], row["r_factor"]) == (j,q,r)]
        combined = np.array([0.4*row["tremor_nrmse"] + 0.3*row["voluntary_state_nrmse"]
                             + 0.3*row["transient_nrmse"] for row in subset])
        scores.append(dict(sigma_j_factor=j, qw_factor=q, r_factor=r,
                           selection_score=float(0.7*np.median(combined)+0.3*np.percentile(combined, 90)),
                           median_tremor_nrmse=float(np.median([row["tremor_nrmse"] for row in subset])),
                           median_voluntary_state_nrmse=float(np.median([row["voluntary_state_nrmse"] for row in subset]))))
    write_csv(report / "joint_selection_scores.csv", scores)
    chosen = min(scores, key=lambda row: row["selection_score"])
    # Select two-stage parameters on the same synthetic development signals.
    two_rows = []
    for j in (1000.0, 10000.0, 100000.0):
        for r in (0.1, 1.0, 10.0):
            cfg = {**base, "two_stage_sigma_j_factor": j, "two_stage_r_factor": r}
            for scale, ratio, dk, vk, signal in cases:
                estimator = make_method("kf_then_bmflc", signal.config.sample_rate_hz, scale, cfg)
                output, diag, _ = run_stream(estimator, signal.mixed_observation_deg_s)
                vstate = np.array([item["voluntary_velocity_deg_s"][0] for item in diag])
                mask = (signal.time_s >= 3) & (signal.time_s < 5)
                two_rows.append(dict(sigma_j_factor=j, r_factor=r, tremor_kind=dk,
                                     voluntary_kind=vk, scale_deg_s=scale, rho=ratio,
                                     tremor_nrmse=rms(output[:,0]-signal.tremor_true_deg_s[:,0])/rms(signal.tremor_true_deg_s),
                                     voluntary_state_nrmse=rms(vstate-signal.voluntary_true_deg_s[:,0])/rms(signal.tremor_true_deg_s),
                                     transient_nrmse=rms(output[mask,0]-signal.tremor_true_deg_s[mask,0])/rms(signal.tremor_true_deg_s[mask])))
            print("development two-stage", j, r, flush=True)
    write_csv(report / "two_stage_grid.csv", two_rows)
    two_scores = []
    for j in (1000.0, 10000.0, 100000.0):
        for r in (0.1, 1.0, 10.0):
            subset = [row for row in two_rows if (row["sigma_j_factor"],row["r_factor"])==(j,r)]
            combined = np.array([0.4*row["tremor_nrmse"]+0.3*row["voluntary_state_nrmse"]
                                 +0.3*row["transient_nrmse"] for row in subset])
            two_scores.append(dict(sigma_j_factor=j, r_factor=r,
                                   selection_score=float(0.7*np.median(combined)+0.3*np.percentile(combined,90))))
    write_csv(report / "two_stage_selection_scores.csv", two_scores)
    chosen_two = min(two_scores, key=lambda row: row["selection_score"])
    frozen = {**base, "sigma_j_factor": chosen["sigma_j_factor"],
              "qw_factor": chosen["qw_factor"], "r_factor": chosen["r_factor"],
              "two_stage_sigma_j_factor": chosen_two["sigma_j_factor"],
              "two_stage_r_factor": chosen_two["r_factor"],
              "development": "controlled synthetic only: 8 tremor/voluntary combinations × 2 scales × 3 rho",
              "score": "0.7 median + 0.3 p90 of (0.4 tremor nRMSE + 0.3 voluntary-state nRMSE + 0.3 transient nRMSE)",
              "rho_definition": "RMS voluntary / RMS tremor over 3-5 s",
              "final_windows_used_for_selection": False}
    write_json(result / "frozen_parameters.json", frozen)
    return frozen


def load_frozen():
    path = RESULT / "development/frozen_parameters.json"
    if not path.is_file():
        raise FileNotFoundError("Run --part development first")
    return json.loads(path.read_text())


def run_one(name, signal, scale, frozen, modal=6):
    method = make_method(name, signal.config.sample_rate_hz, scale, frozen, modal)
    output, diag, elapsed = run_stream(method, signal.mixed_observation_deg_s)
    score = evaluate(signal, output, elapsed)
    vstate = np.array([item["voluntary_velocity_deg_s"][0] for item in diag]) if name in (
        "kf_then_bmflc", "joint_kinematic_bmflc_kf", "joint_random_walk_bmflc_kf") else None
    weights = np.stack([item["adaptive_weights"][0] for item in diag]) if name in (
        "bmflc_lms", "bmflc_huber", "kf_then_bmflc", "joint_kinematic_bmflc_kf",
        "joint_random_walk_bmflc_kf") else None
    if weights is not None and name in ("bmflc_lms", "bmflc_huber", "kf_then_bmflc"):
        weights = weights[:, :-1]  # exclude original BMFLC bias
    return output, diag, score, vstate, weights


def summary_row(signal, name, score, output, vstate, weights):
    t = signal.time_s
    d = signal.tremor_true_deg_s[:,0]
    v = signal.voluntary_true_deg_s[:,0]
    pre = (t >= 0) & (t < 3)
    transient = (t >= 3) & (t < 5)
    post = (t >= 5) & (t < t[-1])
    error = output[:,0] - d
    verror = (vstate-v) if vstate is not None else (signal.mixed_observation_deg_s[:,0]-output[:,0]-v)
    vn = np.linalg.norm(weights, axis=1) if weights is not None else None
    vnorm = rms(v[transient])
    dnorm = rms(d[transient])
    excess = (output[:,0]-d)[transient]
    voluntary_energy = float(np.sum(v[transient]**2))
    # Energy fractions are diagnostics, not a conserved physical partition:
    # errors/noise can make their sum differ from one.
    return dict(method=name, case=signal.case_name,
                injected_proxy_rms_deg_s=rms(d), rho=vnorm/dnorm if dnorm else None,
                rmse_deg_s=score["tremor_rmse_deg_s"],
                nrmse=score["tremor_nrmse_by_true_rms"],
                voluntary_reconstruction_rmse_deg_s=score["voluntary_reconstruction_rmse_deg_s"],
                voluntary_state_rmse_deg_s=rms(verror),
                pre_rmse_deg_s=rms(error[pre]),
                transient_rmse_deg_s=rms(error[transient]),
                post_rmse_deg_s=rms(error[post]),
                band_power_reduction_db=score["band_power_reduction_db"],
                diagnostic_lag_ms=score["diagnostic_lag_ms"],
                mean_update_us=score["mean_update_us"], max_update_us=score["max_update_us"],
                mean_fourier_weight_norm=float(np.mean(vn)) if vn is not None else None,
                peak_fourier_weight_norm=float(np.max(vn)) if vn is not None else None,
                peak_transient_weight_excursion=float(np.max(vn[transient])-np.median(vn[pre])) if vn is not None else None,
                transient_false_tremor_energy_fraction=float(np.sum(excess**2)/voluntary_energy) if voluntary_energy>0 else None,
                transient_false_tremor_projection_fraction=float(np.dot(excess,v[transient])/voluntary_energy) if voluntary_energy>0 else None,
                transient_voluntary_state_energy_fraction=float(np.sum(vstate[transient]**2)/voluntary_energy) if vstate is not None and voluntary_energy>0 else None,
                transient_voluntary_state_projection_fraction=float(np.dot(vstate[transient],v[transient])/voluntary_energy) if vstate is not None and voluntary_energy>0 else None,
                recovery_time_s=recovery_time(t, error, pre, after=3.65 if "fast_smooth" in signal.case_name else 3.35) if "fast_transition" in signal.case_name or "fast_smooth" in signal.case_name else None)


def recovery_time(t, error, pre, after):
    fs = 1/np.median(np.diff(t))
    width = max(1, round(0.5*fs))
    rolling = np.sqrt(np.convolve(error**2, np.ones(width)/width, mode="valid"))
    threshold = 1.5*rms(error[pre])
    for idx in range(np.searchsorted(t, after), len(rolling)):
        if rolling[idx] <= threshold:
            return float(t[idx]-after)
    return None


def save_joint_trace(path, signal, output, diag):
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = REPORT / "diagnostics" / path.relative_to(RESULT)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fields = dict(time_s=signal.time_s,
                  observation_deg_s=signal.mixed_observation_deg_s[:,0],
                  true_voluntary_deg_s=signal.voluntary_true_deg_s[:,0],
                  estimated_voluntary_deg_s=np.array([d["voluntary_velocity_deg_s"][0] for d in diag]),
                  injected_proxy_deg_s=signal.tremor_true_deg_s[:,0],
                  estimated_tremor_deg_s=output[:,0],
                  innovation_deg_s=np.array([d["innovation_deg_s"][0] for d in diag]),
                  acceleration_deg_s2=np.array([d["voluntary_acceleration_deg_s2"][0] for d in diag]),
                  voluntary_gain=np.stack([d["kalman_gain_voluntary"][0] for d in diag]),
                  tremor_gain_norm=np.array([d["kalman_gain_tremor_norm"][0] for d in diag]),
                  voluntary_variance=np.stack([d["posterior_voluntary_variance"][0] for d in diag]),
                  tremor_weight_variance_sum=np.array([d["posterior_weight_variance_sum"][0] for d in diag]),
                  fourier_weights=np.stack([d["adaptive_weights"][0] for d in diag]))
    np.savez_compressed(path, **fields)
    fig, axes = plt.subplots(4,1,figsize=(11,9),sharex=True)
    axes[0].plot(signal.time_s,fields["observation_deg_s"],label="observed",alpha=.5)
    axes[0].plot(signal.time_s,fields["true_voluntary_deg_s"],label="true voluntary")
    axes[0].plot(signal.time_s,fields["estimated_voluntary_deg_s"],label="estimated voluntary")
    axes[0].legend(fontsize=7);axes[0].set_ylabel("deg/s")
    axes[1].plot(signal.time_s,fields["injected_proxy_deg_s"],label="injected proxy")
    axes[1].plot(signal.time_s,fields["estimated_tremor_deg_s"],label="estimated tremor")
    axes[1].legend(fontsize=7);axes[1].set_ylabel("deg/s")
    axes[2].plot(signal.time_s,fields["innovation_deg_s"],label="innovation")
    axes[2].plot(signal.time_s,fields["acceleration_deg_s2"],label="angular acceleration")
    axes[2].legend(fontsize=7)
    axes[3].plot(signal.time_s,np.linalg.norm(fields["fourier_weights"],axis=1),label="weight norm")
    axes[3].plot(signal.time_s,fields["voluntary_gain"][:,0],label="voluntary gain")
    axes[3].plot(signal.time_s,fields["tremor_gain_norm"],label="tremor gain norm")
    axes[3].legend(fontsize=7);axes[3].set_xlabel("s")
    fig.tight_layout();fig.savefig(plot_path.with_suffix(".png"),dpi=140);plt.close(fig)
    fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True)
    w = fields["fourier_weights"]
    for idx in (25,60,100,140):
        axes[0].plot(signal.time_s,w[:,idx],label=f"coefficient {idx}")
    axes[0].legend(fontsize=7);axes[0].set_ylabel("deg/s")
    axes[1].plot(signal.time_s,fields["voluntary_variance"][:,0],label="voluntary velocity")
    axes[1].plot(signal.time_s,fields["voluntary_variance"][:,1],label="voluntary acceleration")
    axes[1].legend(fontsize=7);axes[1].set_ylabel("posterior variance")
    axes[2].plot(signal.time_s,fields["tremor_weight_variance_sum"])
    axes[2].set(ylabel="sum tremor variance",xlabel="s")
    fig.tight_layout();fig.savefig(plot_path.with_name(plot_path.stem+"_states.png"),dpi=140);plt.close(fig)


def run_synthetic():
    frozen = load_frozen()
    rows = []
    designs = (("constant","slow"),("drift","point"),("step","fast_smooth"),
               ("amplitude_modulated","slow"),("multifrequency","point"),
               ("constant","overlap"),("multifrequency","fast_smooth"))
    for dk,vk in designs:
        signal = synthetic_case(dk,vk,5,1.0, duration=12)
        for name in METHODS:
            output,diag,score,vstate,weights=run_one(name,signal,5,frozen)
            rows.append({"tremor_kind":dk,"voluntary_kind":vk,**summary_row(signal,name,score,output,vstate,weights)})
            if name=="joint_kinematic_bmflc_kf" and (dk,vk) in (("step","fast_smooth"),("constant","overlap")):
                save_joint_trace(RESULT/f"synthetic/{dk}_{vk}_joint.npz",signal,output,diag)
        print("synthetic",dk,vk,flush=True)
    write_csv(REPORT/"synthetic/controlled_trials.csv",rows)
    return rows


def run_ratio():
    frozen=load_frozen();rows=[]
    # Fixed controlled multi-tone tremor; every level generated before scoring.
    for rho in (0.25,0.5,1.0,2.0,4.0,8.0,16.0):
        signal=synthetic_case("multifrequency","fast_smooth",5,rho,duration=12)
        for name in METHODS:
            output,diag,score,vstate,weights=run_one(name,signal,5,frozen)
            rows.append({"target_rho":rho,**summary_row(signal,name,score,output,vstate,weights)})
        print("ratio",rho,flush=True)
    write_csv(REPORT/"amplitude_ratio/sweep.csv",rows)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in METHODS:
        subset=[row for row in rows if row["method"]==name]
        ax.plot([r["rho"] for r in subset],[r["nrmse"] for r in subset],marker="o",label=name)
    ax.set(xscale="log",yscale="log",xlabel="rho: voluntary/tremor RMS, 3-5 s",ylabel="tremor nRMSE")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(REPORT/"amplitude_ratio/sweep.png",dpi=150);plt.close(fig)
    return rows


def run_final():
    frozen=load_frozen();rows=[];windows=[]
    for subject in SUBJECTS:
        table=f"{subject}_1"
        for start in WINDOWS:
            for ia,axis in enumerate("XYZ"):
                window=load_pinheiro_window(table,start,12,(3,12),1,(ia,))
                fs=window.sample_rate_hz
                scale=calibration_band_rms(table,ia,fs)
                # Same independent modal calibration as Round 3.
                from .calibration import calibrate_pinheiro_modal
                modal=float(calibrate_pinheiro_modal(table,(ia,),start,12).modal_frequency_hz[0])
                windows.append(dict(subject=subject,table=table,axis=axis,start_s=start,
                                    end_s=start+12,sample_rate_hz=fs,calibration_scale_deg_s=scale,
                                    modal_hz=modal,source_sha256=window.source_sha256))
                cfg=BenchmarkCaseConfig(sample_rate_hz=fs,duration_s=12,tremor_scale=1,
                                        noise_std_deg_s=0,random_seed=20261026)
                for case in CASES:
                    signal=generate_case(case,window.tremor_band_proxy_deg_s,cfg)
                    for name in METHODS:
                        output,diag,score,vstate,weights=run_one(name,signal,scale,frozen,modal)
                        rows.append(dict(subject=subject,table=table,axis=axis,start_s=start,
                                         **summary_row(signal,name,score,output,vstate,weights)))
                        if name=="joint_kinematic_bmflc_kf" and start==20 and axis=="X" and case in ("fast_transition","spectral_overlap"):
                            save_joint_trace(RESULT/f"representative/{subject}_{case}_joint.npz",signal,output,diag)
        print("final subject",subject,flush=True)
    write_csv(REPORT/"final/baselines.csv",[r for r in rows if r["method"] in METHODS[:5]])
    write_csv(REPORT/"final/new_methods.csv",[r for r in rows if r["method"] in METHODS[5:]])
    write_csv(REPORT/"final/windows_and_calibration.csv",windows)
    summary=[]
    for subject in SUBJECTS:
        for case in CASES:
            for name in METHODS:
                subset=[r for r in rows if r["subject"]==subject and r["case"]==case and r["method"]==name]
                summary.append(dict(subject=subject,case=case,method=name,n_axis_windows=len(subset),
                                    **{key+"_median":float(np.median([r[key] for r in subset])) for key in
                                       ("rmse_deg_s","nrmse","voluntary_reconstruction_rmse_deg_s",
                                        "voluntary_state_rmse_deg_s","pre_rmse_deg_s","transient_rmse_deg_s",
                                        "post_rmse_deg_s","mean_update_us","band_power_reduction_db")}))
    write_csv(REPORT/"final/by_subject_medians.csv",summary)
    distributions=[]
    for subject in SUBJECTS:
        for case in CASES:
            for name in METHODS:
                subset=[r for r in rows if r["subject"]==subject and r["case"]==case and r["method"]==name]
                record=dict(subject=subject,case=case,method=name,n_axis_windows=len(subset))
                for key in ("nrmse","rmse_deg_s","voluntary_state_rmse_deg_s",
                            "transient_rmse_deg_s","post_rmse_deg_s","mean_update_us"):
                    values=np.array([r[key] for r in subset],dtype=float)
                    record.update({f"{key}_{label}":float(np.percentile(values,p))
                                   for label,p in (("q25",25),("median",50),("q75",75))})
                distributions.append(record)
    write_csv(REPORT/"final/by_subject_distribution.csv",distributions)
    across=[]
    for case in CASES:
        for name in METHODS:
            subset=[r for r in summary if r["case"]==case and r["method"]==name]
            across.append(dict(case=case,method=name,n_subjects=len(subset),
                               **{key+"_across_subject_median":float(np.median([r[key+"_median"] for r in subset])) for key in
                                  ("nrmse","rmse_deg_s","voluntary_reconstruction_rmse_deg_s",
                                   "transient_rmse_deg_s","post_rmse_deg_s","mean_update_us")}))
    write_csv(REPORT/"final/across_subject_summary.csv",across)
    across_distribution=[]
    for case in CASES:
        for name in METHODS:
            subset=[r for r in summary if r["case"]==case and r["method"]==name]
            values=np.array([r["nrmse_median"] for r in subset],dtype=float)
            across_distribution.append(dict(case=case,method=name,n_subjects=len(subset),
                                            nrmse_subject_q25=float(np.percentile(values,25)),
                                            nrmse_subject_median=float(np.median(values)),
                                            nrmse_subject_q75=float(np.percentile(values,75))))
    write_csv(REPORT/"final/across_subject_distribution.csv",across_distribution)
    write_json(RESULT/"final/protocol.json",dict(subjects=SUBJECTS,table_rule="predeclared _1 per subject",
               calibration_s=[0,15],windows_s=WINDOWS,axes="XYZ",cases=CASES,
               noise_std_deg_s=0,proxy_band_hz=[3,12],random_seed=20261026,
               frozen_parameters=frozen,methods=METHODS))
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for ax,case in zip(axes.flat,CASES):
        matrix=np.array([[next(r["nrmse_median"] for r in summary if r["subject"]==subject and r["case"]==case and r["method"]==name)
                          for name in METHODS] for subject in SUBJECTS])
        im=ax.imshow(np.log10(np.maximum(matrix,1e-3)),vmin=-1.5,vmax=1.5,aspect="auto")
        ax.set_yticks(range(5),SUBJECTS);ax.set_xticks(range(len(METHODS)),METHODS,rotation=45,ha="right",fontsize=7)
        ax.set_title(case)
        for i in range(5):
            for j in range(len(METHODS)):
                ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if matrix[i,j]<.3 or matrix[i,j]>3 else "black")
    fig.colorbar(im,ax=axes,label="log10 nRMSE",shrink=.8)
    fig.savefig(REPORT/"final/subject_scenario_nrmse.png",dpi=150);plt.close(fig)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=("all","development","synthetic","ratio","final"),default="all")
    part=parser.parse_args().part
    started=time.monotonic()
    if part in ("all","development"):
        print("frozen",run_development(),flush=True)
    if part in ("all","synthetic"):
        print("synthetic rows",len(run_synthetic()),flush=True)
    if part in ("all","ratio"):
        print("ratio rows",len(run_ratio()),flush=True)
    if part in ("all","final"):
        print("final rows",len(run_final()),flush=True)
    print(f"completed {time.monotonic()-started:.1f} s",flush=True)


if __name__=="__main__":
    main()
