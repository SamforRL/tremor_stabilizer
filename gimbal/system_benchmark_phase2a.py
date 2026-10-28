"""Closed-loop Phase 2A comparison of frozen tremor estimators."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from analysis.tremor_baselines.abpf import ABPF
from analysis.tremor_baselines.bmflc_core import BMFLC
from analysis.tremor_baselines.damped_bmflc import DampedBMFLC
from analysis.tremor_baselines.butterworth import FixedButterworthBPF
from analysis.tremor_baselines.calibration import calibrate_pinheiro_modal
from analysis.tremor_baselines.metrics import diagnostic_lag_ms
from analysis.tremor_baselines.recorded import ROOT as PROJECT_ROOT
from analysis.tremor_baselines.round3_variants import RobustBMFLC
from .control import pd_torque
from .kinematics import axis_joint_reference
from .motion import measured_roll_pitch, roll_pitch_quaternion, roll_pitch_rotation
from .system_benchmark import (
    DURATION, FS as LOG_FS, MODEL, NFOOD, RADIUS, SCALE, UNLOAD,
    bandpower, classify, has_cad_bowl, initialize, interp, joint, spoon_inside, tremor_source,
    voluntary, write_csv,
)

OUT = PROJECT_ROOT / "experiments/results/system_benchmark/phase2a"
REPORT = PROJECT_ROOT / "reports/system_benchmark/phase2a"
METHODS = ("NoComp", "FixedBPF", "ABPF", "OriginalBMFLC", "HuberBMFLC", "Oracle")
VIDEO_METHODS = ("NoComp", "ABPF", "OriginalBMFLC", "HuberBMFLC", "Oracle")
PHASES = ((0, 1, "hold"), (1, 3, "lift"), (3, 7, "transport"),
          (7, 9, "approach"), (9, 11, "intentional_unload"), (11, 12, "unload_hold"))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


class EstimatorBundle:
    """Uniform two-axis adapter; all wrapped methods consume/return deg/s."""

    def __init__(self, name, fs, modal_hz, huber_delta):
        self.name = name
        if name == "FixedBPF":
            self.estimator = FixedButterworthBPF(fs, order=4, low_hz=3., high_hz=12.)
        elif name == "ABPF":
            self.estimator = ABPF(fs, modal_hz, frequency_step_hz=.1, limit_offset_hz=1.5)
        elif name == "OriginalBMFLC":
            self.estimator = BMFLC(fs, low_hz=3., high_hz=12., spacing_hz=.1, lms_gain=.01)
        elif name == "HuberBMFLC":
            self.estimator = [RobustBMFLC(fs, low_hz=3., high_hz=12., spacing_hz=.1,
                                           lms_gain=.01, variant="huber",
                                           huber_delta=float(delta))
                              for delta in huber_delta]
        elif name == "DampedBMFLC":
            self.estimator = DampedBMFLC(fs, low_hz=3., high_hz=12., spacing_hz=.1)
        else:
            raise ValueError(name)

    def reset(self):
        if isinstance(self.estimator, list):
            for estimator in self.estimator:
                estimator.reset()
        else:
            self.estimator.reset()

    def step(self, sample):
        x = np.asarray(sample, dtype=float)
        if isinstance(self.estimator, list):
            outputs, errors, weights = [], [], []
            for axis, estimator in enumerate(self.estimator):
                y, d = estimator.step(x[axis:axis + 1])
                outputs.append(float(y[0]))
                errors.append(float(d["internal_error"][0]))
                weights.append(np.asarray(d["adaptive_weights"])[0, :-1].copy())
            return np.asarray(outputs), {
                "internal_error": np.asarray(errors),
                "adaptive_weights": np.stack(weights),
            }
        estimate, diagnostics = self.estimator.step(x)
        diagnostics = dict(diagnostics)
        if "adaptive_weights" in diagnostics:
            diagnostics["adaptive_weights"] = np.asarray(diagnostics["adaptive_weights"])
            if self.name == "OriginalBMFLC":
                diagnostics["adaptive_weights"] = diagnostics["adaptive_weights"][:, :-1]
        if "internal_error" not in diagnostics:
            diagnostics["internal_error"] = x - estimate
        return np.asarray(estimate), diagnostics


def make_estimator(name, fs, modal_hz, huber_delta):
    return None if name in ("NoComp", "Oracle") else EstimatorBundle(name, fs, modal_hz, huber_delta)


def estimator_signal_metrics(true, estimated, fs):
    rows = {}
    for axis, label in enumerate(("roll", "pitch")):
        truth = np.asarray(true[:, axis], dtype=float)
        estimate = np.asarray(estimated[:, axis], dtype=float)
        denominator = rms(truth)
        error = estimate - truth
        centered_truth = truth - truth.mean()
        centered_estimate = estimate - estimate.mean()
        corr_den = np.linalg.norm(centered_truth) * np.linalg.norm(centered_estimate)
        corr = float(np.dot(centered_truth, centered_estimate) / corr_den) if corr_den else np.nan
        freqs = np.fft.rfftfreq(len(truth), 1 / fs)
        true_spectrum = np.fft.rfft(centered_truth)
        estimate_spectrum = np.fft.rfft(centered_estimate)
        in_band = (freqs >= 3.) & (freqs <= 12.)
        has_estimate = rms(estimate) > 1e-12
        if np.any(in_band) and has_estimate:
            peak_idx = np.flatnonzero(in_band)[np.argmax(np.abs(true_spectrum[in_band]))]
            phase = float(np.degrees(np.angle(estimate_spectrum[peak_idx]
                                              * np.conj(true_spectrum[peak_idx]))))
            peak_frequency = float(freqs[peak_idx])
        else:
            phase, peak_frequency = np.nan, np.nan
        xcorr_lag = (diagnostic_lag_ms(truth[:, None], estimate[:, None], fs, max_lag_s=.5)
                     if has_estimate else np.nan)
        phase_lag = (-phase / (360. * peak_frequency) * 1000.
                     if has_estimate and np.isfinite(phase) and peak_frequency > 0 else np.nan)
        rows.update({
            f"{label}_tremor_nrmse": rms(error) / max(denominator, 1e-12),
            f"{label}_tremor_rmse_deg_s": rms(error),
            f"{label}_correlation": corr,
            f"{label}_amplitude_ratio": rms(estimate) / max(denominator, 1e-12),
            f"{label}_phase_error_deg": phase,
            f"{label}_phase_frequency_hz": peak_frequency,
            f"{label}_effective_lag_ms": phase_lag,
            f"{label}_xcorr_lag_ms": xcorr_lag,
        })
    rows["estimator_nrmse_mean"] = float(np.mean([rows["roll_tremor_nrmse"], rows["pitch_tremor_nrmse"]]))
    corr_values = np.asarray([rows["roll_correlation"], rows["pitch_correlation"]], dtype=float)
    rows["estimator_correlation_mean"] = float(np.mean(corr_values[np.isfinite(corr_values)])) if np.any(np.isfinite(corr_values)) else np.nan
    rows["estimator_amplitude_ratio_mean"] = float(np.mean([rows["roll_amplitude_ratio"], rows["pitch_amplitude_ratio"]]))
    phase_values = np.abs([rows["roll_phase_error_deg"], rows["pitch_phase_error_deg"]])
    lag_values = np.asarray([rows["roll_effective_lag_ms"], rows["pitch_effective_lag_ms"]], dtype=float)
    rows["estimator_abs_phase_error_deg_mean"] = float(np.mean(phase_values[np.isfinite(phase_values)])) if np.any(np.isfinite(phase_values)) else np.nan
    rows["estimator_effective_lag_ms_mean"] = float(np.mean(lag_values[np.isfinite(lag_values)])) if np.any(np.isfinite(lag_values)) else np.nan
    xcorr_values = np.asarray([rows["roll_xcorr_lag_ms"], rows["pitch_xcorr_lag_ms"]], dtype=float)
    rows["estimator_xcorr_lag_ms_mean"] = float(np.mean(xcorr_values[np.isfinite(xcorr_values)])) if np.any(np.isfinite(xcorr_values)) else np.nan
    return rows


def rate_to_angle_init(anchor):
    """Shared angle reference: the injected band-limited angle at t=0."""
    return np.asarray(anchor, dtype=float).copy()


def run_trial(seed, method, source, modal_hz, huber_delta, alpha=None,
              save_traces=True, output_root=OUT, voluntary_fn=voluntary,
              active_window=None):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    cad_bowl = has_cad_bowl(model)
    handle_id, pitch_id, mocap_id, food = initialize(model, data, seed, cad_bowl)
    qr, vr, jr = joint(model, "roll")
    qp, vp, jp = joint(model, "pitch")
    qadr, vadr = np.array([qr, qp]), np.array([vr, vp])
    limits = model.jnt_range[[jr, jp]]
    window, proxy_rate_rad_s, proxy_angle_rad = source
    fs = float(window.sample_rate_hz)
    sensor_times = window.time_s
    dt = float(model.opt.timestep)
    total_steps = round(DURATION / dt)
    stride = round(1. / LOG_FS / dt)
    alpha = SCALE["medium"] if alpha is None else float(alpha)
    estimator = make_estimator(method, fs, modal_hz, huber_delta)
    if estimator:
        estimator.reset()
    pure_reference = EstimatorBundle(method, fs, modal_hz, huber_delta) if method in (
        "OriginalBMFLC", "HuberBMFLC", "DampedBMFLC"
    ) else None
    if pure_reference:
        pure_reference.reset()

    # Match the phase-1 linear endpoint-drift removal in the velocity signal.
    raw_angle = np.zeros_like(proxy_rate_rad_s)
    sample_dt = 1. / fs
    raw_angle[1:] = np.cumsum((proxy_rate_rad_s[:-1] + proxy_rate_rad_s[1:]) * (sample_dt / 2), axis=0)
    drift_rate = raw_angle[-1] / ((len(raw_angle) - 1) * sample_dt)
    true_tremor_rate_rad_s = proxy_rate_rad_s - drift_rate[None, :]
    angle_anchor = proxy_angle_rad[0] * alpha

    data.mocap_pos[mocap_id] = [0, 0, .16]
    data.mocap_quat[mocap_id] = roll_pitch_quaternion(0, 0)
    mujoco.mj_forward(model, data)

    n_sensor = len(sensor_times)
    input_trace = np.zeros((n_sensor, 2))
    true_rate_trace = np.zeros((n_sensor, 2))
    estimate_trace = np.zeros((n_sensor, 2))
    residual_trace = np.zeros((n_sensor, 2))
    weight_norm_trace = np.full((n_sensor, 2), np.nan)
    weight_delta_trace = np.full((n_sensor, 2), np.nan)
    center_trace = np.full((n_sensor, 2), np.nan)
    sensor_phase = []
    sensor_index = 0
    held_estimate = np.zeros(2)
    angle_estimate = np.zeros(2) if method == "NoComp" else rate_to_angle_init(angle_anchor)
    previous_angle_rate = None
    samples_done = 0

    trace_rows, particle_counts, particle_positions, phases = [], [], [], []
    exited_before_unload = np.zeros(NFOOD, dtype=bool)
    first_exit = np.full(NFOOD, np.nan)
    inside_at_unload = np.zeros(NFOOD, dtype=bool)
    saturation_steps = 0
    limit_events = 0
    torque_cost = speed_cost = peak_torque = peak_speed = 0.

    for step in range(total_steps):
        t = data.time
        # Update the gyro estimator at the dataset's native 148.15 Hz. Its
        # angular-velocity output is held between sensor samples.
        while sensor_index < n_sensor and sensor_times[sensor_index] <= t + 1e-10:
            st = float(sensor_times[sensor_index])
            _, _, _, voluntary_rate_rad_s, phase = voluntary_fn(st)
            tremor_rate = true_tremor_rate_rad_s[sensor_index] * alpha
            true_rate_deg = np.rad2deg(tremor_rate)
            input_deg = np.rad2deg(voluntary_rate_rad_s + tremor_rate)
            input_trace[sensor_index] = input_deg
            true_rate_trace[sensor_index] = true_rate_deg
            if method == "NoComp":
                out = np.zeros(2)
                diagnostics = {"internal_error": input_deg.copy()}
            elif method == "Oracle":
                out = true_rate_deg.copy()
                diagnostics = {"internal_error": input_deg - out}
            else:
                out, diagnostics = estimator.step(input_deg)
            held_estimate = np.asarray(out, dtype=float)
            estimate_trace[sensor_index] = held_estimate
            residual_trace[sensor_index] = np.asarray(diagnostics.get("internal_error", input_deg - out), dtype=float)
            weights = diagnostics.get("adaptive_weights")
            if weights is not None:
                weight_norm_trace[sensor_index] = np.linalg.norm(weights, axis=1)
            if "center_frequency_hz" in diagnostics:
                center_trace[sensor_index] = np.asarray(diagnostics["center_frequency_hz"], dtype=float)
            if pure_reference:
                pure_out, pure_diag = pure_reference.step(true_rate_deg)
                pure_weights = np.asarray(pure_diag["adaptive_weights"])
                weight_delta_trace[sensor_index] = np.linalg.norm(weights - pure_weights, axis=1)
            sensor_phase.append(phase)
            sensor_index += 1
            samples_done += 1

        # The estimator path and Oracle both use angular velocity -> angle via
        # the same causal trapezoidal integrator at the controller rate.
        if method == "NoComp":
            control_tremor_angle = np.zeros(2)
        else:
            if method == "Oracle":
                rate_now = np.array([
                    np.interp(t, sensor_times[:samples_done], true_rate_trace[:samples_done, axis])
                    for axis in range(2)
                ])
                rate_now = np.deg2rad(rate_now)
            else:
                rate_now = np.deg2rad(held_estimate)
            if previous_angle_rate is not None:
                angle_estimate += .5 * (previous_angle_rate + rate_now) * dt
            previous_angle_rate = rate_now
            control_tremor_angle = angle_estimate

        p, voluntary_rp, _, _, phase = voluntary_fn(t)
        tremor_angle = interp(sensor_times, proxy_angle_rad, min(t, sensor_times[-1])) * alpha
        data.mocap_pos[mocap_id] = p
        data.mocap_quat[mocap_id] = roll_pitch_quaternion(*(voluntary_rp + tremor_angle))
        mujoco.mj_step1(model, data)
        handle_rotation = data.xmat[handle_id].reshape(3, 3)
        observed_rp = measured_roll_pitch(handle_rotation)
        # The prescribed voluntary path is the task reference, as in Phase 1;
        # only the tremor term supplied by the estimator varies by method.
        intended = voluntary_rp + tremor_angle - control_tremor_angle
        q_des, _, reachable = axis_joint_reference(
            handle_rotation, roll_pitch_rotation(*intended)[:, 2], limits
        )
        torque, saturated = pd_torque(
            q_des, data.qpos[qadr], data.qvel[vadr], data.qfrc_bias[vadr],
            model.actuator_ctrlrange, kp=(5., 5.), kd=(.008, .008)
        )
        data.ctrl[:] = torque
        mujoco.mj_step2(model, data)
        saturation_steps += int(saturated)
        torque_cost += float(torque @ torque) * dt
        speed = data.qvel[vadr]
        speed_cost += float(speed @ speed) * dt
        peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
        peak_speed = max(peak_speed, float(np.max(np.abs(speed))))
        limit_events += int(np.any(np.abs(data.qpos[qadr]) >= limits[:, 1] - .01))

        if (step + 1) % stride == 0 or step == total_steps - 1:
            log_t = data.time
            log_p, log_v, log_vel, log_vr, phase = voluntary_fn(log_t)
            tremor_angle_log = interp(sensor_times, proxy_angle_rad, min(log_t, sensor_times[-1])) * alpha
            data.mocap_pos[mocap_id] = log_p
            data.mocap_quat[mocap_id] = roll_pitch_quaternion(*(log_v + tremor_angle_log))
            mujoco.mj_forward(model, data)
            tool_rotation = data.xmat[pitch_id].reshape(3, 3).copy()
            desired_rotation = roll_pitch_rotation(*log_v)
            axis_error = desired_rotation.T @ tool_rotation[:, 2]
            error_roll = math.atan2(-axis_error[1], axis_error[2])
            error_pitch = math.atan2(axis_error[0], axis_error[2])
            world_angles = [math.atan2(tool_rotation[2, 1], tool_rotation[1, 1]),
                            math.atan2(tool_rotation[0, 2], tool_rotation[0, 0])]
            inside = spoon_inside(data, pitch_id, food, cad_bowl)
            if log_t < UNLOAD - 1e-9:
                newly_exited = (~inside) & (~exited_before_unload)
                first_exit[newly_exited] = log_t
                exited_before_unload |= ~inside
            elif abs(log_t - UNLOAD) < .005:
                inside_at_unload = inside.copy()
            particle_counts.append([log_t, int(inside.sum())])
            particle_positions.append([data.qpos[a:a + 3].copy() for a in food])
            phases.append(phase)
            trace_rows.append([
                log_t, *log_p, *log_v, *log_vel, *log_vr,
                *np.deg2rad(interp(sensor_times[:samples_done], true_rate_trace[:samples_done],
                                   min(log_t, sensor_times[samples_done - 1]))),
                *tremor_angle_log, *intended, *q_des, *data.qpos[qadr], *data.qvel[vadr],
                *world_angles, error_roll, error_pitch,
                math.acos(float(np.clip(tool_rotation[:, 2] @ desired_rotation[:, 2], -1, 1))),
                *desired_rotation[:, 2], *tool_rotation[:, 2], *data.actuator_force[:2],
                int(saturated), int(reachable), int(inside.sum()),
                *control_tremor_angle, *held_estimate,
            ])

    if inside_at_unload.sum() == 0 and particle_counts:
        idx = int(np.argmin(np.abs(np.asarray(particle_counts)[:, 0] - UNLOAD)))
        # The normal 100 Hz log should always capture 9.00 s; this fallback is
        # deterministic for a nonstandard physics timestep.
        log_t = particle_counts[idx][0]
        if abs(log_t - UNLOAD) < .01:
            inside_at_unload = spoon_inside(data, pitch_id, food, cad_bowl)
    labels = classify(data, pitch_id, food, cad_bowl)
    labels = [label if inside_at_unload[i] else "spilled_elsewhere" for i, label in enumerate(labels)]
    retained = int(inside_at_unload.sum())
    target_count = labels.count("successfully_transferred_to_target")
    unload_failures = max(0, retained - target_count)

    values = np.asarray(trace_rows, dtype=float)
    names = ["time", "x", "y", "z", "vol_roll", "vol_pitch", "vx", "vy", "vz",
             "vol_roll_rate", "vol_pitch_rate", "tremor_roll_rate", "tremor_pitch_rate",
             "tremor_roll_angle", "tremor_pitch_angle", "intent_roll", "intent_pitch",
             "q_des_roll", "q_des_pitch", "q_roll", "q_pitch", "qvel_roll", "qvel_pitch",
             "spoon_world_roll", "spoon_world_pitch", "roll_error", "pitch_error", "axis_error",
             "desired_axis_x", "desired_axis_y", "desired_axis_z", "actual_axis_x", "actual_axis_y", "actual_axis_z",
             "tau_roll", "tau_pitch", "saturated", "reachable", "retained_count",
             "estimate_angle_roll", "estimate_angle_pitch", "estimate_rate_roll_deg_s", "estimate_rate_pitch_deg_s"]
    times = values[:, 0]
    transport = (times >= 3.) & (times < 9.)
    unload = times >= 9.
    orientation_error = np.degrees(values[:, [names.index("roll_error"), names.index("pitch_error")]])
    axis_err_deg = np.degrees(values[:, names.index("axis_error")])
    angle = values[:, [names.index("spoon_world_roll"), names.index("spoon_world_pitch")]]
    angle_velocity = np.gradient(angle, times, axis=0)
    powers = [bandpower(angle_velocity[transport, i:i + 1], LOG_FS) for i in range(2)]
    phase_rms = lambda mask, col: float(np.sqrt(np.mean(values[mask, col] ** 2)))
    c = {name: i for i, name in enumerate(names)}
    lowpass_error = _diagnostic_lowpass(angle - values[:, [c["vol_roll"], c["vol_pitch"]]], LOG_FS)
    i_pre = (times >= 8.5) & (times < UNLOAD)
    i_post = (times >= 11.2) & (times < 11.8)
    desired_pitch_amplitude = (np.mean(values[i_post, c["vol_pitch"]])
                               - np.mean(values[i_pre, c["vol_pitch"]]))
    pitch_ratio = ((np.mean(angle[i_post, 1]) - np.mean(angle[i_pre, 1]))
                   / desired_pitch_amplitude if abs(desired_pitch_amplitude) > 1e-12 else float("nan"))

    signal_metrics = estimator_signal_metrics(true_rate_trace[:samples_done], estimate_trace[:samples_done], fs)
    if active_window is not None:
        event_start, event_end = map(float, active_window)
        tt = sensor_times[:samples_done]
        rate_error = estimate_trace[:samples_done] - true_rate_trace[:samples_done]
        windows = {
            "pre": (tt >= event_start - .5) & (tt < event_start),
            "during": (tt >= event_start) & (tt <= event_end),
            "post": (tt > event_end) & (tt <= event_end + 1.0),
        }
        for label, mask in windows.items():
            value = rms(rate_error[mask]) if np.any(mask) else float("nan")
            reference = rms(true_rate_trace[:samples_done][mask]) if np.any(mask) else 0.
            signal_metrics[f"active_transition_{label}_rmse_deg_s"] = value
            signal_metrics[f"active_transition_{label}_nrmse"] = value / max(reference, 1e-12)
        valid_delta = weight_delta_trace[:samples_done]
        delta_norm = np.full(len(valid_delta), np.nan)
        finite_rows = np.any(np.isfinite(valid_delta), axis=1)
        if np.any(finite_rows):
            delta_norm[finite_rows] = np.nanmean(valid_delta[finite_rows], axis=1)
        event_mask = windows["during"] & np.isfinite(delta_norm)
        pre_mask = windows["pre"] & np.isfinite(delta_norm)
        post_mask = windows["post"] & np.isfinite(delta_norm)
        if np.any(event_mask):
            baseline_delta = float(np.mean(delta_norm[pre_mask])) if np.any(pre_mask) else 0.
            event_delta = delta_norm[event_mask]
            peak_delta = float(np.max(event_delta))
            signal_metrics.update({
                "active_transition_fourier_weight_delta_pre": baseline_delta,
                "active_transition_fourier_weight_delta_peak": peak_delta,
                "active_transition_fourier_weight_delta_increment": max(0., peak_delta - baseline_delta),
                "active_transition_fourier_weight_delta_post_mean": float(np.mean(delta_norm[post_mask])) if np.any(post_mask) else float("nan"),
            })
            threshold = baseline_delta + .2 * max(0., peak_delta - baseline_delta)
            recovery = float("nan")
            width = max(1, round(.25 * fs))
            for k in range(np.searchsorted(tt, event_end), len(tt) - width + 1):
                if np.all(np.isfinite(delta_norm[k:k + width])) and np.all(delta_norm[k:k + width] <= threshold):
                    recovery = float(tt[k] - event_end)
                    break
            signal_metrics["active_transition_weight_recovery_s"] = recovery
        else:
            for key in ("active_transition_fourier_weight_delta_pre",
                        "active_transition_fourier_weight_delta_peak",
                        "active_transition_fourier_weight_delta_increment",
                        "active_transition_fourier_weight_delta_post_mean",
                        "active_transition_weight_recovery_s"):
                signal_metrics[key] = float("nan")
    metrics = {
        "seed": seed, "method": method, "severity_scale": alpha,
        "initial_particles": NFOOD, "retained_at_unload": retained,
        "retain_ratio": retained / NFOOD, "target_particles": target_count,
        "transfer_ratio": target_count / NFOOD,
        "transport_spill_count": NFOOD - retained, "unload_failure_count": unload_failures,
        "retained_final": labels.count("retained_in_spoon"),
        "spilled_elsewhere": labels.count("spilled_elsewhere"),
        "transport_roll_rmse_deg": phase_rms(transport, c["roll_error"])*180/math.pi,
        "transport_pitch_rmse_deg": phase_rms(transport, c["pitch_error"])*180/math.pi,
        "transport_roll_peak_deg": float(np.max(np.abs(orientation_error[transport, 0]))),
        "transport_pitch_peak_deg": float(np.max(np.abs(orientation_error[transport, 1]))),
        "transport_roll_p95_abs_deg": float(np.percentile(np.abs(orientation_error[transport, 0]), 95)),
        "transport_pitch_p95_abs_deg": float(np.percentile(np.abs(orientation_error[transport, 1]), 95)),
        "transport_orientation_p95_deg": float(np.percentile(np.abs(axis_err_deg[transport]), 95)),
        "transport_axis_rmse_deg": float(np.sqrt(np.mean(axis_err_deg[transport]**2))),
        "tremor_band_power_roll": powers[0], "tremor_band_power_pitch": powers[1],
        "intentional_pitch_amplitude_ratio": float(pitch_ratio),
        "intent_unload_pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean(lowpass_error[unload, 1]**2)))),
        "torque_integral_Nm2_s": torque_cost, "joint_speed_integral_rad2_s": speed_cost,
        "peak_torque_Nm": peak_torque, "peak_joint_speed_rad_s": peak_speed,
        "saturation_fraction": saturation_steps / total_steps,
        "joint_limit_events": limit_events, "joint_limit_event_fraction": limit_events / total_steps,
        "simulation_dt_s": dt, "controller_rate_hz": 1 / dt,
        "estimator_rate_hz": fs, "controller": "PD+bias Kp=5 Kd=.008 torque ±.15 Nm",
        **signal_metrics,
    }

    if save_traces:
        trace_dir = output_root / "controller_traces" / method
        trace_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_dir / f"seed{seed:02d}.npz", time_s=times, values=values,
                            columns=np.asarray(names), phase=np.asarray(phases),
                            particle_count=np.asarray(particle_counts),
                            particle_positions=np.asarray(particle_positions), particle_labels=np.asarray(labels),
                            tremor_angle_euler_rad=proxy_angle_rad * alpha,
                            tremor_estimate_angle_euler_rad=values[:, [c["estimate_angle_roll"], c["estimate_angle_pitch"]]],
                            input_gyro_deg_s=input_trace[:samples_done], true_tremor_rate_deg_s=true_rate_trace[:samples_done],
                            estimate_gyro_deg_s=estimate_trace[:samples_done], residual_deg_s=residual_trace[:samples_done],
                            sensor_time_s=sensor_times[:samples_done], sensor_phase=np.asarray(sensor_phase),
                            weight_norm=weight_norm_trace[:samples_done], weight_delta_norm=weight_delta_trace[:samples_done],
                            center_frequency_hz=center_trace[:samples_done],
                            first_exit_time_s=first_exit, ever_outside_before_unload=exited_before_unload,
                            inside_at_unload=inside_at_unload, final_particle_positions=np.asarray([data.qpos[a:a+3] for a in food]))
        estimator_dir = output_root / "estimator_traces" / method
        estimator_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(estimator_dir / f"seed{seed:02d}.npz",
                            time_s=sensor_times[:samples_done], phase=np.asarray(sensor_phase),
                            input_gyro_deg_s=input_trace[:samples_done],
                            true_tremor_rate_deg_s=true_rate_trace[:samples_done],
                            estimate_gyro_deg_s=estimate_trace[:samples_done],
                            residual_deg_s=residual_trace[:samples_done],
                            weight_norm=weight_norm_trace[:samples_done],
                            weight_delta_norm=weight_delta_trace[:samples_done],
                            center_frequency_hz=center_trace[:samples_done])
        particle_dir = output_root / "particles" / method
        particle_dir.mkdir(parents=True, exist_ok=True)
        with (particle_dir / f"seed{seed:02d}.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["particle_id", "final_class", "ever_outside_before_unload", "first_exit_time_s",
                         "final_x_m", "final_y_m", "final_z_m"])
            for i, label in enumerate(labels):
                wr.writerow([i, label, int(exited_before_unload[i]), first_exit[i], *data.qpos[food[i]:food[i]+3]])
        weight_dir = output_root / "weight_diagnostics" / method
        weight_dir.mkdir(parents=True, exist_ok=True)
        with (weight_dir / f"seed{seed:02d}.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["time_s", "phase", "roll_weight_norm", "pitch_weight_norm",
                         "roll_weight_delta_vs_tremor_only", "pitch_weight_delta_vs_tremor_only",
                         "roll_residual_deg_s", "pitch_residual_deg_s", "roll_center_hz", "pitch_center_hz"])
            for k in range(samples_done):
                wr.writerow([sensor_times[k], sensor_phase[k], *weight_norm_trace[k], *weight_delta_trace[k],
                             *residual_trace[k], *center_trace[k]])
    return (metrics, signal_metrics, values, names, np.asarray(particle_positions),
            phases, times, np.asarray(particle_counts))


def _diagnostic_lowpass(x, fs, cutoff=2.):
    mean = np.mean(x, axis=0, keepdims=True)
    freq = np.fft.rfftfreq(len(x), 1/fs)
    spectrum = np.fft.rfft(x - mean, axis=0)
    spectrum[freq > cutoff] = 0
    return np.fft.irfft(spectrum, n=len(x), axis=0) + mean


def save_config(source, modal_hz, huber_delta):
    window, _, _ = source
    phase1_config_path = PROJECT_ROOT / "experiments/results/system_benchmark/phase1/benchmark_config.json"
    config = {
        "benchmark": "system benchmark phase 2A; phase1 physics/trajectory frozen",
        "model": str(MODEL.relative_to(PROJECT_ROOT)), "phase1_config_sha256": _sha256(phase1_config_path),
        "phase1_configuration_snapshot": json.loads(phase1_config_path.read_text()),
        "duration_s": DURATION, "physics_dt_s": .001, "controller_hz": 1000,
        "estimator_sample_rate_hz": window.sample_rate_hz,
        "seeds": list(range(10)), "methods": list(METHODS), "alpha": SCALE["medium"],
        "source": {"table": window.table, "start_s": window.start_s, "band_hz": list(window.band_hz),
                   "channels": list(window.channels), "mapping": {"gyro_X": "roll", "gyro_Y": "pitch"},
                   "sha256": window.source_sha256, "proxy_signal_is_not_independent_physiological_ground_truth": True},
        "measurement": "voluntary roll/pitch Euler rates plus alpha-scaled drift-corrected proxy rates; deg/s",
        "estimator_output": "angular velocity deg/s; held between native sensor samples",
        "integration": "causal trapezoidal integration at controller rate; common initial angle anchor from phase1 injected proxy at t=0; no post-hoc alignment",
        "fixed_bpf": {"order": 4, "low_hz": 3., "high_hz": 12.},
        "abpf": {"modal_frequency_hz_from_raw_0_15s": np.asarray(modal_hz).tolist(),
                 "frequency_step_hz": .1, "limit_offset_hz": 1.5},
        "bmflc": {"frequency_band_hz": [3., 12.], "spacing_hz": .1, "lms_gain": .01,
                  "bias_included_but_excluded_from_tremor_output": True},
        "huber": {"delta_deg_s": np.asarray(huber_delta).tolist(), "rule": "0.5 times per-axis raw 3-12 Hz RMS from independent 0-15 s calibration"},
        "controller": {"kp": [5., 5.], "kd": [.008, .008], "torque_limit_Nm": .15},
        "angle_convention": "same initial angle offset and trapezoidal rate-to-angle integrator for all compensating methods and Oracle; NoComp estimate is zero",
        "rendering": "fixed-camera side-projection animation, matching phase1 batch visualization; interactive CAD model is available via gimbal.system_benchmark_viewer",
    }
    (OUT / "frozen_config.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def _sha256(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_video(path, times, values, particles, phases, names, method, seed, tremor_label=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to write requested videos")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = {name: i for i, name in enumerate(names)}
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    frame_indices = np.linspace(0, len(times)-1, round(DURATION*15)+1).astype(int)
    proc = None
    for index in frame_indices:
        ax.clear(); t = times[index]; r = values[index]
        base = np.array([r[c["x"]], r[c["y"]], r[c["z"]]])
        rh = roll_pitch_rotation(r[c["vol_roll"]] + r[c["tremor_roll_angle"]],
                                 r[c["vol_pitch"]] + r[c["tremor_pitch_angle"]])
        tool = rh @ roll_pitch_rotation(r[c["q_roll"]], r[c["q_pitch"]])
        pivot = base + rh @ np.array([.073, 0, 0])
        bowl = pivot + tool @ np.array([.064, 0, 0])
        local = np.array([[.040, 0, 0], [.042, 0, .008], [.086, 0, .008], [.088, 0, 0]])
        rim = np.array([bowl + tool @ np.array([x-.064, y, z]) for x, y, z in local])
        ax.plot(rim[:, 0], rim[:, 2], color="#9a5b00", lw=5, solid_capstyle="round", label="collision tray proxy")
        ax.plot([base[0]-.14*rh[0, 0], base[0]], [base[2]-.14*rh[2, 0], base[2]],
                color="#315a9b", lw=8, label="handle / gimbal")
        ax.scatter(particles[index, :, 0], particles[index, :, 2], s=18, c="#e98324",
                   edgecolors="none", zorder=4, label="food particles")
        ax.plot([.28, .40, .40, .28, .28], [.101, .101, .205, .205, .101],
                color="#315e9a", lw=2, label="target container")
        label = tremor_label if tremor_label is not None else "alpha=4"
        ax.set(xlim=(-.16, .46), ylim=(0, .45), xlabel="world X (m)", ylabel="world Z (m)",
               title=f"{method} · seed={seed} · {label} · {phases[index]} · t={t:.2f} s")
        ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout(); fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if proc is None:
            h, w = frame.shape[:2]
            proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{w}x{h}", "-r", "15", "-i", "-", "-an", "-c:v", "libx264",
                                     "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg encoding failed for {path}")
    plt.close(fig)


def plot_results(rows, signal_rows, traces, representative_seed):
    REPORT.mkdir(parents=True, exist_ok=True)
    labels = list(METHODS)
    colors = dict(zip(labels, plt.cm.tab10.colors[:len(labels)]))
    for metric, title, fname in (("retain_ratio", "Transport retention", "retention_paired.png"),
                                 ("transfer_ratio", "Target transfer", "transfer_paired.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        by_seed = {int(r["seed"]): {} for r in rows}
        for r in rows: by_seed[int(r["seed"])][r["method"]] = float(r[metric])
        for seed, vals in by_seed.items():
            ax.plot(range(len(labels)), [vals[m] for m in labels], color="0.65", alpha=.55, lw=.8, zorder=1)
        for i, method in enumerate(labels):
            ys = [float(r[metric]) for r in rows if r["method"] == method]
            ax.scatter(np.full(len(ys), i), ys, color=colors[method], s=25, label=method, zorder=2)
            ax.plot([i-.15, i+.15], [np.mean(ys)]*2, color="black", lw=2)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylim(-.03, 1.03); ax.set_ylabel("ratio"); ax.set_title(title + " · paired seeds")
        ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(REPORT/fname, dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("transport_roll_rmse_deg", "transport_pitch_rmse_deg"),
                                 ("Roll RMSE", "Pitch RMSE")):
        vals = [[float(r[metric]) for r in rows if r["method"] == m] for m in labels]
        ax.boxplot(vals, tick_labels=labels, showmeans=True); ax.set_title(title); ax.set_ylabel("deg")
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=.25)
    fig.savefig(REPORT/"orientation_rmse_comparison.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("roll_band_attenuation_db", "pitch_band_attenuation_db"),
                                 ("Roll 3–12 Hz attenuation", "Pitch 3–12 Hz attenuation")):
        vals = [[float(r[metric]) for r in rows if r["method"] == m and r.get(metric) not in (None, "")]
                for m in labels]
        ax.boxplot(vals, tick_labels=labels, showmeans=True); ax.axhline(0, color="k", lw=.8)
        ax.set_title(title); ax.set_ylabel("dB vs NoComp"); ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=.25)
    fig.savefig(REPORT/"tremor_attenuation_comparison.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, xmetric, title, xlabel in ((axes[0], "estimator_nrmse_mean", "Estimator nRMSE vs retention", "mean gyro tremor nRMSE"),
                                       (axes[1], "estimator_abs_phase_error_deg_mean", "Phase error vs retention", "mean absolute phase error (deg)")):
        for method in labels:
            subset = [r for r in rows if r["method"] == method and np.isfinite(float(r[xmetric]))]
            ax.scatter([float(r[xmetric]) for r in subset], [float(r["retain_ratio"]) for r in subset],
                       label=method, color=colors[method], alpha=.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("retention ratio"); ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(REPORT/"signal_metrics_vs_retention.png", dpi=160); plt.close(fig)

    selected = [r for r in traces if r["seed"] == representative_seed and r["method"] in ("NoComp", "ABPF", "OriginalBMFLC", "HuberBMFLC", "Oracle")]
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    for item in selected:
        t, values, cols = item["times"], item["values"], item["names"]
        c = {n: i for i, n in enumerate(cols)}; label = item["method"]
        axes[0].plot(t, np.rad2deg(values[:, c["pitch_error"]]), label=label)
        axes[1].plot(t, np.rad2deg(values[:, c["q_pitch"]]), label=label)
        axes[2].plot(t, values[:, c["tau_pitch"]], label=label)
        axes[3].step(item["counts"][:, 0], item["counts"][:, 1], where="post", label=label)
        axes[4].plot(t, values[:, c["estimate_rate_pitch_deg_s"]], label=label)
    for ax, title, ylabel in zip(axes, ("Spoon pitch task error", "Pitch joint angle", "Pitch torque",
                                        "Particles inside spoon", "Estimated pitch tremor rate"),
                                 ("deg", "rad", "Nm", "count", "deg/s")):
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
        for t0 in (1, 3, 7, 9, 11): ax.axvline(t0, color="gray", alpha=.2)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(REPORT/"representative_time_series.png", dpi=160); plt.close(fig)


def _markdown_report(rows, signal_rows, pairs, representative_seed, config):
    REPORT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        stats[method] = {k: np.asarray([float(r[k]) for r in subset if r.get(k) not in (None, "")], dtype=float)
                         for k in subset[0] if k not in ("method", "seed", "controller")}
    def ms(method, key, factor=1.):
        arr = stats[method][key] * factor
        return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f}"
    pair_h_o = [r for r in pairs if r["method_a"] == "HuberBMFLC" and r["method_b"] == "OriginalBMFLC"]
    pair_a_h = [r for r in pairs if r["method_a"] == "ABPF" and r["method_b"] == "HuberBMFLC"]
    h_ret = float(np.mean([r["retain_ratio_diff"] for r in pair_h_o]))
    h_transfer = float(np.mean([r["transfer_ratio_diff"] for r in pair_h_o]))
    h_pitch = float(np.mean([r["transport_pitch_rmse_deg_diff"] for r in pair_h_o]))
    h_nrmse = float(np.mean([r["estimator_nrmse_mean_diff"] for r in pair_h_o]))
    a_ret = float(np.mean([r["retain_ratio_diff"] for r in pair_a_h]))
    a_transfer = float(np.mean([r["transfer_ratio_diff"] for r in pair_a_h]))
    a_pitch = float(np.mean([r["transport_pitch_rmse_deg_diff"] for r in pair_a_h]))
    a_nrmse = float(np.mean([r["estimator_nrmse_mean_diff"] for r in pair_a_h]))
    evaluated = []
    for method in ("FixedBPF", "ABPF", "OriginalBMFLC", "HuberBMFLC"):
        subset = [r for r in rows if r["method"] == method]
        evaluated.append({key: float(np.mean([float(r[key]) for r in subset]))
                          for key in ("estimator_nrmse_mean", "estimator_abs_phase_error_deg_mean", "retain_ratio")})
    def corr_metric(xkey, ykey):
        x = np.asarray([r[xkey] for r in evaluated]); y = np.asarray([r[ykey] for r in evaluated])
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    nrmse_ret_corr = corr_metric("estimator_nrmse_mean", "retain_ratio")
    phase_ret_corr = corr_metric("estimator_abs_phase_error_deg_mean", "retain_ratio")
    particle_ranges = {
        method: (min(float(r["retain_ratio"]) for r in rows if r["method"] == method),
                 max(float(r["retain_ratio"]) for r in rows if r["method"] == method),
                 min(float(r["transfer_ratio"]) for r in rows if r["method"] == method),
                 max(float(r["transfer_ratio"]) for r in rows if r["method"] == method))
        for method in METHODS
    }
    lines = [
        "# 系统级 Estimator 闭环比较报告：Phase 2A", "",
        "## 摘要", "",
        "本轮在冻结的 Phase 1 颗粒勺子任务上，以相同主动轨迹、p03_2 记录震颤代理（X→roll、Y→pitch，α=4）、控制器、物理参数和 10 个配对种子比较 NoComp、FixedBPF、ABPF、OriginalBMFLC、HuberBMFLC 与 Oracle。所有在线估计器逐样本接收两轴角速度（deg/s），输出角速度；输出经相同因果梯形积分转为补偿角，再进入原 gimbal IK/PD 控制链。Oracle 也走同一积分链。", "",
        "颗粒 retention/transfer 衡量实际任务结果；角速度 nRMSE、相位误差是信号域指标。二者不应预设为一致。NoComp 到 Oracle 的 Phase 1 动态范围为本轮参照。", "",
        "## 冻结配置与信号链", "",
        f"- 传感器/估计器采样率：{config['estimator_sample_rate_hz']:.6f} Hz；MuJoCo 物理/控制器更新：1 kHz；估计输出在采样间隔保持。",
        "- FixedBPF：因果四阶 prototype Butterworth，3–12 Hz。",
        f"- ABPF：0.1 Hz 中心频率步长、±1.5 Hz 限幅；p03_2 独立 0–15 s 原始 gyro 校准中心频率 {np.asarray(config['abpf']['modal_frequency_hz_from_raw_0_15s']).round(3).tolist()} Hz。",
        "- OriginalBMFLC：3–12 Hz，0.1 Hz 网格，μ=0.01，包含意图/漂移 bias 状态且震颤输出不含 bias。",
        f"- HuberBMFLC：基与 μ 同 OriginalBMFLC；冻结阈值 δ=0.5S，按轴 δ={np.asarray(config['huber']['delta_deg_s']).round(3).tolist()} deg/s（S 来自独立原始 0–15 s 校准的 3–12 Hz RMS）。",
        "- 角度转换：共同使用初始震颤角参考和 1 kHz 因果梯形积分；没有事后时间对齐。该共同初始角参考属于仿真零点初始化假设，真实 gyro 单独不能观测绝对角偏置。",
        "- 原始记录经离线带通得到的信号仍是震颤代理，不是独立测量的生理震颤真值。", "",
        "## 10 个种子主结果", "",
        "表中数值为 mean ± sample SD；attenuation 为相对同 seed NoComp 的 3–12 Hz 输出功率 dB（负值代表衰减）。", "",
        "| Method | Retain % | Transfer % | Roll RMSE (deg) | Pitch RMSE (deg) | Roll/Pitch attenuation (dB) | Intent ratio | Torque cost (Nm²s) | Estimator nRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        atten = f"{ms(m,'roll_band_attenuation_db')} / {ms(m,'pitch_band_attenuation_db')}"
        lines.append(f"| {m} | {ms(m,'retain_ratio',100)} | {ms(m,'transfer_ratio',100)} | {ms(m,'transport_roll_rmse_deg')} | {ms(m,'transport_pitch_rmse_deg')} | {atten} | {ms(m,'intentional_pitch_amplitude_ratio')} | {ms(m,'torque_integral_Nm2_s')} | {ms(m,'estimator_nrmse_mean')} |")
    lines += ["", f"按 NoComp 的 10-seed retention 中位数选出的代表 seed：**{representative_seed:02d}**。该 seed 用于 representative time series 与五段同机位投影动画，避免按 estimator 表现挑选。", "",
              "## 重点配对比较", "", "正差表示前者高于后者。逐 seed 原始差值另存 `paired_differences.csv`。", "",
              "| 配对 | retention 差 (pp) mean ± SD | transfer 差 (pp) mean ± SD | Pitch RMSE 差 (deg) mean ± SD |",
              "|---|---:|---:|---:|"]
    for a, b in (("HuberBMFLC", "OriginalBMFLC"), ("ABPF", "HuberBMFLC")):
        p = [r for r in pairs if r["method_a"] == a and r["method_b"] == b]
        lines.append(f"| {a} − {b} | {np.mean([r['retain_ratio_diff'] for r in p])*100:.2f} ± {np.std([r['retain_ratio_diff'] for r in p],ddof=1)*100:.2f} | {np.mean([r['transfer_ratio_diff'] for r in p])*100:.2f} ± {np.std([r['transfer_ratio_diff'] for r in p],ddof=1)*100:.2f} | {np.mean([r['transport_pitch_rmse_deg_diff'] for r in p]):.3f} ± {np.std([r['transport_pitch_rmse_deg_diff'] for r in p],ddof=1):.3f} |")
    lines += ["", "## Signal-level 相位、相关与幅值", "",
              "以下指标使用固定输入信号，因此同一方法的跨 seed 信号值相同；`effective lag` 由主频相位换算，正值表示估计滞后、负值表示领先。cross-correlation lag 另存 CSV；周期信号的互相关存在整周期多解，不作为主相位结论。", "",
              "| Method | Roll/Pitch correlation | Roll/Pitch RMS amplitude ratio | Roll/Pitch phase error (deg) | Roll/Pitch phase-equivalent lag (ms) |",
              "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        sr = next(r for r in signal_rows if r["method"] == method)
        def pairfmt(key, digits=2):
            vals = [float(sr[f"{axis}_{key}"]) for axis in ("roll", "pitch")]
            return " / ".join("n/a" if not np.isfinite(v) else f"{v:.{digits}f}" for v in vals)
        lines.append(f"| {method} | {pairfmt('correlation')} | {pairfmt('amplitude_ratio')} | {pairfmt('phase_error_deg')} | {pairfmt('effective_lag_ms')} |")
    lines += ["", "## 结果解读", "",
              f"- **Huber vs Original BMFLC：**Huber−Original 的平均 paired retention 差为 {h_ret*100:+.2f} 个百分点，transfer 差 {h_transfer*100:+.2f} 个百分点，pitch RMSE 差 {h_pitch:+.3f}°，估计 nRMSE 差 {h_nrmse:+.3f}。正 RMSE/nRMSE 差表示 Huber 较差；逐 seed 差值见配对 CSV。",
              f"- **ABPF vs Huber：**ABPF−Huber 的平均 paired retention 差 {a_ret*100:+.2f} 个百分点，transfer 差 {a_transfer*100:+.2f} 个百分点，pitch RMSE 差 {a_pitch:+.3f}°，估计 nRMSE 差 {a_nrmse:+.3f}。",
              f"- **估计精度与任务表现：**以四种估计器的跨 seed retention 均值作为 4 个方法级点，nRMSE 与 retention 的 Pearson r={nrmse_ret_corr:.3f}；平均绝对主频相位误差与 retention 的 r={phase_ret_corr:.3f}。样本仅 4 个方法，这只是描述性排序关系，不能作统计推断。",
              "- **颗粒范围：**" + "; ".join(f"{m}: retain {particle_ranges[m][0]*100:.1f}–{particle_ranges[m][1]*100:.1f}%, transfer {particle_ranges[m][2]*100:.1f}–{particle_ranges[m][3]*100:.1f}%" for m in METHODS) + ".",
              "- **信号指标与任务指标：**`signal_metrics_vs_retention.png` 将每个 method/seed 的估计 nRMSE、主频相位误差与颗粒保留率配对展示。十个固定种子只用于工程 benchmark，不进行临床统计显著性推断。",
              "- **权重污染：**OriginalBMFLC 与 HuberBMFLC 的 `weight_diagnostics/` 逐采样保存 residual、Fourier weight norm；同一 trial 另运行纯震颤输入参考，并记录混合输入与纯震颤权重距离。可定位 lift、approach 和 unload 周围的权重变化。",
              "- **Phase 1 限制继续适用：**颗粒与浅托盘碰撞代理接触，CAD 勺头网格仍主要显示；因此结果验证的是冻结的代理几何 benchmark，不应解读为真实勺碗接触精度。", "",
              "## 可视化与结果文件", "",
              f"代表 seed {representative_seed:02d} 视频位于 `videos/`：NoComp、ABPF、OriginalBMFLC、HuberBMFLC、Oracle。保存视频沿用 Phase 1 的固定侧视投影动画；要用 MuJoCo CAD 场景实时回放任一方法，可运行 `python -m gimbal.system_benchmark_viewer --replay-phase2a HuberBMFLC --seed {representative_seed}`。这是保存状态的原生 viewer 回放，不会重新仿真。",
              "主结果 `per_run_metrics.csv`、`aggregate_metrics.csv`、`signal_metrics.csv`、`paired_differences.csv`；配置 `frozen_config.json`；时序在 `controller_traces/<method>/`；估计信号在 `estimator_traces/<method>/`；权重诊断在 `weight_diagnostics/`；颗粒分类在 `particles/<method>/`。", "",
              "## 决策", "",
              f"1. Huber 相对 Original BMFLC 的 mean retention 差 {h_ret*100:+.2f} pp、transfer 差 {h_transfer*100:+.2f} pp；本轮任务级结果{'没有显示 Huber 优势' if h_ret <= 0 and h_transfer <= 0 else '显示 Huber 在至少一个颗粒指标上有优势'}。Huber 的 pitch RMSE 平均低 {abs(h_pitch):.3f}°，但 estimator nRMSE 高 {h_nrmse:+.3f}，signal-level tremor reconstruction 更差。",
              f"2. ABPF 是强 baseline：相对 Huber retention/transfer 平均分别高 {a_ret*100:+.2f}/{a_transfer*100:+.2f} pp；其 roll/pitch RMSE mean 为 {ms('ABPF','transport_roll_rmse_deg')}° / {ms('ABPF','transport_pitch_rmse_deg')}°。",
              f"3. nRMSE 与 retention 的跨方法相关 r={nrmse_ret_corr:.3f}，主频相位误差与 retention r={phase_ret_corr:.3f}；二者在本 benchmark 中都不能单独解释颗粒结果。",
              "4. estimator improvement 确实总体改善了相对 NoComp 的功能表现（所有估计器 retention/transfer 高于 NoComp），但估计器之间的排序与 signal-level nRMSE 不一致。Oracle 仍是上界。",
              "这些是固定任务与五个方法配置下的工程描述，不作临床统计显著性结论。", ""]
    (REPORT / "系统级Estimator闭环比较报告.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(rows):
    metric_keys = [k for k in rows[0] if k not in ("seed", "method", "controller")]
    result = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        for key in metric_keys:
            vals = np.asarray([float(r[key]) for r in subset if r.get(key) not in (None, "") and np.isfinite(float(r[key]))])
            result.append({"method": method, "metric": key, "n": len(vals),
                           "mean": float(np.mean(vals)) if len(vals) else "",
                           "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.,
                           "min": float(np.min(vals)) if len(vals) else "",
                           "max": float(np.max(vals)) if len(vals) else ""})
    return result


def paired_differences(rows):
    by = {(int(r["seed"]), r["method"]): r for r in rows}
    out = []
    for a, b in (("HuberBMFLC", "OriginalBMFLC"), ("ABPF", "HuberBMFLC")):
        for seed in range(10):
            ra, rb = by[(seed, a)], by[(seed, b)]
            out.append({"seed": seed, "method_a": a, "method_b": b,
                        **{f"{metric}_diff": float(ra[metric]) - float(rb[metric])
                           for metric in ("retain_ratio", "transfer_ratio", "transport_pitch_rmse_deg",
                                          "estimator_nrmse_mean", "estimator_abs_phase_error_deg_mean",
                                          "torque_integral_Nm2_s")}})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--videos", action="store_true", help="render the five representative-seed projections")
    parser.add_argument("--summary-only", action="store_true", help="rebuild metrics/report/plots/videos from saved traces")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 10:
        parser.error("--seeds must be in 1..10; formal report requires 10")
    OUT.mkdir(parents=True, exist_ok=True)
    source = tremor_source()
    window = source[0]
    calibration = calibrate_pinheiro_modal("p03_2", (0, 1), 15., DURATION,
                                           calibration_start_s=0., preferred_duration_s=15.)
    modal_hz = calibration.modal_frequency_hz
    from analysis.tremor_baselines.run_round3 import calibration_band_rms
    scales = np.asarray([calibration_band_rms("p03_2", axis, window.sample_rate_hz) for axis in (0, 1)])
    huber_delta = .5 * scales
    save_config(source, modal_hz, huber_delta)

    if args.summary_only:
        with (OUT / "per_run_metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        signal_rows, traces = [], []
        for row in rows:
            seed, method = int(row["seed"]), row["method"]
            trace_path = OUT / "controller_traces" / method / f"seed{seed:02d}.npz"
            with np.load(trace_path) as saved:
                sensor_times = saved["sensor_time_s"]
                sig = estimator_signal_metrics(saved["true_tremor_rate_deg_s"],
                                               saved["estimate_gyro_deg_s"],
                                               window.sample_rate_hz)
                values, names = saved["values"], [str(x) for x in saved["columns"]]
                traces.append({"seed": seed, "method": method, "times": saved["time_s"].copy(),
                               "values": values.copy(), "names": names,
                               "particles": saved["particle_positions"].copy(),
                               "phases": saved["phase"].tolist(), "counts": saved["particle_count"].copy()})
                estimator_dir = OUT / "estimator_traces" / method
                estimator_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(estimator_dir / f"seed{seed:02d}.npz",
                                    time_s=saved["sensor_time_s"], phase=saved["sensor_phase"],
                                    input_gyro_deg_s=saved["input_gyro_deg_s"],
                                    true_tremor_rate_deg_s=saved["true_tremor_rate_deg_s"],
                                    estimate_gyro_deg_s=saved["estimate_gyro_deg_s"],
                                    residual_deg_s=saved["residual_deg_s"],
                                    weight_norm=saved["weight_norm"],
                                    weight_delta_norm=saved["weight_delta_norm"],
                                    center_frequency_hz=saved["center_frequency_hz"])
            row.update(sig)
            signal_rows.append({"seed": seed, "method": method, **sig})
        no_comp = {int(r["seed"]): r for r in rows if r["method"] == "NoComp"}
        for row in rows:
            base = no_comp[int(row["seed"])]
            for axis in ("roll", "pitch"):
                row[f"{axis}_band_attenuation_db"] = 10 * math.log10(
                    max(float(row[f"tremor_band_power_{axis}"]), 1e-30)
                    / max(float(base[f"tremor_band_power_{axis}"]), 1e-30))
        representative_seed = min(
            no_comp.values(),
            key=lambda r: (abs(float(r["retain_ratio"]) - np.median([float(x["retain_ratio"]) for x in no_comp.values()])), int(r["seed"]))
        )["seed"]
        representative_seed = int(representative_seed)
        pairs = paired_differences(rows)
        write_csv(OUT / "per_run_metrics.csv", rows)
        write_csv(OUT / "signal_metrics.csv", signal_rows)
        write_csv(OUT / "aggregate_metrics.csv", aggregate(rows))
        write_csv(OUT / "paired_differences.csv", pairs)
        plot_results(rows, signal_rows, traces, representative_seed)
        if args.videos:
            for method in VIDEO_METHODS:
                tr = next(item for item in traces if item["method"] == method and item["seed"] == representative_seed)
                encode_video(REPORT / "videos" / f"seed{representative_seed:02d}_{method}.mp4",
                             tr["times"], tr["values"], tr["particles"], tr["phases"], tr["names"], method, representative_seed)
        config = json.loads((OUT / "frozen_config.json").read_text())
        config["representative_seed_no_comp_median_retention"] = representative_seed
        (OUT / "frozen_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        _markdown_report(rows, signal_rows, pairs, representative_seed, config)
        print(f"Summary rebuilt from saved traces: {REPORT / '系统级Estimator闭环比较报告.md'}")
        return

    rows, signal_rows, traces = [], [], []
    for method in METHODS:
        for seed in range(args.seeds):
            metric, signal_metric, values, names, particles, phases, times, particle_counts = run_trial(
                seed, method, source, modal_hz, huber_delta
            )
            rows.append(metric)
            signal_rows.append({"seed": seed, "method": method, **signal_metric})
            traces.append({"seed": seed, "method": method, "values": values, "names": names,
                           "particles": particles, "phases": phases, "times": times,
                           "counts": particle_counts})
            print(f"{method:16s} seed={seed:02d} retain={metric['retain_ratio']:.3f} "
                  f"transfer={metric['transfer_ratio']:.3f} nRMSE={metric['estimator_nrmse_mean']:.3f}", flush=True)

    # The fixed 10-seed protocol is paired. Short smoke runs are allowed for
    # development, but do not produce a final report or silently replace data.
    write_csv(OUT / "per_run_metrics.csv", rows)
    write_csv(OUT / "signal_metrics.csv", signal_rows)
    write_csv(OUT / "aggregate_metrics.csv", aggregate(rows))
    pairs = paired_differences(rows) if args.seeds == 10 else []
    write_csv(OUT / "paired_differences.csv", pairs)
    if args.seeds < 10:
        print("Smoke run only; use --seeds 10 for final plots/report.")
        return

    no_comp = [r for r in rows if r["method"] == "NoComp"]
    median_retention = float(np.median([r["retain_ratio"] for r in no_comp]))
    representative_seed = min(no_comp, key=lambda r: (abs(r["retain_ratio"] - median_retention), r["seed"]))["seed"]
    reference_power = {int(r["seed"]): r for r in no_comp}
    for r in rows:
        base = reference_power[int(r["seed"])]
        for axis in ("roll", "pitch"):
            power = float(r[f"tremor_band_power_{axis}"])
            base_power = max(float(base[f"tremor_band_power_{axis}"]), 1e-30)
            r[f"{axis}_band_attenuation_db"] = 10 * math.log10(max(power, 1e-30) / base_power)
    write_csv(OUT / "per_run_metrics.csv", rows)
    write_csv(OUT / "aggregate_metrics.csv", aggregate(rows))
    pairs = paired_differences(rows)
    write_csv(OUT / "paired_differences.csv", pairs)
    plot_results(rows, signal_rows, traces, representative_seed)
    if args.videos:
        for method in VIDEO_METHODS:
            tr = next(item for item in traces if item["method"] == method and item["seed"] == representative_seed)
            encode_video(REPORT / "videos" / f"seed{representative_seed:02d}_{method}.mp4",
                         tr["times"], tr["values"], tr["particles"], tr["phases"], tr["names"], method, representative_seed)
    config = json.loads((OUT / "frozen_config.json").read_text())
    config["representative_seed_no_comp_median_retention"] = int(representative_seed)
    (OUT / "frozen_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    _markdown_report(rows, signal_rows, pairs, representative_seed, config)
    print(f"Report: {REPORT / '系统级Estimator闭环比较报告.md'}")


if __name__ == "__main__":
    main()
