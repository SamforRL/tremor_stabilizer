"""Round 5: real recorded task motion for voluntary prior identification.

Phase A freezes a model using only development subjects. Phase B and its
held-out voluntary subjects must not be run before that freeze exists.
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

from .recorded import ROOT
from .round5_motion import MotionKF,prediction_trace
from .round4_estimators import JointKinematicBMFLCKF
from .round3_variants import RobustBMFLC
from .interface import create_estimator
from .metrics import run_stream,evaluate
from .signals import SignalComponents,BenchmarkCaseConfig
from .calibration import calibrate_pinheiro_modal
from .run_round4 import calibration_band_rms


REPORT=ROOT/"reports/tremor_baselines/round5"
RESULT=ROOT/"experiments/results/tremor_baselines/round5"
DATA=ROOT/"data/processed"
ACTIVITIES=("spiral","typing","rest","mixing","brushing","walking")
PRIMARY=("spiral","typing","mixing","brushing")
FS=50.0
SEED=20261027
HORIZONS=(1,5,12)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader();writer.writerows(rows)


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")


def rms(x):
    x=np.asarray(x,dtype=float)
    return float(np.sqrt(np.mean(x*x)))


def subject_split():
    path=RESULT/"round5_subject_split.json"
    if path.is_file():
        split=json.loads(path.read_text())
    else:
        subjects=sorted(int(p.stem.split("_")[1][1:]) for p in DATA.glob("bioclitee_s*_spiral.npz"))
        if len(subjects)!=20 or len(set(subjects))!=20:
            raise ValueError("Expected 20 unique subjects before splitting")
        shuffled=np.random.default_rng(SEED).permutation(subjects).tolist()
        development=shuffled[:15];heldout=shuffled[15:]
        folds=[development[i::3] for i in range(3)]
        split=dict(seed=SEED,development_subjects=development,
                   heldout_subjects=heldout,development_cv_folds=folds,
                   rule="subject-disjoint 15/5 split; no final test tuning")
        write_json(path,split)
    if set(split["development_subjects"]) & set(split["heldout_subjects"]):
        raise ValueError("Subject leakage in saved split")
    return split


def valid_windows(subject:int,activity:str,window_s:float=10,maximum:int=2,
                  minimum_s:float=8):
    """Nonoverlapping contiguous raw samples; no smoothing or zero padding."""
    with np.load(DATA/f"bioclitee_s{subject:02d}_{activity}.npz") as d:
        t=np.asarray(d["time_ms"],dtype=float)
        gyro=np.asarray(d["gyro_deg_s"],dtype=float)
    if gyro.shape!=(len(t),3) or not np.all(np.isfinite(gyro)):
        raise ValueError("Invalid gyro data")
    cuts=np.r_[0,np.flatnonzero(np.diff(t)>30)+1,len(t)]
    windows=[]
    n=round(window_s*FS)
    # A genuinely short activity may use its complete valid record. A short
    # fragment caused by a timestamp gap in an otherwise long trial is not a
    # stand-in for the desired 8–12 s activity window.
    required=round((minimum_s if len(t)<round(8*FS) else 8)*FS)
    for a,b in zip(cuts[:-1],cuts[1:]):
        length=b-a
        if length<required:
            continue
        if length<n:
            windows.append((t[a:b],gyro[a:b],a,b))
        else:
            for start in range(a,b-n+1,n):
                windows.append((t[start:start+n],gyro[start:start+n],start,start+n))
                if len(windows)>=maximum:
                    return windows
        if len(windows)>=maximum:
            break
    return windows


def candidates():
    rows=[]
    for r in (0.1,1.0,5.0):
        for q in (0.5,5.0,50.0):
            rows.append(dict(model="random_walk",process_scale=q,measurement_sd_deg_s=r,
                             correlation_time_s=None))
        for j in (100.0,1000.0,10000.0):
            rows.append(dict(model="constant_acceleration",process_scale=j,
                             measurement_sd_deg_s=r,correlation_time_s=None))
        for tau in (0.2,1.0):
            for sigma_a in (10.0,100.0,1000.0):
                rows.append(dict(model="singer",process_scale=sigma_a,
                                 measurement_sd_deg_s=r,correlation_time_s=tau))
    return rows


def motion_metrics(y,config):
    method=MotionKF(**config,fs=FS)
    filtered,innovation,forecasts,acceleration=prediction_trace(y,method,HORIZONS)
    start=round(FS)
    out=dict(filter_rmse_deg_s=rms(filtered[start:]-y[start:]),
             innovation_rmse_deg_s=rms(innovation[start:]))
    for horizon in HORIZONS:
        idx=max(start,horizon)
        error=forecasts[horizon][idx:]-y[idx:]
        naive=y[idx-horizon:-horizon]-y[idx:]
        out[f"pred_{horizon}_rmse_deg_s"]=rms(error)
        out[f"persistence_{horizon}_rmse_deg_s"]=rms(naive)
    z=innovation[start:]-np.mean(innovation[start:])
    denom=float(z@z)
    acf=[float(z[:-lag]@z[lag:]/denom) if denom>0 else 0.0 for lag in range(1,11)]
    out.update({f"innovation_acf_lag{lag}":value for lag,value in enumerate(acf,1)})
    out["innovation_acf_energy"]=float(np.sum(np.square(acf)))
    return out,innovation,filtered,acceleration


def _all_subject_windows(subjects,activities=ACTIVITIES,maximum=2):
    for subject in subjects:
        for activity in activities:
            for window_index,(time_ms,gyro,start,end) in enumerate(valid_windows(subject,activity,maximum=maximum)):
                for axis in range(3):
                    yield subject,activity,window_index,"XYZ"[axis],time_ms,gyro[:,axis],start,end


def _score(row):
    # Per-window persistence ratio removes amplitude dominance and rewards
    # predictive value at all horizons. Filtering RMSE is never selected on.
    weights={1:0.3,5:0.35,12:0.35}
    return float(sum(weight*np.log(max(row[f"pred_{h}_rmse_deg_s"],1e-9)/
                                   max(row[f"persistence_{h}_rmse_deg_s"],1e-9))
                     for h,weight in weights.items()))


def run_phase_a_development():
    split=subject_split()
    rows=[]
    windows=list(_all_subject_windows(split["development_subjects"],maximum=2))
    for candidate_id,config in enumerate(candidates()):
        for subject,activity,index,axis,t,y,start,end in windows:
            metrics,_,_,_=motion_metrics(y,config)
            rows.append(dict(candidate_id=candidate_id,subject=subject,activity=activity,
                             window_index=index,axis=axis,start_sample=start,samples=len(y),
                             **config,**metrics,selection_log_ratio=_score(metrics)))
        print("phase A candidate",candidate_id+1,"/",len(candidates()),flush=True)
    write_csv(REPORT/"phase_a/development_grid.csv",rows)
    scores=[]
    for candidate_id,config in enumerate(candidates()):
        subset=[r for r in rows if r["candidate_id"]==candidate_id and r["activity"] in PRIMARY]
        subject_means={subject:float(np.mean([r["selection_log_ratio"] for r in subset if r["subject"]==subject]))
                       for subject in split["development_subjects"]}
        fold_scores=[float(np.mean([subject_means[s] for s in fold])) for fold in split["development_cv_folds"]]
        scores.append(dict(candidate_id=candidate_id,**config,
                           fold0=fold_scores[0],fold1=fold_scores[1],fold2=fold_scores[2],
                           cv_mean=float(np.mean(fold_scores)),cv_std=float(np.std(fold_scores))))
    write_csv(REPORT/"phase_a/development_cv_scores.csv",scores)
    best_by_model={model:min((r for r in scores if r["model"]==model),key=lambda r:r["cv_mean"])
                   for model in ("random_walk","constant_acceleration","singer")}
    selected=min(best_by_model.values(),key=lambda r:r["cv_mean"])
    frozen=dict(selected_model=selected["model"],selected_candidate_id=selected["candidate_id"],
                selected_parameters={k:selected[k] for k in ("model","process_scale","measurement_sd_deg_s","correlation_time_s")},
                best_per_model={model:{k:row[k] for k in ("candidate_id","process_scale","measurement_sd_deg_s","correlation_time_s","cv_mean","cv_std")}
                                for model,row in best_by_model.items()},
                selection="lowest mean 3-fold subject-wise CV weighted log prediction RMSE / persistence RMSE; primary activities only",
                folds=split["development_cv_folds"],development_subjects=split["development_subjects"],
                heldout_subjects=split["heldout_subjects"],fs_hz=FS,
                horizons_samples=HORIZONS,initial_velocity_sd_deg_s=50,
                initial_acceleration_sd_deg_s2=500,
                final_subjects_used_for_selection=False)
    write_json(RESULT/"phase_a/frozen_voluntary_prior.json",frozen)
    # Optional oracle-best development subject only; never used in final.
    oracle=[]
    for subject in split["development_subjects"]:
        options=[]
        for candidate_id,config in enumerate(candidates()):
            subset=[r for r in rows if r["subject"]==subject and r["candidate_id"]==candidate_id
                    and r["activity"] in PRIMARY]
            options.append((float(np.mean([r["selection_log_ratio"] for r in subset])),candidate_id,config))
        score,candidate_id,config=min(options)
        oracle.append(dict(subject=subject,candidate_id=candidate_id,score=score,**config))
    write_csv(REPORT/"phase_a/development_subject_oracles.csv",oracle)
    return frozen


def load_frozen():
    path=RESULT/"phase_a/frozen_voluntary_prior.json"
    if not path.is_file():
        raise FileNotFoundError("Run --part phase_a_development before held-out tests")
    return json.loads(path.read_text())


def _welch_psd(x,fs=FS,nperseg=256):
    x=np.asarray(x,dtype=float)
    if len(x)<nperseg:return None
    win=np.hanning(nperseg)
    scale=fs*np.sum(win**2)
    power=[]
    for start in range(0,len(x)-nperseg+1,nperseg//2):
        z=x[start:start+nperseg]-np.mean(x[start:start+nperseg])
        spec=np.abs(np.fft.rfft(z*win))**2/scale
        spec[1:-1]*=2
        power.append(spec)
    return np.fft.rfftfreq(nperseg,1/fs),np.mean(power,axis=0)


def run_phase_a_heldout():
    frozen=load_frozen();split=subject_split();rows=[];psds={}
    configs=[]
    for model,best in frozen["best_per_model"].items():
        configs.append(dict(model=model,process_scale=best["process_scale"],
                            measurement_sd_deg_s=best["measurement_sd_deg_s"],
                            correlation_time_s=best["correlation_time_s"]))
    for subject,activity,index,axis,t,y,start,end in _all_subject_windows(split["heldout_subjects"],maximum=2):
        for config in configs:
            metric,innovation,filtered,accel=motion_metrics(y,config)
            rows.append(dict(subject=subject,activity=activity,window_index=index,axis=axis,
                             start_sample=start,samples=len(y),model=config["model"],
                             **metric,selection_log_ratio=_score(metric)))
            result=_welch_psd(innovation[round(FS):])
            if result is not None:
                key=(activity,config["model"])
                psds.setdefault(key,[]).append(result[1])
            if subject==split["heldout_subjects"][0] and index==0 and axis=="X":
                directory=RESULT/"phase_a/representative"
                directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/f"{activity}_{config['model']}.npz",
                                    time_s=(t-t[0])/1000,gyro_deg_s=y,filtered_deg_s=filtered,
                                    innovation_deg_s=innovation,acceleration_deg_s2=accel)
    write_csv(REPORT/"phase_a/heldout_trials.csv",rows)
    summary=[]
    for activity in ACTIVITIES:
        for model in ("random_walk","constant_acceleration","singer"):
            subset=[r for r in rows if r["activity"]==activity and r["model"]==model]
            if not subset:continue
            subject_rows=[]
            for subject in split["heldout_subjects"]:
                sr=[r for r in subset if r["subject"]==subject]
                if sr:subject_rows.append({key:float(np.median([r[key] for r in sr])) for key in
                                           ("pred_1_rmse_deg_s","pred_5_rmse_deg_s","pred_12_rmse_deg_s",
                                            "innovation_acf_energy","filter_rmse_deg_s","selection_log_ratio")})
            summary.append(dict(activity=activity,model=model,n_subjects=len(subject_rows),
                                **{key+"_subject_median":float(np.median([r[key] for r in subject_rows])) for key in subject_rows[0]}))
    write_csv(REPORT/"phase_a/heldout_activity_summary.csv",summary)
    psd_rows=[]
    for (activity,model),spectra in psds.items():
        mean=np.mean(spectra,axis=0)
        for f,power in zip(np.fft.rfftfreq(256,1/FS),mean):
            psd_rows.append(dict(activity=activity,model=model,frequency_hz=float(f),
                                 innovation_psd_deg2_s=float(power),n_windows=len(spectra)))
    write_csv(REPORT/"phase_a/innovation_psd.csv",psd_rows)
    return rows,summary


PHASE_B_METHODS=("bmflc_lms","bmflc_huber","abpf_adaptive",
                 "joint_ca_round4","joint_rw_round4","joint_data_prior")


def _phase_b_method(name,subject,fs,scale,modal,round4,frozen):
    bm=dict(sample_rate_hz=fs,low_hz=3,high_hz=12,spacing_hz=.1,lms_gain=.01)
    if name=="bmflc_lms":return create_estimator("bmflc",bm)
    if name=="bmflc_huber":return RobustBMFLC(**bm,variant="huber",huber_delta=round4["huber_factor"]*scale)
    if name=="abpf_adaptive":
        return create_estimator("abpf",dict(sample_rate_hz=fs,modal_frequency_hz=modal,
                                            frequency_step_hz=.1,limit_offset_hz=1.5))
    p_v=round4["initial_voluntary_variance_factor"]*scale**2
    p_a=round4["initial_acceleration_variance_factor"]*scale**2
    p_w=round4["initial_weight_variance_factor"]*scale**2/91
    q_w=round4["qw_factor"]*scale**2/91
    r=round4["r_factor"]*scale**2
    sigma=round4["sigma_j_factor"]*scale
    model="constant_acceleration" if name=="joint_ca_round4" else "random_walk"
    if name=="joint_rw_round4":sigma*=.5/fs**2
    if name=="joint_data_prior":
        selected=frozen["selected_parameters"]
        model=selected["model"]
        if model not in ("random_walk","constant_acceleration"):
            raise RuntimeError("Frozen Singer prior needs an explicit Singer joint implementation")
        sigma=selected["process_scale"]
    if name not in ("joint_ca_round4","joint_rw_round4","joint_data_prior"):
        raise ValueError(name)
    return JointKinematicBMFLCKF(sample_rate_hz=fs,sigma_j_deg_s3=sigma,
        weight_process_variance=q_w,measurement_variance=r,
        initial_voluntary_variance=p_v,initial_acceleration_variance=p_a,
        initial_weight_variance=p_w,low_hz=3,high_hz=12,spacing_hz=.1,
        voluntary_model=model)


def _proxy(subject,axis,start):
    path=RESULT/f"resampled_proxy/{subject}_{axis}_{start}s.npz"
    with np.load(path) as data:
        out=np.asarray(data["proxy_deg_s"],dtype=float)
        if float(data["fs_hz"])!=FS:raise ValueError("Wrong proxy rate")
    return out


def _mixture(task,proxy,amplitude_scale=1.0,case="recorded_task"):
    n=len(task)
    if len(proxy)<n:raise ValueError("Proxy is too short; no zero padding allowed")
    v=np.asarray(task,dtype=float)
    d=amplitude_scale*np.asarray(proxy[:n],dtype=float)
    config=BenchmarkCaseConfig(sample_rate_hz=FS,duration_s=n/FS,
                               noise_std_deg_s=0,random_seed=SEED)
    return SignalComponents(np.arange(n)/FS,v[:,None],d[:,None],np.zeros((n,1)),
                            (v+d)[:,None],case,config)


def _score_separation(signal,name,estimator):
    output,diag,elapsed=run_stream(estimator,signal.mixed_observation_deg_s)
    metric=evaluate(signal,output,elapsed,(3,12))
    v=signal.voluntary_true_deg_s[:,0]
    d=signal.tremor_true_deg_s[:,0]
    dhat=output[:,0]
    if name.startswith("joint_"):
        vhat=np.array([item["voluntary_velocity_deg_s"][0] for item in diag])
        weights=np.stack([item["adaptive_weights"][0] for item in diag])
    else:
        vhat=signal.mixed_observation_deg_s[:,0]-dhat
        weights=(np.stack([item["adaptive_weights"][0,:-1] for item in diag])
                 if name in ("bmflc_lms","bmflc_huber") else None)
    denom_v=float(v@v);denom_d=float(d@d)
    return dict(tremor_rmse_deg_s=rms(dhat-d),tremor_nrmse=rms(dhat-d)/max(rms(d),1e-9),
                task_rmse_deg_s=rms(vhat-v),task_nrmse=rms(vhat-v)/max(rms(v),1e-9),
                complement_task_rmse_deg_s=metric["voluntary_reconstruction_rmse_deg_s"],
                voluntary_to_tremor_leakage=float(np.dot(dhat-d,v)/denom_v) if denom_v else None,
                tremor_to_voluntary_leakage=float(np.dot(vhat-v,d)/denom_d) if denom_d else None,
                mean_weight_norm=float(np.mean(np.linalg.norm(weights,axis=1))) if weights is not None else None,
                peak_weight_norm=float(np.max(np.linalg.norm(weights,axis=1))) if weights is not None else None,
                band_power_reduction_db=metric["band_power_reduction_db"],
                mean_update_us=metric["mean_update_us"],
                max_update_us=metric["max_update_us"]),output,diag,vhat


def _save_separation_trace(label,signal,name,output,diag,vhat):
    raw=RESULT/"phase_b/representative";plot=REPORT/"phase_b/diagnostics"
    raw.mkdir(parents=True,exist_ok=True);plot.mkdir(parents=True,exist_ok=True)
    path=raw/f"{label}_{name}.npz"
    info=dict(time_s=signal.time_s,mixture_deg_s=signal.mixed_observation_deg_s[:,0],
              task_reference_deg_s=signal.voluntary_true_deg_s[:,0],
              injected_proxy_deg_s=signal.tremor_true_deg_s[:,0],
              estimated_task_deg_s=vhat,estimated_tremor_deg_s=output[:,0])
    if name.startswith("joint_"):
        info.update(innovation_deg_s=np.array([z["innovation_deg_s"][0] for z in diag]),
                    acceleration_deg_s2=np.array([z["voluntary_acceleration_deg_s2"][0] for z in diag]),
                    weight_norm=np.array([np.linalg.norm(z["adaptive_weights"][0]) for z in diag]),
                    voluntary_gain=np.stack([z["kalman_gain_voluntary"][0] for z in diag]),
                    tremor_gain_norm=np.array([z["kalman_gain_tremor_norm"][0] for z in diag]),
                    voluntary_variance=np.stack([z["posterior_voluntary_variance"][0] for z in diag]),
                    weight_variance=np.array([z["posterior_weight_variance_sum"][0] for z in diag]))
    np.savez_compressed(path,**info)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for key in ("mixture_deg_s","task_reference_deg_s","estimated_task_deg_s"):
        axes[0].plot(info["time_s"],info[key],label=key,alpha=.8)
    for key in ("injected_proxy_deg_s","estimated_tremor_deg_s"):
        axes[1].plot(info["time_s"],info[key],label=key,alpha=.8)
    if name.startswith("joint_"):
        axes[2].plot(info["time_s"],info["innovation_deg_s"],label="innovation")
        axes[2].plot(info["time_s"],info["weight_norm"],label="Fourier weight norm")
    for ax in axes:ax.legend(fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout();fig.savefig(plot/f"{label}_{name}.png",dpi=130);plt.close(fig)


def _check_resampling():
    rows=list(csv.DictReader(open(REPORT/"data_audit/proxy_resampling.csv")))
    if len(rows)!=45:raise ValueError("Need 45 checked proxy resamples")
    for row in rows:
        if abs(float(row["rms_relative_change"]))>.02 or abs(float(row["band_power_relative_change"]))>.1 or abs(float(row["dominant_change_hz"]))>.25:
            raise ValueError("Proxy resampling distortion exceeds fixed audit thresholds")


def run_phase_b():
    frozen=load_frozen();split=subject_split();_check_resampling()
    round4=json.loads((ROOT/"experiments/results/tremor_baselines/round4/development/frozen_parameters.json").read_text())
    pairing={subject:f"p{i:02d}" for i,subject in enumerate(sorted(split["heldout_subjects"]),1)}
    baseline=[];new=[];ratio_rows=[];window_rows=[]
    for subject in split["heldout_subjects"]:
        pinheiro=pairing[subject]
        for activity in PRIMARY:
            for index,(time_ms,gyro,start,end) in enumerate(valid_windows(subject,activity,minimum_s=3)):
                proxy_start=(20,33)[index]
                for ia,axis in enumerate("XYZ"):
                    task=gyro[:,ia]
                    proxy=_proxy(pinheiro,axis,proxy_start)
                    signal=_mixture(task,proxy,1.0,f"{activity}_natural")
                    s=calibration_band_rms(f"{pinheiro}_1",ia,148.14814814814815)
                    modal=float(calibrate_pinheiro_modal(f"{pinheiro}_1",(ia,),20,12).modal_frequency_hz[0])
                    common=dict(voluntary_subject=subject,pinheiro_subject=pinheiro,
                                activity=activity,axis=axis,window_index=index,
                                task_start_sample=start,samples=len(task),duration_s=len(task)/FS,
                                task_rms_deg_s=rms(task),proxy_rms_deg_s=rms(proxy[:len(task)]),
                                rho=rms(task)/max(rms(proxy[:len(task)]),1e-9),
                                source_proxy_start_s=proxy_start,
                                source_calibration_rms_deg_s=s,
                                source_modal_hz=modal)
                    window_rows.append(common)
                    for method_name in PHASE_B_METHODS:
                        estimator=_phase_b_method(method_name,subject,FS,s,modal,round4,frozen)
                        metrics,output,diag,vhat=_score_separation(signal,method_name,estimator)
                        row=dict(**common,method=method_name,protocol="natural",**metrics)
                        (baseline if method_name in PHASE_B_METHODS[:3] else new).append(row)
                        if index==0 and axis=="X" and activity=="mixing" and method_name in ("joint_ca_round4","joint_rw_round4","joint_data_prior"):
                            _save_separation_trace(f"s{subject:02d}_{activity}",signal,method_name,output,diag,vhat)
                    # Fixed representative subset: first valid 10 s window,
                    # X axis, all five held-out people and four activities.
                    if index==0 and axis=="X" and len(task)>=round(8*FS):
                        for target_rho in (.25,1.,4.,8.):
                            factor=rms(task)/max(target_rho*rms(proxy[:len(task)]),1e-9)
                            swept=_mixture(task,proxy,factor,f"{activity}_rho{target_rho}")
                            for method_name in PHASE_B_METHODS:
                                estimator=_phase_b_method(method_name,subject,FS,s,modal,round4,frozen)
                                metric,_,_,_=_score_separation(swept,method_name,estimator)
                                ratio_rows.append(dict(**common,method=method_name,protocol="ratio_sweep",
                                                       target_rho=target_rho,proxy_amplitude_factor=factor,
                                                       actual_rho=rms(task)/rms(swept.tremor_true_deg_s),**metric))
        print("phase B heldout subject",subject,flush=True)
    write_csv(REPORT/"phase_b/natural_baselines.csv",baseline)
    write_csv(REPORT/"phase_b/natural_joint_methods.csv",new)
    write_csv(REPORT/"phase_b/ratio_sweep.csv",ratio_rows)
    write_csv(REPORT/"phase_b/windows_and_pairing.csv",window_rows)
    write_json(RESULT/"phase_b/protocol.json",dict(heldout_subjects=split["heldout_subjects"],
               pinheiro_pairing=pairing,pairing_rule="sorted heldout subject IDs map to p01..p05",
               selected_voluntary_prior=frozen["selected_parameters"],
               round4_tremor_side={k:round4[k] for k in ("qw_factor","r_factor","initial_weight_variance_factor")},
               methods=PHASE_B_METHODS,activities=PRIMARY,axes="XYZ",windows_max=2,
               window_s=10,minimum_s=3,noise_std_deg_s=0,ratio_levels=[.25,1,4,8],
               ratio_subset="first valid activity window, X axis, all heldout people",
               fs_hz=FS,seed=SEED,no_final_tuning=True))
    all_rows=baseline+new
    summary=[]
    for activity in PRIMARY:
        for method_name in PHASE_B_METHODS:
            subset=[r for r in all_rows if r["activity"]==activity and r["method"]==method_name]
            person=[]
            for subject in split["heldout_subjects"]:
                p=[r for r in subset if r["voluntary_subject"]==subject]
                if p:person.append({k:float(np.median([r[k] for r in p])) for k in
                                    ("tremor_nrmse","task_nrmse","tremor_rmse_deg_s","task_rmse_deg_s",
                                     "voluntary_to_tremor_leakage","tremor_to_voluntary_leakage","mean_update_us")})
            summary.append(dict(activity=activity,method=method_name,n_subjects=len(person),
                                **{k+"_subject_median":float(np.median([p[k] for p in person])) for k in person[0]}))
    write_csv(REPORT/"phase_b/activity_subject_summary.csv",summary)
    leakage_summary=[]
    for activity in PRIMARY:
        for method_name in PHASE_B_METHODS:
            subset=[r for r in all_rows if r["activity"]==activity and r["method"]==method_name]
            subject_values=[]
            for subject in split["heldout_subjects"]:
                person=[r for r in subset if r["voluntary_subject"]==subject]
                if person:
                    subject_values.append(dict(subject=subject,
                         abs_l_v_to_d=float(np.median([abs(r["voluntary_to_tremor_leakage"]) for r in person])),
                         abs_l_d_to_v=float(np.median([abs(r["tremor_to_voluntary_leakage"]) for r in person]))))
            leakage_summary.append(dict(activity=activity,method=method_name,n_subjects=len(subject_values),
                 abs_l_v_to_d_subject_median=float(np.median([p["abs_l_v_to_d"] for p in subject_values])),
                 abs_l_d_to_v_subject_median=float(np.median([p["abs_l_d_to_v"] for p in subject_values]))))
    write_csv(REPORT/"phase_b/activity_absolute_leakage.csv",leakage_summary)
    by_subject=[]
    for subject in split["heldout_subjects"]:
        for method_name in PHASE_B_METHODS:
            person=[r for r in all_rows if r["voluntary_subject"]==subject and r["method"]==method_name]
            if not person:
                continue
            record=dict(voluntary_subject=subject,method=method_name,n_axis_windows=len(person))
            for key in ("tremor_nrmse","task_nrmse","tremor_rmse_deg_s","task_rmse_deg_s",
                        "band_power_reduction_db","mean_update_us"):
                record[key+"_median"]=float(np.median([r[key] for r in person]))
            for key in ("voluntary_to_tremor_leakage","tremor_to_voluntary_leakage"):
                record["abs_"+key+"_median"]=float(np.median([abs(r[key]) for r in person]))
            by_subject.append(record)
    write_csv(REPORT/"phase_b/by_subject_summary.csv",by_subject)
    return all_rows,ratio_rows,summary


def render_plots():
    def read(path):return list(csv.DictReader(path.open()))
    held=read(REPORT/"phase_a/heldout_activity_summary.csv")
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    models=("random_walk","constant_acceleration","singer")
    for ax,h in zip(axes,HORIZONS):
        x=np.arange(len(PRIMARY));width=.25
        for j,model in enumerate(models):
            vals=[float(next(r[f"pred_{h}_rmse_deg_s_subject_median"] for r in held
                             if r["activity"]==activity and r["model"]==model)) for activity in PRIMARY]
            ax.bar(x+(j-1)*width,vals,width,label=model)
        ax.set_xticks(x,PRIMARY,rotation=25)
        ax.set_yscale("log");ax.set_title(f"causal {h/FS:.2f} s prediction")
        ax.set_ylabel("RMSE (deg/s)")
    axes[0].legend(fontsize=7)
    fig.savefig(REPORT/"phase_a/heldout_prediction.png",dpi=150);plt.close(fig)
    psd=read(REPORT/"phase_a/innovation_psd.csv")
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,activity in zip(axes.flat,PRIMARY):
        for model in models:
            subset=[r for r in psd if r["activity"]==activity and r["model"]==model]
            ax.semilogy([float(r["frequency_hz"]) for r in subset],
                        [float(r["innovation_psd_deg2_s"]) for r in subset],label=model)
        ax.set(title=activity,xlabel="Hz",ylabel="innovation PSD")
    axes[0,0].legend(fontsize=7)
    fig.savefig(REPORT/"phase_a/innovation_psd.png",dpi=150);plt.close(fig)
    summary=read(REPORT/"phase_b/activity_subject_summary.csv")
    leakage=read(REPORT/"phase_b/activity_absolute_leakage.csv")
    methods=PHASE_B_METHODS
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,key,title,data in zip(axes,("tremor_nrmse_subject_median","abs_l_d_to_v_subject_median"),
                                 ("tremor nRMSE","median |tremor → task leakage|"),
                                 (summary,leakage)):
        matrix=np.array([[float(next(r[key] for r in data if r["activity"]==a and r["method"]==m))
                          for m in methods] for a in PRIMARY])
        im=ax.imshow(matrix,aspect="auto",cmap="viridis")
        ax.set_yticks(range(len(PRIMARY)),PRIMARY)
        ax.set_xticks(range(len(methods)),methods,rotation=45,ha="right",fontsize=7)
        ax.set_title(title)
        for i in range(len(PRIMARY)):
            for j in range(len(methods)):
                ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=7,
                        color="white" if matrix[i,j]<.3 else "black")
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(REPORT/"phase_b/activity_comparison.png",dpi=150);plt.close(fig)
    ratios=read(REPORT/"phase_b/ratio_sweep.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for method in methods:
        xs=[];ys=[];zs=[]
        for rho in (.25,1.,4.,8.):
            subset=[r for r in ratios if r["method"]==method and float(r["target_rho"])==rho]
            xs.append(rho)
            ys.append(float(np.median([float(r["tremor_nrmse"]) for r in subset])))
            zs.append(float(np.median([abs(float(r["tremor_to_voluntary_leakage"])) for r in subset])))
        axes[0].plot(xs,ys,marker="o",label=method)
        axes[1].plot(xs,zs,marker="o",label=method)
    axes[0].set(xscale="log",yscale="log",xlabel="target rho",ylabel="tremor nRMSE")
    axes[1].set(xscale="log",xlabel="target rho",ylabel="median |tremor → task leakage|")
    axes[0].legend(fontsize=7)
    fig.savefig(REPORT/"phase_b/ratio_sweep.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=("phase_a_development","phase_a_heldout","phase_b","plots","all"),default="all")
    part=parser.parse_args().part;start=time.monotonic()
    if part in ("all","phase_a_development"):
        print("frozen",run_phase_a_development(),flush=True)
    if part in ("all","phase_a_heldout"):
        rows,summary=run_phase_a_heldout()
        print("heldout",len(rows),len(summary),flush=True)
    if part in ("all","phase_b"):
        rows,ratios,summary=run_phase_b()
        print("phase B",len(rows),len(ratios),len(summary),flush=True)
    if part in ("all","plots"):
        render_plots()
    print("completed",round(time.monotonic()-start,1),"s",flush=True)


if __name__=="__main__":
    main()
