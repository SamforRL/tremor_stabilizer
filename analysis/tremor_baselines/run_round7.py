"""Round 7: causal spectral prefilter × unchanged BMFLC update law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .butterworth import sos_response
from .metrics import run_stream
from .recorded import ROOT,load_pinheiro_window
from .round3_variants import RobustBMFLC
from .run_round3 import calibration_band_rms
from .round7_prefilter import PrefilteredBMFLC,design_prefilter,filter_trace,response_metrics
from .signals import BenchmarkCaseConfig,SignalComponents,controlled_tremor,generate_case

RESULT=ROOT/"experiments/results/tremor_baselines/round7"
REPORT=ROOT/"reports/tremor_baselines/round7"
FS=148.14814814814815
SUBJECTS=tuple(f"p{i:02d}" for i in range(1,6))
WINDOWS=(20.,33.,46.)
SCENARIOS=("slow_smooth","point_to_point","fast_transition","spectral_overlap")
TREMORS=("constant","drift","step","amplitude_modulated","multifrequency")
RHO=(.25,.5,1.,2.,4.,8.,16.)
CORE=("raw_lms","raw_huber","hpf2_lms","hpf2_huber")
SECONDARY=("bpf_lms","bpf_huber")
METHODS=CORE+SECONDARY

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")

def rms(x):
    x=np.asarray(x,dtype=float)
    return float(np.sqrt(np.mean(x*x)))

def method_config(name):
    if name.startswith("raw_"):return "none","huber" if name.endswith("huber") else "lms"
    if name.startswith("hpf2_"):return "hpf","huber" if name.endswith("huber") else "lms"
    if name.startswith("bpf_"):return "bpf","huber" if name.endswith("huber") else "lms"
    raise ValueError(name)

def estimator(name,fs,scale,cutoff_hz=2.):
    kind,update=method_config(name)
    return PrefilteredBMFLC(fs,kind=kind,cutoff_hz=cutoff_hz,update=update,
                            huber_delta=.5*scale if update=="huber" else None)

def calibration_scales(subject,axis,fs=FS):
    table=f"{subject}_1"
    raw=calibration_band_rms(table,axis,fs)
    with np.load(ROOT/f"data/processed/pinheiro_{table}_lumbrical_gyro.npz") as d:
        t=np.asarray(d["time_s"],dtype=float)
        x=np.asarray(d["gyro_deg_s"],dtype=float)[:,axis]
    uniform=np.arange(round(15*fs))/fs
    values=np.interp(uniform,t,x)
    scales={"raw":raw}
    for label,kind in (("hpf2","hpf"),("bpf","bpf")):
        filtered=filter_trace(values,design_prefilter(kind,fs,2.))
        f=np.fft.rfftfreq(len(filtered),1/fs)
        sp=np.fft.rfft(filtered)
        sp[(f<3)|(f>12)]=0
        scales[label]=rms(np.fft.irfft(sp,n=len(filtered)))
    return scales

def synthetic(tremor,scenario,rho=1.,scale=5.):
    t,d,_=controlled_tremor(tremor,FS,12.,amplitude_deg_s=scale)
    config=BenchmarkCaseConfig(sample_rate_hz=FS,duration_s=12.,noise_std_deg_s=0)
    original=generate_case(scenario,d,config)
    v=original.voluntary_true_deg_s[:,0]
    d=d[:,0]
    mask=(t>=3)&(t<5)
    v=v*(rho*rms(d[mask])/max(rms(v[mask]),1e-9))
    return SignalComponents(t,v[:,None],d[:,None],np.zeros((len(t),1)),
                            (v+d)[:,None],f"{tremor}_{scenario}_rho{rho}",config)

def recorded(subject,start,axis,scenario,rho=None):
    window=load_pinheiro_window(table=f"{subject}_1",start_s=start,duration_s=12.,
                                channel_indices=(axis,))
    config=BenchmarkCaseConfig(sample_rate_hz=window.sample_rate_hz,duration_s=12.,noise_std_deg_s=0)
    original=generate_case(scenario,window.tremor_band_proxy_deg_s,config)
    if rho is None:return original,window
    t=original.time_s;mask=(t>=3)&(t<5)
    v=original.voluntary_true_deg_s[:,0]
    d=original.tremor_true_deg_s[:,0]
    v=v*(rho*rms(d[mask])/max(rms(v[mask]),1e-9))
    return SignalComponents(t,v[:,None],d[:,None],np.zeros((len(t),1)),
                            (v+d)[:,None],f"{scenario}_rho{rho}",config),window

def weight_metrics(t,norm):
    pre=norm[(t>=0)&(t<3)]
    active=norm[(t>=3)&(t<5)]
    post=norm[(t>=5)&(t<12)]
    pre_mean=float(np.mean(pre));peak=float(np.max(active));increment=max(peak-pre_mean,0.)
    threshold=pre_mean+.2*increment
    width=round(.5/(t[1]-t[0]))
    recovery=None
    if increment>0:
        start=np.searchsorted(t,5.)
        for k in range(start,len(norm)-width+1):
            if np.all(norm[k:k+width]<=threshold):
                recovery=float(t[k]-5.)
                break
    return dict(weight_pre_mean=pre_mean,weight_transition_peak=peak,
                weight_peak_increment=increment,weight_post_mean=float(np.mean(post)),
                weight_post_residual=float(np.mean(post)-pre_mean),
                recovery_time_s=recovery)

def score(signal,name,scale,save_trace=None,cutoff_hz=2.):
    fs=signal.config.sample_rate_hz
    method=estimator(name,fs,scale,cutoff_hz)
    output,diagnostics,elapsed=run_stream(method,signal.mixed_observation_deg_s)
    output=output[:,0];t=signal.time_s
    d=signal.tremor_true_deg_s[:,0];v=signal.voluntary_true_deg_s[:,0]
    x=np.array([item["raw_input"][0] for item in diagnostics])
    z=np.array([item["prefiltered_input"][0] for item in diagnostics])
    error=np.array([item["internal_error"][0] for item in diagnostics])
    update=np.array([item.get("robust_update_error",item["internal_error"])[0] for item in diagnostics])
    weights=np.stack([item["adaptive_weights"][0,:-1] for item in diagnostics])
    norm=np.linalg.norm(weights,axis=1)
    kind,_=method_config(name)
    sos=design_prefilter(kind,fs,cutoff_hz)
    vd=filter_trace(v,sos);dd=filter_trace(d,sos)
    mask=(t>=3)&(t<5)
    records=dict(method=name,prefilter=kind,prefilter_cutoff_hz=cutoff_hz if kind=="hpf" else None,
                 update_rule=method.update,
                 tremor_rmse_deg_s=rms(output-d),tremor_nrmse=rms(output-d)/max(rms(d),1e-9),
                 pre_rmse_deg_s=rms((output-d)[t<3]),
                 transition_rmse_deg_s=rms((output-d)[mask]),
                 recovery_rmse_deg_s=rms((output-d)[t>=5]),
                 residual_transition_rms_deg_s=rms(error[mask]),
                 update_error_transition_rms_deg_s=rms(update[mask]),
                 input_transition_rms_deg_s=rms(z[mask]),
                 raw_input_transition_rms_deg_s=rms(x[mask]),
                 voluntary_transmission=rms(vd)/max(rms(v),1e-9),
                 tremor_transmission=rms(dd)/max(rms(d),1e-9),
                 voluntary_transition_transmission=rms(vd[mask])/max(rms(v[mask]),1e-9),
                 tremor_transition_transmission=rms(dd[mask])/max(rms(d[mask]),1e-9),
                 mean_update_us=float(np.mean(elapsed)/1e3),
                 max_update_us=float(np.max(elapsed)/1e3),
                 **weight_metrics(t,norm))
    if save_trace is not None:
        save_trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save_trace,time_s=t,voluntary_deg_s=v,tremor_deg_s=d,
                            mixture_deg_s=x,prefiltered_deg_s=z,estimated_tremor_deg_s=output,
                            prediction_error_deg_s=error,update_error_deg_s=update,
                            fourier_weight_norm=norm,filtered_voluntary_deg_s=vd,
                            filtered_tremor_deg_s=dd)
    return records

def run_response():
    rows=[];curves=[]
    for label,kind,cutoff in (("HPF 1.5","hpf",1.5),("HPF 2.0","hpf",2.),
                               ("HPF 2.5","hpf",2.5),("HPF 3.0","hpf",3.),
                               ("BPF 3-12","bpf",2.)):
        sos=design_prefilter(kind,FS,cutoff)
        m=response_metrics(sos,FS)
        for i,freq in enumerate((4.,6.,8.,10.)):
            rows.append(dict(filter=label,kind=kind,cutoff_hz=cutoff,fs_hz=FS,
                             butterworth_prototype_order=2,digital_filter_order=2 if kind=="hpf" else 4,
                             mean_band_magnitude_error=m["mean_band_magnitude_error"],
                             frequency_hz=freq,magnitude=m["magnitude"][i],
                             phase_rad=m["phase_rad"][i],
                             phase_delay_ms=m["phase_delay_ms"][i],
                             group_delay_ms=m["group_delay_ms"][i]))
        grid=np.linspace(.05,20,300)
        h=sos_response(sos,grid,FS)
        for f,value in zip(grid,h):
            curves.append(dict(filter=label,frequency_hz=float(f),magnitude=float(abs(value)),
                               phase_rad=float(np.angle(value))))
    write_csv(REPORT/"filter_response/summary.csv",rows)
    write_csv(REPORT/"filter_response/curves.csv",curves)
    write_json(RESULT/"frozen_protocol.json",dict(main_prefilter="2nd-order causal HPF 2 Hz",
        secondary_prefilter="Butterworth bandpass prototype order 2, digital order 4, 3-12 Hz",
        highpass_sensitivity_hz=[1.5,2,2.5,3],huber_delta="0.5 * independent raw 0-15 s calibration-band RMS",
        bmflc=dict(low_hz=3,high_hz=12,spacing_hz=.1,lms_gain=.01),
        source="unchanged p01-p05 table _1; windows 20,33,46 s; 3 gyro axes; 4 designed voluntary cases",
        ratio_sweep=RHO,ratio_definition="RMS(voluntary)/RMS(proxy) over 3-5 s",
        no_final_tuning=True))

def run_synthetic():
    rows=[];ratios=[];sensitivity=[]
    for tremor in TREMORS:
        for scenario in SCENARIOS:
            signal=synthetic(tremor,scenario,1.)
            scale=rms(signal.tremor_true_deg_s[:,0])
            for method in METHODS:
                rows.append(dict(tremor=tremor,scenario=scenario,rho=1.,
                                 **score(signal,method,scale)))
            print("synthetic",tremor,scenario,flush=True)
    for scenario in ("fast_transition","spectral_overlap"):
        for rho in RHO:
            signal=synthetic("multifrequency",scenario,rho)
            scale=rms(signal.tremor_true_deg_s[:,0])
            for method in METHODS:
                ratios.append(dict(tremor="multifrequency",scenario=scenario,rho=rho,
                                   **score(signal,method,scale)))
    for scenario in SCENARIOS:
        signal=synthetic("multifrequency",scenario,1.)
        v=signal.voluntary_true_deg_s[:,0];d=signal.tremor_true_deg_s[:,0]
        for cutoff in (1.5,2.,2.5,3.):
            sos=design_prefilter("hpf",FS,cutoff)
            sensitivity.append(dict(scenario=scenario,cutoff_hz=cutoff,
                 voluntary_transmission=rms(filter_trace(v,sos))/rms(v),
                 tremor_transmission=rms(filter_trace(d,sos))/rms(d),
                 mean_band_magnitude_error=response_metrics(sos,FS)["mean_band_magnitude_error"]))
    write_csv(REPORT/"synthetic/controlled_cases.csv",rows)
    write_csv(REPORT/"synthetic/amplitude_ratio.csv",ratios)
    write_csv(REPORT/"synthetic/hpf_cutoff_sensitivity.csv",sensitivity)

def run_hpf_sensitivity():
    rows=[]
    for scenario in ("fast_transition","spectral_overlap"):
        for rho in (1.,4.):
            signal=synthetic("multifrequency",scenario,rho)
            scale=rms(signal.tremor_true_deg_s[:,0])
            for cutoff in (1.5,2.,2.5,3.):
                for method in ("hpf2_lms","hpf2_huber"):
                    rows.append(dict(scenario=scenario,rho=rho,cutoff_hz=cutoff,
                                     **score(signal,method,scale,cutoff_hz=cutoff)))
    write_csv(REPORT/"synthetic/hpf_cutoff_estimator_sensitivity.csv",rows)

def run_recorded():
    rows=[];ratio_rows=[];scales=[]
    for subject in SUBJECTS:
        for axis in range(3):
            cal=calibration_scales(subject,axis)
            scales.append(dict(subject=subject,axis="XYZ"[axis],s_raw_deg_s=cal["raw"],
                               s_hpf_filtered_deg_s=cal["hpf2"],s_bpf_filtered_deg_s=cal["bpf"],
                               fixed_huber_delta_deg_s=.5*cal["raw"]))
            for start in WINDOWS:
                for scenario in SCENARIOS:
                    signal,window=recorded(subject,start,axis,scenario)
                    for method in METHODS:
                        trace=(RESULT/f"traces/{subject}_X_20_fast_{method}.npz"
                               if subject=="p02" and axis==0 and start==20 and scenario=="fast_transition" else None)
                        rows.append(dict(subject=subject,axis="XYZ"[axis],start_s=start,
                                         scenario=scenario,source_sha256=window.source_sha256,
                                         s_raw_deg_s=cal["raw"],s_filtered_deg_s=cal["hpf2"] if method.startswith("hpf2") else cal["bpf"] if method.startswith("bpf") else cal["raw"],
                                         huber_delta_deg_s=.5*cal["raw"],
                                         **score(signal,method,cal["raw"],trace)))
                print("recorded",subject,"XYZ"[axis],start,flush=True)
            # Predeclared ratio subset: first window, fast and overlap.
            for scenario in ("fast_transition","spectral_overlap"):
                for rho in RHO:
                    signal,window=recorded(subject,20.,axis,scenario,rho)
                    for method in CORE:
                        ratio_rows.append(dict(subject=subject,axis="XYZ"[axis],start_s=20.,
                                               scenario=scenario,rho=rho,source_sha256=window.source_sha256,
                                               **score(signal,method,cal["raw"])))
    write_csv(REPORT/"recorded/core_trials.csv",[r for r in rows if r["method"] in CORE])
    write_csv(REPORT/"recorded/bpf_secondary_trials.csv",[r for r in rows if r["method"] in SECONDARY])
    write_csv(REPORT/"recorded/amplitude_ratio.csv",ratio_rows)
    write_csv(REPORT/"recorded/calibration_scales.csv",scales)

def tone_metrics(x,y,fs,freq,start=4.,end=10.):
    t=np.arange(len(x))/fs
    mask=(t>=start)&(t<end)
    a=np.dot(x[mask],np.exp(-2j*np.pi*freq*t[mask]))
    b=np.dot(y[mask],np.exp(-2j*np.pi*freq*t[mask]))
    h=b/a
    phase=float(np.angle(h))
    return dict(amplitude_ratio=float(abs(h)),phase_error_rad=phase,
                effective_phase_delay_ms=-phase/(2*np.pi*freq)*1000,
                rmse_deg_s=rms(y[mask]-x[mask]))

def run_phase():
    rows=[]
    for freq in (4.,6.,8.,10.):
        t=np.arange(round(12*FS))/FS
        d=5*np.sin(2*np.pi*freq*t)
        config=BenchmarkCaseConfig(sample_rate_hz=FS,duration_s=12.,noise_std_deg_s=0)
        signal=SignalComponents(t,np.zeros((len(t),1)),d[:,None],np.zeros((len(t),1)),
                                d[:,None],f"tone_{freq}",config)
        for kind in ("none","hpf","bpf"):
            sos=design_prefilter(kind,FS,2.)
            filtered=filter_trace(d,sos)
            rows.append(dict(frequency_hz=freq,stage="prefilter",kind=kind,method="filter_only",
                             **tone_metrics(d,filtered,FS,freq)))
        for name in METHODS:
            method=estimator(name,FS,rms(d))
            output,_,_=run_stream(method,signal.mixed_observation_deg_s)
            rows.append(dict(frequency_hz=freq,stage="final_bmflc",kind=method.kind,method=name,
                             **tone_metrics(d,output[:,0],FS,freq)))
    write_csv(REPORT/"phase/tones.csv",rows)

def run_mechanism():
    """Compare weights against the same method on the same proxy without task motion."""
    rows=[]
    for subject in SUBJECTS:
        for axis in range(3):
            scale=calibration_band_rms(f"{subject}_1",axis,FS)
            pure_cache={}
            for scenario in ("fast_transition","spectral_overlap"):
                signal,_=recorded(subject,20.,axis,scenario)
                for name in CORE:
                    method=estimator(name,FS,scale)
                    output,diag,_=run_stream(method,signal.mixed_observation_deg_s)
                    w=np.stack([item["adaptive_weights"][0,:-1] for item in diag])
                    if name not in pure_cache:
                        _,pure_diag,_=run_stream(estimator(name,FS,scale),signal.tremor_true_deg_s)
                        pure_cache[name]=np.stack([item["adaptive_weights"][0,:-1] for item in pure_diag])
                    pure=pure_cache[name]
                    diff=np.linalg.norm(w-pure,axis=1)
                    t=signal.time_s
                    pre=diff[t<3];active=diff[(t>=3)&(t<5)];post=diff[t>=5]
                    pre_mean=float(np.mean(pre));peak=float(np.max(active));increment=max(peak-pre_mean,0.)
                    threshold=pre_mean+.2*increment
                    width=round(.5*FS)
                    recovery=None
                    for k in range(np.searchsorted(t,5.),len(diff)-width+1):
                        if np.all(diff[k:k+width]<=threshold):
                            recovery=float(t[k]-5.)
                            break
                    rows.append(dict(subject=subject,axis="XYZ"[axis],start_s=20.,scenario=scenario,
                                     method=name,pre_difference_mean=pre_mean,
                                     transition_difference_peak=peak,peak_excess_over_pre=increment,
                                     recovery_difference_mean=float(np.mean(post)),
                                     recovery_time_s=recovery,
                                     tremor_rmse_deg_s=rms(output[:,0]-signal.tremor_true_deg_s[:,0])))
                    if subject=="p02" and axis==0:
                        out=RESULT/f"mechanism/p02_X_{scenario}_{name}.npz"
                        out.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(out,time_s=t,weight_difference_norm=diff,
                                            mixed_weight_norm=np.linalg.norm(w,axis=1),
                                            pure_weight_norm=np.linalg.norm(pure,axis=1),
                                            mixed_weights=w,pure_weights=pure)
            print("mechanism",subject,"XYZ"[axis],flush=True)
    write_csv(REPORT/"mechanism/weight_contamination.csv",rows)

def summarize():
    core=list(csv.DictReader((REPORT/"recorded/core_trials.csv").open()))
    bpf=list(csv.DictReader((REPORT/"recorded/bpf_secondary_trials.csv").open()))
    by_subject=[];interaction=[]
    for scenario in SCENARIOS:
        for subject in SUBJECTS:
            method_values={}
            for method in METHODS:
                part=[r for r in core+bpf if r["scenario"]==scenario and r["subject"]==subject and r["method"]==method]
                record=dict(subject=subject,scenario=scenario,method=method,n_axis_windows=len(part))
                for metric in ("tremor_nrmse","transition_rmse_deg_s","recovery_rmse_deg_s",
                               "weight_peak_increment","weight_post_residual",
                               "residual_transition_rms_deg_s","voluntary_transmission",
                               "tremor_transmission","mean_update_us"):
                    record[metric+"_median"]=float(np.median([float(r[metric]) for r in part]))
                by_subject.append(record)
                method_values[method]=record["tremor_nrmse_median"]
            v=method_values
            interaction.append(dict(subject=subject,scenario=scenario,
                prefilter_effect_with_lms=v["hpf2_lms"]-v["raw_lms"],
                huber_effect_without_filter=v["raw_huber"]-v["raw_lms"],
                huber_effect_after_filter=v["hpf2_huber"]-v["hpf2_lms"],
                combined_minus_raw_huber=v["hpf2_huber"]-v["raw_huber"],
                difference_in_differences=(v["hpf2_huber"]-v["hpf2_lms"])
                                        -(v["raw_huber"]-v["raw_lms"])))
    write_csv(REPORT/"recorded/by_subject_summary.csv",by_subject)
    write_csv(REPORT/"recorded/paired_factor_effects.csv",interaction)
    ratios=list(csv.DictReader((REPORT/"recorded/amplitude_ratio.csv").open()))
    ratio_summary=[]
    for scenario in ("fast_transition","spectral_overlap"):
        for rho in RHO:
            for method in CORE:
                per_subject=[]
                for subject in SUBJECTS:
                    part=[r for r in ratios if r["scenario"]==scenario and r["method"]==method
                          and float(r["rho"])==rho and r["subject"]==subject]
                    per_subject.append(float(np.median([float(r["tremor_nrmse"]) for r in part])))
                ratio_summary.append(dict(scenario=scenario,rho=rho,method=method,n_subjects=5,
                    subject_median_tremor_nrmse=float(np.median(per_subject)),
                    min_subject_median=min(per_subject),max_subject_median=max(per_subject)))
    write_csv(REPORT/"recorded/ratio_subject_summary.csv",ratio_summary)

def plots():
    response=list(csv.DictReader((REPORT/"filter_response/curves.csv").open()))
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
    for label in sorted(set(r["filter"] for r in response)):
        a=[r for r in response if r["filter"]==label]
        f=[float(r["frequency_hz"]) for r in a]
        axes[0].plot(f,[float(r["magnitude"]) for r in a],label=label)
        axes[1].plot(f,np.unwrap([float(r["phase_rad"]) for r in a]),label=label)
    axes[0].axvspan(3,12,color="gray",alpha=.1);axes[0].set_ylabel("magnitude")
    axes[1].set_ylabel("phase (rad)");axes[1].set_xlabel("Hz");axes[0].legend()
    fig.savefig(REPORT/"filter_response/response.png",dpi=150);plt.close(fig)
    rows=list(csv.DictReader((REPORT/"recorded/core_trials.csv").open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,scenario in zip(axes.flat,SCENARIOS):
        for name in CORE:
            x=[float(r["tremor_nrmse"]) for r in rows if r["scenario"]==scenario and r["method"]==name]
            ax.boxplot(x,positions=[CORE.index(name)+1],widths=.6)
        ax.set_xticks(range(1,5),CORE,rotation=25,fontsize=7)
        ax.set(title=scenario,ylabel="tremor nRMSE")
    fig.savefig(REPORT/"recorded/core_scenarios.png",dpi=150);plt.close(fig)
    ratios=list(csv.DictReader((REPORT/"recorded/amplitude_ratio.csv").open()))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,scenario in zip(axes,("fast_transition","spectral_overlap")):
        for name in CORE:
            values=[]
            for rho in RHO:
                part=[float(r["tremor_nrmse"]) for r in ratios if r["scenario"]==scenario and r["method"]==name and float(r["rho"])==rho]
                values.append(np.median(part))
            ax.plot(RHO,values,marker="o",label=name)
        ax.set(xlabel="rho",ylabel="median tremor nRMSE",title=scenario,xscale="log")
    axes[0].legend(fontsize=7);fig.savefig(REPORT/"recorded/amplitude_ratio.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    for name in CORE:
        with np.load(RESULT/f"traces/p02_X_20_fast_{name}.npz") as d:
            t=d["time_s"]
            axes[0].plot(t,d["estimated_tremor_deg_s"],label=name,alpha=.8)
            axes[1].plot(t,d["fourier_weight_norm"],label=name)
    with np.load(RESULT/"traces/p02_X_20_fast_raw_lms.npz") as d:
        axes[0].plot(t,d["tremor_deg_s"],color="black",lw=1,label="injected proxy")
    axes[0].set_ylabel("tremor estimate (deg/s)");axes[1].set_ylabel("Fourier weight norm")
    axes[1].set_xlabel("s");axes[0].legend(fontsize=7)
    for ax in axes:ax.axvspan(3,5,color="gray",alpha=.15)
    fig.savefig(REPORT/"recorded/representative_fast_transition.png",dpi=150);plt.close(fig)
    mechanism=REPORT/"mechanism/weight_contamination.csv"
    if mechanism.exists():
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
        for ax,scenario in zip(axes,("fast_transition","spectral_overlap")):
            for name in CORE:
                with np.load(RESULT/f"mechanism/p02_X_{scenario}_{name}.npz") as d:
                    ax.plot(d["time_s"],d["weight_difference_norm"],label=name)
            ax.set(title=scenario,ylabel="||weights(mixture)-weights(proxy)||")
            ax.axvspan(3,5,color="gray",alpha=.15)
        axes[0].legend(fontsize=7);axes[-1].set_xlabel("s")
        fig.savefig(REPORT/"mechanism/representative_weight_difference.png",dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=("response","synthetic","hpf_sensitivity","recorded","phase","mechanism","summary","plots","all"),default="all")
    args=parser.parse_args()
    if args.part in ("response","all"):run_response()
    if args.part in ("synthetic","all"):run_synthetic()
    if args.part in ("hpf_sensitivity","all"):run_hpf_sensitivity()
    if args.part in ("recorded","all"):run_recorded()
    if args.part in ("phase","all"):run_phase()
    if args.part in ("mechanism","all"):run_mechanism()
    if args.part in ("summary","all"):summarize()
    if args.part in ("plots","all"):plots()

if __name__=="__main__":main()
