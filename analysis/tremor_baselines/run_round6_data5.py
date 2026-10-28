"""Round-6 ProMP protocol on repeated P01 peg-transfer trials, separately saved."""

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
from .round5_motion import MotionKF,prediction_trace
from .round6_promp import CausalProMP,phase_basis,ridge_weights,fit_population,distance_squared
from .run_round6 import write_csv,write_json,rms

RESULT=ROOT/"experiments/results/tremor_baselines/round6_data5"
REPORT=ROOT/"reports/tremor_baselines/round6_data5"
AUDIT=REPORT/"data_audit/trials.csv"
SENSORS=("wrist_rh","tool_rh","wrist_lh","tool_lh")
AXES="XYZ"
K=100
M_CHOICES=(4,6,8)
RIDGES=(.001,.1,1.)
SHRINK=.5
RHOS=(.25,1.,4.)
CONTAMINATION_S=30

def manifest():
    rows=list(csv.DictReader(AUDIT.open()))
    usable=[r for r in rows if r["status"]=="ok" and r["sensor"] in SENSORS]
    groups={}
    for row in usable:
        key=(int(row["session"]),row["timestamp"],row["trial_id"])
        groups.setdefault(key,{})[row["sensor"]]=row
    complete={key:parts for key,parts in groups.items() if all(sensor in parts for sensor in SENSORS)}
    if not complete or not {1,2,3}<={key[0] for key in complete}:
        raise ValueError("missing complete data5 trials")
    return complete

def split():
    trials=manifest()
    development=[key for key in sorted(trials) if key[0] in (1,2)]
    heldout=[key for key in sorted(trials) if key[0]==3]
    payload=dict(participant="P01",development_sessions=[1,2],heldout_sessions=[3],
                 development_trials=[list(k) for k in development],
                 heldout_trials=[list(k) for k in heldout],
                 rule="session chronological split; same participant in both; no cross-subject claim",
                 repeated_archives="expanded session 00/01 directories duplicate ZIPs and are not counted twice")
    path=RESULT/"session_split.json"
    if path.exists() and json.loads(path.read_text())!=payload:
        raise ValueError("frozen data5 split changed")
    write_json(path,payload)
    return trials,development,heldout

def stream(row,axis):
    with np.load(ROOT/row["processed_file"]) as d:
        t=np.asarray(d["time_ms"],dtype=float)
        y=np.asarray(d["gyro_deg_s"],dtype=float)[:,AXES.index(axis)]
    if np.any(np.diff(t)<=0) or len(y)<100 or not np.all(np.isfinite(y)):
        raise ValueError("invalid extracted gyro")
    phase=(t-t[0])/(t[-1]-t[0])
    fs=1000/float(np.median(np.diff(t)))
    return phase,y,fs,float((t[-1]-t[0])/1000)

def demo(row,axis):
    phase,y,_,_=stream(row,axis)
    return np.interp(np.linspace(0,1,K),phase,y)

def develop():
    trials,dev,_=split();grid=[];frozen=[];weights_out=[]
    for sensor in SENSORS:
        for axis in AXES:
            curves=np.stack([demo(trials[key][sensor],axis) for key in dev])
            norms=np.maximum(np.sqrt(np.mean(curves**2,axis=1)),1.)
            options=[]
            for m in M_CHOICES:
                phi=phase_basis(np.linspace(0,1,K),m)
                for ridge in RIDGES:
                    weights=np.stack([ridge_weights(y,phi,ridge) for y in curves])
                    fitted=weights@phi.T
                    loo=((weights.sum(axis=0)[None,:]-weights)/(len(dev)-1))@phi.T
                    rep=np.sqrt(np.mean((fitted-curves)**2,axis=1))/norms
                    pop=np.sqrt(np.mean((loo-curves)**2,axis=1))/norms
                    score=float(np.median(.5*rep+.5*pop))
                    row=dict(sensor=sensor,axis=axis,basis_count=m,ridge=ridge,
                             development_fitted_nrmse_median=float(np.median(rep)),
                             development_loo_nrmse_median=float(np.median(pop)),score=score)
                    grid.append(row);options.append(row)
            best=min(options,key=lambda r:(r["score"],r["basis_count"],r["ridge"]))
            m=best["basis_count"];ridge=best["ridge"]
            phi=phase_basis(np.linspace(0,1,K),m)
            weights=np.stack([ridge_weights(y,phi,ridge) for y in curves])
            mean,cov,obs_var=fit_population(weights,[y-phi@w for y,w in zip(curves,weights)],SHRINK)
            model_dir=RESULT/"models";model_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(model_dir/f"{sensor}_{axis}.npz",basis_count=m,ridge=ridge,
                                mean=mean,covariance=cov,observation_variance=obs_var,
                                development_weights=weights)
            frozen.append(dict(**best,shrinkage=SHRINK,
                               observation_sd_deg_s=float(np.sqrt(obs_var)),
                               n_demonstrations=len(dev),basis_width=1/(m-1)))
            for key,w in zip(dev,weights):
                weights_out.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],
                                        sensor=sensor,axis=axis,
                                        mahalanobis_squared=distance_squared(w,mean,cov),
                                        **{f"w{i}":float(w[i]) for i in range(m)}))
    write_csv(REPORT/"development/choice_grid.csv",grid)
    write_csv(REPORT/"development/frozen_models.csv",frozen)
    write_csv(REPORT/"weights/development_weights.csv",weights_out)
    write_json(RESULT/"frozen_method.json",dict(dataset="data5 P01 FLS Task A",development_sessions=[1,2],
               heldout_sessions=[3],basis_counts=M_CHOICES,ridge_grid=RIDGES,phase_grid_size=K,
               basis_width="1/(M-1)",covariance_shrinkage=SHRINK,
               selection="per sensor/axis: median development trial of half fitted normalized RMSE plus half leave-one-trial-out population normalized RMSE",
               phase="full-trial oracle normalized phase; known trial end",
               no_heldout_selection=True))
    print("frozen",len(frozen),"data5 models from",len(dev),"development trials",flush=True)

def model(sensor,axis):
    with np.load(RESULT/f"models/{sensor}_{axis}.npz") as d:
        return int(d["basis_count"]),float(d["ridge"]),np.array(d["mean"]),np.array(d["covariance"]),float(d["observation_variance"])

def causal_predictions(y,phi,mean,cov,obs_var,horizons):
    est=CausalProMP(mean,cov,obs_var);n=len(y)
    future={h:np.full(n,np.nan) for h in horizons}
    nll=np.empty(n);z=np.empty(n);current=np.empty(n)
    remaining={}
    landmarks={round(frac*(n-1)):frac for frac in (.2,.4,.6)}
    for k,value in enumerate(y):
        current[k],_,z[k],nll[k]=est.update(phi[k],float(value))
        for h in horizons:
            if k+h<n:future[h][k+h]=phi[k+h]@est.mean
        if k in landmarks:remaining[landmarks[k]]=est.mean.copy()
    return current,nll,z,future,remaining

def clean():
    trials,_,held=split()
    rw=json.loads((ROOT/"experiments/results/tremor_baselines/round5/phase_a/frozen_voluntary_prior.json").read_text())["selected_parameters"]
    if rw["model"]!="random_walk":raise ValueError("Round-5 RW changed")
    rows=[];weight_rows=[]
    for key in held:
        for sensor in SENSORS:
            for axis in AXES:
                phase,y,fs,duration=stream(trials[key][sensor],axis)
                m,ridge,mean,cov,obs_var=model(sensor,axis)
                phi=phase_basis(phase,m)
                fitted_weight=ridge_weights(demo(trials[key][sensor],axis),
                                            phase_basis(np.linspace(0,1,K),m),ridge)
                horizons=(round(.1*fs),round(.24*fs))
                current,nll,z,future,remaining=causal_predictions(y,phi,mean,cov,obs_var,horizons)
                # Preserve RW process variance per physical second when Fs changes.
                rw_scale=rw["process_scale"]*np.sqrt(50/fs)
                method=MotionKF(model="random_walk",fs=fs,process_scale=rw_scale,
                                measurement_sd_deg_s=rw["measurement_sd_deg_s"])
                _,_,rw_future,_=prediction_trace(y,method,horizons)
                warmup=round(fs)
                row=dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,axis=axis,
                         fs_hz=fs,duration_s=duration,samples=len(y),
                         basis_count=m,ridge=ridge,signal_rms_deg_s=rms(y),
                         population_rmse_deg_s=rms(phi@mean-y),
                         fitted_rmse_deg_s=rms(phi@fitted_weight-y),
                         mahalanobis_squared=distance_squared(fitted_weight,mean,cov),
                         clean_nll_mean=float(np.mean(nll[warmup:])),
                         clean_abs_z_mean=float(np.mean(abs(z[warmup:]))))
                for h,label in zip(horizons,("100ms","240ms")):
                    row[f"promp_{label}_rmse_deg_s"]=rms(future[h][warmup:]-y[warmup:])
                    row[f"rw_{label}_rmse_deg_s"]=rms(rw_future[h][warmup:]-y[warmup:])
                for frac in (.2,.4,.6):
                    k=round(frac*(len(y)-1))
                    row[f"remaining_{round(frac*100)}_rmse_deg_s"]=rms(phi[k+1:]@remaining[frac]-y[k+1:])
                rows.append(row)
                weight_rows.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,axis=axis,
                                        mahalanobis_squared=row["mahalanobis_squared"],
                                        **{f"w{i}":float(fitted_weight[i]) for i in range(m)}))
                if key==held[0] and axis=="X":
                    out=RESULT/f"representative/clean_{sensor}_X.npz";out.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out,phase=phase,recorded_deg_s=y,
                                        population_deg_s=phi@mean,fitted_deg_s=phi@fitted_weight,
                                        causal_prediction_deg_s=current,nll=nll,z=z,
                                        promp_100ms=future[horizons[0]],rw_100ms=rw_future[horizons[0]])
        print("clean trial",key,flush=True)
    write_csv(REPORT/"clean/heldout_trials.csv",rows)
    write_csv(REPORT/"weights/heldout_weights.csv",weight_rows)
    by_trial=[]
    for key in held:
        for sensor in SENSORS:
            part=[r for r in rows if (r["session"],r["timestamp"],r["trial_id"])==key and r["sensor"]==sensor]
            by_trial.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,
                 **{metric+"_axis_median":float(np.median([r[metric] for r in part])) for metric in
                    ("population_rmse_deg_s","fitted_rmse_deg_s","signal_rms_deg_s",
                     "promp_100ms_rmse_deg_s","rw_100ms_rmse_deg_s",
                     "promp_240ms_rmse_deg_s","rw_240ms_rmse_deg_s")}))
    write_csv(REPORT/"clean/by_trial_summary.csv",by_trial)
    print("clean evaluated",len(rows),"sensor-axis trials",flush=True)

def proxy(pinheiro,sensor,axis,fs):
    family=sensor.split("_")[0]
    with np.load(RESULT/f"proxy/{pinheiro}_{family}_{axis}.npz") as d:
        if abs(float(d["fs_hz"])-fs)>.1:raise ValueError("proxy/sample rate mismatch")
        return np.asarray(d["proxy_deg_s"],dtype=float)

def segment_trace(y,phi,mean,cov,obs_var,stop):
    """Causal prefix through end of fixed injection segment; no future gyro."""
    est=CausalProMP(mean,cov,obs_var)
    nll=np.empty(stop);z=np.empty(stop)
    for k in range(stop):
        _,_,z[k],nll[k]=est.update(phi[k],float(y[k]))
    return nll,z

def discrimination():
    trials,_,held=split()
    audit=list(csv.DictReader((REPORT/"data_audit/proxy_resampling.csv").open()))
    if len(audit)!=30 or max(abs(float(r["band_power_relative_change"])) for r in audit)>.1:
        raise ValueError("proxy audit failed")
    pairing={key:f"p{(i%5)+1:02d}" for i,key in enumerate(held)}
    rows=[];weight_rows=[]
    for key in held:
        for sensor in SENSORS:
            for axis in AXES:
                phase,v,fs,duration=stream(trials[key][sensor],axis)
                n=round(CONTAMINATION_S*fs)
                if n>len(v):raise ValueError("held-out trial shorter than contamination window")
                start=(len(v)-n)//2;stop=start+n
                m,ridge,mean,cov,obs_var=model(sensor,axis)
                phi=phase_basis(phase,m)
                phi_grid=phase_basis(np.linspace(0,1,K),m)
                clean_grid=np.interp(np.linspace(0,1,K),phase,v)
                clean_weight=ridge_weights(clean_grid,phi_grid,ridge)
                clean_distance=distance_squared(clean_weight,mean,cov)
                clean_nll,clean_z=segment_trace(v,phi,mean,cov,obs_var,stop)
                source=proxy(pairing[key],sensor,axis,fs)[:n]
                for rho in RHOS:
                    d=source*rms(v[start:stop])/max(rho*rms(source),1e-9)
                    mixed=v.copy();mixed[start:stop]+=d
                    cont_nll,cont_z=segment_trace(mixed,phi,mean,cov,obs_var,stop)
                    weight=ridge_weights(np.interp(np.linspace(0,1,K),phase,mixed),phi_grid,ridge)
                    dist=distance_squared(weight,mean,cov)
                    window=slice(start,stop)
                    row=dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,axis=axis,
                             pinheiro_subject=pairing[key],fs_hz=fs,duration_s=duration,
                             injection_start_s=start/fs,injection_duration_s=n/fs,
                             target_rho=rho,actual_rho=rms(v[window])/rms(d),
                             clean_nll_mean=float(np.mean(clean_nll[window])),
                             contaminated_nll_mean=float(np.mean(cont_nll[window])),
                             clean_abs_z_mean=float(np.mean(abs(clean_z[window]))),
                             contaminated_abs_z_mean=float(np.mean(abs(cont_z[window]))),
                             clean_mahalanobis_squared=clean_distance,
                             contaminated_mahalanobis_squared=dist)
                    rows.append(row)
                    weight_rows.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,axis=axis,
                                            target_rho=rho,clean_mahalanobis_squared=clean_distance,
                                            contaminated_mahalanobis_squared=dist,
                                            **{f"clean_w{i}":float(clean_weight[i]) for i in range(m)},
                                            **{f"contaminated_w{i}":float(weight[i]) for i in range(m)}))
                    if key==held[0] and sensor in ("wrist_rh","tool_rh") and axis=="X" and rho==1:
                        out=RESULT/f"representative/contamination_{sensor}_X.npz"
                        np.savez_compressed(out,phase=phase[:stop],clean_deg_s=v[:stop],mixed_deg_s=mixed[:stop],
                                            injection_start_sample=start,clean_nll=clean_nll,contaminated_nll=cont_nll,
                                            clean_z=clean_z,contaminated_z=cont_z)
        print("contamination trial",key,flush=True)
    write_csv(REPORT/"discrimination/contamination_trials.csv",rows)
    write_csv(REPORT/"weights/contaminated_weights.csv",weight_rows)
    by_trial=[]
    for key in held:
        for sensor in SENSORS:
            for rho in RHOS:
                part=[r for r in rows if (r["session"],r["timestamp"],r["trial_id"])==key
                      and r["sensor"]==sensor and r["target_rho"]==rho]
                by_trial.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,target_rho=rho,
                      **{metric+"_axis_median":float(np.median([r[metric] for r in part])) for metric in
                         ("clean_nll_mean","contaminated_nll_mean","clean_abs_z_mean",
                          "contaminated_abs_z_mean","clean_mahalanobis_squared",
                          "contaminated_mahalanobis_squared")}))
    write_csv(REPORT/"discrimination/by_trial_summary.csv",by_trial)
    print("contamination evaluated",len(rows),"sensor-axis-rho trials",flush=True)
    return rows

def gate():
    clean=list(csv.DictReader((REPORT/"clean/heldout_trials.csv").open()))
    contam=list(csv.DictReader((REPORT/"discrimination/contamination_trials.csv").open()))
    primary=[r for r in clean if r["sensor"]=="wrist_rh"]
    fit_ratio=float(np.median([float(r["fitted_rmse_deg_s"])/max(float(r["signal_rms_deg_s"]),1e-9) for r in primary]))
    pop_ratio=float(np.median([float(r["population_rmse_deg_s"])/max(float(r["signal_rms_deg_s"]),1e-9) for r in primary]))
    pred100=float(np.mean([float(r["promp_100ms_rmse_deg_s"])<float(r["rw_100ms_rmse_deg_s"]) for r in primary]))
    pred240=float(np.mean([float(r["promp_240ms_rmse_deg_s"])<float(r["rw_240ms_rmse_deg_s"]) for r in primary]))
    d=[r for r in contam if r["sensor"]=="wrist_rh" and float(r["target_rho"])==1]
    nll=float(np.mean([float(r["contaminated_nll_mean"])>float(r["clean_nll_mean"]) for r in d]))
    z=float(np.mean([float(r["contaminated_abs_z_mean"])>float(r["clean_abs_z_mean"]) for r in d]))
    dist=float(np.mean([float(r["contaminated_mahalanobis_squared"])>float(r["clean_mahalanobis_squared"]) for r in d]))
    conditions=dict(representation=bool(fit_ratio<.75 and pop_ratio<1),
                    prediction=bool(pred100>=.6 and pred240>=.6),
                    discrimination=bool(nll>=.8 and z>=.8 and dist>=.7))
    payload=dict(primary="right wrist",n_heldout_trials=len(d)//3,
                 fitted_nrmse_median=fit_ratio,population_nrmse_median=pop_ratio,
                 promp_win_fraction_100ms=pred100,promp_win_fraction_240ms=pred240,
                 nll_increase_fraction_rho1=nll,abs_z_increase_fraction_rho1=z,
                 mahalanobis_increase_fraction_rho1=dist,conditions=conditions,
                 execute_separation=all(conditions.values()),
                 thresholds=dict(fitted_nrmse_lt=.75,population_nrmse_lt=1.,prediction_win_gte=.6,
                                 nll_z_gte=.8,weight_distance_gte=.7),
                 note="Operational gates; thresholds were encoded after clean representation inspection but before reading contamination results. Not a preregistered statistical test. Same-subject session holdout.")
    write_json(RESULT/"gate.json",payload)
    print("gate",conditions,"execute separation",payload["execute_separation"],flush=True)
    return payload["execute_separation"]

def separation():
    status=json.loads((RESULT/"gate.json").read_text())
    if not status["execute_separation"]:
        print("separation skipped: representation, prediction, or discrimination condition failed",flush=True)
        return
    # Previous Round-6 protocol permits this only after the three gates pass.
    from .round3_variants import RobustBMFLC
    from .run_round4 import calibration_band_rms
    trials,_,held=split();round4=json.loads((ROOT/"experiments/results/tremor_baselines/round4/development/frozen_parameters.json").read_text())
    pairing={key:f"p{(i%5)+1:02d}" for i,key in enumerate(held)}
    rows=[]
    for key in held:
        for sensor in ("tool_rh","wrist_rh"):
            for axis_index,axis in enumerate(AXES):
                phase,v,fs,duration=stream(trials[key][sensor],axis)
                n=round(CONTAMINATION_S*fs);start=(len(v)-n)//2;stop=start+n
                m,_,mean,cov,obs_var=model(sensor,axis);phi=phase_basis(phase,m)
                source=proxy(pairing[key],sensor,axis,fs)[:n]
                scale=calibration_band_rms(f"{pairing[key]}_1",axis_index,148.14814814814815)
                for rho in RHOS:
                    d=source*rms(v[start:stop])/max(rho*rms(source),1e-9)
                    mixed=v.copy();mixed[start:stop]+=d
                    estimator=CausalProMP(mean,cov,obs_var)
                    predicted=np.empty(stop)
                    for k in range(stop):predicted[k]=estimator.update(phi[k],float(mixed[k]))[0]
                    promp_d=mixed[start:stop]-predicted[start:stop]
                    huber=RobustBMFLC(sample_rate_hz=fs,low_hz=3,high_hz=12,spacing_hz=.1,
                                      lms_gain=.01,variant="huber",huber_delta=round4["huber_factor"]*scale)
                    huber_d=np.empty(stop)
                    for k in range(stop):huber_d[k]=huber.step(np.array([mixed[k]]))[0][0]
                    for name,dhat,vhat in (("causal_promp_residual",promp_d,predicted[start:stop]),
                                            ("huber_bmflc",huber_d[start:stop],mixed[start:stop]-huber_d[start:stop])):
                        task=v[start:stop]
                        rows.append(dict(session=key[0],timestamp=key[1],trial_id=key[2],sensor=sensor,axis=axis,
                                         rho=rho,method=name,tremor_nrmse=rms(dhat-d)/rms(d),
                                         task_nrmse=rms(vhat-task)/rms(task),
                                         abs_l_v_to_d=abs(float(np.dot(dhat-d,task)/np.dot(task,task))),
                                         abs_l_d_to_v=abs(float(np.dot(vhat-task,d)/np.dot(d,d)))))
    write_csv(REPORT/"separation/trials.csv",rows)

def plots():
    clean=list(csv.DictReader((REPORT/"clean/heldout_trials.csv").open()))
    contam=list(csv.DictReader((REPORT/"discrimination/contamination_trials.csv").open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,sensor in zip(axes.flat,SENSORS):
        part=[r for r in clean if r["sensor"]==sensor]
        for h in ("100ms","240ms"):
            ax.scatter([float(r[f"rw_{h}_rmse_deg_s"]) for r in part],
                       [float(r[f"promp_{h}_rmse_deg_s"]) for r in part],label=h,alpha=.65)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color="black",lw=1)
        ax.set(title=sensor,xlabel="RW RMSE (deg/s)",ylabel="ProMP RMSE (deg/s)")
    axes[0,0].legend();fig.savefig(REPORT/"clean/prediction_comparison.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,sensor in zip(axes.flat,SENSORS):
        for rho in RHOS:
            part=[r for r in contam if r["sensor"]==sensor and float(r["target_rho"])==rho]
            ax.scatter([float(r["clean_nll_mean"]) for r in part],
                       [float(r["contaminated_nll_mean"]) for r in part],label=f"rho={rho}",alpha=.6)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color="black",lw=1)
        ax.set(title=sensor,xlabel="clean NLL",ylabel="contaminated NLL")
    axes[0,0].legend();fig.savefig(REPORT/"discrimination/nll_comparison.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    for ax,sensor in zip(axes,("wrist_rh","tool_rh")):
        with np.load(RESULT/f"representative/clean_{sensor}_X.npz") as d:
            ax.plot(d["phase"],d["recorded_deg_s"],label="recorded gyro",alpha=.7)
            ax.plot(d["phase"],d["population_deg_s"],label="population")
            ax.plot(d["phase"],d["fitted_deg_s"],label="oracle fitted")
        ax.set(title=sensor,ylabel="deg/s")
    axes[0].legend();axes[-1].set_xlabel("oracle full-trial phase")
    fig.savefig(REPORT/"clean/representative_trajectory.png",dpi=150);plt.close(fig)
    dev=list(csv.DictReader((REPORT/"weights/development_weights.csv").open()))
    held=list(csv.DictReader((REPORT/"weights/heldout_weights.csv").open()))
    fig,axes=plt.subplots(4,3,figsize=(12,12),constrained_layout=True)
    for i,sensor in enumerate(SENSORS):
        for j,axis in enumerate(AXES):
            ax=axes[i,j];m,_,_,_,_=model(sensor,axis)
            train=np.array([[float(r[f"w{k}"]) for k in range(m)] for r in dev if r["sensor"]==sensor and r["axis"]==axis])
            test=np.array([[float(r[f"w{k}"]) for k in range(m)] for r in held if r["sensor"]==sensor and r["axis"]==axis])
            center=train.mean(axis=0);_,_,vh=np.linalg.svd(train-center,full_matrices=False)
            ax.scatter(*((train-center)@vh[:2].T).T,label="session 1-2",alpha=.6)
            ax.scatter(*((test-center)@vh[:2].T).T,label="session 3",marker="x",alpha=.7)
            ax.set_title(f"{sensor} {axis}")
    axes[0,0].legend(fontsize=7)
    fig.savefig(REPORT/"weights/pca_sessions.png",dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=("development","clean","discrimination","separation","plots","all"),default="all")
    args=parser.parse_args()
    if args.part in ("development","all"):develop()
    if args.part in ("clean","all"):clean()
    if args.part in ("discrimination","all"):
        discrimination();gate()
    if args.part in ("separation","all"):separation()
    if args.part in ("plots","all"):plots()

if __name__=="__main__":main()
