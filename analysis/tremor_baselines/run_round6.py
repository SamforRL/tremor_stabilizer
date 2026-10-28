"""Round 6: population ProMP representation, prediction, and contamination.

Activity boundaries and normalized phase are known from the recorded labels.
They are oracle inputs for this representation study, not an online detector.
"""

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
from .round3_variants import RobustBMFLC
from .round6_promp import phase_basis,ridge_weights,fit_population,distance_squared,CausalProMP

RESULT=ROOT/"experiments/results/tremor_baselines/round6"
REPORT=ROOT/"reports/tremor_baselines/round6"
DATA=ROOT/"data/processed"
ROUND5=ROOT/"experiments/results/tremor_baselines/round5"
ACTIVITIES=("spiral","typing","mixing","brushing")
FS=50.0
K=100
M_CHOICES=(4,6,8)
RIDGES=(1e-3,0.1,1.0)
SHRINKAGE=0.5
RHO_LEVELS=(0.25,1.0,4.0)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)))
            w.writeheader();w.writerows(rows)

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")

def load_split():
    original=ROUND5/"round5_subject_split.json"
    if not original.is_file():raise FileNotFoundError(original)
    source=original.read_bytes()
    split=json.loads(source)
    if len(split["development_subjects"])!=15 or len(split["heldout_subjects"])!=5 or \
       set(split["development_subjects"]) & set(split["heldout_subjects"]):
        raise ValueError("Round-5 split is invalid")
    target=RESULT/"round6_subject_split.json"
    if target.exists() and target.read_bytes()!=source:
        raise ValueError("Round-6 split differs from frozen Round-5 split")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(source)
    return split

def trial(subject,activity,axis):
    with np.load(DATA/f"bioclitee_s{subject:02d}_{activity}.npz") as d:
        t=np.asarray(d["time_ms"],dtype=float)
        y=np.asarray(d["gyro_deg_s"],dtype=float)[:,"XYZ".index(axis)]
    if len(t)<50 or np.any(np.diff(t)<=0) or not np.all(np.isfinite(y)):
        raise ValueError("invalid activity gyro trial")
    phase=(t-t[0])/(t[-1]-t[0])
    return phase,y,(t[-1]-t[0])/1000

def phase_samples(subject,activity,axis):
    phase,y,duration=trial(subject,activity,axis)
    grid=np.linspace(0,1,K)
    return np.interp(grid,phase,y),duration

def rms(x):return float(np.sqrt(np.mean(np.square(x))))

def develop():
    split=load_split();dev=split["development_subjects"]
    grids=[];selection=[];weight_rows=[]
    for activity in ACTIVITIES:
        for axis in "XYZ":
            demos=np.stack([phase_samples(s,activity,axis)[0] for s in dev])
            norm=np.maximum(np.sqrt(np.mean(demos*demos,axis=1)),1.0)
            trials=[]
            for m in M_CHOICES:
                phi=phase_basis(np.linspace(0,1,K),m)
                for ridge in RIDGES:
                    weights=np.stack([ridge_weights(y,phi,ridge) for y in demos])
                    fitted=weights@phi.T
                    representation=np.sqrt(np.mean((fitted-demos)**2,axis=1))/norm
                    # Leave-one-subject-out population mean; no held-out people.
                    loo=(weights.sum(axis=0)[None,:]-weights)/(len(dev)-1)
                    population=loo@phi.T
                    pop_error=np.sqrt(np.mean((population-demos)**2,axis=1))/norm
                    score=float(np.median(0.5*representation+0.5*pop_error))
                    row=dict(activity=activity,axis=axis,basis_count=m,ridge=ridge,
                             representation_nrmse_dev_median=float(np.median(representation)),
                             population_loo_nrmse_dev_median=float(np.median(pop_error)),
                             selection_score=score)
                    grids.append(row);trials.append(row)
            best=min(trials,key=lambda row:(row["selection_score"],row["basis_count"],row["ridge"]))
            m=best["basis_count"];ridge=best["ridge"]
            phi=phase_basis(np.linspace(0,1,K),m)
            weights=np.stack([ridge_weights(y,phi,ridge) for y in demos])
            residuals=[y-phi@w for y,w in zip(demos,weights)]
            mean,cov,obs_var=fit_population(weights,residuals,SHRINKAGE)
            model_dir=RESULT/"models";model_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(model_dir/f"{activity}_{axis}.npz",weight_mean=mean,
                                weight_covariance=cov,observation_variance=obs_var,
                                basis_count=m,ridge=ridge,phase_grid=np.linspace(0,1,K),
                                development_weights=weights,development_subjects=dev)
            selection.append(dict(**best,shrinkage=SHRINKAGE,
                                  observation_sd_deg_s=float(np.sqrt(obs_var)),
                                  weight_cov_trace=float(np.trace(cov)),n_demonstrations=len(dev),
                                  basis_width=1/(m-1)))
            for subject,w,y in zip(dev,weights,demos):
                weight_rows.append(dict(subject=subject,role="development",activity=activity,axis=axis,
                                        duration_s=phase_samples(subject,activity,axis)[1],
                                        mahalanobis_squared=distance_squared(w,mean,cov),
                                        representation_rmse_deg_s=rms(phi@w-y),
                                        **{f"w{i}":float(w[i]) for i in range(m)}))
    write_csv(REPORT/"development/choice_grid.csv",grids)
    write_csv(REPORT/"development/frozen_models.csv",selection)
    write_csv(REPORT/"weights/development_weights.csv",weight_rows)
    write_json(RESULT/"frozen_method.json",dict(round5_split_file=str(ROUND5/"round5_subject_split.json"),
               fs_hz=FS,phase_grid_size=K,phase="(t-t0)/(t_end-t0), oracle activity boundary",
               basis_counts=M_CHOICES,ridge_grid=RIDGES,basis_width_rule="h=1/(M-1)",
               covariance_regularization="fixed diagonal shrinkage",shrinkage=SHRINKAGE,
               observation_variance="mean squared development fitted residual on 100-point grid, floor 1e-6",
               selection="per activity/axis; median over development subjects of 0.5 normalized fitted RMSE + 0.5 leave-one-subject-out population RMSE",
               models=selection,heldout_used_for_selection=False))
    print("frozen",len(selection),"activity-axis ProMP models",flush=True)

def model(activity,axis):
    path=RESULT/f"models/{activity}_{axis}.npz"
    with np.load(path) as d:
        return int(d["basis_count"]),float(d["ridge"]),np.array(d["weight_mean"]),np.array(d["weight_covariance"]),float(d["observation_variance"])

def causal_trace(y,phi,mean,cov,obs_var,horizons=(5,12)):
    estimator=CausalProMP(mean,cov,obs_var)
    n=len(y)
    predicted=np.empty(n);variance=np.empty(n);z=np.empty(n);nll=np.empty(n)
    future={h:np.full(n,np.nan) for h in horizons}
    posterior_means={}
    for k,value in enumerate(y):
        predicted[k],variance[k],z[k],nll[k]=estimator.update(phi[k],float(value))
        for h in horizons:
            if k+h<n:future[h][k+h]=phi[k+h]@estimator.mean
        if k in {round(0.2*(n-1)),round(0.4*(n-1)),round(0.6*(n-1))}:
            posterior_means[k]=estimator.mean.copy()
    return predicted,variance,z,nll,future,posterior_means

def evaluate_clean():
    split=load_split();rw=json.loads((ROUND5/"phase_a/frozen_voluntary_prior.json").read_text())["selected_parameters"]
    if rw["model"]!="random_walk":raise ValueError("Round-5 baseline changed")
    rows=[];weights=[]
    for subject in split["heldout_subjects"]:
        for activity in ACTIVITIES:
            for axis in "XYZ":
                phase,y,duration=trial(subject,activity,axis)
                m,ridge,mean,cov,obs_var=model(activity,axis)
                phi=phase_basis(phase,m)
                grid_y,_=phase_samples(subject,activity,axis)
                fit=ridge_weights(grid_y,phase_basis(np.linspace(0,1,K),m),ridge)
                predicted,var,z,nll,future,posterior=causal_trace(y,phi,mean,cov,obs_var)
                rw_method=MotionKF(**rw,fs=FS)
                _,_,rw_future,_=prediction_trace(y,rw_method,(5,12))
                cut=round(FS)
                record=dict(subject=subject,activity=activity,axis=axis,samples=len(y),duration_s=duration,
                            basis_count=m,ridge=ridge,observation_sd_deg_s=float(np.sqrt(obs_var)),
                            population_rmse_deg_s=rms(phi@mean-y),
                            representation_rmse_deg_s=rms(phi@fit-y),
                            signal_rms_deg_s=rms(y),mahalanobis_squared=distance_squared(fit,mean,cov),
                            clean_nll_median=float(np.median(nll[cut:])),
                            clean_abs_z_median=float(np.median(abs(z[cut:]))))
                for h in (5,12):
                    record[f"promp_pred_{h}_rmse_deg_s"]=rms(future[h][cut:]-y[cut:])
                    record[f"rw_pred_{h}_rmse_deg_s"]=rms(rw_future[h][cut:]-y[cut:])
                for fraction in (0.2,0.4,0.6):
                    k=round(fraction*(len(y)-1))
                    record[f"remaining_{int(fraction*100)}_rmse_deg_s"]=rms(phi[k+1:]@posterior[k]-y[k+1:])
                rows.append(record)
                weights.append(dict(subject=subject,role="heldout",activity=activity,axis=axis,
                                    duration_s=duration,mahalanobis_squared=record["mahalanobis_squared"],
                                    representation_rmse_deg_s=record["representation_rmse_deg_s"],
                                    **{f"w{i}":float(fit[i]) for i in range(m)}))
                trace=RESULT/f"clean_traces/s{subject:02d}_{activity}_{axis}.npz"
                trace.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(trace,phase=phase,reference_deg_s=y,population_mean_deg_s=phi@mean,
                                    oracle_fit_deg_s=phi@fit,causal_prediction_deg_s=predicted,
                                    predictive_variance=var,standardized_innovation=z,predictive_nll=nll,
                                    forecast_100ms_deg_s=future[5],forecast_240ms_deg_s=future[12],
                                    rw_100ms_deg_s=rw_future[5],rw_240ms_deg_s=rw_future[12])
    write_csv(REPORT/"clean/heldout_representation_prediction.csv",rows)
    write_csv(REPORT/"weights/heldout_weights.csv",weights)
    print("evaluated",len(rows),"clean heldout activity-axis trials",flush=True)

def proxy(subject,axis,n):
    with np.load(RESULT/f"resampled_proxy/{subject}_{axis}.npz") as d:
        values=np.asarray(d["proxy_deg_s"],dtype=float)
        if float(d["fs_hz"])!=FS:raise ValueError("incorrect proxy sample rate")
    if n>len(values):raise ValueError("full activity exceeds independent proxy")
    return values[:n]

def _pairing():
    split=load_split()
    return {s:f"p{i:02d}" for i,s in enumerate(sorted(split["heldout_subjects"]),1)}

def discrimination():
    split=load_split();pairs=_pairing();rows=[];weight_rows=[]
    audit=list(csv.DictReader((REPORT/"data_audit/proxy_resampling.csv").open()))
    if len(audit)!=15 or any(abs(float(r["band_power_relative_change"]))>.1 for r in audit):
        raise ValueError("Round-6 proxy resampling audit failed")
    for subject in split["heldout_subjects"]:
        pinheiro=pairs[subject]
        for activity in ACTIVITIES:
            for axis in "XYZ":
                phase,v,duration=trial(subject,activity,axis)
                m,ridge,mean,cov,obs_var=model(activity,axis)
                phi=phase_basis(phase,m)
                phi_grid=phase_basis(np.linspace(0,1,K),m)
                clean_grid,_=phase_samples(subject,activity,axis)
                clean_weight=ridge_weights(clean_grid,phi_grid,ridge)
                clean_distance=distance_squared(clean_weight,mean,cov)
                _,_,clean_z,clean_nll,_,_=causal_trace(v,phi,mean,cov,obs_var)
                source=proxy(pinheiro,axis,len(v))
                for rho in RHO_LEVELS:
                    d=source*rms(v)/max(rho*rms(source),1e-9)
                    y=v+d
                    prediction,variance,z,nll,_,_=causal_trace(y,phi,mean,cov,obs_var)
                    contaminated_grid=np.interp(np.linspace(0,1,K),phase,y)
                    contaminated_weight=ridge_weights(contaminated_grid,phi_grid,ridge)
                    contaminated_distance=distance_squared(contaminated_weight,mean,cov)
                    cut=round(FS)
                    row=dict(subject=subject,pinheiro_subject=pinheiro,activity=activity,axis=axis,
                             samples=len(v),duration_s=duration,target_rho=rho,
                             actual_rho=rms(v)/rms(d),task_rms_deg_s=rms(v),proxy_rms_deg_s=rms(d),
                             clean_nll_mean=float(np.mean(clean_nll[cut:])),
                             contaminated_nll_mean=float(np.mean(nll[cut:])),
                             clean_nll_median=float(np.median(clean_nll[cut:])),
                             contaminated_nll_median=float(np.median(nll[cut:])),
                             clean_abs_z_mean=float(np.mean(abs(clean_z[cut:]))),
                             contaminated_abs_z_mean=float(np.mean(abs(z[cut:]))),
                             clean_abs_z_median=float(np.median(abs(clean_z[cut:]))),
                             contaminated_abs_z_median=float(np.median(abs(z[cut:]))),
                             clean_mahalanobis_squared=clean_distance,
                             contaminated_mahalanobis_squared=contaminated_distance)
                    rows.append(row)
                    weight_rows.append(dict(subject=subject,activity=activity,axis=axis,target_rho=rho,
                                            clean_mahalanobis_squared=clean_distance,
                                            contaminated_mahalanobis_squared=contaminated_distance,
                                            **{f"clean_w{i}":float(clean_weight[i]) for i in range(m)},
                                            **{f"contaminated_w{i}":float(contaminated_weight[i]) for i in range(m)}))
                    out=RESULT/f"likelihood/s{subject:02d}_{activity}_{axis}_rho{rho:g}.npz"
                    out.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out,phase=phase,clean_deg_s=v,injected_proxy_deg_s=d,
                                        mixture_deg_s=y,causal_prior_prediction_deg_s=prediction,
                                        predictive_variance=variance,clean_nll=clean_nll,
                                        contaminated_nll=nll,clean_z=clean_z,contaminated_z=z)
    write_csv(REPORT/"discrimination/contamination_trials.csv",rows)
    write_csv(REPORT/"weights/contaminated_weights.csv",weight_rows)
    # Operational gate for the conditional separation experiment: require
    # most spiral trials at rho=1 to increase both NLL and |z|. Weight
    # distance is reported separately and is not part of this gate.
    spiral=[r for r in rows if r["activity"]=="spiral" and r["target_rho"]==1]
    nll_count=sum(r["contaminated_nll_mean"]>r["clean_nll_mean"] for r in spiral)
    z_count=sum(r["contaminated_abs_z_mean"]>r["clean_abs_z_mean"] for r in spiral)
    discrimination_gate=(nll_count>=12 and z_count>=12) # 80% of 5 people x 3 axes
    clean=list(csv.DictReader((REPORT/"clean/heldout_representation_prediction.csv").open()))
    spiral_clean=[r for r in clean if r["activity"]=="spiral"]
    fitted_ratio=float(np.median([float(r["representation_rmse_deg_s"])/
                                  max(float(r["signal_rms_deg_s"]),1e-9) for r in spiral_clean]))
    population_ratio=float(np.median([float(r["population_rmse_deg_s"])/
                                      max(float(r["signal_rms_deg_s"]),1e-9) for r in spiral_clean]))
    # A fitted model that does not beat the zero waveform by a substantial
    # margin is not a useful representation, even if contamination increases
    # raw residual NLL. The zero waveform has nRMSE 1 by this definition.
    representation_gate=fitted_ratio<0.75 and population_ratio<1.0
    gate=discrimination_gate and representation_gate
    write_json(RESULT/"discrimination_gate.json",dict(activity="spiral",rho=1,n_trials=len(spiral),
               nll_increase_count=nll_count,abs_z_increase_count=z_count,
               required_count=12,discrimination_gate=discrimination_gate,
               fitted_nrmse_median=fitted_ratio,population_nrmse_median=population_ratio,
               representation_gate=representation_gate,execute_separation=gate,
               discrimination_rule="at least 80% of 15 spiral subject-axis trials increase both mean NLL and mean abs z",
               representation_rule="spiral median fitted nRMSE < 0.75 and population nRMSE < 1.0; zero waveform nRMSE=1"))
    print("discrimination gate",discrimination_gate,"representation gate",representation_gate,
          "execute separation",gate,flush=True)
    return gate

def separation():
    gate=json.loads((RESULT/"discrimination_gate.json").read_text())
    if not gate["execute_separation"]:
        print("separation skipped: representation or discrimination gate failed",flush=True)
        return
    split=load_split();pairs=_pairing()
    round4=json.loads((ROOT/"experiments/results/tremor_baselines/round4/development/frozen_parameters.json").read_text())
    from .run_round4 import calibration_band_rms
    rows=[]
    for subject in split["heldout_subjects"]:
        pinheiro=pairs[subject]
        for activity in ACTIVITIES:
            for axis_index,axis in enumerate("XYZ"):
                phase,v,duration=trial(subject,activity,axis)
                m,_,mean,cov,obs_var=model(activity,axis)
                phi=phase_basis(phase,m)
                source=proxy(pinheiro,axis,len(v))
                scale=calibration_band_rms(f"{pinheiro}_1",axis_index,148.14814814814815)
                for rho in RHO_LEVELS:
                    d=source*rms(v)/max(rho*rms(source),1e-9)
                    y=v+d
                    prediction,_,_,_,_,_=causal_trace(y,phi,mean,cov,obs_var)
                    promp_d=y-prediction
                    huber=RobustBMFLC(sample_rate_hz=FS,low_hz=3,high_hz=12,spacing_hz=.1,
                                      lms_gain=.01,variant="huber",
                                      huber_delta=round4["huber_factor"]*scale)
                    huber_d=np.empty(len(y))
                    for k,value in enumerate(y):huber_d[k]=huber.step(np.array([value]))[0][0]
                    for name,dhat,vhat in (("causal_promp_residual",promp_d,prediction),
                                            ("huber_bmflc",huber_d,y-huber_d)):
                        result=dict(subject=subject,pinheiro_subject=pinheiro,activity=activity,
                                    axis=axis,target_rho=rho,samples=len(y),duration_s=duration,
                                    task_rms_deg_s=rms(v),proxy_rms_deg_s=rms(d),
                                    method=name,tremor_rmse_deg_s=rms(dhat-d),
                                    tremor_nrmse=rms(dhat-d)/max(rms(d),1e-9),
                                    task_rmse_deg_s=rms(vhat-v),
                                    task_nrmse=rms(vhat-v)/max(rms(v),1e-9),
                                    voluntary_to_tremor_leakage=float(np.dot(dhat-d,v)/np.dot(v,v)),
                                    tremor_to_voluntary_leakage=float(np.dot(vhat-v,d)/np.dot(d,d)))
                        rows.append(result)
                        if subject==split["heldout_subjects"][0] and axis=="X" and rho==1:
                            out=RESULT/f"separation_traces/{activity}_{name}.npz"
                            out.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(out,phase=phase,task_reference_deg_s=v,
                                                injected_proxy_deg_s=d,mixture_deg_s=y,
                                                estimated_task_deg_s=vhat,estimated_proxy_deg_s=dhat)
    write_csv(REPORT/"separation/separation_trials.csv",rows)
    print("separation evaluated",len(rows),"method trials",flush=True)

def plots():
    rows=list(csv.DictReader((REPORT/"clean/heldout_representation_prediction.csv").open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,activity in zip(axes.flat,ACTIVITIES):
        subset=[r for r in rows if r["activity"]==activity]
        ax.scatter([float(r["rw_pred_5_rmse_deg_s"]) for r in subset],
                   [float(r["promp_pred_5_rmse_deg_s"]) for r in subset],label="100 ms",alpha=.7)
        ax.scatter([float(r["rw_pred_12_rmse_deg_s"]) for r in subset],
                   [float(r["promp_pred_12_rmse_deg_s"]) for r in subset],label="240 ms",alpha=.7)
        limit=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,limit],[0,limit],color="black",lw=1)
        ax.set(title=activity,xlabel="RW RMSE (deg/s)",ylabel="ProMP RMSE (deg/s)")
    axes[0,0].legend();fig.savefig(REPORT/"clean/prediction_comparison.png",dpi=150);plt.close(fig)
    contamination=list(csv.DictReader((REPORT/"discrimination/contamination_trials.csv").open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,activity in zip(axes.flat,ACTIVITIES):
        subset=[r for r in contamination if r["activity"]==activity]
        for rho in RHO_LEVELS:
            part=[r for r in subset if float(r["target_rho"])==rho]
            ax.scatter([float(r["clean_nll_mean"]) for r in part],
                       [float(r["contaminated_nll_mean"]) for r in part],label=f"rho={rho}",alpha=.7)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color="black",lw=1)
        ax.set(title=activity,xlabel="clean mean NLL",ylabel="contaminated mean NLL")
    axes[0,0].legend();fig.savefig(REPORT/"discrimination/nll_comparison.png",dpi=150);plt.close(fig)
    # PCA only for display; axes are computed on development weights.
    fig,axes=plt.subplots(3,4,figsize=(13,9),constrained_layout=True)
    development=list(csv.DictReader((REPORT/"weights/development_weights.csv").open()))
    held=list(csv.DictReader((REPORT/"weights/heldout_weights.csv").open()))
    for i,axis in enumerate("XYZ"):
        for j,activity in enumerate(ACTIVITIES):
            ax=axes[i,j];m,_,_,_,_=model(activity,axis)
            dev=np.array([[float(r[f"w{k}"]) for k in range(m)] for r in development if r["activity"]==activity and r["axis"]==axis])
            test=np.array([[float(r[f"w{k}"]) for k in range(m)] for r in held if r["activity"]==activity and r["axis"]==axis])
            center=dev.mean(axis=0);_,_,vh=np.linalg.svd(dev-center,full_matrices=False)
            ax.scatter(*((dev-center)@vh[:2].T).T,label="development")
            ax.scatter(*((test-center)@vh[:2].T).T,label="heldout",marker="x")
            ax.set_title(f"{activity} {axis}");ax.set_xlabel("PC1");ax.set_ylabel("PC2")
    axes[0,0].legend(fontsize=7);fig.savefig(REPORT/"weights/pca_development_heldout.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    subject=load_split()["heldout_subjects"][0]
    for ax,activity in zip(axes.flat,ACTIVITIES):
        with np.load(RESULT/f"clean_traces/s{subject:02d}_{activity}_X.npz") as d:
            phase=d["phase"]
            ax.plot(phase,d["reference_deg_s"],label="recorded task",alpha=.75)
            ax.plot(phase,d["population_mean_deg_s"],label="population mean")
            ax.plot(phase,d["oracle_fit_deg_s"],label="full-trial fit")
        ax.set(title=f"held-out s{subject:02d} {activity} X",xlabel="oracle phase",ylabel="deg/s")
    axes[0,0].legend(fontsize=7)
    fig.savefig(REPORT/"clean/representative_trajectories.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,activity in zip(axes.flat,ACTIVITIES):
        for rho in RHO_LEVELS:
            part=[r for r in contamination if r["activity"]==activity and float(r["target_rho"])==rho]
            ax.scatter([float(r["clean_mahalanobis_squared"]) for r in part],
                       [float(r["contaminated_mahalanobis_squared"]) for r in part],
                       label=f"rho={rho}",alpha=.7)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color="black",lw=1)
        ax.set(title=activity,xlabel="clean Mahalanobis squared",ylabel="contaminated Mahalanobis squared")
    axes[0,0].legend();fig.savefig(REPORT/"discrimination/weight_distance.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    with np.load(RESULT/f"likelihood/s{subject:02d}_spiral_X_rho1.npz") as d:
        phase=d["phase"]
        axes[0].plot(phase,d["clean_deg_s"],label="recorded task")
        axes[0].plot(phase,d["mixture_deg_s"],label="task + proxy",alpha=.7)
        axes[1].plot(phase,d["clean_nll"],label="clean predictive NLL")
        axes[1].plot(phase,d["contaminated_nll"],label="contaminated predictive NLL",alpha=.7)
    axes[0].set_ylabel("deg/s");axes[1].set_ylabel("NLL");axes[1].set_xlabel("oracle phase")
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(REPORT/"discrimination/representative_spiral_likelihood.png",dpi=150);plt.close(fig)
    sep=REPORT/"separation/separation_trials.csv"
    if sep.exists():
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        separation_rows=list(csv.DictReader(sep.open()))
        for ax,activity in zip(axes.flat,ACTIVITIES):
            for method in ("causal_promp_residual","huber_bmflc"):
                part=[r for r in separation_rows if r["activity"]==activity and r["method"]==method]
                x=[rho for rho in RHO_LEVELS]
                y=[float(np.median([float(r["tremor_nrmse"]) for r in part if float(r["target_rho"])==rho])) for rho in x]
                ax.plot(x,y,marker="o",label=method)
            ax.set(title=activity,xlabel="rho",ylabel="tremor nRMSE")
        axes[0,0].legend(fontsize=7);fig.savefig(REPORT/"separation/activity_comparison.png",dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=("development","clean","discrimination","separation","plots","all"),default="all")
    args=parser.parse_args()
    if args.part in ("development","all"):develop()
    if args.part in ("clean","all"):evaluate_clean()
    if args.part in ("discrimination","all"):discrimination()
    if args.part in ("separation","all"):separation()
    if args.part in ("plots","all"):plots()

if __name__=="__main__":main()
