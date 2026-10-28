"""Tables and common plots for the corrected physical scoop comparison."""
import csv
import json
from pathlib import Path
import numpy as np
from .system_benchmark import ROOT

OUT=ROOT/'experiments/results/system_benchmark/corrected_algorithm_comparison'
METHODS=('NoComp','FixedBPF','ABPF','OriginalBMFLC','HuberBMFLC','DampedBMFLC','Oracle')


def export_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def summarize(alpha,output_root=OUT):
    root=Path(output_root)/f'alpha{alpha:g}'
    runs=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('*/*.json'))]
    cfg=json.loads((root/'frozen_protocol.json').read_text())
    if any(r['fingerprint']!=cfg['fingerprint'] for r in runs):
        raise RuntimeError('Mixed protocol fingerprints')
    rows=[]
    by={(r['method'],r['seed']):r for r in runs}
    for r in runs:
        t=r['task'];base=by.get(('NoComp',r['seed']))
        row=dict(method=r['method'],seed=r['seed'],alpha=alpha)
        row.update({k:v for k,v in t.items() if not isinstance(v,list)})
        row.update(r['signal'])
        for axis in ('roll','pitch'):
            k=f'output_{axis}_bandpower'
            row[f'{axis}_attenuation_db']=float(10*np.log10(max(t[k],1e-30)/max(base['task'][k],1e-30))) if base else None
        rows.append(row)
    export_csv(root/'per_run_metrics.csv',rows)
    aggregates=[]
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        if not group:continue
        a=dict(method=method,n=len(group))
        for k in group[0]:
            if k in ('method','seed','alpha'):continue
            v=np.array([r[k] if r[k] is not None else np.nan for r in group],float)
            v=v[np.isfinite(v)]
            for suffix,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1) if len(x)>1 else 0.),('min',np.min),('max',np.max)]:
                a[k+'_'+suffix]=float(fn(v)) if len(v) else None
        aggregates.append(a)
    export_csv(root/'aggregate_metrics.csv',aggregates)
    paired=[]
    for first,second in [('HuberBMFLC','OriginalBMFLC'),('ABPF','HuberBMFLC'),('DampedBMFLC','HuberBMFLC')]:
        for s in sorted(set(r['seed'] for r in runs)):
            if (first,s) not in by or (second,s) not in by:continue
            x=next(r for r in rows if r['method']==first and r['seed']==s)
            y=next(r for r in rows if r['method']==second and r['seed']==s)
            a=dict(first=first,second=second,seed=s)
            for k in ('pickup_count','transferred_pickup_count','transport_retention_ratio','transport_pitch_rmse_deg','torque_integral_Nm2_s','estimator_nrmse_mean'):
                a[k+'_difference']=x[k]-y[k] if x[k] is not None and y[k] is not None else None
            paired.append(a)
    export_csv(root/'paired_comparisons.csv',paired)
    phase_diagnostics(root, runs, cfg)
    plot_results(root,rows,runs)
    return aggregates


def phase_diagnostics(root, runs, cfg):
    from .system_benchmark_phase2a import estimator_signal_metrics
    phases=[('initial_hold',0.,1.05),('approach',1.05,4.55),
            ('scoop_and_level',4.55,8.05),('lift',8.05,11.2),
            ('transport',11.2,15.75),('intentional_unload',15.75,21.)]
    rows=[]
    for run in runs:
        with np.load(ROOT/run['trace'],allow_pickle=False) as d:
            S=d['sensor'];L=d['controller']
            C={str(k):i for i,k in enumerate(d['controller_columns'])}
            for name,start,end in phases:
                keep=(S[:,0]>=start)&(S[:,0]<end)
                a=S[keep]
                row=dict(method=run['method'],seed=run['seed'],phase=name,start_s=start,end_s=end)
                row.update(estimator_signal_metrics(a[:,3:5],a[:,5:7],cfg['sampling']['estimator_hz']))
                for j,axis in enumerate(('roll','pitch')):
                    if np.isfinite(a[:,9+j]).any():
                        row[axis+'_weight_norm_mean']=float(np.nanmean(a[:,9+j]))
                        row[axis+'_contamination_mean']=float(np.nanmean(a[:,11+j]))
                        row[axis+'_contamination_peak']=float(np.nanmax(a[:,11+j]))
                b=L[(L[:,0]>=start)&(L[:,0]<end)]
                row['integrated_estimate_roll_final_deg']=float(np.rad2deg(b[-1,C['estimate_angle_roll']]))
                row['integrated_estimate_pitch_final_deg']=float(np.rad2deg(b[-1,C['estimate_angle_pitch']]))
                rows.append(row)
    export_csv(root/'phase_diagnostics.csv',rows)


def plot_results(root,rows,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=root/'plots';plots.mkdir(exist_ok=True)
    methods=[m for m in METHODS if any(r['method']==m for r in rows)]
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,key,title in zip(axes,('pickup_count','transferred_pickup_count','transport_retention_ratio'),('Picked grains','Delivered grains','Retention of picked grains')):
        for seed in sorted(set(r['seed'] for r in rows)):
            rs={r['method']:r for r in rows if r['seed']==seed}
            ax.plot(range(len(methods)),[rs[m][key] if m in rs else np.nan for m in methods],alpha=.4,marker='o')
        ax.set_xticks(range(len(methods)),methods,rotation=45,ha='right');ax.set_title(title);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(plots/'paired_particle_outcomes.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,keys,title in zip(axes,[['transport_roll_rmse_deg','transport_pitch_rmse_deg'],['roll_attenuation_db','pitch_attenuation_db'],['torque_integral_Nm2_s']],['Carry orientation RMSE (deg)','3-12 Hz output attenuation (dB)','Torque squared integral (Nm²s)']):
        for j,k in enumerate(keys):
            groups=[[r[k] for r in rows if r['method']==m and r[k] is not None] for m in methods]
            mean=[np.mean(g) if g else np.nan for g in groups];sd=[np.std(g,ddof=1) if len(g)>1 else 0 for g in groups]
            ax.errorbar(np.arange(len(methods))+(j-.5)*.1,mean,yerr=sd,fmt='o',label=k.split('_')[0])
        ax.set_xticks(range(len(methods)),methods,rotation=45,ha='right');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(plots/'stability_and_effort.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m in methods:
        rs=[r for r in rows if r['method']==m]
        for ax,k in zip(axes,('estimator_nrmse_mean','estimator_abs_phase_error_deg_mean')):
            ax.scatter([r[k] for r in rs],[r['transferred_pickup_count'] for r in rs],s=22,label=m)
            ax.set_xlabel(k);ax.set_ylabel('Delivered grains');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(plots/'signal_vs_delivery.png',dpi=160);plt.close(fig)
    baselines=[r for r in rows if r['method']=='NoComp']
    if not baselines:return
    med=np.median([r['transferred_pickup_count'] for r in baselines])
    seed=min(baselines,key=lambda r:(abs(r['transferred_pickup_count']-med),r['seed']))['seed']
    (root/'representative_seed.json').write_text(json.dumps(dict(seed=seed,rule='NoComp delivered count nearest median; ties lowest seed'),indent=2)+'\n')
    spoon_position_diagnostic(root, runs, seed)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    desired_plotted=False
    for r in runs:
        if r['seed']!=seed:continue
        with np.load(ROOT/r['trace']) as d:
            c={str(k):i for i,k in enumerate(d['controller_columns'])};L=d['controller'];S=d['sensor']
            if not desired_plotted:
                axes[0].plot(L[:,0],np.rad2deg(L[:,c['desired_roll']]),'k--',label='Desired',lw=1)
                axes[1].plot(L[:,0],np.rad2deg(L[:,c['desired_pitch']]),'k--',lw=1)
                desired_plotted=True
            axes[0].plot(L[:,0],np.rad2deg(L[:,c['spoon_roll']]),label=r['method'],lw=.8)
            axes[1].plot(L[:,0],np.rad2deg(L[:,c['spoon_pitch']]),lw=.8)
            axes[2].plot(L[:,0],L[:,c['in_spoon_count']],lw=.8)
            if r['method'] in ('OriginalBMFLC','HuberBMFLC','DampedBMFLC'):
                axes[3].plot(S[:,0],np.linalg.norm(S[:,11:13],axis=1),label=r['method'],lw=.8)
    for ax,title in zip(axes,['Spoon roll (deg)','Spoon pitch (deg)','Grains inside spoon','Fourier weight deviation\n(deg/s)']):
        ax.set_ylabel(title);ax.grid(alpha=.2)
        for t in (11.2,15.75):ax.axvline(t,color='k',ls=':',alpha=.4)
    axes[0].legend(ncol=4,fontsize=8)
    if axes[3].get_legend_handles_labels()[0]:axes[3].legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(plots/'representative_time_series.png',dpi=160);plt.close(fig)


def spoon_position_diagnostic(root, runs, seed):
    """FK of saved joint/root states, independent of particle outcomes."""
    import mujoco
    from .motion import roll_pitch_rotation
    from .system_benchmark import initialize, joint
    from .system_benchmark_scenes import load_scene_model, scene_trajectory, BOWL_X, BOWL_BOTTOM
    selected=[r for r in runs if r['seed']==seed]
    if not selected:return
    first=np.load(ROOT/selected[0]['trace']);count=first['particle_positions'].shape[1];first.close()
    cfg=json.loads((root/'frozen_protocol.json').read_text())
    model,tmp=load_scene_model(count,bowl_depth_m=cfg['plant'].get('bowl_depth_m',.008));Path(tmp).unlink(missing_ok=True)
    data=mujoco.MjData(model);_,pid,_,_=initialize(model,data,seed,particle_count=count)
    qa=[joint(model,name)[0] for name in ('roll','pitch')]
    local=np.array([BOWL_X,0.,BOWL_BOTTOM]);offset=np.array([.073+BOWL_X,0.,BOWL_BOTTOM])
    rows=[]
    for run in selected:
        with np.load(ROOT/run['trace']) as saved:
            if 'replay_time_s' not in saved:continue
            for i,t in enumerate(saved['replay_time_s']):
                data.qpos[:7]=saved['handle_qpos'][i];data.qpos[qa]=saved['joint_qpos'][i]
                mujoco.mj_kinematics(model,data)
                actual=data.xpos[pid]+data.xmat[pid].reshape(3,3)@local
                p,rp,phase=scene_trajectory(float(t),'scoop_transfer')
                desired=p+roll_pitch_rotation(*rp)@offset
                error=actual-desired
                rows.append(dict(method=run['method'],seed=seed,time_s=float(t),phase=phase,
                                 x_error_mm=error[0]*1000,y_error_mm=error[1]*1000,z_error_mm=error[2]*1000,
                                 norm_error_mm=float(np.linalg.norm(error)*1000)))
    export_csv(root/'representative_position_diagnostic.csv',rows)
