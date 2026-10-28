"""Curated final shallow-spoon metrics, exact seed pairing and common plots."""
import json,shutil
import numpy as np
from .common import ROOT,DATA,REPORT,CONFIG,write_csv,write_json,sha


def prepare_system():
    source=ROOT/CONFIG['system']['source_root']/'alpha5';dest=DATA/'system';dest.mkdir(parents=True,exist_ok=True)
    runs=[json.loads(p.read_text()) for p in sorted((source/'runs').glob('*/*.json'))]
    methods=CONFIG['system']['all_methods'];assert len(runs)==70
    cfg=json.loads((source/'frozen_protocol.json').read_text());by={(r['method'],r['seed']):r for r in runs}
    for m in methods:assert sorted(r['seed'] for r in runs if r['method']==m)==list(range(10))
    assert all(r['fingerprint']==cfg['fingerprint'] for r in runs)
    shutil.copyfile(source/'frozen_protocol.json',dest/'frozen_protocol.json')
    provenance=[];rows=[];particle_rows=[];representative={}
    no=[r for r in runs if r['method']=='NoComp'];median=np.median([r['task']['transferred_pickup_count'] for r in no])
    seed=min(no,key=lambda r:(abs(r['task']['transferred_pickup_count']-median),r['seed']))['seed']
    for r in runs:
        row=dict(method=r['method'],seed=r['seed'],split='development' if r['seed']<3 else 'validation')
        row.update({k:v for k,v in r['task'].items() if not isinstance(v,list)})
        row.update(r['signal']);row['trace']=r['trace'];row['spill_ratio_final']=1-row['transfer_ratio_of_picked']
        base=by['NoComp',r['seed']]['task']
        for axis in ('roll','pitch'):
            k=f'output_{axis}_bandpower';row[axis+'_attenuation_db']=float(10*np.log10(max(row[k],1e-30)/max(base[k],1e-30)))
        with np.load(ROOT/r['trace'],allow_pickle=False) as d:
            L=d['controller'];S=d['sensor'];C={str(k):i for i,k in enumerate(d['controller_columns'])}
            assert np.isfinite(L).all() and np.isfinite(S[:,:9]).all() and np.isfinite(d['particle_positions']).all()
            p=d['particle_positions'][-1];picked=d['picked_mask'];unload=d['unload_mask']
            target=(abs(p[:,0]-.34)<.060)&(abs(p[:,1])<.060)&(p[:,2]>.107)&(p[:,2]<.207)
            assert int((picked&target).sum())==row['transferred_pickup_count']
            row['unload_failure_count']=int((picked&unload&~target).sum())
            for i in np.flatnonzero(picked):
                particle_rows.append(dict(method=r['method'],seed=r['seed'],particle_id=int(i),
                    retained_at_unload=bool(unload[i]),final_in_target=bool(target[i]),
                    transport_loss_at_checkpoint=bool(not unload[i]),
                    unload_failure=bool(unload[i] and not target[i])))
            for label,lo,hi in [('shared_scoop',0.,11.2),('transport',11.2,15.75),('unload',15.75,21.00001)]:
                a=L[(L[:,0]>=lo)&(L[:,0]<hi)]
                row[label+'_sampled_saturation_fraction']=float(a[:,C['saturated']].mean())
                row[label+'_sampled_torque_cost_Nm2_s']=float(np.trapezoid(np.sum(a[:,[C['tau_roll'],C['tau_pitch']]]**2,axis=1),a[:,0]))
            # Saved Damped diagnostics use rad/s; do not compare raw values
            # against the other Fourier methods' deg/s coefficients.
            scale=180/np.pi if r['method']=='DampedBMFLC' else 1.
            if np.isfinite(S[:,11:13]).any():
                v=np.linalg.norm(S[:,11:13]*scale,axis=1)
                row['fourier_contamination_peak_deg_s']=float(v.max())
                row['fourier_contamination_transport_mean_deg_s']=float(v[(S[:,0]>=11.2)&(S[:,0]<15.75)].mean())
            if r['seed']==seed:representative[r['method']]={k:d[k].copy() for k in ('controller','controller_columns','sensor')}
        rows.append(row)
        path=source/'runs'/r['method']/f'seed{r["seed"]:02d}.json'
        provenance.append(dict(method=r['method'],seed=r['seed'],source=str(path.relative_to(ROOT)),sha256=sha(path),trace=r['trace'],trace_sha256=sha(ROOT/r['trace'])))
    # The pre-disturbance physics and picked IDs must match for every method.
    pairing=[]
    for s in range(10):
        with np.load(ROOT/by['NoComp',s]['trace']) as reference:
            keep=reference['replay_time_s']<=11.2+1e-8
            for m in methods:
                with np.load(ROOT/by[m,s]['trace']) as other:
                    for key in ('particle_positions','joint_qpos','handle_qpos'):
                        assert np.array_equal(reference[key][keep],other[key][keep]),(s,m,key)
                    assert np.array_equal(reference['picked_mask'],other['picked_mask'])
        pairing.append(dict(seed=s,all_seven_methods_identical_pre_onset_physics=True))
    zero=ROOT/CONFIG['system']['source_root']/'alpha0/runs/NoComp'
    zr=[json.loads(p.read_text()) for p in zero.glob('*.json')];assert len(zr)==10
    assert all(r['task']['transport_retention_ratio']==1 and r['task']['transfer_ratio_of_picked']==1 for r in zr)
    write_csv(dest/'per_run_metrics.csv',rows);write_csv(dest/'particle_outcomes.csv',particle_rows)
    write_csv(dest/'provenance.csv',provenance)
    aggregates=[]
    for split in ('validation','development','all'):
        for m in methods:
            group=[r for r in rows if r['method']==m and (split=='all' or r['split']==split)]
            a=dict(split=split,method=m,n=len(group))
            for k in group[0]:
                if k in ('method','seed','split','trace'):continue
                values=np.array([r.get(k) if r.get(k) is not None else np.nan for r in group],float)
                values=values[np.isfinite(values)]
                if not len(values):continue
                a.update({k+'_mean':float(values.mean()),k+'_sd':float(values.std(ddof=1)),
                          k+'_min':float(values.min()),k+'_max':float(values.max())})
            aggregates.append(a)
    write_csv(dest/'aggregate_metrics.csv',aggregates)
    paired=[]
    for first,second in [('ABPF','NoComp'),('ABPF','FixedBPF'),('ABPF','HuberBMFLC'),('ABPF','DampedBMFLC'),('HuberBMFLC','OriginalBMFLC')]:
        for s in range(10):
            a=next(r for r in rows if r['method']==first and r['seed']==s)
            b=next(r for r in rows if r['method']==second and r['seed']==s)
            p=dict(first=first,second=second,seed=s,split=a['split'])
            for k in ('transport_retention_ratio','transfer_ratio_of_picked','transport_pitch_rmse_deg','unload_pitch_rmse_deg','torque_integral_Nm2_s'):
                p[k+'_difference']=a[k]-b[k]
            paired.append(p)
    write_csv(dest/'paired_differences.csv',paired)
    write_json(dest/'verification.json',dict(runs=70,baseline_runs=10,pairing=pairing,finite_traces=True,
        no_tremor_all_retained_and_transferred=True,representative_seed=seed,
        representative_rule='NoComp final delivered count nearest median over ten seeds; ties lowest seed',
        warmup='Same NoComp scoop; only transport onwards enables compensation',
        diagnostic_units='Fourier coefficient metrics converted to deg/s; no bias components',
        phase_saturation='100 Hz sampled rate, distinct from 1 kHz whole-run saturation count'))
    plot_system(rows,aggregates,representative,methods)
    return rows,aggregates


def plot_system(rows,aggregates,traces,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=REPORT/'system_figures';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    for ax,key,title in zip(axes,('transport_retention_ratio','transfer_ratio_of_picked'),('Transport retention (%)','Final transfer (%)')):
        for seed in range(10):
            rs={r['method']:r for r in rows if r['seed']==seed}
            ax.plot(range(7),[100*rs[m][key] for m in methods],'-o',alpha=.4,lw=1)
        ax.set_xticks(range(7),methods,rotation=40,ha='right');ax.set_ylim(-3,103);ax.set_title(title);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'paired_retention_transfer.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    keys=[('transport_roll_rmse_deg','transport_pitch_rmse_deg'),('roll_attenuation_db','pitch_attenuation_db'),('transport_sampled_torque_cost_Nm2_s',)]
    titles=('Carry orientation RMSE (deg)','Output 3–12 Hz attenuation (dB)','Transport torque cost (100 Hz log)')
    for ax,ks,title in zip(axes,keys,titles):
        for j,k in enumerate(ks):
            group=[next(r for r in aggregates if r['split']=='validation' and r['method']==m) for m in methods]
            ax.errorbar(np.arange(7)+j*.09,[r[k+'_mean'] for r in group],yerr=[r[k+'_sd'] for r in group],fmt='o',label=k.split('_')[0])
        ax.set_xticks(range(7),methods,rotation=40,ha='right');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'stability_suppression_effort.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
    for m,d in traces.items():
        L=d['controller'];S=d['sensor'];C={str(k):i for i,k in enumerate(d['controller_columns'])}
        axes[0].plot(L[:,0],np.rad2deg(L[:,C['spoon_roll']]),label=m,lw=.8)
        axes[1].plot(L[:,0],np.rad2deg(L[:,C['spoon_pitch']]),lw=.8)
        axes[2].plot(L[:,0],L[:,C['in_spoon_count']],lw=.8)
        if m in ('OriginalBMFLC','HuberBMFLC','DampedBMFLC'):
            scale=180/np.pi if m=='DampedBMFLC' else 1.
            axes[3].plot(S[:,0],np.linalg.norm(S[:,11:13]*scale,axis=1),label=m,lw=.8)
    axes[1].plot(L[:,0],np.rad2deg(L[:,C['desired_pitch']]),'k--',label='Task reference')
    for ax,title in zip(axes,('Roll (deg)','Pitch (deg)','Grains in spoon','Fourier deviation\n(deg/s)')):
        ax.set_ylabel(title);ax.grid(alpha=.2)
        for t in (11.2,15.75):ax.axvline(t,color='k',ls=':',alpha=.5)
    axes[0].legend(ncol=4,fontsize=8);axes[1].legend(fontsize=8);axes[3].legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(folder/'representative_full_task.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for m in methods:
        group=[r for r in rows if r['method']==m]
        for ax,key in zip(axes,('estimator_nrmse_mean','estimator_abs_phase_error_deg_mean')):
            ax.scatter([r[key] for r in group],[100*r['transport_retention_ratio'] for r in group],label=m,s=20)
            ax.set_xlabel(key);ax.set_ylabel('Transport retention (%)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'signal_vs_function.png',dpi=160);plt.close(fig)
