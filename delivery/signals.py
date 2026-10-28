"""Curate existing signal evidence; audit outputs and correct diagnostics only."""
import shutil
import numpy as np
from .common import ROOT,DATA,REPORT,CONFIG,sha,write_csv,write_json,read_csv
from gimbal.estimators import METHODS,EstimatorConfig,create_estimator


def replay(method,x):
    estimator=create_estimator(method,EstimatorConfig.reference())
    y=[];weights=[]
    for sample in x:
        out,d=estimator.step(sample);y.append(out)
        if 'adaptive_weights' in d:weights.append(d['adaptive_weights'])
    return np.asarray(y),np.asarray(weights) if weights else None


def weights_metrics(time,mixed,pure):
    if mixed is None:return {},None
    delta=np.linalg.norm(mixed-pure,axis=(1,2));event=(time>=3)&(time<5);post=time>=5
    pre=float(delta[time<3].mean());peak=float(delta[event].max())
    threshold=pre+.2*(peak-pre);n=round(.5*CONFIG['sample_rate_hz']);recovery=None
    for i in range(np.searchsorted(time,5.),len(time)-n+1):
        if np.all(delta[i:i+n]<=threshold):recovery=float(time[i]-5.);break
    return dict(weight_contamination_peak_deg_s=peak,weight_contamination_post_deg_s=float(delta[post].mean()),
                weight_recovery_s=recovery),delta


def prepare_signals():
    from gimbal.system_benchmark_phase2a import estimator_signal_metrics
    history=ROOT/CONFIG['signal']['source_root'];dest=DATA/'signals';dest.mkdir(parents=True,exist_ok=True)
    for name in ('signal_level_metrics.csv','amplitude_ratio_metrics.csv','frozen_config.json','Damped_BMFLC_reproduction_notes.md'):
        shutil.copyfile(history/name,dest/('historical_'+name))
    rows=[];provenance=[];cache={};audit=[]
    for case in ('steady','fast_transition','amplitude_ratio'):
        for rho in ((.5,1.,2.,4.,8.,16.) if case=='amplitude_ratio' else (1.,)):
            for method in METHODS:
                stem=f'{case}_moderate_{method}_rho{rho:g}'
                # Historical ratio traces exist at rho=4 and 16. Other ratios
                # use the same saved voluntary waveform scaled analytically.
                source_rho=rho if case!='amplitude_ratio' or rho in (4.,16.) else 16.
                source=history/'signal_traces'/f'{case}_moderate_{method}_rho{source_rho:g}.npz'
                with np.load(source,allow_pickle=False) as saved:
                    t=saved['time_s'];d=saved['tremor'];v=saved['voluntary']*(rho/source_rho)
                    x=v+d;out,w=replay(method,x)
                    reused=case!='amplitude_ratio' or rho in (4.,16.)
                    if reused:
                        assert np.array_equal(out,saved['estimate']),f'Frozen output changed: {stem}'
                        out=saved['estimate'].copy()
                    pure,wp=replay(method,d) if w is not None else (None,None)
                    metric=estimator_signal_metrics(d,out,CONFIG['sample_rate_hz'])
                    event=(t>=3)&(t<5);err=out-d
                    ratio=float(np.sqrt(np.mean(v[event]**2))/np.sqrt(np.mean(d[event]**2)))
                    weight,delta=weights_metrics(t,w,wp) if case!='steady' else ({},None)
                    row=dict(case=case,method=method,condition='moderate',nominal_rho=rho if case=='amplitude_ratio' else None,
                             actual_event_rho=ratio,source='p03_1',start_s=20.,duration_s=12.,
                             nrmse=metric['estimator_nrmse_mean'],correlation=metric['estimator_correlation_mean'],
                             amplitude_ratio=metric['estimator_amplitude_ratio_mean'],phase_error_deg=metric['estimator_abs_phase_error_deg_mean'],
                             transition_rmse_deg_s=float(np.sqrt(np.mean(err[event]**2))),
                             recovery_rmse_deg_s=float(np.sqrt(np.mean(err[t>=5]**2))),
                             output_origin='historical_exact_match' if reused else 'same_saved_inputs_ratio_replay',**weight)
                    rows.append(row);audit.append(dict(case=stem,historical_output_checked=reused,finite_output=bool(np.isfinite(out).all())))
                    extra={}
                    if delta is not None:extra['weight_delta_deg_s']=delta
                    if w is not None:
                        extra['weight_norm_deg_s']=np.linalg.norm(w,axis=(1,2))
                        if case=='fast_transition':extra['fourier_weights_deg_s']=w.astype(np.float32)
                    target=dest/'traces'/f'{stem}.npz';target.parent.mkdir(exist_ok=True)
                    np.savez_compressed(target,time_s=t,voluntary=v,tremor=d,mixture=x,estimate=out,**extra)
                    provenance.append(dict(case=stem,source_file=str(source.relative_to(ROOT)),source_sha256=sha(source),
                                           input_scaling=rho/source_rho,curated_trace=str(target.relative_to(ROOT))))
                    cache[(case,method,rho)]=dict(t=t,d=d,v=v,y=out,delta=delta)
    # Check inherited numerical scores wherever historical rows exist.
    old=read_csv(history/'signal_level_metrics.csv')
    oldratio=read_csv(history/'amplitude_ratio_metrics.csv')
    for row in rows:
        source=oldratio if row['case']=='amplitude_ratio' else old
        candidates=[r for r in source if r['condition']=='moderate' and r['method']==row['method']
                    and (row['case']=='amplitude_ratio' and float(r['rho'])==row['nominal_rho']
                         or row['case']!='amplitude_ratio' and r['experiment']==row['case'])]
        assert len(candidates)==1
        historical=candidates[0];key='estimator_nrmse_mean' if row['case']=='steady' else 'full_nrmse'
        assert np.isclose(row['nrmse'],float(historical[key]),rtol=1e-12,atol=1e-12)
    write_csv(dest/'metrics.csv',rows);write_csv(dest/'provenance.csv',provenance)
    write_json(dest/'audit.json',dict(checked_cases=audit,nrmse_agrees_with_historical_tables=True,
        corrections=['Fourier bias removed exactly once; retain 182 components',
                     'Damped native rad/s Fourier weights converted to deg/s before diagnostic comparisons'],
        no_new_data_or_parameter_selection=True,lag_alignment_applied=False))
    plot_signals(cache,rows)
    prepare_mechanism()
    return rows


def welch(x,fs):
    n=min(len(x),round(2*fs));win=np.hanning(n);ps=[]
    for i in range(0,len(x)-n+1,max(1,n//2)):
        a=x[i:i+n];z=np.fft.rfft((a-a.mean())*win);p=abs(z)**2/(fs*(win@win));p[1:-1]*=2;ps.append(p)
    return np.fft.rfftfreq(n,1/fs),np.mean(ps,axis=0)


def plot_signals(cache,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=REPORT/'signal_figures';folder.mkdir(parents=True,exist_ok=True)
    for case in ('steady','fast_transition'):
        fig,axes=plt.subplots(5,2,figsize=(14,12),sharex=True)
        for i,m in enumerate(METHODS):
            a=cache[(case,m,1.)]
            for j in range(2):
                ax=axes[i,j];ax.plot(a['t'],a['d'][:,j],color='k',lw=.8,label='Injected proxy')
                ax.plot(a['t'],a['y'][:,j],lw=.8,label=m);ax.set_ylabel(m+'\n(deg/s)');ax.grid(alpha=.2)
                if case=='fast_transition':ax.axvspan(3,5,color='orange',alpha=.12)
                if i==0:ax.set_title('Gyro '+('X','Y')[j]);ax.legend(fontsize=8)
        axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
        fig.tight_layout();fig.savefig(folder/(case+'_tremor_estimates.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,12),sharex=True)
    for i,m in enumerate(METHODS):
        a=cache[('fast_transition',m,1.)]
        for j in range(2):
            ax=axes[i,j];ax.plot(a['t'],a['v'][:,j],'k',lw=1,label='Designed voluntary')
            ax.plot(a['t'],a['v'][:,j]+a['d'][:,j]-a['y'][:,j],lw=.8,label='Reconstructed voluntary')
            ax.set_ylabel(m+'\n(deg/s)');ax.axvspan(3,5,color='orange',alpha=.12);ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=8);ax.set_title(('Gyro X','Gyro Y')[j])
    axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(folder/'fast_voluntary_reconstruction.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for i,case in enumerate(('steady','fast_transition')):
        for j in range(2):
            ax=axes[i,j];a=cache[(case,METHODS[0],1.)];f,p=welch(a['d'][:,j],CONFIG['sample_rate_hz'])
            ax.semilogy(f,np.maximum(p,1e-12),'k',label='Injected proxy')
            for m in METHODS:
                a=cache[(case,m,1.)];f,p=welch(a['y'][:,j],CONFIG['sample_rate_hz']);ax.semilogy(f,np.maximum(p,1e-12),label=m)
            ax.set_xlim(0,16);ax.set_title(case+' / '+('X','Y')[j]);ax.set_xlabel('Hz');ax.set_ylabel('(deg/s)²/Hz');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'common_welch_psd.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for m in METHODS:
        a=sorted([r for r in rows if r['case']=='amplitude_ratio' and r['method']==m],key=lambda r:r['nominal_rho'])
        axes[0].plot([r['nominal_rho'] for r in a],[r['nrmse'] for r in a],'-o',label=m)
        if 'weight_contamination_peak_deg_s' in a[0]:
            axes[1].plot([r['nominal_rho'] for r in a],[r['weight_contamination_peak_deg_s'] for r in a],'-o',label=m)
    for ax,title in zip(axes,('Tremor nRMSE','Peak Fourier contamination (deg/s)')):
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('rho: RMS(v)/RMS(d), 3–5 s');ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'amplitude_ratio_robustness.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for m in METHODS:
        a=cache[('fast_transition',m,1.)]
        axes[0].plot(a['t'],np.sqrt(np.mean((a['y']-a['d'])**2,axis=1)),label=m,lw=.9)
        if a['delta'] is not None:axes[1].plot(a['t'],a['delta'],label=m,lw=.9)
    for ax in axes:ax.axvspan(3,5,color='orange',alpha=.12);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Instantaneous error RMS (deg/s)');axes[1].set_ylabel('Fourier deviation (deg/s)');axes[1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(folder/'fast_error_and_weight_contamination.png',dpi=160);plt.close(fig)


def prepare_mechanism():
    """Small inherited mechanism appendix; no new simulation or tuning."""
    import matplotlib.pyplot as plt
    dest=DATA/'mechanism';dest.mkdir(exist_ok=True)
    sources=[ROOT/'reports/tremor_baselines/round3/abpf_ablation'/name
             for name in ('controlled_cases.csv','recorded_pairs.csv')]
    sources.append(ROOT/'reports/tremor_baselines/round3/abpf_response/frequency_response.csv')
    provenance=[]
    for p in sources:
        shutil.copyfile(p,dest/p.name)
        provenance.append(dict(source=str(p.relative_to(ROOT)),sha256=sha(p)))
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for j,case in enumerate(('drift','step')):
        for variant in ('fixed','adaptive'):
            p=ROOT/f'experiments/results/tremor_baselines/round3/abpf_ablation/{case}_abpf_{variant}.npz'
            shutil.copyfile(p,dest/p.name);provenance.append(dict(source=str(p.relative_to(ROOT)),sha256=sha(p)))
            with np.load(p) as d:
                t=d['time_s'];axes[0,j].plot(t,d['center_frequency_hz'],label=variant)
                axes[1,j].plot(t,d['estimate_deg_s']-d['true_tremor_deg_s'],label=variant,lw=.8)
                truth=d['true_frequency_hz']
        axes[0,j].plot(t,truth,'k--',label='Known frequency');axes[0,j].set_title(case+' (historical 100 Hz verification)')
        for i in range(2):axes[i,j].legend(fontsize=8);axes[i,j].grid(alpha=.2)
    axes[0,0].set_ylabel('Frequency (Hz)');axes[1,0].set_ylabel('Tremor error (deg/s)')
    for ax in axes[1]:ax.set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(REPORT/'signal_figures/abpf_fixed_adaptive_mechanism.png',dpi=160);plt.close(fig)
    write_csv(dest/'provenance.csv',provenance)
