"""Stable team entrypoint: check, demo, build, view, system, package."""
import argparse,os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tremor_delivery_mpl'))
from .common import ROOT,DATA,CONFIG,write_json


def reproduce_system(workers,seeds):
    import json
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from gimbal.system_benchmark_corrected import run_trial,protocol
    from gimbal.system_benchmark_corrected_summary import summarize
    output=DATA/'reproduction_system'
    source=ROOT/CONFIG['system']['source_root']
    frozen=json.loads((source/'alpha5/frozen_protocol.json').read_text())
    depth=frozen['plant']['bowl_depth_m']
    onset=frozen['tremor_envelope']['onset_s']
    for alpha,methods in ((5.,CONFIG['system']['all_methods']),(0.,['NoComp'])):
        p,_=protocol(alpha,depth,onset)
        shipped=json.loads((source/f'alpha{alpha:g}/frozen_protocol.json').read_text())
        if p['fingerprint']!=shipped['fingerprint']:
            raise RuntimeError('Reproduction does not match the shipped full-precision protocol; run check before recomputing')
        target=output/f'alpha{alpha:g}/frozen_protocol.json'
        if target.exists():
            if json.loads(target.read_text())['fingerprint']!=p['fingerprint']:
                raise RuntimeError('Reproduction directory belongs to different code; archive it first')
        write_json(target,p)
        jobs=[(s,m) for s in range(seeds) for m in methods]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_trial,s,m,alpha,output,True,True,depth,onset) for s,m in jobs]
            for i,future in enumerate(as_completed(futures),1):
                r=future.result();print(f'{i}/{len(jobs)} alpha={alpha:g} {r["method"]} seed={r["seed"]}: retain={r["task"]["transport_retention_ratio"]}, transfer={r["task"]["transfer_ratio_of_picked"]}',flush=True)
        summarize(alpha,output)
    print('New simulation results:',output.relative_to(ROOT),'; shipped results unchanged')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='check public API, frozen files and model compilation')
    sub.add_parser('demo',help='minimal streaming estimator example without simulation')
    sub.add_parser('build',help='audit/reuse signal history, curate completed system data and regenerate report/plots')
    view=sub.add_parser('view',help='native MuJoCo replay of the frozen representative seed')
    view.add_argument('--method',choices=CONFIG['system']['all_methods'],default='ABPF')
    view.add_argument('--seed',type=int,choices=range(10),default=3)
    view.add_argument('--speed',type=float,default=1.)
    system=sub.add_parser('system',help='recompute system trials in an isolated reproduction directory')
    system.add_argument('--workers',type=int,default=4);system.add_argument('--seeds',type=int,default=10)
    sub.add_parser('package',help='verify and create the self-contained team ZIP')
    args=p.parse_args()
    if args.command=='demo':
        import numpy as np
        from gimbal.estimators import create_estimator,EstimatorConfig
        c=EstimatorConfig.reference(unit='rad/s');e=create_estimator('ABPF',c)
        t=np.arange(round(c.sample_rate_hz))/c.sample_rate_hz
        for ti in t:
            observation=.1*np.sin(2*np.pi*6*ti)*np.ones(2)
            tremor,diagnostics=e.step(observation)
        print('Input/output unit:',c.unit,'; final estimate:',tremor)
        print('Center frequencies (Hz):',diagnostics['center_frequency_hz'])
    elif args.command=='check':
        from .verify import verify
        verify()
    elif args.command=='build':
        from .signals import prepare_signals
        from .system import prepare_system
        from .report import write_report
        from .bundle import freeze
        from .verify import verify
        print('Auditing frozen signal evidence...',flush=True);prepare_signals()
        print('Curating completed seven-condition system evidence...',flush=True);prepare_system()
        write_report();freeze();verify()
        print('Ready: reports/final_delivery/课设交付实验报告.md')
    elif args.command=='view':
        if args.speed<=0:p.error('--speed must be positive')
        import sys
        trace=ROOT/CONFIG['system']['source_root']/f'alpha5/traces/{args.method}/seed{args.seed:02d}.npz'
        if not trace.exists():raise FileNotFoundError(trace)
        from gimbal.system_benchmark_viewer import main as viewer
        sys.argv=['mujoco_replay','--replay-run',str(trace),'--speed',str(args.speed)];viewer()
    elif args.command=='system':
        if args.workers<1 or not 1<=args.seeds<=10:p.error('workers>=1 and seeds between 1 and 10 required')
        reproduce_system(args.workers,args.seeds)
    else:
        from .bundle import package
        package()


if __name__=='__main__':main()
