"""Team entrypoints: check, demo, view, system."""
import argparse,os,tempfile,sys
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
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('check');commands.add_parser('demo')
    view=commands.add_parser('view');view.add_argument('--method',choices=CONFIG['system']['all_methods'],default='ABPF');view.add_argument('--speed',type=float,default=1.)
    system=commands.add_parser('system');system.add_argument('--workers',type=int,default=4);system.add_argument('--seeds',type=int,default=10)
    a=p.parse_args()
    if a.command=='check':
        from .verify import verify
        verify()
    elif a.command=='demo':
        from gimbal.estimators import create_estimator,EstimatorConfig
        e=create_estimator('ABPF',EstimatorConfig.reference(unit='rad/s'))
        print(e.step([0.1,-0.05]))
    elif a.command=='view':
        if a.speed<=0:p.error('speed must be positive')
        from gimbal.system_benchmark_viewer import main as viewer
        trace=ROOT/CONFIG['system']['source_root']/f'alpha5/traces/{a.method}/seed03.npz'
        sys.argv=['viewer','--replay-run',str(trace),'--speed',str(a.speed)];viewer()
    else:
        if a.workers<1 or not 1<=a.seeds<=10:p.error('workers>=1 and 1<=seeds<=10 required')
        reproduce_system(a.workers,a.seeds)

if __name__=='__main__':main()
