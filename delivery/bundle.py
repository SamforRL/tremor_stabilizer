"""Whitelist-based portable handoff. Never includes raw datasets or secrets."""
from pathlib import Path
import zipfile
from .common import ROOT,DATA,REPORT,CONFIG,sha,write_json

GIMBAL_FILES=('__init__.py','control.py','kinematics.py','motion.py','estimators.py',
              'system_benchmark.py','system_benchmark_scenes.py','system_benchmark_corrected.py',
              'system_benchmark_corrected_summary.py','system_benchmark_phase2a.py','system_benchmark_viewer.py')


def selected_files():
    files=set()
    def add(p):
        p=ROOT/p
        if p.is_file():files.add(p)
    def tree(p):
        p=ROOT/p
        if p.exists():
            files.update(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts
                         and f.suffix not in ('.pyc','.tmp'))
    for p in ('README.md','requirements.txt','AGENTS.md','analysis/__init__.py','data/README.md','reports/history/工程历史说明.md'):
        add(p)
    tree(Path('delivery'))
    for name in GIMBAL_FILES:add(Path('gimbal')/name)
    for p in (ROOT/'analysis/tremor_baselines').glob('*.py'):files.add(p)
    for name in ('README.md','REPRODUCTION_NOTES.md','LITERATURE_AMBIGUITIES.md'):add(Path('analysis/tremor_baselines')/name)
    add(Path('models/system_benchmark_spoon.xml'));tree(Path('models/cad_spoon'))
    add(Path('data/processed/pinheiro_p03_2_lumbrical_gyro.npz'))
    for p in ('experiments/results/system_benchmark/phase2a/frozen_config.json',
              'experiments/results/system_benchmark/geometry_refinement/scoop_baseline_v3/frozen_config.json'):
        add(p)
    history=Path(CONFIG['signal']['source_root'])
    for name in ('frozen_config.json','signal_level_metrics.csv','amplitude_ratio_metrics.csv','Damped_BMFLC_reproduction_notes.md'):
        add(history/name)
    tree(history/'signal_traces')
    for name in ('controlled_cases.csv','recorded_pairs.csv'):
        add(Path('reports/tremor_baselines/round3/abpf_ablation')/name)
    add(Path('reports/tremor_baselines/round3/abpf_response/frequency_response.csv'))
    for case in ('drift','step'):
        for variant in ('adaptive','fixed'):
            add(Path('experiments/results/tremor_baselines/round3/abpf_ablation')/f'{case}_abpf_{variant}.npz')
    for alpha in ('alpha0','alpha5'):
        base=Path(CONFIG['system']['source_root'])/alpha
        tree(base/'runs');tree(base/'traces');add(base/'frozen_protocol.json')
    tree(DATA.relative_to(ROOT));tree(REPORT.relative_to(ROOT))
    return sorted(p for p in files if p.name not in ('manifest.json','verification.json') or p.parent!=ROOT/'delivery')


def freeze():
    excluded={DATA/'verification.json',ROOT/'delivery/manifest.json'}
    files={str(p.relative_to(ROOT)):sha(p) for p in selected_files() if p not in excluded}
    write_json(ROOT/'delivery/manifest.json',dict(version=CONFIG['version'],files=files,
        primary_signal_data='experiments/results/final_delivery/signals/metrics.csv',
        primary_system_data='experiments/results/final_delivery/system/aggregate_metrics.csv',
        note='Hashes protect the shipped snapshot; no tuning or automatic parameter changes'))
    return files


def package():
    from .verify import verify
    verify(manifest=True)
    files=selected_files()+[ROOT/'delivery/manifest.json']
    target=ROOT/'dist/tremor_stabilizer_course_v1.zip';target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for p in sorted(set(files)):
            archive.write(p,'tremor_stabilizer_course_v1/'+str(p.relative_to(ROOT)))
    checksum=sha(target)
    target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n')
    print(f'Bundle: {target.relative_to(ROOT)} ({target.stat().st_size/1e6:.1f} MB)')
    return target
