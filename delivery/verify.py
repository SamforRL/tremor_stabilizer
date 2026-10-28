"""One maintained delivery check, instead of scattered ad-hoc test scripts."""
from dataclasses import replace
import importlib.metadata
import sys
import numpy as np
from .common import ROOT,DATA,CONFIG,write_json,sha
from gimbal.estimators import EstimatorConfig,create_estimator,METHODS


def verify(manifest=True):
    checks=[];c=EstimatorConfig.reference();rng=np.random.default_rng(5418)
    x=rng.normal(0,20,(300,2))
    from gimbal.system_benchmark_phase2a import make_estimator
    for method in METHODS:
        a=create_estimator(method,c);b=make_estimator(method,c.sample_rate_hz,c.modal_frequency_hz,c.huber_delta_deg_s)
        y=[]
        for sample in x:
            out,d=a.step(sample);legacy,_=b.step(sample)
            assert np.array_equal(out,legacy),method
            if 'adaptive_weights' in d:assert d['adaptive_weights'].shape==(2,182)
            y.append(out)
        y=np.asarray(y);assert y.shape==x.shape and np.isfinite(y).all()
        a.reset();assert np.array_equal(y,np.asarray([a.step(s)[0] for s in x]))
        a.reset();prefix=np.asarray([a.step(s)[0] for s in x[:150]])
        a.reset();changed=x.copy();changed[150:]*=20
        assert np.array_equal(prefix,np.asarray([a.step(s)[0] for s in changed])[:150])
        rad=create_estimator(method,replace(c,unit='rad/s'))
        converted=np.asarray([rad.step(np.deg2rad(s))[0] for s in x])
        assert np.allclose(converted,np.deg2rad(y),rtol=1e-11,atol=1e-12)
        for axes in (1,3):
            cc=replace(c,axes=axes,modal_frequency_hz=(6.,)*axes,huber_delta_deg_s=(8.,)*axes)
            est=create_estimator(method,cc);assert est.step(np.zeros(axes))[0].shape==(axes,)
        a.reset()
        for invalid in (np.zeros(3),np.array([np.nan,0.]),np.array([np.inf,0.])):
            try:a.step(invalid)
            except ValueError:pass
            else:raise AssertionError('Invalid input was accepted')
        checks.append(dict(method=method,finite_dimensions=True,reset=True,causal_prefix=True,
                           exact_legacy_output=True,degree_radian_equivalence=True,one_three_axes=True))
    from gimbal.system_benchmark_corrected import checked_kinematics
    checked_kinematics()
    import json
    frozen=json.loads((ROOT/CONFIG['system']['source_root']/'alpha5/frozen_protocol.json').read_text())
    assert CONFIG['system']['tremor_onset_s']==frozen['tremor_envelope']['onset_s']
    assert CONFIG['system']['bowl_depth_m']==frozen['plant']['bowl_depth_m']
    assert c.sample_rate_hz==frozen['sampling']['estimator_hz']
    from gimbal.system_benchmark_scenes import load_scene_model
    import mujoco
    model,path=load_scene_model(180,bowl_depth_m=.004)
    from pathlib import Path
    Path(path).unlink(missing_ok=True)
    assert model.nu==2 and model.neq==1 and model.nmocap==1 and model.opt.timestep==.001
    depthid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_NUMERIC,'scene_bowl_depth_m')
    assert np.isclose(model.numeric_data[model.numeric_adr[depthid]],.004)
    for filename in ('signals/audit.json','system/verification.json'):
        if not (DATA/filename).exists():raise FileNotFoundError('Run python -m delivery build first')
    checked=0
    if manifest:
        p=ROOT/'delivery/manifest.json'
        if not p.exists():raise FileNotFoundError('Missing delivery manifest; run build')
        m=json.loads(p.read_text())
        for rel,expected in m['files'].items():
            if sha(ROOT/rel)!=expected:raise RuntimeError(f'Frozen file changed: {rel}')
            checked+=1
    result=dict(passed=True,python=sys.version.split()[0],dependencies={name:importlib.metadata.version(name) for name in ('numpy','mujoco','matplotlib')},
                estimator_checks=checks,kinematics=True,shallow_model_compiled=True,verified_manifest_files=checked)
    write_json(DATA/'verification.json',result)
    print(f'PASS: five estimators, reset/causality/units/dimensions, 2-DoF MuJoCo model, {checked} frozen files')
    return result
