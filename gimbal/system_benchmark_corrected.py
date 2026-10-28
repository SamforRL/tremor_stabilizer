"""Frozen estimators on the corrected dynamic-handle scoop benchmark.

Signal input is a causal kinematic projection of the actual simulated gyro.
Oracle knows the prescribed voluntary rate; no estimator receives that truth.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import mujoco
import numpy as np
from .control import pd_torque
from .kinematics import axis_joint_reference
from .motion import measured_roll_pitch, roll_pitch_quaternion, roll_pitch_rotation
from .system_benchmark import ROOT, initialize, joint, mjdot, spoon_inside, tremor_source, interp, bandpower
from .system_benchmark_scenes import (
    load_scene_model, initialize_source_particles, scene_trajectory, scene_duration,
    SCOOP_DEFAULT_PARTICLES, SCOOP_PICKUP_TIME, SCOOP_UNLOAD_TIME, SCOOP_WAYPOINTS,
    SCENE_PD_KP, SCENE_PD_KD, scoop_task_metrics,
)
from .system_benchmark_phase2a import make_estimator, estimator_signal_metrics
from .system_benchmark_corrected_summary import summarize

OUT = ROOT/'experiments/results/system_benchmark/corrected_algorithm_comparison'
REPORT = ROOT/'reports/system_benchmark/corrected_algorithm_comparison'
METHODS = ('NoComp','FixedBPF','ABPF','OriginalBMFLC','HuberBMFLC','DampedBMFLC','Oracle')
FOURIER = ('OriginalBMFLC','HuberBMFLC','DampedBMFLC')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def protocol(alpha, bowl_depth_m=None, tremor_onset_s=None):
    baseline=json.loads((ROOT/'experiments/results/system_benchmark/phase2a/frozen_config.json').read_text())
    plant=json.loads((ROOT/'experiments/results/system_benchmark/geometry_refinement/scoop_baseline_v3/frozen_config.json').read_text())
    plant['bowl_depth_m']=.008 if bowl_depth_m is None else float(bowl_depth_m)
    duration=scene_duration('scoop_transfer');source=tremor_source(duration_s=duration)
    w=source[0]
    p=dict(protocol='corrected_scoop_estimators_v1',alpha=float(alpha),source=dict(table=w.table,
           start_s=w.start_s,duration_s=w.duration_s,band_hz=w.band_hz,channels=w.channels,
           sha256=w.source_sha256,sample_rate_hz=w.sample_rate_hz),plant=plant,
           methods=METHODS,seeds=list(range(10)),duration_s=duration,
           estimator_parameters={k:baseline[k] for k in ('fixed_bpf','abpf','bmflc','huber')},
           damped_bmflc=dict(eta=.020,lambda_ref=.99999,reference_fs_hz=1000.,kdmp=350.,xdmp_rad_s=.009),
           interface=dict(input='actual world gyro -> exact instantaneous derivative of measured RxRy roll/pitch coordinates, deg/s',
             gyro_frames='mj_objectVelocity world angular velocity; full 3D gyro logged',
             true_system_disturbance='actual projected handle angular rate minus known prescribed voluntary angular rate',
             oracle='perfect present system-disturbance rate, sampled/held/integrated identically to estimators',
             command_proxy='recorded 3-12 Hz gyro proxy; angle offset removed so command disturbance starts at zero',
             integration='zero initial correction; common causal trapezoidal integration at 1 kHz; no leakage or final phase alignment',
             reference='actual projected handle angles minus integrated estimate; preserve residual unactuated orientation',
             initial_bias='common zero correction; small settled drive bias is not removed by an oracle-only angle anchor',
             note='system disturbance includes physical drive/contact tracking errors; not an isolated physiological tremor label'),
           sampling=dict(estimator_hz=w.sample_rate_hz,physics_hz=1000.,logging_hz=100.,replay_hz=50.),
           source_proxy_is_not_physiological_ground_truth=True)
    p['tremor_envelope']=dict(onset_s=tremor_onset_s,ramp_s=.5,
                             compensation_enabled_from_s=tremor_onset_s,
                             definition='all-task if onset is null; otherwise shared NoComp scoop before checkpoint, then quintic tremor ramp and enable compensation; identical initial transport load, no method-specific gate')
    paths=['gimbal/system_benchmark_corrected.py','gimbal/system_benchmark_scenes.py','gimbal/control.py','gimbal/kinematics.py',
           'gimbal/system_benchmark_phase2a.py','analysis/tremor_baselines/abpf.py','analysis/tremor_baselines/bmflc_core.py',
           'analysis/tremor_baselines/round3_variants.py','analysis/tremor_baselines/damped_bmflc.py','analysis/tremor_baselines/butterworth.py']
    p['code_sha256']={s:sha(ROOT/s) for s in paths}
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    return p,source


def voluntary_rate(t):
    for (ta,_,a,_),(tb,_,b,_) in zip(SCOOP_WAYPOINTS[:-1],SCOOP_WAYPOINTS[1:]):
        if ta<=t<tb:
            return np.array([0.,(b-a)*mjdot((t-ta)/(tb-ta),tb-ta)])
    return np.zeros(2)


def coordinate_rates(rotation, world_omega):
    """Analytic derivative of measured_roll_pitch; never differentiate samples."""
    R=np.asarray(rotation).reshape(3,3);wx,wy,wz=world_omega
    W=np.array([[0.,-wz,wy],[wz,0.,-wx],[-wy,wx,0.]])
    D=W@R
    return np.array([(R[1,1]*D[2,1]-R[2,1]*D[1,1])/max(R[1,1]**2+R[2,1]**2,1e-12),
                     (R[0,0]*D[0,2]-R[0,2]*D[0,0])/max(R[0,0]**2+R[0,2]**2,1e-12)])


def make_compensation_reference(handle_R, observed_angles, estimate_angle):
    residual_R=handle_R@roll_pitch_rotation(*observed_angles).T
    return residual_R@roll_pitch_rotation(*(observed_angles-estimate_angle))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def run_trial(seed,method,alpha,output_root=OUT,resume=True,save_states=True,bowl_depth_m=None,tremor_onset_s=None):
    p,source=protocol(alpha,bowl_depth_m,tremor_onset_s);root=Path(output_root)/f'alpha{alpha:g}'
    result_path=root/'runs'/method/f'seed{seed:02d}.json'
    if resume and result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous.get('fingerprint')==p['fingerprint'] and previous.get('completed'):
            return previous
    started=time.perf_counter();model,tmp=load_scene_model(SCOOP_DEFAULT_PARTICLES,bowl_depth_m=p['plant']['bowl_depth_m']);Path(tmp).unlink(missing_ok=True)
    data=mujoco.MjData(model);hid,pid,mid,food=initialize(model,data,seed,particle_count=SCOOP_DEFAULT_PARTICLES)
    initialize_source_particles(model,data,seed,food)
    qr,vr,jr=joint(model,'roll');qp,vp,jp=joint(model,'pitch')
    q=np.array([qr,qp]);v=np.array([vr,vp]);limits=model.jnt_range[[jr,jp]]
    fs=source[0].sample_rate_hz;duration=p['duration_s'];dt=float(model.opt.timestep)
    cfg=p['estimator_parameters'];modal=np.array(cfg['abpf']['modal_frequency_hz_from_raw_0_15s'])
    delta=np.array(cfg['huber']['delta_deg_s'])
    estimator=make_estimator(method,fs,modal,delta)
    pure=make_estimator(method,fs,modal,delta) if method in FOURIER else None
    if estimator:estimator.reset()
    if pure:pure.reset()
    sensor_times=np.arange(math.ceil(duration*fs))/fs;sensor_index=0
    window,command_rates,command_angles=source
    command_angles=(command_angles-command_angles[0])*alpha
    est_angle=np.zeros(2);last_rate=None;held=np.zeros(2);picked=unload=None
    gyro=np.zeros(6);tool_vel=np.zeros(6);sensor=[];logs=[];states=[];weight_records=[];cpu=[]
    tau_cost=speed_cost=0.;peak_tau=peak_qv=0.;saturations=limit_steps=0
    for k in range(round(duration/dt)):
        t=float(data.time);position,vol_rp,phase=scene_trajectory(t,'scoop_transfer')
        injected=interp(window.time_s,command_angles,min(t,window.time_s[-1]))
        if tremor_onset_s is not None:
            u=float(np.clip((t-tremor_onset_s)/.5,0.,1.))
            injected*=u**3*(10.+u*(-15.+6.*u))
        data.mocap_pos[mid]=position;data.mocap_quat[mid]=roll_pitch_quaternion(*(vol_rp+injected))
        mujoco.mj_step1(model,data)
        Rh=data.xmat[hid].reshape(3,3).copy();observed=measured_roll_pitch(Rh)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,hid,gyro,0)
        rate=coordinate_rates(Rh,gyro[:3]);truth=rate-voluntary_rate(t)
        if sensor_index<len(sensor_times) and sensor_times[sensor_index]<=t+1e-10:
            x=np.rad2deg(rate);truth_deg=np.rad2deg(truth)
            if method=='NoComp':out=np.zeros(2);diagnostics={'internal_error':x.copy()}
            elif method=='Oracle':out=truth_deg.copy();diagnostics={'internal_error':x-truth_deg}
            else:
                tic=time.perf_counter_ns();out,diagnostics=estimator.step(x);cpu.append((time.perf_counter_ns()-tic)*1e-9)
            held=np.asarray(out,dtype=float)
            weights=diagnostics.get('adaptive_weights');norm=np.full(2,np.nan);contamination=np.full(2,np.nan)
            if weights is not None:
                weights=np.asarray(weights);norm=np.linalg.norm(weights,axis=1)
                if pure:
                    _,pure_diag=pure.step(truth_deg)
                    contamination=np.linalg.norm(weights-np.asarray(pure_diag['adaptive_weights']),axis=1)
                if seed==0:weight_records.append(weights.astype(np.float32))
            center=np.asarray(diagnostics.get('center_frequency_hz',[np.nan,np.nan]))
            err=np.asarray(diagnostics.get('internal_error',x-held))
            sensor.append([t,*x,*truth_deg,*held,*err,*norm,*contamination,*center,*gyro[:3],*voluntary_rate(t)])
            sensor_index+=1
        rate_now=np.deg2rad(held)
        if tremor_onset_s is None or t>=tremor_onset_s:
            if last_rate is not None:est_angle+=.5*(last_rate+rate_now)*dt
            last_rate=rate_now.copy()
        else:
            est_angle.fill(0.);last_rate=None
        desired=make_compensation_reference(Rh,observed,est_angle)
        qdes,_,reachable=axis_joint_reference(Rh,desired[:,2],limits)
        tau,saturated=pd_torque(qdes,data.qpos[q],data.qvel[v],data.qfrc_bias[v],model.actuator_ctrlrange,kp=SCENE_PD_KP,kd=SCENE_PD_KD)
        data.ctrl[:]=tau;mujoco.mj_step2(model,data)
        saturations+=int(saturated);tau_cost+=float(tau@tau)*dt;speed_cost+=float(data.qvel[v]@data.qvel[v])*dt
        peak_tau=max(peak_tau,float(max(abs(tau))));peak_qv=max(peak_qv,float(max(abs(data.qvel[v]))))
        limit_steps+=int(np.any(data.qpos[q]<=limits[:,0]+.005) or np.any(data.qpos[q]>=limits[:,1]-.005))
        if picked is None and data.time>=SCOOP_PICKUP_TIME-1e-9:picked=spoon_inside(data,pid,food).copy()
        if unload is None and data.time>=SCOOP_UNLOAD_TIME-1e-9:unload=spoon_inside(data,pid,food).copy()
        if (k+1)%10==0:
            Rt=data.xmat[pid].reshape(3,3).copy();task_R=roll_pitch_rotation(*vol_rp);z=task_R.T@Rt[:,2]
            errors=np.array([math.atan2(-z[1],z[2]),math.atan2(z[0],z[2])])
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pid,tool_vel,0)
            output_rate=coordinate_rates(Rt,tool_vel[:3])
            logs.append([data.time,*vol_rp,*observed,*measured_roll_pitch(Rt),*errors,*output_rate,
                         *data.qpos[q],*qdes,*data.qvel[v],*tau,*est_angle,*rate,*truth,*rate_now,
                         int(spoon_inside(data,pid,food).sum()),int(saturated),int(reachable),*data.qpos[:3]])
        if save_states and (k+1)%20==0:
            states.append((float(data.time),data.qpos[:7].copy(),data.qpos[q].copy(),
                           np.array([data.qpos[a:a+3] for a in food]),data.mocap_pos[mid].copy(),data.mocap_quat[mid].copy()))
    if not np.isfinite(data.qpos).all():raise RuntimeError('non-finite physical state')
    task=scoop_task_metrics(data,pid,food,picked,unload);L=np.asarray(logs);S=np.asarray(sensor)
    carry=(L[:,0]>=SCOOP_PICKUP_TIME)&(L[:,0]<SCOOP_UNLOAD_TIME);pour=L[:,0]>=SCOOP_UNLOAD_TIME
    # Column conventions below are frozen with every trace.
    columns=['time_s','desired_roll','desired_pitch','handle_roll','handle_pitch','spoon_roll','spoon_pitch',
             'roll_error','pitch_error','spoon_roll_rate','spoon_pitch_rate','q_roll','q_pitch','qdes_roll','qdes_pitch',
             'qvel_roll','qvel_pitch','tau_roll','tau_pitch','estimate_angle_roll','estimate_angle_pitch',
             'handle_roll_rate','handle_pitch_rate','true_disturbance_roll_rate','true_disturbance_pitch_rate',
             'estimate_roll_rate','estimate_pitch_rate','in_spoon_count','saturated','reachable','handle_x','handle_y','handle_z']
    # Use explicit names rather than index arithmetic for evaluation.
    C={s:i for i,s in enumerate(columns)}
    errors=L[carry][:,[C['roll_error'],C['pitch_error']]]
    task.update(transport_roll_rmse_deg=float(np.degrees(_rms(errors[:,0]))),
                transport_pitch_rmse_deg=float(np.degrees(_rms(errors[:,1]))),
                orientation_peak_deg=float(np.degrees(np.max(np.linalg.norm(errors,axis=1)))),
                orientation_p95_deg=float(np.degrees(np.percentile(np.linalg.norm(errors,axis=1),95))),
                output_roll_bandpower=bandpower(L[carry,C['spoon_roll_rate']:C['spoon_roll_rate']+1],100.),
                output_pitch_bandpower=bandpower(L[carry,C['spoon_pitch_rate']:C['spoon_pitch_rate']+1],100.),
                torque_integral_Nm2_s=tau_cost,joint_speed_integral_rad2_s=speed_cost,peak_torque_Nm=peak_tau,
                peak_joint_speed_rad_s=peak_qv,saturation_fraction=saturations/round(duration/dt),
                joint_limit_fraction=limit_steps/round(duration/dt))
    pre=(L[:,0]>=SCOOP_UNLOAD_TIME-.5)&(L[:,0]<SCOOP_UNLOAD_TIME)
    post=L[:,0]>duration-.5
    achieved=L[post,C['spoon_pitch']].mean()-L[pre,C['spoon_pitch']].mean()
    intended=L[post,C['desired_pitch']].mean()-L[pre,C['desired_pitch']].mean()
    task['intentional_pitch_amplitude_ratio']=float(achieved/intended)
    task['unload_pitch_rmse_deg']=float(np.degrees(_rms(L[pour,C['spoon_pitch']]-L[pour,C['desired_pitch']])))
    signal=estimator_signal_metrics(S[:,3:5],S[:,5:7],fs)
    cpu=np.asarray(cpu)
    task['estimator_mean_us']=float(cpu.mean()*1e6) if len(cpu) else 0.
    task['estimator_p99_us']=float(np.percentile(cpu,99)*1e6) if len(cpu) else 0.
    task['estimator_max_us']=float(cpu.max()*1e6) if len(cpu) else 0.
    task['sensor_sample_jitter_max_ms']=float(max(abs(S[:,0]-sensor_times[:len(S)])))*1000
    result=dict(completed=True,seed=seed,method=method,alpha=alpha,fingerprint=p['fingerprint'],
                task=task,signal=signal,wall_s=time.perf_counter()-started)
    trace=root/'traces'/method/f'seed{seed:02d}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
    sensor_columns=['time_s','input_roll_deg_s','input_pitch_deg_s','true_roll_deg_s','true_pitch_deg_s',
                    'estimate_roll_deg_s','estimate_pitch_deg_s','residual_roll','residual_pitch',
                    'weight_norm_roll','weight_norm_pitch','contamination_roll','contamination_pitch',
                    'center_roll_hz','center_pitch_hz','gyro_world_x_rad_s','gyro_world_y_rad_s','gyro_world_z_rad_s',
                    'voluntary_roll_rate_rad_s','voluntary_pitch_rate_rad_s']
    extra={}
    if states:
        extra=dict(replay_time_s=np.array([x[0] for x in states]),handle_qpos=np.array([x[1] for x in states],np.float32),
                   joint_qpos=np.array([x[2] for x in states],np.float32),particle_positions=np.array([x[3] for x in states],np.float32),
                   mocap_pos=np.array([x[4] for x in states],np.float32),mocap_quat=np.array([x[5] for x in states],np.float32))
    if weight_records:extra['fourier_weights']=np.asarray(weight_records)
    np.savez_compressed(trace,controller=L.astype(np.float32),controller_columns=columns,sensor=S.astype(np.float32),
                        sensor_columns=sensor_columns,picked_mask=picked,unload_mask=unload,
                        config_json=json.dumps(p),method=method,seed=seed,alpha=alpha,**extra)
    result['trace']=str(trace.relative_to(ROOT));write_json(result_path,result)
    return result


def checked_kinematics():
    for r,p in ((0.,0.),(.2,.8),(-.2,.95)):
        rate=np.array([.3,-.4]);R=roll_pitch_rotation(r,p)
        omega=np.array([rate[0],math.cos(r)*rate[1],math.sin(r)*rate[1]])
        assert np.allclose(coordinate_rates(R,omega),rate,atol=1e-10)
        assert np.allclose(make_compensation_reference(R,np.array([r,p]),np.zeros(2)),R,atol=1e-10)
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('pilot','run','summary'),default='pilot')
    ap.add_argument('--alpha',type=float,default=4.)
    ap.add_argument('--seeds',type=int,default=10)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    ap.add_argument('--no-resume',action='store_true')
    ap.add_argument('--bowl-depth-mm',type=float,help='separate geometry-sensitivity protocol; original depth is 8 mm')
    ap.add_argument('--transport-tremor-only',action='store_true',help='diagnostic: smoothly introduce tremor only after the fixed pickup checkpoint')
    args=ap.parse_args();checked_kinematics()
    depth=None if args.bowl_depth_mm is None else args.bowl_depth_mm/1000
    if depth is not None and not .002<=depth<=.012:ap.error('bowl depth must be 2–12 mm')
    output_root=OUT if depth is None else ROOT/f'experiments/results/system_benchmark/geometry_sensitivity/depth{args.bowl_depth_mm:g}mm'
    onset=SCOOP_PICKUP_TIME if args.transport_tremor_only else None
    if onset is not None:output_root=output_root/'transport_onset_v2_matched_load'
    p,_=protocol(args.alpha,depth,onset);root=output_root/f'alpha{args.alpha:g}'
    if args.mode=='summary':
        summarize(args.alpha,output_root);return
    existing=root/'frozen_protocol.json'
    if existing.exists() and json.loads(existing.read_text())['fingerprint']!=p['fingerprint']:
        raise RuntimeError('Protocol fingerprint changed; choose a new output root before mixing runs')
    write_json(existing,p)
    methods=('NoComp','Oracle') if args.mode=='pilot' else args.methods
    seeds=range(1) if args.mode=='pilot' else range(args.seeds)
    jobs=[(seed,method) for seed in seeds for method in methods]
    with ProcessPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(run_trial,s,m,args.alpha,output_root,not args.no_resume,True,depth,onset):(s,m) for s,m in jobs}
        for i,future in enumerate(as_completed(futures),1):
            r=future.result();q=r['task']
            print(f"{i}/{len(jobs)} alpha={args.alpha:g} seed={r['seed']} {r['method']}: "
                  f"pickup={q['pickup_count']} retained={q['retained_before_unload_count']} "
                  f"delivered={q['transferred_pickup_count']} transfer={q['transfer_ratio_of_picked']} "
                  f"roll/pitch RMSE={q['transport_roll_rmse_deg']:.3f}/{q['transport_pitch_rmse_deg']:.3f} deg",flush=True)
    if args.mode=='run':summarize(args.alpha,output_root)


if __name__=='__main__':main()
