"""MuJoCo particle-spoon transport benchmark."""
from pathlib import Path
from analysis.tremor_baselines.recorded import load_pinheiro_window
from .control import pd_torque
from .kinematics import axis_joint_reference
from .motion import roll_pitch_quaternion, roll_pitch_rotation
import argparse, csv, json, math, os, shutil, subprocess
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tremor-system-benchmark-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models/system_benchmark_spoon.xml"
CAD_BOWL_MODEL=ROOT/"models/system_benchmark_spoon_cad_bowl.xml"
CAD_BOWL_TILES=ROOT/"models/cad_spoon/collision_tiles.json"
OUT=ROOT/"experiments/results/system_benchmark/phase1"
REPORT=ROOT/"reports/system_benchmark/phase1"
FS,DURATION,NFOOD=100.,12.,30
RADIUS,MASS,UNLOAD=.0032,.00015,9.
CAD_BOWL_VERTICAL_RADIUS,CAD_BOWL_FOOTPRINT_RADIUS,CAD_BOWL_MASS=.0015,.0022,.0000228
ELLIPSOID_BOWL_A,ELLIPSOID_BOWL_B,ELLIPSOID_BOWL_DEPTH,ELLIPSOID_FOOD_RADIUS=.024,.018,.008,.002
SEEDS=range(10)
SCALE={"mild":2.,"medium":4.,"severe":6.}

def mj(u):
    u=float(np.clip(u,0,1))
    return 10*u**3-15*u**4+6*u**5

def mjdot(u,T):
    return (30*u*u-60*u**3+30*u**4)/T if 0<u<1 else 0.

def voluntary(t):
    x=.20*mj((t-3)/4)+.024*mj((t-7)/2)
    z=.16+.14*mj((t-1)/2)
    pitch=.95*mj((t-9)/2.)
    vel=[.20*mjdot((t-3)/4,4)+.024*mjdot((t-7)/2,2),0,.14*mjdot((t-1)/2,2)]
    pr=.95*mjdot((t-9)/2.,2.)
    phase="hold" if t<1 else "lift" if t<3 else "transport" if t<7 else "approach" if t<9 else "intentional_unload"
    return np.array([x,0,z]),np.array([0,pitch]),np.array(vel),np.array([0,pr]),phase

def tremor_source(duration_s=DURATION):
    w=load_pinheiro_window("p03_2",15.,duration_s,(3.,12.),1.,(0,1))
    rate=np.deg2rad(w.tremor_band_proxy_deg_s);dt=1/w.sample_rate_hz
    angle=np.zeros_like(rate);angle[1:]=np.cumsum((rate[:-1]+rate[1:])*(dt/2),axis=0)
    angle-=np.linspace(0,1,len(angle))[:,None]*angle[-1]
    angle-=angle.mean(axis=0,keepdims=True)
    return w,rate,angle

def interp(t0,x,t):
    return np.array([np.interp(t,t0,x[:,i]) for i in range(x.shape[1])])

def lowpass(x,fs,cut=2.):
    mean=x.mean(axis=0,keepdims=True);f=np.fft.rfftfreq(len(x),1/fs);z=np.fft.rfft(x-mean,axis=0);z[f>cut]=0
    return np.fft.irfft(z,n=len(x),axis=0)+mean

def bandpower(x,fs):
    n=min(round(2*fs),len(x));hop=n//2;win=np.hanning(n);psd=[]
    for st in range(0,len(x)-n+1,max(1,hop)):
        z=np.fft.rfft((x[st:st+n]-x[st:st+n].mean(axis=0))*win[:,None],axis=0)
        p=abs(z)**2/(fs*np.sum(win**2));p[1:-1]*=2;psd.append(p)
    f=np.fft.rfftfreq(n,1/fs);m=(f>=3)&(f<=12)
    return float(np.mean(np.trapezoid(np.mean(psd,axis=0)[m],f[m],axis=0)))

def joint(model,name):
    j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    return int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),j

def has_cad_bowl(model):
    return mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"bowl_cad_tile_00") >= 0

def has_ellipsoid_bowl(model):
    return mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"half_ellipsoid_bowl_visual") >= 0

def _cad_bowl_surface_z(x,y):
    if not hasattr(_cad_bowl_surface_z,"samples"):
        with CAD_BOWL_TILES.open() as f:
            spec=json.load(f)
        _cad_bowl_surface_z.samples=np.asarray([tile["surface_point_m"] for tile in spec["tiles"]])
    samples=_cad_bowl_surface_z.samples
    return float(samples[np.argmin(np.sum((samples[:,:2]-[x,y])**2,axis=1)),2])

def initialize(model,data,seed,cad_bowl=None,particle_count=None):
    particle_count=NFOOD if particle_count is None else int(particle_count)
    if particle_count < 1:
        raise ValueError("particle_count must be positive")
    cad_bowl=has_cad_bowl(model) if cad_bowl is None else bool(cad_bowl)
    rng=np.random.default_rng(seed);hid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"handle")
    pid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pitch_frame");mid=int(model.body_mocapid[hid])
    if mid < 0:
        drive=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"handle_drive")
        if drive < 0:
            raise RuntimeError("Dynamic handle needs a handle_drive mocap target")
        mid=int(model.body_mocapid[drive])
        qa,_,_=joint(model,"handle_free")
        data.qpos[qa:qa+3]=[0,0,.16];data.qpos[qa+3:qa+7]=roll_pitch_quaternion(0,0)
    data.mocap_pos[mid]=[0,0,.16];data.mocap_quat[mid]=roll_pitch_quaternion(0,0);mujoco.mj_forward(model,data)
    p=data.xpos[pid].copy();R=data.xmat[pid].reshape(3,3).copy();addresses=[]
    if cad_bowl:
        xs=np.linspace(.048,.0705,6);ys=np.linspace(-.009,.009,5)
        layout=[(x,y,0) for y in ys for x in xs]
        if len(layout)!=particle_count:
            raise RuntimeError(f"CAD bowl particle layout has {len(layout)} entries, expected {particle_count}")
    else:
        layout=None
    for i in range(particle_count):
        qa,_,_=joint(model,f"food_{i:02d}_free");addresses.append(qa)
        if cad_bowl:
            x,y,layer=layout[i]
            x+=rng.uniform(-.0002,.0002);y+=rng.uniform(-.0002,.0002)
            surface=_cad_bowl_surface_z(x,y)
            local=np.array([x,y,surface+CAD_BOWL_VERTICAL_RADIUS+layer*(2*CAD_BOWL_VERTICAL_RADIUS)+.00010])
        else:
            ix=i%6;iy=(i//6)%5
            local=np.array([.064+np.linspace(-.0175,.0175,6)[ix]+rng.uniform(-.00035,.00035),
                            np.linspace(-.014,.014,5)[iy]+rng.uniform(-.00035,.00035),RADIUS+.0001])
        data.qpos[qa:qa+3]=p+R@local;a=rng.uniform(-math.pi,math.pi)
        data.qpos[qa+3:qa+7]=[math.cos(a/2),0,0,math.sin(a/2)]
    mujoco.mj_forward(model,data)
    return hid,pid,mid,addresses

def spoon_inside(data,pid,food,cad_bowl=False):
    p=data.xpos[pid];R=data.xmat[pid].reshape(3,3)
    x=np.array([R.T@(data.qpos[a:a+3]-p) for a in food])
    if has_ellipsoid_bowl(data.model):
        dx=(x[:,0]-.064)/ELLIPSOID_BOWL_A;dy=x[:,1]/ELLIPSOID_BOWL_B
        rho2=dx*dx+dy*dy
        k2=.65**2
        depth_id=mujoco.mj_name2id(data.model,mujoco.mjtObj.mjOBJ_NUMERIC,'scene_bowl_depth_m')
        depth=(data.model.numeric_data[data.model.numeric_adr[depth_id]]
               if depth_id>=0 else ELLIPSOID_BOWL_DEPTH)
        surface=-.002+depth*(1.-np.sqrt(np.maximum(0.,1.-k2*rho2)))/(1.-np.sqrt(1.-k2))
        return (rho2<=1.04)&(x[:,2]>=surface-.004)&(x[:,2]<=.020)
    if cad_bowl:
        radius=CAD_BOWL_VERTICAL_RADIUS
        footprint=((x[:,0]-.058)/(.024-CAD_BOWL_FOOTPRINT_RADIUS))**2+(x[:,1]/(.0155-CAD_BOWL_FOOTPRINT_RADIUS))**2<=1.
        surface=np.asarray([_cad_bowl_surface_z(q[0],q[1]) for q in x])
        return footprint&(x[:,2]>=surface+radius-.004)&(x[:,2]<=surface+.040)
    return ((x[:,0]>=.0432)&(x[:,0]<=.0848)&(abs(x[:,1])<=.0168)&(x[:,2]>=-.0032)&(x[:,2]<=.0112))

def classify(data,pid,food,cad_bowl=False):
    inside=spoon_inside(data,pid,food,cad_bowl);labels=[]
    radius=CAD_BOWL_FOOTPRINT_RADIUS if cad_bowl else RADIUS
    for i,a in enumerate(food):
        x=data.qpos[a:a+3]-[.34,0,0]
        cup=abs(x[0])<=.060-radius+.0004 and abs(x[1])<=.060-radius+.0004 and .109-radius<=x[2]<=.205+radius
        labels.append("successfully_transferred_to_target" if cup else "retained_in_spoon" if inside[i] else "spilled_elsewhere")
    return labels

def run_one(seed,condition,alpha,src,video=False):
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    cad_bowl=has_cad_bowl(model)
    hid,pid,mid,food=initialize(model,data,seed,cad_bowl)
    qr,vr,jr=joint(model,"roll");qp,vp,jp=joint(model,"pitch")
    qadr=np.array([qr,qp]);vadr=np.array([vr,vp]);limits=model.jnt_range[[jr,jp]]
    w,rates,angles=src;dt=float(model.opt.timestep);steps=round(DURATION/dt);stride=round(1/FS/dt)
    rows=[];counts=[];ppos=[];sat=limits_hit=0;taucost=qvcost=peak_tau=peak_qv=0.
    exited_before_unload=np.zeros(NFOOD,dtype=bool);first_exit=np.full(NFOOD,np.nan)
    inside_at_unload=np.zeros(NFOOD,dtype=bool)
    renderer=None
    def at(t):
        p,v,vel,vr,phase=voluntary(t)
        tr=interp(w.time_s,rates,min(t,w.time_s[-1]))*alpha;ta=interp(w.time_s,angles,min(t,w.time_s[-1]))*alpha
        return p,v,vel,vr,tr,ta,phase
    p,v,*_=at(0);data.mocap_pos[mid]=p;data.mocap_quat[mid]=roll_pitch_quaternion(*v);mujoco.mj_forward(model,data)
    initial=np.array([data.qpos[a:a+3] for a in food])
    for k in range(steps):
        t=data.time;p,v,vel,vr,tr,ta,phase=at(t);data.mocap_pos[mid]=p
        data.mocap_quat[mid]=roll_pitch_quaternion(*(v+ta));mujoco.mj_step1(model,data)
        Rh=data.xmat[hid].reshape(3,3);estimate=ta if condition=="Oracle" else np.zeros(2);intent=v+ta-estimate
        qdes,_,reachable=axis_joint_reference(Rh,roll_pitch_rotation(*intent)[:,2],limits)
        tau,saturated=pd_torque(qdes,data.qpos[qadr],data.qvel[vadr],data.qfrc_bias[vadr],
                                model.actuator_ctrlrange,kp=(5.,5.),kd=(.008,.008))
        data.ctrl[:]=tau;mujoco.mj_step2(model,data);sat+=int(saturated)
        peak_tau=max(peak_tau,float(max(abs(tau))));taucost+=float(tau@tau)*dt
        qv=data.qvel[vadr];qvcost+=float(qv@qv)*dt;peak_qv=max(peak_qv,float(max(abs(qv))))
        limits_hit+=int(np.any(abs(data.qpos[qadr])>=limits[:,1]-.01))
        if (k+1)%stride==0 or k==steps-1:
            t=data.time;p,v,vel,vr,tr,ta,phase=at(t);data.mocap_pos[mid]=p
            data.mocap_quat[mid]=roll_pitch_quaternion(*(v+ta));mujoco.mj_forward(model,data)
            Rt=data.xmat[pid].reshape(3,3).copy();Rd=roll_pitch_rotation(*v);ae=Rd.T@Rt[:,2]
            er=math.atan2(-ae[1],ae[2]);ep=math.atan2(ae[0],ae[2])
            world=[math.atan2(Rt[2,1],Rt[1,1]),math.atan2(Rt[0,2],Rt[0,0])]
            inside=spoon_inside(data,pid,food,cad_bowl)
            if t<UNLOAD-1e-9:
                newly_exited=(~inside)&(~exited_before_unload)
                first_exit[newly_exited]=t
                exited_before_unload|=~inside
            elif abs(t-UNLOAD)<.005:
                inside_at_unload=inside.copy()
            retained=int(inside.sum());counts.append([t,retained]);ppos.append([data.qpos[a:a+3].copy() for a in food])
            rows.append([t,*p,*v,*vel,*vr,*tr,*ta,*intent,*qdes,*data.qpos[qadr],*data.qvel[vadr],*world,er,ep,
                         math.acos(float(np.clip(Rt[:,2]@Rd[:,2],-1,1))),*Rd[:,2],*Rt[:,2],
                         *data.actuator_force[:2],int(saturated),int(reachable),retained,phase])
    if renderer:renderer.close()
    arr=np.array([r[:-1] for r in rows],dtype=float);times=arr[:,0]
    names=["time","x","y","z","vol_roll","vol_pitch","vx","vy","vz","vol_roll_rate","vol_pitch_rate",
           "tremor_roll_rate","tremor_pitch_rate","tremor_roll_angle","tremor_pitch_angle","intent_roll","intent_pitch",
           "q_des_roll","q_des_pitch","q_roll","q_pitch","qvel_roll","qvel_pitch","spoon_world_roll","spoon_world_pitch",
           "roll_error","pitch_error","axis_error","desired_axis_x","desired_axis_y","desired_axis_z","actual_axis_x",
           "actual_axis_y","actual_axis_z","tau_roll","tau_pitch","saturated","reachable","retained_count"]
    c={x:i for i,x in enumerate(names)};transport=(times>=3)&(times<9);unload=times>=9
    err=arr[:,[c["roll_error"],c["pitch_error"]]][transport];orient=arr[:,[c["spoon_world_roll"],c["spoon_world_pitch"]]]
    omega=np.gradient(orient,times,axis=0);power=[bandpower(omega[transport,i:i+1],FS) for i in range(2)]
    desired=arr[:,[c["vol_roll"],c["vol_pitch"]]]
    intent_error_lf=lowpass(orient-desired,FS)
    i0=int(np.argmin(abs(times-UNLOAD)));i1=int(np.argmin(abs(times-DURATION)))
    retained=int(np.sum(inside_at_unload))
    labels=classify(data,pid,food,cad_bowl)
    # Success is credited only to particles retained through transport and
    # then found inside the target after the intentional unload.
    labels=[label if inside_at_unload[i] else "spilled_elsewhere" for i,label in enumerate(labels)]
    target=labels.count("successfully_transferred_to_target")
    pre_pitch=np.mean(orient[(times>=8.5)&(times<UNLOAD),1]);post_pitch=np.mean(orient[(times>=11.2)&(times<11.8),1])
    pre_des=np.mean(desired[(times>=8.5)&(times<UNLOAD),1]);post_des=np.mean(desired[(times>=11.2)&(times<11.8),1])
    metrics=dict(seed=int(seed),condition=condition,severity_scale=alpha,initial_particles=NFOOD,retained_at_unload=retained,
        retain_ratio=retained/NFOOD,target_particles=target,retained_final=labels.count("retained_in_spoon"),
        spilled_elsewhere=labels.count("spilled_elsewhere"),transfer_ratio=target/NFOOD,spill_ratio=1-target/NFOOD,
        transport_roll_rmse_deg=float(np.degrees(np.sqrt(np.mean(err[:,0]**2)))),
        transport_pitch_rmse_deg=float(np.degrees(np.sqrt(np.mean(err[:,1]**2)))),
        transport_roll_peak_deg=float(np.degrees(np.max(abs(err[:,0])))),
        transport_pitch_peak_deg=float(np.degrees(np.max(abs(err[:,1])))),
        transport_axis_rmse_deg=float(np.degrees(np.sqrt(np.mean(arr[transport,c["axis_error"]]**2)))),
        tremor_band_power_roll=power[0],tremor_band_power_pitch=power[1],
        intent_unload_roll_rmse_deg=float(np.degrees(np.sqrt(np.mean(intent_error_lf[unload,0]**2)))),
        intent_unload_pitch_rmse_deg=float(np.degrees(np.sqrt(np.mean(intent_error_lf[unload,1]**2)))),
        intentional_pitch_amplitude_ratio=float((post_pitch-pre_pitch)/(post_des-pre_des)),
        torque_integral_Nm2_s=taucost,joint_speed_integral_rad2_s=qvcost,peak_torque_Nm=peak_tau,
        peak_joint_speed_rad_s=peak_qv,saturation_fraction=sat/steps,joint_limit_event_fraction=limits_hit/steps,
        simulation_dt_s=dt,logging_rate_hz=FS,controller="existing PD+bias Kp=5 Kd=.008 torque ±.15 Nm")
    path=OUT/"traces"/f"seed{seed:02d}_{condition}_alpha{alpha:g}.npz";path.parent.mkdir(parents=True,exist_ok=True)
    voluntary_angles=arr[:,[c["vol_roll"],c["vol_pitch"]]]
    voluntary_R=np.array([roll_pitch_rotation(*q) for q in voluntary_angles])
    omega_v_world=np.column_stack((np.zeros(len(times)),arr[:,c["vol_pitch_rate"]],np.zeros(len(times))))
    tremor_true=arr[:,[c["tremor_roll_angle"],c["tremor_pitch_angle"]]]
    tremor_estimate=tremor_true.copy() if condition=="Oracle" else np.zeros_like(tremor_true)
    np.savez_compressed(path,time_s=times,columns=names,values=arr,phase=np.array([r[-1] for r in rows]),
                        particle_count=np.asarray(counts),particle_positions=np.asarray(ppos),particle_labels=labels,
                        voluntary_position_m=arr[:,[c["x"],c["y"],c["z"]]],voluntary_rotation_world=voluntary_R,
                        voluntary_angular_velocity_world_rad_s=omega_v_world,
                        voluntary_linear_velocity_world_m_s=arr[:,[c["vx"],c["vy"],c["vz"]]],
                        tremor_rate_euler_rad_s=arr[:,[c["tremor_roll_rate"],c["tremor_pitch_rate"]]],
                        tremor_angle_euler_rad=tremor_true,tremor_estimate_euler_rad=tremor_estimate,
                        first_exit_time_s=first_exit,ever_outside_before_unload=exited_before_unload,
                        inside_at_unload=inside_at_unload,initial_particles=initial)
    path=OUT/"particles"/f"seed{seed:02d}_{condition}_alpha{alpha:g}.csv";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["particle_id","final_class","ever_outside_before_unload","first_exit_time_s","final_x_m","final_y_m","final_z_m"])
        for i,(label,a) in enumerate(zip(labels,food)):wr.writerow([i,label,int(exited_before_unload[i]),first_exit[i],*data.qpos[a:a+3]])
    if video:encode_video(times,arr,np.asarray(ppos),[r[-1] for r in rows],names,condition,alpha)
    return metrics

def encode_video(times,values,particles,phases,names,condition,alpha):
    ff=shutil.which("ffmpeg")
    if not ff:raise RuntimeError("ffmpeg is required for videos")
    REPORT.mkdir(parents=True,exist_ok=True)
    suffix="no_tremor" if alpha==0 else f"{condition.lower()}_alpha{alpha:g}"
    path=REPORT/f"{suffix}.mp4"
    c={name:i for i,name in enumerate(names)}
    fig,ax=plt.subplots(figsize=(9,6),dpi=100)
    fig.tight_layout()
    proc=None
    frame_indices=np.linspace(0,len(times)-1,round(DURATION*15)+1).astype(int)
    for k in frame_indices:
        ax.clear();t=times[k];r=values[k]
        base=np.array([r[c["x"]],r[c["y"]],r[c["z"]]])
        rh=roll_pitch_rotation(r[c["vol_roll"]]+r[c["tremor_roll_angle"]],
                               r[c["vol_pitch"]]+r[c["tremor_pitch_angle"]])
        tool=rh@roll_pitch_rotation(r[c["q_roll"]],r[c["q_pitch"]])
        pivot=base+rh@np.array([.073,0,0])
        bowl=pivot+tool@np.array([.064,0,0])
        # Side projection of the bowl floor and lip; camera and axes are fixed.
        local=np.array([[.040,0,0],[.042,0,.008],[.086,0,.008],[.088,0,0]])
        rim=np.array([bowl+tool@np.array([x-.064,y,z]) for x,y,z in local])
        ax.plot(rim[:,0],rim[:,2],color="#9a5b00",lw=5,solid_capstyle="round",label="spoon bowl")
        ax.plot([base[0]-.14*rh[0,0],base[0]],[base[2]-.14*rh[2,0],base[2]],color="#315a9b",lw=8,label="handle")
        ax.scatter(particles[k,:,0],particles[k,:,2],s=18,c="#e98324",edgecolors="none",zorder=4,label="food particles")
        ax.plot([.28,.40,.40,.28,.28],[.101,.101,.205,.205,.101],color="#315e9a",lw=2,label="target container")
        ax.set(xlim=(-.16,.46),ylim=(0,.45),xlabel="world X (m)",ylabel="world Z (m)",title=f"{condition} · α={alpha:g} · {phases[k]} · t={t:.2f} s")
        ax.grid(alpha=.2);ax.legend(loc="upper left",fontsize=8,ncol=2)
        fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
        if k==0:
            h,w=frame.shape[:2]
            proc=subprocess.Popen([ff,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}",
                "-r","15","-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",str(path)],stdin=subprocess.PIPE)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():raise RuntimeError("ffmpeg encoding failed")
    plt.close(fig)

def save_config(src):
    OUT.mkdir(parents=True,exist_ok=True);w,rates,_=src
    cfg=dict(model=str(MODEL.relative_to(ROOT)),same_2dof_topology=True,step_s=.001,logging_hz=FS,duration_s=DURATION,
        phases={"hold":[0,1],"lift":[1,3],"transport":[3,7],"approach":[7,9],"intentional_unload":[9,11],"unload_hold":[11,12]},
        trajectory={"lift_m":.14,"transport_x_m":.20,"approach_x_m":.024,"unload_pitch_rad":.95,"profile":"minimum jerk"},
        particles={"count":NFOOD,"shape":"identical rigid spheres","radius_m":RADIUS,"mass_kg":MASS,"initial_layout":"6 by 5"},
        bowl_collision_proxy="shallow tray with floor and four short walls; CAD meshes remain visual",
        target_cup={"center_m":[.34,0,0],"inner_half_width_m":.06},tremor={"table":"p03_2","start_s":15,"band_hz":[3,12],
        "mapping":{"gyro_X":"roll rate","gyro_Y":"pitch rate"},
        "input_units":"deg/s in processed NPZ; converted to rad/s before integration",
        "rate_rms_rad_s":np.sqrt(np.mean(rates**2,axis=0)).tolist(),
        "severity_scale":{"mild":2,"medium":4,"severe":6},"medium":4,"source_file":w.source_file,
        "source_sha256":w.source_sha256,"rate_to_angle":"trapezoidal integration after deg/s to rad/s; finite-window drift removed"},
        seeds=list(SEEDS),oracle="exact injected angle subtracted; same IK and torque controller",
        no_comp="zero estimate; inferred intent follows observed handle; neutral joint reference")
    (OUT/"benchmark_config.json").write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+"\n")

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def plot(rows):
    REPORT.mkdir(parents=True,exist_ok=True);spec=[("retain_ratio","Transport retention"),("transfer_ratio","Target transfer"),
        ("transport_roll_rmse_deg","Roll task error (deg)"),("transport_pitch_rmse_deg","Pitch task error (deg)"),
        ("intentional_pitch_amplitude_ratio","Intentional pitch ratio")]
    cases=[("NoComp",0.),("NoComp",4.),("Oracle",4.)];labels=["No tremor","NoComp","Oracle"]
    fig,axes=plt.subplots(1,len(spec),figsize=(15,4),constrained_layout=True)
    for ax,(key,title) in zip(axes,spec):
        v=[[float(r[key]) for r in rows if r["condition"]==name and float(r["severity_scale"])==alpha
            and r.get(key) not in (None,"")] for name,alpha in cases]
        ax.boxplot(v,tick_labels=labels);ax.set_title(title);ax.tick_params(axis="x",rotation=30);ax.grid(axis="y",alpha=.2)
    fig.savefig(REPORT/"phase1_outcomes.png",dpi=150);plt.close(fig)
    files=[("NoComp",0,"no tremor"),("NoComp",4,"NoComp"),("Oracle",4,"Oracle")]
    traces=[]
    for condition,alpha,label in files:
        p=OUT/f"traces/seed00_{condition}_alpha{alpha}.npz"
        if p.exists():
            with np.load(p) as d:
                traces.append((label,d["time_s"],d["values"],list(d["columns"]),d["particle_count"],
                               d["tremor_angle_euler_rad"],d["tremor_estimate_euler_rad"]))
    if len(traces)==3:
        fig,axes=plt.subplots(5,1,figsize=(11,11),sharex=True,constrained_layout=True)
        for label,t,x,names,counts,dtrue,dhat in traces:
            c={n:i for i,n in enumerate(names)}
            axes[0].plot(t,x[:,c["vol_pitch"]],label=f"{label}: voluntary pitch",alpha=.8)
            axes[0].plot(t,dtrue[:,1],label=f"{label}: injected tremor",alpha=.55)
            axes[0].plot(t,dhat[:,1],"--",label=f"{label}: compensation input",alpha=.7)
            axes[1].plot(t,np.degrees(x[:,c["roll_error"]]),label=f"{label}: roll error")
            axes[1].plot(t,np.degrees(x[:,c["pitch_error"]]),label=f"{label}: pitch error")
            axes[2].plot(t,np.degrees(x[:,c["q_roll"]]),label=f"{label}: roll joint")
            axes[2].plot(t,np.degrees(x[:,c["q_pitch"]]),label=f"{label}: pitch joint")
            axes[3].plot(t,x[:,c["tau_roll"]],label=f"{label}: roll torque")
            axes[3].plot(t,x[:,c["tau_pitch"]],label=f"{label}: pitch torque")
            axes[4].step(counts[:,0],counts[:,1],where="post",label=label)
        axes[0].set_ylabel("angle (rad)");axes[1].set_ylabel("task error (deg)")
        axes[2].set_ylabel("joint angle (deg)");axes[3].set_ylabel("torque (N m)")
        axes[4].set_ylabel("retained");axes[4].set_xlabel("time (s)")
        for ax in axes:
            for t0 in (1,3,7,9,11):ax.axvline(t0,color="gray",alpha=.25)
            ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
        fig.savefig(REPORT/"representative_timeseries.png",dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("sanity","medium","all","summary"),default="all")
    p.add_argument("--seeds",type=int,default=10);p.add_argument("--videos",action="store_true");args=p.parse_args()
    src=tremor_source();save_config(src);rows=[]
    if args.mode=="summary":
        with (OUT/"per_run_metrics.csv").open() as f:rows=list(csv.DictReader(f))
        plot(rows);return
    if args.mode in ("sanity","all"):
        for seed in list(SEEDS)[:args.seeds]:
            r=run_one(seed,"NoComp",0,src,video=args.videos and seed==0);rows.append(r);print("sanity",seed,r["retain_ratio"],r["transfer_ratio"],flush=True)
    if args.mode in ("medium","all"):
        for seed in list(SEEDS)[:args.seeds]:
            for cond in ("NoComp","Oracle"):
                r=run_one(seed,cond,SCALE["medium"],src,video=args.videos and seed==0);rows.append(r)
                print("medium",cond,seed,r["retain_ratio"],r["transfer_ratio"],flush=True)
    path=OUT/"per_run_metrics.csv";merged={}
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):merged[(int(r["seed"]),r["condition"],float(r["severity_scale"]))]=r
    for r in rows:merged[(r["seed"],r["condition"],r["severity_scale"])]=r
    all_rows=list(merged.values());agg=[]
    no_comp={int(r["seed"]):r for r in all_rows if r["condition"]=="NoComp" and float(r["severity_scale"])==SCALE["medium"]}
    for r in all_rows:
        if r["condition"]=="Oracle" and float(r["severity_scale"])==SCALE["medium"] and int(r["seed"]) in no_comp:
            base=no_comp[int(r["seed"])]
            r["roll_band_attenuation_db"]=10*math.log10(float(r["tremor_band_power_roll"])/max(float(base["tremor_band_power_roll"]),1e-30))
            r["pitch_band_attenuation_db"]=10*math.log10(float(r["tremor_band_power_pitch"])/max(float(base["tremor_band_power_pitch"]),1e-30))
    write_csv(path,all_rows)
    for cond,alpha in (("NoComp",0.),("NoComp",SCALE["medium"]),("Oracle",SCALE["medium"])):
        subset=[r for r in all_rows if r["condition"]==cond and float(r["severity_scale"])==alpha]
        for key in ("retain_ratio","transfer_ratio","transport_roll_rmse_deg","transport_pitch_rmse_deg",
                    "transport_axis_rmse_deg","intentional_pitch_amplitude_ratio","intent_unload_pitch_rmse_deg",
                    "peak_torque_Nm","torque_integral_Nm2_s","saturation_fraction",
                    "roll_band_attenuation_db","pitch_band_attenuation_db"):
            v=np.array([float(r[key]) for r in subset if r.get(key) not in (None,"")])
            agg.append(dict(condition=cond,severity_scale=alpha,metric=key,n=len(v),
                mean=v.mean() if len(v) else None,std=v.std(ddof=1) if len(v)>1 else 0,
                min=v.min() if len(v) else None,max=v.max() if len(v) else None))
    write_csv(OUT/"aggregate_metrics.csv",agg)
    if all_rows:plot(all_rows)

if __name__=="__main__":main()
