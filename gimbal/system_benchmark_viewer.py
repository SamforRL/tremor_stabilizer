"""Run one particle-spoon benchmark trial in the interactive MuJoCo viewer."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .system_benchmark_scenes import (
    SCENE_CONTACT_FRICTION,
    SCENE_CONTACT_SOLIMP,
    SCENE_CONTACT_SOLREF,
    SCENE_PD_KP,
    SCENE_PD_KD,
    SPOON_CONTACT_FRICTION,
    HANDLE_WELD_SOLREF,
    HANDLE_WELD_TORQUE_SCALE,
    SCOOP_DEFAULT_PARTICLES,
    SCOOP_PICKUP_TIME,
    SCOOP_UNLOAD_TIME,
    SCOOP_WAYPOINTS,
    SCOOP_SOURCE_DEPTH,
    initialize_ellipsoid_bowl_particles,
    initialize_source_particles,
    load_scene_model,
    scene_trajectory,
    scene_duration,
    scoop_task_metrics,
)
from .control import pd_torque
from .kinematics import axis_joint_reference
from .motion import roll_pitch_quaternion, roll_pitch_rotation
from .system_benchmark import (
    SCALE,
    initialize,
    interp,
    joint,
    spoon_inside,
    tremor_source,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("NoComp", "Oracle"), default="NoComp")
    parser.add_argument("--severity", choices=("none", "mild", "medium", "severe"),
                        help="default: none for scoop_transfer, medium for other scenes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--particles", type=int, default=None,
                        help=f"food grain count (default: {SCOOP_DEFAULT_PARTICLES} for scoop_transfer, 30 otherwise)")
    parser.add_argument("--bowl-depth-mm", type=float, default=8.,
                        help="live spoon depth, 2–12 mm; saved replay reads its own geometry")
    parser.add_argument("--metrics-json", type=Path,
                        help="save scoop counts, picked particle IDs and task configuration as JSON")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="simulation speed relative to wall time; 1 is real time")
    parser.add_argument("--scene", choices=("pitch_pour", "roll_pour", "scoop_transfer"),
                        default="pitch_pour",
                        help="pitch tilt, roll tilt, or scoop from the source bowl and transfer")
    parser.add_argument("--replay-phase2a", choices=("NoComp", "FixedBPF", "ABPF", "OriginalBMFLC",
                                                     "HuberBMFLC", "Oracle"),
                        help="replay a saved Phase 2A MuJoCo state trace in the native viewer")
    parser.add_argument("--replay-run", type=Path,
                        help="replay a corrected scoop experiment NPZ in the native MuJoCo viewer")
    args = parser.parse_args()
    args.severity = args.severity or ("none" if args.scene == "scoop_transfer" else "medium")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if not 2. <= args.bowl_depth_mm <= 12.:
        parser.error("--bowl-depth-mm must be between 2 and 12")
    if args.replay_run:
        if args.replay_phase2a or args.metrics_json:
            parser.error("--replay-run cannot be combined with --replay-phase2a or --metrics-json")
        _replay_corrected(args.replay_run, args.speed)
        return
    if args.metrics_json and (args.scene != "scoop_transfer" or args.replay_phase2a):
        parser.error("--metrics-json is available for live scoop_transfer trials")

    particle_count = args.particles if args.particles is not None else (
        SCOOP_DEFAULT_PARTICLES if args.scene == "scoop_transfer" else 30)
    if particle_count < 1 or particle_count > 200:
        parser.error("--particles must be between 1 and 200 (the tested range)")
    if args.replay_phase2a:
        if args.particles not in (None, 30):
            parser.error("saved Phase 2A replay contains exactly 30 particles")
        particle_count = 30
    model, temporary_model_path = load_scene_model(particle_count,
                                                  dynamic_handle=not bool(args.replay_phase2a),
                                                  bowl_depth_m=args.bowl_depth_mm/1000)
    # Meshes have already been loaded into MjModel; keep no generated XML files.
    Path(temporary_model_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    handle_id, pitch_id, mocap_id, food = initialize(
        model, data, args.seed, particle_count=particle_count)
    if args.scene == "scoop_transfer" and not args.replay_phase2a:
        print("Settling randomly placed grains under gravity...")
        settling = initialize_source_particles(model, data, args.seed, food)
        print(f"Settled for {settling['settling_s']:.1f} s; peak grain–grain "
              f"contact count={settling['peak_grain_grain_contacts']}.")
    elif not args.replay_phase2a:
        initialize_ellipsoid_bowl_particles(model, data, pitch_id, food, args.seed)
    q_roll, v_roll, j_roll = joint(model, "roll")
    q_pitch, v_pitch, j_pitch = joint(model, "pitch")
    qadr = np.array([q_roll, q_pitch])
    vadr = np.array([v_roll, v_pitch])
    limits = model.jnt_range[[j_roll, j_pitch]]

    from mujoco import viewer as mjviewer
    if args.replay_phase2a:
        _replay_phase2a(mjviewer, model, data, mocap_id, q_roll, q_pitch, food,
                        args.replay_phase2a, args.seed, args.speed)
        return

    duration = scene_duration(args.scene)
    source, rates, angles = tremor_source(duration_s=duration) if args.severity != "none" else (None, None, None)
    alpha = SCALE[args.severity] if args.severity != "none" else 0.0
    dt = float(model.opt.timestep)
    steps_per_frame = max(1, round(1.0 / (60 * dt)))
    status_interval = max(1, round(1.0 / (steps_per_frame * dt)))
    total_steps = round(duration / dt)
    initial_position, initial_rp, _ = scene_trajectory(0.0, args.scene)
    data.mocap_pos[mocap_id] = initial_position
    data.mocap_quat[mocap_id] = roll_pitch_quaternion(*initial_rp)
    mujoco.mj_forward(model, data)

    print(f"MuJoCo interactive task: {args.scene}, condition={args.condition}, "
          f"severity={args.severity}, seed={args.seed}; {duration:g} s at {args.speed:g}x speed.")
    print("Scene contains a tabletop, source bowl and target cup. "
          "Scoop transfer is a contact demonstration, not a validated pickup benchmark.")
    print("Close the viewer window to exit. Press Space in the viewer to pause/resume.")
    picked_mask = unload_mask = None
    with mjviewer.launch_passive(model, data) as viewer:
        # Table, vessels, and collision proxies are assigned to geom group 3.
        viewer.opt.geomgroup[:] = 1
        viewer.opt.geomgroup[4] = 0  # hide collision-only ellipsoid patches
        viewer.cam.lookat[:] = [0.18, 0.0, 0.12]
        viewer.cam.distance = 0.70
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        frame_count = 0
        step_index = 0
        while viewer.is_running() and step_index < total_steps:
            start_wall = time.monotonic()
            batch_end = min(total_steps, step_index + steps_per_frame)
            while step_index < batch_end:
                t = data.time
                p, voluntary_rp, phase = scene_trajectory(t, args.scene)
                tremor_angle = (np.zeros(2) if source is None else
                                interp(source.time_s, angles, min(t, source.time_s[-1])) * alpha)
                data.mocap_pos[mocap_id] = p
                data.mocap_quat[mocap_id] = roll_pitch_quaternion(*(voluntary_rp + tremor_angle))
                mujoco.mj_step1(model, data)

                handle_rotation = data.xmat[handle_id].reshape(3, 3)
                estimated_tremor = tremor_angle if args.condition == "Oracle" else np.zeros(2)
                intended = voluntary_rp + tremor_angle - estimated_tremor
                q_des, _, _ = axis_joint_reference(
                    handle_rotation, roll_pitch_rotation(*intended)[:, 2], limits
                )
                torque, _ = pd_torque(
                    q_des, data.qpos[qadr], data.qvel[vadr], data.qfrc_bias[vadr],
                    model.actuator_ctrlrange, kp=SCENE_PD_KP, kd=SCENE_PD_KD
                )
                data.ctrl[:] = torque
                mujoco.mj_step2(model, data)
                step_index += 1
                if args.scene == "scoop_transfer":
                    if picked_mask is None and data.time >= SCOOP_PICKUP_TIME - 1e-9:
                        picked_mask = spoon_inside(data, pitch_id, food).copy()
                        print(f"Pickup after lift: {picked_mask.sum()}/{particle_count} grains.")
                    if unload_mask is None and data.time >= SCOOP_UNLOAD_TIME - 1e-9:
                        unload_mask = spoon_inside(data, pitch_id, food).copy()
                        retained = int((picked_mask & unload_mask).sum())
                        print(f"Retained before unloading: {retained}/{picked_mask.sum()} picked grains.")

            viewer.sync()
            frame_count += 1
            if frame_count % status_interval == 0:
                retained = int(spoon_inside(data, pitch_id, food).sum())
                print(f"t={data.time:5.2f} s | {phase:28s} | particles in spoon region={retained:3d}/{particle_count}")
            frame_duration = steps_per_frame * dt / args.speed
            time.sleep(max(0.0, frame_duration - (time.monotonic() - start_wall)))

        if args.scene == "scoop_transfer":
            metrics = scoop_task_metrics(data, pitch_id, food, picked_mask, unload_mask)
            print(f"Stopped at t={data.time:.2f}s: picked={metrics['pickup_count']}, "
                  f"transferred picked grains={metrics['transferred_pickup_count']}, "
                  f"left in source={metrics['remaining_in_source_count']}.")
            if args.metrics_json:
                record = dict(
                    completed=bool(data.time >= duration-dt/2), simulation_time_s=float(data.time),
                    config=dict(scene=args.scene, condition=args.condition, severity=args.severity,
                                tremor_scale=alpha, seed=args.seed, particle_count=particle_count,
                                bowl_depth_m=args.bowl_depth_mm/1000,
                                physics_dt_s=dt, duration_s=duration,
                                pickup_time_s=SCOOP_PICKUP_TIME, unload_time_s=SCOOP_UNLOAD_TIME,
                                spoon_world_waypoints=SCOOP_WAYPOINTS,
                                contact_solref=SCENE_CONTACT_SOLREF, contact_solimp=SCENE_CONTACT_SOLIMP,
                                friction=SCENE_CONTACT_FRICTION, solver_iterations=int(model.opt.iterations),
                                solver_tolerance=float(model.opt.tolerance),
                                food_contact_dimensions=3, spoon_friction=SPOON_CONTACT_FRICTION,
                                source_outer_diameter_m=.061, source_depth_m=SCOOP_SOURCE_DEPTH,
                                source_inner_diameter_m=.055, source_floor_height_m=.104,
                                source_shape="flat base and 12 shallow wall boxes",
                                particle_radius_m=.002, particle_mass_kg=.0000369,
                                particle_settling_duration_s=1.6,
                                table_top_height_m=.100, table_front_edge_x_m=.095,
                                spoon_center_thickness_m=.0012, spoon_rim_thickness_m=.0003,
                                handle_drive="dynamic_handle_welded_to_mocap_target",
                                weld_solref=HANDLE_WELD_SOLREF, weld_torque_scale=HANDLE_WELD_TORQUE_SCALE,
                                joint_kp=list(SCENE_PD_KP), joint_kd=list(SCENE_PD_KD),
                                joint_limits_rad=limits.tolist(),
                                torque_limits_Nm=model.actuator_ctrlrange.tolist()),
                    metrics=metrics,
                )
                args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                args.metrics_json.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
                print(f"Saved task metrics to {args.metrics_json}")

        # Leave the completed physical scene visible until the user closes it.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def _replay_phase2a(viewer_module, model, data, mocap_id, q_roll, q_pitch, food,
                    method, seed, speed):
    from .system_benchmark_phase2a import OUT

    trace_path = OUT / "controller_traces" / method / f"seed{seed:02d}.npz"
    if not trace_path.is_file():
        raise FileNotFoundError(f"No saved Phase 2A trace: {trace_path}")
    with np.load(trace_path) as trace:
        times = trace["time_s"].copy()
        values = trace["values"].copy()
        columns = [str(x) for x in trace["columns"]]
        particles = trace["particle_positions"].copy()
    c = {name: i for i, name in enumerate(columns)}
    print(f"Replaying saved MuJoCo states: {method}, seed={seed}, {len(times)} frames, {speed:g}x.")
    print("This replays recorded rigid-body states in the native CAD viewer; close the window to exit.")
    with viewer_module.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.18, 0.0, 0.14]
        viewer.cam.distance = 0.62
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
        for frame, (t, row) in enumerate(zip(times, values)):
            if not viewer.is_running():
                return
            data.mocap_pos[mocap_id] = [row[c["x"]], row[c["y"]], row[c["z"]]]
            data.mocap_quat[mocap_id] = roll_pitch_quaternion(
                row[c["vol_roll"]] + row[c["tremor_roll_angle"]],
                row[c["vol_pitch"]] + row[c["tremor_pitch_angle"]])
            data.qpos[q_roll] = row[c["q_roll"]]
            data.qpos[q_pitch] = row[c["q_pitch"]]
            for i, address in enumerate(food):
                data.qpos[address:address + 3] = particles[frame, i]
                data.qpos[address + 3:address + 7] = [1., 0., 0., 0.]
            mujoco.mj_forward(model, data)
            viewer.sync()
            if frame + 1 < len(times):
                time.sleep(max(0., (times[frame + 1] - t) / speed))
        while viewer.is_running():
            viewer.sync()
            time.sleep(.02)


def _replay_corrected(path, speed):
    """Display saved physical states; no estimator rerun or video encoding."""
    from mujoco import viewer as mjviewer
    with np.load(path, allow_pickle=False) as saved:
        required = ('replay_time_s', 'handle_qpos', 'joint_qpos', 'particle_positions',
                    'mocap_pos', 'mocap_quat')
        if any(k not in saved for k in required):
            raise ValueError('Trace has no corrected-scene replay states')
        state = {k: saved[k].copy() for k in required}
        method = str(saved['method']); seed = int(saved['seed'])
        config = json.loads(str(saved['config_json']))
    count = state['particle_positions'].shape[1]
    model, temporary = load_scene_model(count, bowl_depth_m=config['plant'].get('bowl_depth_m', .008))
    Path(temporary).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    _, _, mid, food = initialize(model, data, seed, particle_count=count)
    qa = [joint(model, name)[0] for name in ('roll', 'pitch')]
    # Forming walls are a settling aid; all recorded task states have them disabled.
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
        if name.startswith('settling_wall_'):
            model.geom_contype[i] = model.geom_conaffinity[i] = 0
            model.geom_rgba[i, 3] = 0
    print(f'Replaying MuJoCo states: {method}, seed={seed}, {count} grains. Close window to exit.')
    with mjviewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 1; viewer.opt.geomgroup[4] = 0
        viewer.cam.lookat[:] = [0.18, 0., 0.12]
        viewer.cam.distance = .70; viewer.cam.azimuth = 90; viewer.cam.elevation = -30
        times = state['replay_time_s']
        for frame, t in enumerate(times):
            if not viewer.is_running():
                break
            started = time.monotonic()
            data.time = float(t); data.qpos[:7] = state['handle_qpos'][frame]
            data.qpos[qa] = state['joint_qpos'][frame]
            data.mocap_pos[mid] = state['mocap_pos'][frame]
            data.mocap_quat[mid] = state['mocap_quat'][frame]
            for i, address in enumerate(food):
                data.qpos[address:address+3] = state['particle_positions'][frame, i]
                data.qpos[address+3:address+7] = [1., 0., 0., 0.]
            mujoco.mj_forward(model, data); viewer.sync()
            if frame+1 < len(times):
                time.sleep(max(0., (times[frame+1]-t)/speed - (time.monotonic()-started)))
        while viewer.is_running():
            viewer.sync(); time.sleep(.02)


if __name__ == "__main__":
    main()
