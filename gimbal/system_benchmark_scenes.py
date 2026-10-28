"""Prescribed task scenes for the interactive MuJoCo spoon viewer.

These scenes are visualization/task demonstrations. They do not replace the
frozen system benchmark model or its formal experiment configuration.
"""
from pathlib import Path
import math
import os
import re
import tempfile

import mujoco
import numpy as np

from .motion import roll_pitch_quaternion, roll_pitch_rotation
from .system_benchmark import MODEL, NFOOD, joint, mj, spoon_inside, voluntary


TABLE_AND_SOURCE = r'''
    <!-- Tabletop and furniture are isolated to the interactive task-scene model. -->
    <!-- Keep the bulky handle in front of the table; only the spoon enters. -->
    <geom name="scene_table_top" type="box" pos="0.3275 0 0.078" size="0.2325 0.28 0.022"
          contype="4" conaffinity="3" group="3" rgba="0.48 0.29 0.16 1" friction="0.8 0.01 0.001"/>
    <geom name="scene_table_leg_xn_yn" type="box" pos="0.12 -0.23 0.038" size="0.018 0.018 0.038" group="3" rgba="0.30 0.20 0.13 1"/>
    <geom name="scene_table_leg_xn_yp" type="box" pos="0.12 0.23 0.038" size="0.018 0.018 0.038" group="3" rgba="0.30 0.20 0.13 1"/>
    <geom name="scene_table_leg_xp_yn" type="box" pos="0.52 -0.23 0.038" size="0.018 0.018 0.038" group="3" rgba="0.30 0.20 0.13 1"/>
    <geom name="scene_table_leg_xp_yp" type="box" pos="0.52 0.23 0.038" size="0.018 0.018 0.038" group="3" rgba="0.30 0.20 0.13 1"/>
    <body name="scene_source_bowl" pos="0.137 0 0">
      <geom name="source_bowl_bottom" type="cylinder" pos="0 0 0.102" size="0.0305 0.002"
            contype="4" conaffinity="3" group="3" rgba="0.18 0.57 0.49 1" friction="0.7 0.01 0.001"/>
      {source_rim}
    </body>
'''

BOWL_A, BOWL_B, BOWL_DEPTH, BOWL_THICKNESS, BOWL_X = 0.024, 0.018, 0.008, 0.0012, 0.064
BOWL_BOTTOM = -0.002
FOOD_RADIUS = 0.002
BOWL_CAP_RATIO = .65
BOWL_RIM_THICKNESS = .0003
# MuJoCo direct-format reference stiffness/damping (not material N/m).
# Overdamped contact avoids storing and releasing large collision impulses.
SCENE_CONTACT_SOLREF = "-100000 -900"
SCENE_CONTACT_SOLIMP = "0.90 0.98 0.0005"
# Last two coefficients are length scales for spin/rolling resistance. They
# must be small relative to the 2 mm grain radius, including on the utensils.
SCENE_CONTACT_FRICTION = "0.6 0.0001 0.00005"
SPOON_CONTACT_FRICTION = "0.2 0.0001 0.00005"
# Critical damping from the neutral fixed-base inertia diagonal:
# I_roll=4.45e-4, I_pitch=3.62e-5 kg m^2, Kd=2*sqrt(Kp*I).
SCENE_PD_KP = (5., 5.)
SCENE_PD_KD = (.095, .027)
HANDLE_WELD_SOLREF = "0.002 1"
HANDLE_WELD_TORQUE_SCALE = 1.
SCOOP_DEFAULT_PARTICLES = 180
SCOOP_SOURCE_DEPTH = .006
SCOOP_TIME_SCALE = 1.75
SCOOP_DURATION = 12.*SCOOP_TIME_SCALE
SCOOP_PICKUP_TIME = 6.4*SCOOP_TIME_SCALE
SCOOP_UNLOAD_TIME = 9.*SCOOP_TIME_SCALE


def _bowl_height(rho2, depth=BOWL_DEPTH):
    k2 = BOWL_CAP_RATIO ** 2
    return BOWL_BOTTOM + depth * (1-math.sqrt(max(1-k2*rho2, 0.))) / (1-math.sqrt(1-k2))


def _bowl_thickness(r):
    return BOWL_THICKNESS + (BOWL_RIM_THICKNESS-BOWL_THICKNESS)*r*r


def _convex_faces(vertices):
    """Small convex hulls using NumPy only; triangulate each supporting plane."""
    from itertools import combinations
    planes = {}
    for a, b, c in combinations(range(len(vertices)), 3):
        normal = np.cross(vertices[b]-vertices[a], vertices[c]-vertices[a])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        distance = (vertices-vertices[a]) @ normal
        if np.max(distance) <= 1e-10:
            pass
        elif np.min(distance) >= -1e-10:
            normal = -normal
        else:
            continue
        offset = normal @ vertices[a]
        planes[tuple(np.round(np.r_[normal, offset], 8))] = (normal, offset)
    faces = []
    for normal, offset in planes.values():
        ids = np.flatnonzero(abs(vertices@normal-offset) < 1e-9)
        center = vertices[ids].mean(axis=0)
        u = vertices[ids[0]]-center
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        coords = vertices[ids]-center
        ids = ids[np.argsort(np.arctan2(coords@v, coords@u))]
        for i in range(1, len(ids)-1):
            a, b, c = ids[0], ids[i], ids[i+1]
            if np.linalg.norm(np.cross(vertices[b]-vertices[a], vertices[c]-vertices[a])) > 1e-12:
                faces.append((a, b, c))
    return faces


def _write_convex_bowl_segments(directory, nsector=12, depth=BOWL_DEPTH):
    """24 small convex solids approximate the concave shell (no hidden tray)."""
    assets, geoms = [], []
    for sector in range(nsector):
        center = 2 * math.pi * (sector + 0.5) / nsector
        angles = np.linspace(center - math.pi / nsector - .003,
                             center + math.pi / nsector + .003, 3)
        for band, (r0, r1) in enumerate(((0., .62), (.60, 1.))):
            verts = []
            for r in (r0, r1):
                z = _bowl_height(r*r, depth)
                for th in angles:
                    for dz in (0., -_bowl_thickness(r)):
                        verts.append((BOWL_X + BOWL_A*r*math.cos(th),
                                      BOWL_B*r*math.sin(th), z+dz))
            verts = np.unique(np.round(verts, 10), axis=0)
            faces = _convex_faces(verts)
            name = f"bowl_segment_{sector:02d}_{band}"
            path = Path(directory) / f"{name}.obj"
            with path.open('w') as f:
                for v in verts:
                    f.write(f'v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n')
                for a, b, c in faces:
                    f.write(f'f {a+1} {b+1} {c+1}\n')
            assets.append(f'<mesh name="{name}" file="{path}"/>')
            geoms.append(f'<geom name="{name}_collision" type="mesh" mesh="{name}" '
                         'contype="1" conaffinity="6" group="4" '
                         'rgba="0 0 0 0" friction="0.7 0.01 0.001" '
                         'condim="3" solref="0.006 1"/>')
    return '\n'.join(assets), '\n'.join(geoms)


def _source_geometry(depth=SCOOP_SOURCE_DEPTH):
    rim = []
    for i in range(12):
        a, b = 2*math.pi*i/12, 2*math.pi*(i+1)/12
        th = (a+b)/2
        rim.append(f'<geom name="source_rim_{i:02d}" type="box" '
                   f'pos="{.0285*math.cos(th)} {.0285*math.sin(th)} {.104+depth/2}" '
                   f'euler="0 0 {th}" size=".001 .0078 {depth/2}" '
                   'contype="4" conaffinity="3" group="3" rgba="0.18 0.57 0.49 1" '
                   'friction="0.7 0.01 0.001"/>')
    for i in range(12):
        th = 2*math.pi*(i+.5)/12
        rim.append(f'<geom name="settling_wall_{i:02d}" type="box" '
                   f'pos="{.0285*math.cos(th)} {.0285*math.sin(th)} .119" '
                   f'euler="0 0 {th}" size=".002 .0078 .015" '
                   'contype="0" conaffinity="0" group="4" rgba="0 0 0 0"/>')
    xml = TABLE_AND_SOURCE.replace('{source_rim}', '\n'.join(rim))
    return xml


def _write_bowl_obj(path, depth=BOWL_DEPTH):
    """Render an ellipsoidal cap with finite rim slope, like a spoon bowl."""
    nt, nr = 64, 14
    verts = [(BOWL_X, 0.0, BOWL_BOTTOM),
             (BOWL_X, 0.0, BOWL_BOTTOM - BOWL_THICKNESS)]
    inner_rings, outer_rings = [], []
    for j in range(1, nr + 1):
        r = j / nr
        inner, outer = [], []
        for k in range(nt):
            th = 2 * math.pi * k / nt
            inner.append(len(verts) + 1)
            inner_z = _bowl_height(r*r, depth)
            verts.append((BOWL_X + BOWL_A * r * math.cos(th),
                          BOWL_B * r * math.sin(th), inner_z))
            outer.append(len(verts) + 1)
            thickness = _bowl_thickness(r)
            verts.append((BOWL_X + (BOWL_A + thickness) * r * math.cos(th),
                          (BOWL_B + thickness) * r * math.sin(th),
                          inner_z - thickness))
        inner_rings.append(inner)
        outer_rings.append(outer)
    faces = []
    for k in range(nt):
        kn = (k + 1) % nt
        faces.append((1, inner_rings[0][k], inner_rings[0][kn]))
        faces.append((2, outer_rings[0][kn], outer_rings[0][k]))
    for j in range(nr - 1):
        a, b = inner_rings[j], inner_rings[j + 1]
        oa, ob = outer_rings[j], outer_rings[j + 1]
        for k in range(nt):
            kn = (k + 1) % nt
            faces.extend(((a[k], b[k], b[kn]), (a[k], b[kn], a[kn])))
            faces.extend(((oa[k], ob[kn], ob[k]), (oa[k], oa[kn], ob[kn])))
    a, b = inner_rings[-1], outer_rings[-1]
    for k in range(nt):
        kn = (k + 1) % nt
        faces.extend(((a[k], b[k], b[kn]), (a[k], b[kn], a[kn])))
    with Path(path).open('w') as f:
        f.write('# Procedural ellipsoidal-cap spoon shell, metres\n')
        for v in verts:
            f.write(f'v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n')
        for face in faces:
            f.write('f ' + ' '.join(map(str, face)) + '\n')


def load_scene_model(particle_count=NFOOD, *, source_depth_m=SCOOP_SOURCE_DEPTH,
                     dynamic_handle=True, bowl_depth_m=BOWL_DEPTH):
    """Load a viewer-only model with an ellipsoidal spoon bowl, table and vessels."""
    if particle_count < 1:
        raise ValueError("particle_count must be positive")
    if not math.isfinite(bowl_depth_m) or not .002 <= bowl_depth_m <= .012:
        raise ValueError("bowl_depth_m must be between 0.002 and 0.012 m")
    xml = MODEL.read_text()
    if dynamic_handle:
        # A directly moved mocap body has zero velocity in contact dynamics.
        # Weld a real free handle to a separate mocap target so friction sees
        # the actual utensil velocity. The two gimbal actuators are unchanged.
        xml = xml.replace('<body name="handle" mocap="true">',
                          '<body name="handle"><freejoint name="handle_free"/>')
        xml = xml.replace('</worldbody>',
                          '<body name="handle_drive" mocap="true"/>\n</worldbody>')
        xml = xml.replace('</worldbody>', '</worldbody>\n<equality>'
                          '<weld name="handle_drive_weld" body1="handle" body2="handle_drive" '
                          f'relpose="0 0 0 1 0 0 0" solref="{HANDLE_WELD_SOLREF}" '
                          f'solimp="0.95 0.99 0.001" torquescale="{HANDLE_WELD_TORQUE_SCALE}"/>'
                          '</equality>')
    meshdir = (MODEL.parent / "cad_spoon" / "meshes").resolve()
    xml = xml.replace('meshdir="cad_spoon/meshes"', f'meshdir="{meshdir}"')
    xml = re.sub(r'\s*<geom name="spoon_head_visual"[^>]*/>', '', xml)
    xml = re.sub(r'\s*<geom name="bowl_floor_proxy"[^>]*/>.*?'
                 r'<geom name="bowl_wall_yp"[^>]*/>', '', xml, flags=re.S)
    xml = xml.replace('iterations="80" tolerance="1e-10"', 'solver="CG" iterations="70" tolerance="1e-6"')
    xml = xml.replace('contype="4" conaffinity="2"', 'contype="4" conaffinity="3"')
    meshdir = tempfile.mkdtemp(prefix="mujoco_spoon_bowl_")
    obj_path = Path(meshdir) / "half_ellipsoid_visual.obj"
    _write_bowl_obj(obj_path, depth=bowl_depth_m)
    collision_assets, collision_geoms = _write_convex_bowl_segments(meshdir, depth=bowl_depth_m)
    xml = xml.replace('</asset>',
                      f'<mesh name="scene_half_ellipsoid_bowl" file="{obj_path}"/>\n'
                      f'{collision_assets}\n  </asset>')
    shell = ('<geom name="half_ellipsoid_bowl_visual" type="mesh" '
             'mesh="scene_half_ellipsoid_bowl" contype="0" conaffinity="0" '
             'group="1" rgba="0.93 0.67 0.29 1"/>\n' + collision_geoms)
    xml = xml.replace('<site name="spoon_tip" pos="0.081 0 0" size="0.003" rgba="1 0.2 0.2 1"/>',
                      '<site name="spoon_tip" pos="0.081 0 0" size="0.003" rgba="1 0.2 0.2 1"/>\n' + shell)
    xml = xml.replace('type="sphere" size="0.0032" mass="0.00015"',
                      'type="sphere" size="0.002" mass="0.0000369"')
    xml = xml.replace('</worldbody>', _source_geometry(source_depth_m) + '\n  </worldbody>')
    if particle_count < NFOOD:
        for i in range(particle_count, NFOOD):
            xml = re.sub(r'\s*<body name="food_' + f'{i:02d}' + r'".*?</body>', '', xml, flags=re.S)
    if particle_count > NFOOD:
        match = re.search(r'(<body name="food_00".*?</body>)', xml, flags=re.S)
        if match is None:
            raise RuntimeError("Could not find the template food body in the MuJoCo model")
        template = match.group(1)
        extra = []
        for i in range(NFOOD, particle_count):
            extra.append(template.replace("food_00", f"food_{i:02d}"))
        xml = xml.replace('</worldbody>', '\n'.join(extra) + '\n  </worldbody>')
    # Set both sides of every physical contact consistently: MuJoCo otherwise
    # takes the larger spin/rolling friction coefficient from either geom.
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    custom = root.find('custom')
    if custom is None:
        custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='scene_bowl_depth_m', data=str(bowl_depth_m))
    for geom in root.iter("geom"):
        if int(geom.get("contype", "0")) or geom.get("name", "").startswith("settling_wall_"):
            geom.set("solref", SCENE_CONTACT_SOLREF)
            geom.set("solimp", SCENE_CONTACT_SOLIMP)
            geom.set("friction", SCENE_CONTACT_FRICTION)
            geom.set("margin", "0.0001")
            if geom.get("name", "").startswith("food_"):
                geom.set("condim", "3")
            elif geom.get("name", "").startswith("bowl_segment_"):
                geom.set("priority", "1")
                geom.set("friction", SPOON_CONTACT_FRICTION)
                geom.set("condim", "3")
    xml = ET.tostring(root, encoding="unicode")
    fd, path = tempfile.mkstemp(prefix="mujoco_spoon_scenes_", suffix=".xml")
    os.close(fd)
    Path(path).write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(path)
        import shutil
        shutil.rmtree(meshdir, ignore_errors=True)
        return model, path
    except Exception:
        Path(path).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(meshdir, ignore_errors=True)
        raise


def initialize_ellipsoid_bowl_particles(model, data, pitch_body, food_addresses, seed=0):
    """Place demo particles on the curved ellipsoidal-cap interior."""
    p = data.xpos[pitch_body].copy()
    R = data.xmat[pitch_body].reshape(3, 3).copy()
    candidates = []
    for y in np.arange(-0.012, 0.0121, 0.004):
        for x in np.arange(-0.020, 0.0201, 0.004):
            rho2 = (x / BOWL_A) ** 2 + (y / BOWL_B) ** 2
            if rho2 <= 0.72:
                candidates.append((x, y, rho2))
    for i, addr in enumerate(food_addresses):
        x, y, rho2 = candidates[i % len(candidates)]
        layer = i // len(candidates)
        depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, 'scene_bowl_depth_m')
        depth = model.numeric_data[model.numeric_adr[depth_id]] if depth_id >= 0 else BOWL_DEPTH
        z = (_bowl_height(rho2, depth)
             + FOOD_RADIUS + 0.0001 + layer * (2 * FOOD_RADIUS))
        data.qpos[addr:addr + 3] = p + R @ np.array([BOWL_X + x, y, z])
        data.qpos[addr + 3:addr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


SCOOP_WAYPOINTS = [
    (0.0, [.075, 0, .155], .70, "settled_source"),
    (0.6, [.075, 0, .155], .70, "above_rim_approach"),
    (1.8, [.097, 0, .130], .70, "diagonal_entry"),
    (2.6, [.105, 0, .1145], .70, "angled_scoop_sweep"),
    (3.6, [.130, 0, .1145], .70, "level_and_collect"),
    (4.4, [.137, 0, .1058], 0., "settle_in_spoon"),
    (4.6, [.137, 0, .1058], 0., "lift_from_source"),
    (6.4, [.137, 0, .240], 0., "transport_to_target"),
    (9.0, [.340, 0, .225], 0., "intentional_pitch_unload"),
    (10.5, [.340, 0, .225], .95, "unload_hold"),
    (12.0, [.340, 0, .225], .95, "unload_hold"),
]
SCOOP_WAYPOINTS = [(t*SCOOP_TIME_SCALE, p, pitch, phase)
                   for t,p,pitch,phase in SCOOP_WAYPOINTS]


def scene_duration(scene):
    return SCOOP_DURATION if scene == "scoop_transfer" else 12.


def scene_trajectory(t: float, scene: str):
    """Return handle position, desired roll/pitch and phase for a scene."""
    if scene in ("pitch_pour", "roll_pour"):
        p, rp, _, _, phase = voluntary(t)
        ramp = mj((t - 9.0) / 2.0)
        if scene == "pitch_pour":
            rp = np.array([0.0, 0.95 * ramp])
        else:
            rp = np.array([0.95 * ramp, 0.0])
        return p, rp, phase

    if scene != "scoop_transfer":
        raise ValueError(f"Unknown task scene: {scene}")

    # Specify the bowl bottom center in world coordinates. Moving a 137 mm
    # offset with a pitched handle otherwise sends the spoon below the table.
    # The desired pose assumes neutral gimbal joints, as does NoComp's command.
    points = SCOOP_WAYPOINTS
    center, pitch, phase = np.array(points[-1][1]), points[-1][2], points[-1][3]
    for (ta, a, pa, label), (tb, b, pb, _) in zip(points[:-1], points[1:]):
        if ta <= t < tb:
            u = mj((t-ta)/(tb-ta))
            center = np.asarray(a) + u*(np.asarray(b)-a)
            pitch, phase = pa+u*(pb-pa), label
            break
    rp = np.array([0., pitch])
    pos = center - roll_pitch_rotation(*rp) @ np.array([.073+BOWL_X, 0., BOWL_BOTTOM])
    return pos, rp, phase


def initialize_source_particles(model, data, seed, food_addresses):
    """Drop non-overlapping random grains, then let real contacts settle them."""
    rng = np.random.default_rng(seed)
    positions = []
    for _ in food_addresses:
        for attempt in range(20000):
            r = .022 * math.sqrt(rng.random())
            angle = rng.uniform(0, 2*math.pi)
            candidate = np.array([.137+r*math.cos(angle), r*math.sin(angle),
                                  rng.uniform(.1065, .128)])
            if all(np.linalg.norm(candidate-p) >= 2*FOOD_RADIUS+.0001 for p in positions):
                positions.append(candidate)
                break
        else:
            raise RuntimeError("Could not place non-overlapping grains; reduce --particles")
    for address, position in zip(food_addresses, positions):
        data.qpos[address:address+3] = position
        data.qpos[address+3:address+7] = [1., 0., 0., 0.]
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle")
    mid = int(model.body_mocapid[hid])
    if mid < 0:
        drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle_drive")
        mid = int(model.body_mocapid[drive])
    p, rp, _ = scene_trajectory(0., "scoop_transfer")
    data.mocap_pos[mid] = p
    data.mocap_quat[mid] = roll_pitch_quaternion(*rp)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handle_free")
    if free >= 0:
        qa, va = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
        data.qpos[qa:qa+3] = p
        data.qpos[qa+3:qa+7] = data.mocap_quat[mid]
        data.qvel[va:va+6] = 0.
    q = np.array([joint(model, name)[0] for name in ("roll", "pitch")])
    v = np.array([joint(model, name)[1] for name in ("roll", "pitch")])
    mujoco.mj_forward(model, data)
    peak_food_contacts = 0
    food_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"food_{i:02d}_geom")
                for i in range(len(food_addresses))}
    settling_walls = [i for i in range(model.ngeom) if
                      (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("settling_wall_")]
    model.geom_contype[settling_walls] = 4
    model.geom_conaffinity[settling_walls] = 2
    for k in range(round(1.6 / model.opt.timestep)):
        if k == round(1.2 / model.opt.timestep):
            model.geom_contype[settling_walls] = 0
            model.geom_conaffinity[settling_walls] = 0
        mujoco.mj_step1(model, data)
        data.ctrl[:] = np.clip(-np.asarray(SCENE_PD_KP)*data.qpos[q]
                               -np.asarray(SCENE_PD_KD)*data.qvel[v]+data.qfrc_bias[v],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step2(model, data)
        if k % 20 == 0:
            peak_food_contacts = max(peak_food_contacts, sum(
                c.geom1 in food_ids and c.geom2 in food_ids for c in data.contact))
    data.time = 0.
    mujoco.mj_forward(model, data)
    return {"settling_s": 1.6, "peak_grain_grain_contacts": int(peak_food_contacts)}


def scoop_task_metrics(data, pitch_body, food_addresses, picked_mask, unload_mask):
    """Count transported food relative to the serving at the pickup checkpoint.

    The masks identify individual grains at fixed checkpoints, so grains left
    in the source do not count as transport spill, and a grain that was never
    picked up cannot receive credit for a successful transfer.
    """
    positions = np.array([data.qpos[a:a+3] for a in food_addresses])
    target = ((abs(positions[:, 0]-.34) < .060) & (abs(positions[:, 1]) < .060)
              & (positions[:, 2] > .107) & (positions[:, 2] < .207))
    source = ((np.linalg.norm(positions[:, :2]-[.137, 0], axis=1) < .0325)
              & (positions[:, 2] > .102) & (positions[:, 2] < .130) & ~target)
    spoon = spoon_inside(data, pitch_body, food_addresses) & ~target & ~source
    picked = None if picked_mask is None else int(picked_mask.sum())
    retained = (None if picked_mask is None or unload_mask is None
                else int((picked_mask & unload_mask).sum()))
    transferred = None if picked_mask is None else int((picked_mask & target).sum())
    return dict(
        initial_source_count=len(food_addresses),
        pickup_count=picked,
        pickup_fraction=None if picked is None else picked/len(food_addresses),
        retained_before_unload_count=retained,
        transport_retention_ratio=None if not picked or retained is None else retained/picked,
        transferred_pickup_count=transferred,
        transfer_ratio_of_picked=None if not picked else transferred/picked,
        transport_spill_count=None if picked is None or retained is None else picked-retained,
        remaining_in_source_count=int(source.sum()),
        in_target_count=int(target.sum()),
        remaining_in_spoon_count=int(spoon.sum()),
        elsewhere_count=int((~(source | target | spoon)).sum()),
        picked_particle_ids=None if picked_mask is None else np.flatnonzero(picked_mask).tolist(),
    )
