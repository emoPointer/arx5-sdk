"""Offline X5 geometric collision screening; NEVER imports SDK or opens CAN.

Inputs: official arm URDF, articulated gripper URDF and their mesh assets;
        joint configurations/trajectories in SDK radians, gripper width in m.
Outputs: a NEW output directory containing model/provenance and screening JSON.
Prerequisites: conda_environments/requirements_pace_collision.txt in an isolated
environment. From any directory:
  python /path/to/sdk/python/pace_collision.py --out /path/to/new/output
Optional --arm-urdf /path/model.urdf --gripper-urdf /path/gripper.urdf
and --floor-z 0 set geometry inputs/support plane relative to the arm base.
Does not change source models or overwrite output directories.

Uses full triangle meshes, with convex hull broad phase, instead of the SDK's
simplified collision cylinders. Parent-child assembly pairs and the closing finger pair are excluded
explicitly; all other robot pairs and link-to-support-plane pairs are checked.
This is a geometric model screen, NOT a hardware safety certification or fitted
dynamics simulation. Cameras, cables, fixtures and payloads absent from supplied
meshes are not represented. All trajectory checks are sampled, not continuous proofs.
"""

import argparse
import hashlib
import itertools
import json
import time
import sys
from concurrent.futures import ProcessPoolExecutor
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def numbers(text, default):
    return np.asarray([float(x) for x in (text or default).split()])


def origin(element):
    node = element.find('origin')
    xyz = numbers(None if node is None else node.get('xyz'), '0 0 0')
    rpy = numbers(None if node is None else node.get('rpy'), '0 0 0')
    xyzw = Rotation.from_euler('xyz', rpy).as_quat()
    return dict(pos=' '.join(map(str, xyz)), quat=' '.join(map(str, xyzw[[3, 0, 1, 2]])))


class CollisionModel:
    def __init__(self, arm_urdf=None, gripper_urdf=None, floor_z=0.):
        self.arm_path = Path(arm_urdf or ROOT/'models/X5.urdf').resolve()
        self.gripper_path = Path(gripper_urdf or ROOT/'models/X5_gripper.urdf').resolve()
        arm = ET.parse(self.arm_path).getroot()
        gripper = ET.parse(self.gripper_path).getroot()
        links = {link.get('name'): (link, self.arm_path.parent) for link in arm.findall('link')}
        links.update({link.get('name'): (link, self.gripper_path.parent) for link in gripper.findall('link')})
        joints = arm.findall('joint')+gripper.findall('joint')
        child_names = {j.find('child').get('link') for j in joints}
        roots = set(links)-child_names
        if roots != {'base_link'}:
            raise ValueError(f'Expected base_link root, got {roots}')
        self.adjacent = {frozenset((j.find('parent').get('link'), j.find('child').get('link'))) for j in joints}
        self.adjacent.add(frozenset(('link7', 'link8')))
        self.sources = {str(self.arm_path): hashlib.sha256(self.arm_path.read_bytes()).hexdigest(),
                        str(self.gripper_path): hashlib.sha256(self.gripper_path.read_bytes()).hexdigest()}
        self.geometry = {}
        self.floor_z = floor_z
        doc = ET.Element('mujoco', model='pace_x5_full_geometry_screen')
        ET.SubElement(doc, 'compiler', angle='radian', autolimits='true', fusestatic='false')
        ET.SubElement(doc, 'option', timestep='0.002', gravity='0 0 -9.807')
        ET.SubElement(doc, 'visual').append(ET.Element('global', offwidth='1200', offheight='900'))
        asset = ET.SubElement(doc, 'asset')
        world = ET.SubElement(doc, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -1 2', dir='0 0 -1', diffuse='0.8 0.8 0.8')
        ET.SubElement(world, 'geom', name='support_plane', type='plane', pos=f'0 0 {floor_z}',
                      size='2 2 .02', rgba='.55 .55 .55 1', contype='0', conaffinity='0')

        def mesh_file(directory, filename):
            path = directory/filename
            if not path.is_file():
                path = ROOT/'models'/filename
            if not path.is_file() and Path(filename).name == 'base_link.STL':
                path = ROOT/'models/meshes/X5/base_link.STL'
            return path.resolve(strict=True)

        def add_link(parent, name, joint=None):
            body = ET.SubElement(parent, 'body', name=name, **(origin(joint) if joint is not None else {}))
            if joint is not None and joint.get('type') != 'fixed':
                joint_type = {'revolute': 'hinge', 'continuous': 'hinge', 'prismatic': 'slide'}[joint.get('type')]
                ET.SubElement(body, 'joint', name=joint.get('name'), type=joint_type,
                              axis=joint.find('axis').get('xyz'), limited='false', damping='0')
            link, directory = links[name]
            for k, visual in enumerate(link.findall('visual')):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError(f'{name}: full geometry requires mesh visual')
                path = mesh_file(directory, mesh.get('filename'))
                mesh_name = f'{name}_mesh_{k}'
                ET.SubElement(asset, 'mesh', name=mesh_name, file=str(path), scale=mesh.get('scale', '1 1 1'))
                self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.geometry[f'{name}_hull_{k}'] = (path, origin(visual), mesh.get('scale', '1 1 1'))
                ET.SubElement(body, 'geom', name=f'{name}_hull_{k}', type='mesh', mesh=mesh_name,
                              rgba='.72 .75 .8 1' if name == 'link3' else '.22 .25 .3 1',
                              contype='0', conaffinity='0', density='1000', **origin(visual))
            for child_joint in joints:
                if child_joint.find('parent').get('link') == name:
                    add_link(body, child_joint.find('child').get('link'), child_joint)

        add_link(world, 'base_link')
        self.xml = ET.tostring(doc, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}') for i in range(1, 9)]
        self.qaddr = self.model.jnt_qposadr[self.joint_ids]
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in self.model.geom_bodyid]
        self.pairs = []
        self.excluded = []
        for a, b in itertools.combinations(range(self.model.ngeom), 2):
            ba, bb = self.body_names[a], self.body_names[b]
            if ba == bb or frozenset((ba, bb)) in self.adjacent or {ba, bb} == {'world', 'base_link'}:
                self.excluded.append([self.geom_names[a], self.geom_names[b]])
            else:
                self.pairs.append((a, b))

    def set_state(self, q, width=0.):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or not 0 <= width <= .088:
            raise ValueError('Need six finite radians and gripper width 0..0.088 m')
        self.data.qpos[self.qaddr[:6]] = q
        self.data.qpos[self.qaddr[6:]] = width/2
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def distances(self, q, width=0., distance_cap=.20):
        self.set_state(q, width)
        distances = np.array([mujoco.mj_geomDistance(self.model, self.data, a, b, distance_cap, None)
                              for a, b in self.pairs])
        return distances

    def inspect(self, q, width=0.):
        d = self.distances(q, width)
        order = np.argsort(d)
        return {'q': list(map(float, q)), 'width': width, 'minimum_distance_m': float(d[order[0]]),
                'closest': [{'pair': [self.geom_names[i] for i in self.pairs[k]], 'distance_m': float(d[k])}
                            for k in order[:5]]}

    def provenance(self):
        return {'mujoco': mujoco.__version__, 'source_sha256': self.sources,
                'collision_geometry': 'full visual mesh convex hulls, SDK URDF kinematics',
                'excluded_assembly_pairs': self.excluded,
                'checked_pairs': [[self.geom_names[a], self.geom_names[b]] for a, b in self.pairs],
                'limitations': ['not fitted dynamic rollout', 'not continuous-time collision proof',
                                'no camera/cable/payload/fixture geometry', 'convex hull false positives possible',
                                'adjacent assembly interference is not certified']}


def initialize_reference(model, path):
    """Load CLab collision meshes read-only, relocating its base to the origin."""
    path = Path(path).resolve()
    doc = ET.parse(path).getroot()
    compiler = doc.find('compiler')
    meshdir = (path.parent/compiler.get('meshdir', '')).resolve()
    compiler.set('meshdir', str(meshdir))
    assets = {m.get('name'): meshdir/m.get('file') for m in doc.findall('asset/mesh')}
    world = doc.find('worldbody')
    root = world.find('body')
    root.set('pos', '0 0 0')
    root.set('quat', '1 0 0 0')
    root.set('name', 'base_link')
    model.geometry = {}
    model.sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    model.adjacent = {frozenset(('link7', 'link8'))}
    for body in root.iter('body'):
        for child in body.findall('body'):
            model.adjacent.add(frozenset((body.get('name'), child.get('name'))))
        for geom in list(body.findall('geom')):
            if geom.get('contype') == '0':
                body.remove(geom)
                continue
            mesh = assets[geom.get('mesh')]
            model.sources[str(mesh)] = hashlib.sha256(mesh.read_bytes()).hexdigest()
            model.geometry[geom.get('name')] = (mesh, {'pos': geom.get('pos', '0 0 0'),
                'quat': geom.get('quat', '1 0 0 0')}, '1 1 1')
    for node in list(doc.findall('contact')):
        doc.remove(node)
    world.insert(0, ET.Element('geom', name='support_plane', type='plane',
                              size='2 2 .02', contype='0', conaffinity='0'))
    model.floor_z = 0.
    model.xml = ET.tostring(doc, encoding='unicode')
    model.model = mujoco.MjModel.from_xml_string(model.xml)
    model.data = mujoco.MjData(model.model)
    model.joint_ids = [mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}') for i in range(1, 9)]
    model.qaddr = model.model.jnt_qposadr[model.joint_ids]
    model.geom_names = [mujoco.mj_id2name(model.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.model.ngeom)]
    model.body_names = [mujoco.mj_id2name(model.model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in model.model.geom_bodyid]
    model.pairs, model.excluded = [], []
    for a, b in itertools.combinations(range(model.model.ngeom), 2):
        ba, bb = model.body_names[a], model.body_names[b]
        if ba == bb or frozenset((ba, bb)) in model.adjacent or {ba, bb} == {'world', 'base_link'}:
            model.excluded.append([model.geom_names[a], model.geom_names[b]])
        else:
            model.pairs.append((a, b))


class ExactCollisionModel(CollisionModel):
    """MuJoCo FK plus triangle-mesh FCL narrow phase, no hull false positives."""
    def __init__(self, *args, reference_mjcf=None, **kwargs):
        import fcl
        import trimesh
        if reference_mjcf is None:
            super().__init__(*args, **kwargs)
        else:
            initialize_reference(self, reference_mjcf)
        self.reference_mjcf = reference_mjcf
        self.fcl = fcl
        self.objects = {}
        self.hull_objects = {}
        self.local_vertices = {}
        self.local_corners = {}
        for geom_name, (path, transform, scale) in self.geometry.items():
            mesh = trimesh.load_mesh(path, process=False)
            quat = numbers(transform['quat'], '1 0 0 0')
            rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            xyz = numbers(transform['pos'], '0 0 0')
            vertices = (mesh.vertices*numbers(scale, '1 1 1'))@rot.T+xyz
            bvh = fcl.BVHModel()
            bvh.beginModel(len(vertices), len(mesh.faces))
            bvh.addSubModel(vertices, mesh.faces)
            bvh.endModel()
            geom_id = self.geom_names.index(geom_name)
            self.objects[geom_id] = fcl.CollisionObject(bvh)
            hull = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False).convex_hull
            hull_vertices = np.asarray(hull.vertices)
            hull_faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).ravel()
            self.hull_objects[geom_id] = fcl.CollisionObject(fcl.Convex(hull_vertices, len(hull.faces), hull_faces))
            self.local_vertices[geom_id] = hull_vertices
            lo, hi = vertices.min(0), vertices.max(0)
            self.local_corners[geom_id] = np.array(list(itertools.product(*zip(lo, hi))))
        self.request = fcl.DistanceRequest(enable_nearest_points=False)

    def distances(self, q, width=0., distance_cap=.20):
        self.set_state(q, width)
        bounds = {}
        floor = {}
        for geom_id, obj in self.objects.items():
            body = int(self.model.geom_bodyid[geom_id])
            rot, pos = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
            obj.setTransform(self.fcl.Transform(rot, pos))
            self.hull_objects[geom_id].setTransform(self.fcl.Transform(rot, pos))
            corners = self.local_corners[geom_id]@rot.T+pos
            bounds[geom_id] = (corners.min(0), corners.max(0))
            floor[geom_id] = float((self.local_vertices[geom_id]@rot[2]).min()+pos[2]-self.floor_z)
        distances = []
        for a, b in self.pairs:
            if a == 0:
                distances.append(min(distance_cap, floor[b]))
                continue
            alo, ahi = bounds[a]
            blo, bhi = bounds[b]
            bound = np.linalg.norm(np.maximum(0, np.maximum(alo-bhi, blo-ahi)))
            if bound >= distance_cap:
                distances.append(distance_cap)
            else:
                value = self.fcl.distance(self.hull_objects[a], self.hull_objects[b], self.request, self.fcl.DistanceResult())
                if value < distance_cap:
                    value = self.fcl.distance(self.objects[a], self.objects[b], self.request, self.fcl.DistanceResult())
                distances.append(min(distance_cap, value))
        return np.asarray(distances)

    def provenance(self):
        result = super().provenance()
        result['collision_geometry'] = 'full visual triangle meshes; MuJoCo FK and FCL BVH narrow phase'
        if self.reference_mjcf:
            result['collision_geometry'] = 'CLab collision triangle meshes and CLab kinematics; base relocated to origin'
        result['limitations'].remove('convex hull false positives possible')
        return result


def screen(model, states, widths, clearance=.01):
    """Check every distinct supplied state; distances are capped at clearance."""
    states = np.asarray(states)
    unique = np.unique(states, axis=0)
    lowest, worst = clearance, None
    for width in widths:
        for index, q in enumerate(unique):
            distances = model.distances(q, width, distance_cap=clearance)
            value = float(np.min(distances))
            if value < lowest:
                k = int(np.argmin(distances))
                lowest = value
                worst = dict(q=q.tolist(), width=width,
                             pair=[model.geom_names[i] for i in model.pairs[k]])
    return dict(passed=lowest >= clearance-1e-9, nominal_states=len(states),
                unique_states=len(unique), widths_m=list(widths),
                clearance_lower_bound_m=lowest, distance_cap_m=clearance, worst=worst)


def trajectory_digest(plan):
    digest = hashlib.sha256()
    for name, t, q in plan:
        digest.update(name.encode())
        digest.update(np.asarray(t, dtype='<f8').tobytes())
        digest.update(np.asarray(q, dtype='<f8').tobytes())
    return digest.hexdigest()


def library_config(name):
    import pace_collection as collector
    library = json.loads(Path(__file__).with_name('pace_pose_library.json').read_text())
    item = library['poses'][name]
    c = collector.template()
    c.update({key: library[key] for key in ('model', 'protocol', 'seed', 'repeats', 'sample_hz')})
    lo, hi = [-3.14, -.05, -.1, -1.6, -1.57, -2], [2.618, 3.5, 3.2, 1.55, 1.57, 2]
    pose = dict(name=name, approved=True, q_min=lo, q_max=hi, **item)
    c['poses'] = [pose]
    c['home'].update(approved=True, q_min=lo, q_max=hi)
    return c, pose, library


def screen_library_job(job):
    import pace_collection as collector
    name, reference = job
    model = ExactCollisionModel(reference_mjcf=reference)
    c, pose, library = library_config(name)
    plan = list(collector.trajectories(c, pose))
    motion = [collector.transition_plan(c, np.zeros(6), pose['q_center'], 'entry'),
              *plan, collector.home_plan(c, pose)]
    result = screen(model, np.concatenate([q for _, _, q in motion]), [0., .044, .088], library['clearance_m'])
    offsets = np.vstack([np.array(list(itertools.product([-.4, 0., .4], repeat=6))),
                         np.random.default_rng(20261009).uniform(-.4, .4, (4096, 6))])
    result['tracking_envelope_sample'] = screen(model, offsets+pose['q_center'], [0., .044, .088], library['clearance_m'])
    model.set_state(pose['q_center'])
    result.update(name=name, model='clab' if reference else 'sdk', q_center=pose['q_center'],
                  link6_position_m=model.data.body('link6').xpos.tolist(),
                  trajectory_sha256=trajectory_digest(plan), provenance=model.provenance())
    print(json.dumps({k: v for k, v in result.items() if k != 'provenance'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-urdf', type=Path)
    parser.add_argument('--gripper-urdf', type=Path)
    parser.add_argument('--floor-z', type=float, default=0.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-mjcf', type=Path)
    parser.add_argument('--library', action='store_true')
    parser.add_argument('--entry-request', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.entry_request:
        import pace_collection as collector
        request = json.loads(args.entry_request.read_text())
        pilot = request.get('protocol') == 'pilot'
        c, pose, library = library_config('pose1' if pilot else request['name'])
        model = ExactCollisionModel(args.arm_urdf, args.gripper_urdf, args.floor_z)
        paths = [collector.transition_plan(c, request['feedback'], request['start'], 'tracking')[2]]
        if pilot:
            c.update(protocol='pilot', repeats=1)
            paths.append(collector.transition_plan(c, request['start'], pose['q_center'], 'entry')[2])
            paths.extend(q for _, _, q in collector.trajectories(c, pose))
            paths.append(collector.home_plan(c, pose)[2])
        else:
            paths.append(collector.transition_plan(c, request['start'], pose['q_center'], 'entry')[2])
        states = np.concatenate(paths)
        result = screen(model, states, [request['width']], library['clearance_m'])
        (args.out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return 0 if result['passed'] else 2
    if args.library:
        library = json.loads(Path(__file__).with_name('pace_pose_library.json').read_text())
        references = [None] + ([str(args.reference_mjcf.resolve())] if args.reference_mjcf else [])
        with ProcessPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(screen_library_job, [(name, ref) for ref in references for name in library['poses']]))
        report = dict(library=library, results=results, code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('pace_collection.py', 'pace_collision.py', 'pace_pose_library.json')})
        report['passed'] = all(r['passed'] and r['tracking_envelope_sample']['passed'] for r in results)
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        return 0 if report['passed'] else 2
    model = ExactCollisionModel(args.arm_urdf, args.gripper_urdf, args.floor_z,
                                reference_mjcf=args.reference_mjcf)
    (args.out/'screen_model.xml').write_text(model.xml)
    (args.out/'provenance.json').write_text(json.dumps(model.provenance(), indent=2))
    for q in ([0]*6, [0,.2,.15,0,0,0], [0,1.,1.5,0,0,0], [0,.8,1.8,-.2,0,0]):
        print(json.dumps(model.inspect(q)), flush=True)


if __name__ == '__main__':
    sys.exit(main())
