"""PACE commands for the existing test_connection console; never opens CAN.

Inputs: already-owned controller, console stop callback, command tokens.
Outputs: unique python/data/pace/<UTC timestamp> directory, no overwrites.
Requires numpy and pace_collection.py alongside this file. Use the deployed
test_connection.py --model X5 --interface can1 --connect --console, then
`pace pilot` or `pace short train pose1` / `pace short validation heldout`.
The operator must review the current pose and complete return-to-zero path.
Only one RUN confirmation per pose; no metadata questionnaire or segment prompts.
Unknown operator/payload/temperature metadata is explicitly recorded as unknown.
No gains, gravity mode, SDK limits or gripper settings are changed. The caller
retains ownership after success/failure and must synchronize its holding target.
"""

import importlib.util
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def selected_pose(config, name, split):
    """Only named, versioned poses; never derive a center from the holding pose."""
    library = json.loads(Path(__file__).with_name('pace_pose_library.json').read_text())
    if any(config[key] != library[key] for key in ('model', 'protocol', 'seed', 'repeats', 'sample_hz')):
        raise ValueError('PACE configuration differs from the collision-screened library')
    if name not in library['poses'] or library['poses'][name]['split'] != split:
        raise ValueError('Use train pose1/pose2/pose3 or validation heldout')
    return np.asarray(library['poses'][name]['q_center'], dtype=float)


def collision_preflight(controller, config, name, start, plan):
    """Read-only geometry subprocess: no SDK import, CAN owner or motor command."""
    directory = Path(__file__).resolve().parent
    root = directory.parent
    report = json.loads((directory/'pace_pose_screen.json').read_text())
    if report['passed'] is not True:
        raise ValueError('Pose library has not passed collision screening')
    for filename, expected in report['code_sha256'].items():
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Collision screen is stale: {filename}')
    digest = hashlib.sha256()
    for label, t, q in plan:
        digest.update(label.encode())
        digest.update(np.asarray(t, dtype='<f8').tobytes())
        digest.update(np.asarray(q, dtype='<f8').tobytes())
    if config['protocol'] == 'short' and digest.hexdigest() != report['trajectory_sha256'][name]:
        raise ValueError('Actual planned trajectory differs from screened trajectory')
    gripper_model = root/'build/pace_geometry/X5_gripper.urdf'
    if not gripper_model.is_file():
        gripper_model = root/'models/X5_gripper.urdf'
    for relative, expected in report['sdk_source_sha256'].items():
        source = gripper_model if relative == 'models/X5_gripper.urdf' else root/relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Collision model differs from screen: {relative}')
    actual_urdf = Path(controller.get_robot_config().urdf_path)
    if hashlib.sha256(actual_urdf.read_bytes()).hexdigest() != report['sdk_source_sha256']['models/X5.urdf']:
        raise ValueError('Live controller URDF differs from screened model')
    state, command = controller.get_joint_state(), controller.get_joint_cmd()
    feedback = state.pos().copy()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = directory/'data/pace_preflight'/stamp
    out.mkdir(parents=True)
    request = out/'request.json'
    request.write_text(json.dumps(dict(name=name, protocol=config['protocol'], start=start.tolist(), feedback=feedback.tolist(),
                                      width=float(command.gripper_pos)), indent=2))
    (out/'pose_screen.json').write_text(json.dumps(report, indent=2))
    executable = root/'build/pace_collision_env/bin/python'
    print('正在离线核对当前位置到固定采集中心的路径（不会发送运动命令）…', flush=True)
    result = subprocess.run([str(executable), str(directory/'pace_collision.py'),
                             '--entry-request', str(request), '--gripper-urdf', str(gripper_model),
                             '--out', str(out/'screen')],
                            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'},
                            capture_output=True, text=True, timeout=180)
    (out/'checker.log').write_text(result.stdout+result.stderr)
    if result.returncode not in (0, 2):
        raise ValueError(f'进入路径未通过几何检查；未开始运动。诊断：{out}')
    if result.returncode == 2:
        print(f'几何间隙低于报告阈值，仅记录、不拦截；请检查现场。诊断：{out}', flush=True)
    fresh = controller.get_joint_state()
    fresh_command = controller.get_joint_cmd()
    if np.max(np.abs(fresh.pos()-feedback)) > .01 or np.max(np.abs(fresh_command.pos()-start)) > 1e-6:
        print('几何检查期间状态有变化，仅记录、不拦截。', flush=True)
    return out


def completed_pilot_in_session(controller):
    """Recover a completed pilot after a command/preflight rejection, not faults."""
    state = controller.get_joint_state()
    session_start = time.time()-float(state.timestamp)
    root = Path(__file__).parent/'data'/'pace'
    if not root.exists():
        return False
    gain, cc = controller.get_gain(), controller.get_controller_config()
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.stat().st_mtime < session_start:
            continue
        for stage in ('approach_pose', 'identification', 'return_zero', 'rest_zero'):
            result = run_dir/stage/'result.json'
            if result.exists() and json.loads(result.read_text())['status'] != 'completed':
                return False
            if (run_dir/stage/'metadata.json').exists() and not result.exists():
                return False
        metadata = run_dir/'identification'/'metadata.json'
        if not metadata.exists():
            continue
        complete = run_dir/'pose_complete.json'
        if not complete.exists():
            return False
        cfg = json.loads(metadata.read_text())['config']
        if cfg['protocol'] == 'pilot':
            return (json.loads(complete.read_text())['status'] == 'completed'
                    and cfg['model'] == controller.get_robot_config().robot_model
                    and np.array_equal(cfg['kp'], gain.kp()) and np.array_equal(cfg['kd'], gain.kd())
                    and cfg['gravity_compensation'] == cc.gravity_compensation)
    return False


def run(controller, words, safety_stop, *, pilot_passed=False):
    """Return True only for a completed pilot; errors propagate to the owner."""
    if words == ['pace', 'pilot']:
        protocol, split, name = 'pilot', 'train', 'pilot'
    elif len(words) == 4 and words[1] == 'short' and words[2] in ('train', 'validation'):
        protocol, split, name = words[1:]
    else:
        raise ValueError('Use pace pilot | pace short train NAME | pace short validation NAME')
    if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
        raise ValueError('Pose name must contain only letters, numbers, underscore or hyphen')
    path = Path(__file__).with_name('pace_collection.py')
    spec = importlib.util.spec_from_file_location('pace_collection', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    c = collector.template()
    rc, cc, gain = controller.get_robot_config(), controller.get_controller_config(), controller.get_gain()
    # pos() exposes SDK-owned memory; keep the JointState owner alive until copy.
    holding_command = controller.get_joint_cmd()
    q = holding_command.pos().copy()
    c.update(model=rc.robot_model, interface_label='existing console owner',
             kp=gain.kp().tolist(), kd=gain.kd().tolist(),
             gravity_compensation=bool(cc.gravity_compensation), protocol=protocol,
             repeats=1 if protocol == 'pilot' else 2,
             max_command_velocity=float(np.min(rc.joint_vel_max)),
             feedback_effort_abort=(.9*np.asarray(rc.joint_torque_max)).tolist())
    c['poses'] = [dict(name=name, split=split, approved=True, q_center=q.tolist(),
                       q_min=rc.joint_pos_min.tolist(), q_max=rc.joint_pos_max.tolist())]
    c['home'].update(approved=True, q_min=rc.joint_pos_min.tolist(), q_max=rc.joint_pos_max.tolist())
    c['condition'] = {key: 'unknown' for key in c['condition']}
    approach = None
    center = (selected_pose(c, name, split) if protocol == 'short' else
              selected_pose({**c, 'protocol': 'short', 'repeats': 2}, 'pose1', 'train'))
    c['poses'][0]['q_center'] = center.tolist()
    if not np.array_equal(center, q):
        approach = q.copy()
        entry = collector.transition_plan(c, q, center, 'approach_pose_and_settle')
        print(f'自动进入采集中心（rad）：{np.round(center, 4).tolist()}，约 {entry[1][-1]:.0f} 秒（含到位停留）。', flush=True)
    if protocol == 'short':
        low, high = collector.CHIRP_BANDS_HZ['low'], collector.CHIRP_BANDS_HZ['high']
        print(f'正式幅度 ±0.25 rad（±14.32°）；low {low[0]:g}–{low[1]:g} Hz / high {high[0]:g}–{high[1]:g} Hz；最高 {high[1]:g} Hz。', flush=True)
    plan = list(collector.trajectories(c, c['poses'][0]))
    home = collector.home_plan(c, c['poses'][0])
    preflight = collision_preflight(controller, c, name, q, plan)
    label = '六关节依次预检查' if protocol == 'pilot' else f'姿态 {name} 采集'
    print(f'{label} {sum(p[1][-1] for p in plan):.0f} 秒 → 缓慢回零约 {home[1][-1]:.0f} 秒 → 零位保持 60 秒。', flush=True)
    print('已检查建模连杆的离散几何路径；相机、线束、外部障碍物仍需人工确认。Ctrl+C 可中止。'
          if preflight else '预检查尚无几何认证；确认当前位置及回零路径无障碍。Ctrl+C 可中止。', flush=True)
    old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if input('输入 RUN 开始全部流程，其他输入取消：').strip() != 'RUN':
            raise ValueError('已取消，未开始采集')
        checked = json.loads((preflight/'request.json').read_text())
        current_state, current_command = controller.get_joint_state(), controller.get_joint_cmd()
        if (np.max(np.abs(current_state.pos()-checked['feedback'])) > .01
                or np.max(np.abs(current_command.pos()-q)) > 1e-6
                or abs(float(current_command.gripper_pos)-checked['width']) > 1e-6):
            print('确认期间状态有变化，仅记录、不拦截。', flush=True)
        output = path.parent/'data'/'pace'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        (preflight/'acquisition.json').write_text(json.dumps({'output_directory': str(output)}, indent=2))
        print(f'PACE output: {output}', flush=True)
        def next_segment(prompt):
            print(f'执行阶段：{prompt.split(":", 1)[0]}', flush=True)
            return 'RUN'

        collector.collect_pose_and_rest(controller, c, name, output,
                                        safety_stop=safety_stop, confirm=next_segment, start_q=approach)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    return protocol == 'pilot'
