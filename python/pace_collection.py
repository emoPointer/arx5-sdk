"""Guarded PACE-oriented acquisition using an ALREADY OWNED joint controller.

Offline: python python/pace_collection.py --template /tmp/pace.json
         python python/pace_collection.py --plan /tmp/pace.json --out python/data/pace-plan
Hardware: in the existing controller process, call collect(controller, config,
          pose_name, output_directory, safety_stop=your_existing_stop_handler).
Requires numpy; hardware mode additionally requires the installed arx5_interface.
Never constructs/destroys a controller or changes gains. collect_pose_and_rest()
explicitly returns to joint zero on an operator-approved path and rests >=60 s.
The caller MUST suspend all other command producers and retain controller ownership
after return/error. A safety_stop(reason) callback is mandatory: it must implement
the site's existing stop policy. Holding is not a substitute for an emergency stop.
Each output directory is NEW; saves raw NPZ, JSON metadata and incomplete runs.
Configuration describes approved joint envelopes, operating condition and splits.
This is not a collision checker or hard-real-time/hardware watchdog.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CHIRP_BANDS_HZ = {'low': (.1, .5), 'high': (.5, 1.)}


def template():
    return {
        "schema_version": 1, "model": "X5", "interface_label": "can1",
        "joint_names": [f"joint{i}" for i in range(1, 7)],
        "condition": {"robot_serial": "FILL", "firmware": "FILL",
                      "payload_and_attachments": "FILL", "base_mount": "FILL",
                      "temperature_notes": "FILL", "operator": "FILL"},
        "kp": [80, 70, 70, 30, 30, 20], "kd": [2, 2, 2, 1, 1, .7],
        "gravity_compensation": True, "sample_hz": 500,
        "seed": 20261009, "repeats": 1, "protocol": "pilot",
        "max_planned_seconds": 7200,
        "max_command_velocity": 5., "max_command_acceleration": 100.,
        "max_feedback_velocity": None, "max_tracking_error": .15,  # QA threshold only; never aborts motion.
        "max_gap_s": .2, "max_duplicate_fraction": .1,
        "start_command_tolerance": .01, "gripper_tolerance": .002,
        "feedback_effort_abort": [None] * 6,
        "home": {"approved": False, "q_min": [None]*6, "q_max": [None]*6,
                 "velocity": .1, "acceleration": .2, "rest_seconds": 60,
                 "arrival_tolerance": .02},
        "poses": [],
        "pose_example_NOT_EXECUTABLE": {
            "name": "working_pose_1", "split": "train", "approved": False,
            "q_center": [None] * 6, "q_min": [None] * 6, "q_max": [None] * 6,
        },
    }


def vector(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected {n} finite numbers")
    return a


def validate(c):
    allowed = set(template())
    if set(c) != allowed:
        raise ValueError(f"Configuration keys mismatch: {set(c) ^ allowed}")
    n = len(c["joint_names"])
    if c["schema_version"] != 1 or n != 6 or len(set(c["joint_names"])) != n:
        raise ValueError("Schema 1 currently supports six named joints only")
    for key in ("kp", "kd", "feedback_effort_abort"):
        if np.any(vector(c[key], n, key) <= 0):
            raise ValueError(f"{key} must be positive")
    for key in ("sample_hz", "max_command_velocity", "max_command_acceleration",
                "max_tracking_error", "max_gap_s",
                "start_command_tolerance", "gripper_tolerance"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if c['max_feedback_velocity'] is not None and (not np.isfinite(c['max_feedback_velocity']) or c['max_feedback_velocity'] <= 0):
        raise ValueError('max_feedback_velocity must be positive or null (SDK protections only)')
    if not np.isfinite(c["max_planned_seconds"]) or c["max_planned_seconds"] <= 0:
        raise ValueError("max_planned_seconds must be positive")
    if not 100 <= c["sample_hz"] <= 500 or not 0 <= c["max_duplicate_fraction"] < 1:
        raise ValueError("Sampling must be 100..500 Hz; invalid duplicate fraction")
    if type(c["repeats"]) is not int or not 1 <= c["repeats"] <= 5:
        raise ValueError("repeats must be an integer in 1..5")
    if type(c["seed"]) is not int or type(c["gravity_compensation"]) is not bool:
        raise ValueError("seed must be integer and gravity_compensation boolean")
    if c["protocol"] not in ("pilot", "short", "extended"):
        raise ValueError("protocol must be pilot, short or extended")
    h = c['home']
    if set(h) != {'approved', 'q_min', 'q_max', 'velocity', 'acceleration', 'rest_seconds', 'arrival_tolerance'}:
        raise ValueError('Invalid home configuration fields')
    if not np.isfinite(h['rest_seconds']) or h['rest_seconds'] < 60:
        raise ValueError('Home rest must be at least 60 seconds')
    for k in ('velocity', 'acceleration', 'arrival_tolerance'):
        if not np.isfinite(h[k]) or h[k] <= 0:
            raise ValueError(f'Invalid home {k}')
    if not c["condition"] or any(not str(v).strip() or v == "FILL" for v in c["condition"].values()):
        raise ValueError("Fill operating condition metadata; unknown values must say 'unknown'")
    if not c["poses"]:
        raise ValueError("Add operator-reviewed poses; no safe pose is assumed")
    names = set()
    for p in c["poses"]:
        if set(p) != {"name", "split", "approved", "q_center", "q_min", "q_max"}:
            raise ValueError("Unexpected pose fields")
        if not p["name"] or p["name"] in names or p["split"] not in ("train", "validation"):
            raise ValueError("Pose names must be unique; split is train or validation")
        names.add(p["name"])
        q, lo, hi = [vector(p[k], n, k) for k in ("q_center", "q_min", "q_max")]
        if not np.all((lo < q) & (q < hi)):
            raise ValueError(f"Invalid envelope: {p['name']}")


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (10 + u * (-15 + 6 * u))


def trajectories(c, pose):
    """Yield (name, t, q). Zero desired velocity is intentional SDK semantics."""
    hz = c["sample_hz"]
    q0 = np.asarray(pose["q_center"], float)
    rng = np.random.default_rng(c["seed"] + (10000 if pose["split"] == "validation" else 0))

    def build(name, duration, signal):
        t = np.arange(round(duration * hz) + 1) / hz
        q = q0 + signal(t)
        lo, hi = np.asarray(pose["q_min"]), np.asarray(pose["q_max"])
        v = np.diff(q, axis=0) * hz
        a = np.diff(v, axis=0) * hz
        return name, t, q

    yield build("static", 10 if c["protocol"] == "pilot" else 20, lambda t: np.zeros((len(t), 6)))
    if c['protocol'] == 'pilot':
        for j in range(6):
            axis = np.eye(6)[j]

            def pilot(t, axis=axis):
                u = np.clip(t-5, 0, 20)
                envelope = smooth(u/3)*smooth((20-u)/3)
                phase = 2*np.pi*(.1*u + (.5-.1)*u*u/40)
                return (.01*envelope*np.sin(phase))[:, None]*axis

            yield build(f'pilot_j{j+1}', 30, pilot)
        return
    if c["protocol"] != "extended":
        for repeat in range(c["repeats"]):
            axis = np.ones(6)
            bands = [
                ("low", 40, .25, *CHIRP_BANDS_HZ['low']),
                ("high", 30, .25, *CHIRP_BANDS_HZ['high'])]
            for label, duration, amp, f0, f1 in bands:
                phases = rng.uniform(-np.pi, np.pi, 6)
                if pose["split"] == "validation":
                    f0, f1 = f1, f0

                def wave(t, duration=duration, amp=amp, f0=f0, f1=f1, phases=phases):
                    u = np.clip(t-5, 0, duration)
                    env = smooth(u/8)*smooth((duration-u)/8)
                    phase = 2*np.pi*(f0*u + (f1-f0)*u*u/(2*duration))
                    return amp*env[:, None]*np.sin(phase[:, None]+phases)*axis

                yield build(f"r{repeat}_{label}", duration+10, wave)
            if c["protocol"] == "short":
                directions = rng.choice([-1., 1.], 6)

                def holds(t):
                    s = .25*(smooth((t-5)/4)-2*smooth((t-17)/4)+smooth((t-29)/4))
                    return s[:, None]*directions

                yield build(f"r{repeat}_holds", 40, holds)
        return
    for repeat in range(c["repeats"]):
        for j in rng.permutation(6):
            axis = np.eye(6)[j]
            for label, duration, amp, f0, f1 in (
                ("low", 40, .04, *CHIRP_BANDS_HZ['low']),
                ("high", 30, .005, *CHIRP_BANDS_HZ['high'])
            ):
                phase = rng.uniform(-np.pi, np.pi)
                if pose["split"] == "validation":
                    f0, f1, amp = f1, f0, amp * .8

                def wave(t, duration=duration, amp=amp, f0=f0, f1=f1, phase=phase, axis=axis):
                    u = np.clip(t - 5, 0, duration)
                    envelope = smooth(u / 3) * smooth((duration - u) / 3)
                    s = amp * envelope * np.sin(phase + 2 * np.pi * (f0 * u + (f1-f0) * u*u / (2*duration)))
                    return s[:, None] * axis

                yield build(f"r{repeat}_j{j+1}_{label}", duration + 10, wave)

            def steps(t, axis=axis):
                # +amplitude hold, -amplitude hold, return; no discontinuous steps.
                s = .04 * (smooth((t-5)/2) - 2*smooth((t-17)/2) + smooth((t-29)/2))
                return s[:, None] * axis

            yield build(f"r{repeat}_j{j+1}_holds", 40, steps)
        phase = rng.uniform(-np.pi, np.pi, 6)

        def mixed(t):
            u = np.clip(t - 5, 0, 40)
            env = smooth(u/3) * smooth((40-u)/3)
            return .015 * env[:, None] * np.sin(2*np.pi*u[:, None]*np.linspace(.17, .73, 6) + phase)

        yield build(f"r{repeat}_mixed", 50, mixed)


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def snapshot(obj, fields):
    result = {}
    for key in fields:
        value = getattr(obj, key)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (list, tuple)):
            value = [v if isinstance(v, (str, int, float)) else str(v) for v in value]
        result[key] = value
    return result


def state_values(s):
    return np.concatenate([s.pos().copy(), s.vel().copy(), s.torque().copy(),
                           [s.gripper_pos, s.gripper_vel, s.gripper_torque, s.timestamp]])


def collect(controller, config, pose_name, output_directory, *, safety_stop,
            confirm=input, sdk=None, clock=time.monotonic, sleep=time.sleep,
            _plan=None):
    """Blocking acquisition IN the owning process. Never opens or closes CAN.

    All other command producers must be suspended. On an exception, including
    Ctrl+C, calls mandatory safety_stop(reason), preserves partial data, re-raises.
    Caller must catch errors and keep its existing controller alive. On success
    leaves last approved target holding, without homing or gain changes.
    """
    validate(config)
    if not callable(safety_stop):
        raise ValueError("An existing site-specific safety_stop callback is required")
    pose = next(p for p in config["poses"] if p["name"] == pose_name)
    if pose["approved"] is not True:
        raise ValueError("Pose envelope must be explicitly operator-approved")
    plan = list(trajectories(config, pose)) if _plan is None else _plan
    # Also check internally generated return/rest trajectories before sending.
    for name, times, q in plan:
        if not np.isfinite(q).all():
            raise ValueError(f'{name}: nonfinite target')
        if len(times) < 3 or not np.all(np.diff(times) > 0):
            raise ValueError(f'{name}: invalid time grid')
        v = np.diff(q, axis=0)/np.diff(times)[:, None]
        a = np.diff(v, axis=0)/np.diff(times[:-1])[:, None]
    rc, cc = controller.get_robot_config(), controller.get_controller_config()
    gain = controller.get_gain()
    q0 = np.asarray(pose["q_center"], float)
    if rc.joint_dof != 6 or rc.robot_model != config["model"]:
        raise ValueError("Robot model mismatch")
    if not cc.background_send_recv or cc.gravity_compensation != config["gravity_compensation"]:
        raise ValueError("Need background joint controller with matching gravity mode")
    if cc.controller_type != "joint_controller" or cc.default_preview_time != 0:
        raise ValueError("Requires joint_controller and zero default_preview_time")
    if not np.array_equal(gain.kp(), config["kp"]) or not np.array_equal(gain.kd(), config["kd"]):
        raise ValueError("Live gains differ; collector never changes gains")
    state, output = controller.get_joint_state(), controller.get_joint_cmd()
    gripper = float(output.gripper_pos)
    start_error = float(np.max(np.abs(state.pos()-q0)))
    if not np.isfinite(state_values(state)).all() or not np.isfinite(state_values(output)).all():
        raise ValueError("Nonfinite initial state/command")
    sdk = importlib.import_module("arx5_interface") if sdk is None else sdk
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    urdf = Path(rc.urdf_path).read_bytes()
    (out / "controller_model.urdf").write_bytes(urdf)
    source = Path(__file__).read_bytes()
    (out / "collector_source.py").write_bytes(source)
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else "unavailable: " + p.stderr
    robot_fields = ("robot_model", "joint_dof", "joint_pos_min", "joint_pos_max", "joint_vel_max",
                    "joint_torque_max", "motor_id", "motor_type", "gravity_vector", "base_link_name",
                    "eef_link_name", "urdf_path", "gripper_width", "gripper_open_readout")
    controller_fields = ("controller_type", "controller_dt", "gravity_compensation", "background_send_recv",
                         "shutdown_to_passive", "interpolation_method", "default_preview_time", "over_current_cnt_max")
    sdk_path = Path(sdk.__file__)
    metadata = dict(config=config, pose=pose, created_utc=datetime.now(timezone.utc).isoformat(),
                    chirp_bands_hz=CHIRP_BANDS_HZ,
                    excitation_profile='pilot_0p1_0p5hz' if config['protocol'] == 'pilot' else 'chirps_up_to_1hz',
                    chirp_sweep_direction='descending' if pose['split'] == 'validation' and config['protocol'] != 'pilot' else 'ascending',
                    robot=snapshot(rc, robot_fields), controller=snapshot(cc, controller_fields),
                    gripper_command=gripper, gripper_kp=gain.gripper_kp, gripper_kd=gain.gripper_kd,
                    start_tracking_error_rad=start_error, tracking_error_action='record_and_flag_only',
                    software_threshold_action='record_and_flag_only',
                    urdf_sha256=hashlib.sha256(urdf).hexdigest(), sdk_path=str(sdk_path),
                    sdk_sha256=hashlib.sha256(sdk_path.read_bytes()).hexdigest(),
                    platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                    git_head=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                    clock="host monotonic seconds + SDK relative seconds; no hardware sensor timestamps",
                    output_semantics="SDK snapshot after interpolation/limits/gravity; before motor conversion; not CAN receipt",
                    effort_semantics="SDK-derived feedback, not calibrated torque or raw current",
                    columns="state/output: q[6], dq[6], effort[6], gripper_pos, gripper_vel, gripper_effort, sdk_timestamp")
    metadata["sample_layout"] = {
        "index": 0, "planned_time_s": 1, "read_start_host_s": 2, "read_end_host_s": 3,
        "send_start_host_s": 4, "send_end_host_s": 5, "python_target_q": [6, 12],
        "feedback_q": [12, 18], "feedback_dq": [18, 24], "feedback_effort": [24, 30],
        "feedback_gripper": [30, 33], "feedback_sdk_timestamp": 33,
        "sdk_output_q": [34, 40], "sdk_output_dq": [40, 46], "sdk_output_effort": [46, 52],
        "sdk_output_gripper": [52, 55], "output_sdk_timestamp": 55,
        "slice_convention": "Python [start, stop); NaN send times mean not submitted",
    }
    write_json(out / "metadata.json", metadata)
    (out / "source_changes.patch").write_text(git("diff", "HEAD"))
    rows, trial_index, results = [], -1, []
    status, error, stop_error = "aborted", None, None
    try:
        for trial_index, (name, times, targets) in enumerate(plan):
            if confirm(f"{pose_name}/{name}: clear space, sole command owner, stop ready? Type RUN: ") != "RUN":
                raise RuntimeError("Operator declined trial")
            current = controller.get_joint_cmd()
            start_command_error = float(np.max(np.abs(current.pos()-q0)))
            rows = []
            threshold_flags = set()
            if start_command_error > config['start_command_tolerance']:
                threshold_flags.add('start_command_error')
            planned_velocity = np.diff(targets, axis=0)/np.diff(times)[:, None]
            planned_acceleration = np.diff(planned_velocity, axis=0)/np.diff(times[:-1])[:, None]
            if np.any(targets < pose['q_min']) or np.any(targets > pose['q_max']):
                threshold_flags.add('target_outside_envelope')
            if np.max(np.abs(planned_velocity)) > config['max_command_velocity']:
                threshold_flags.add('planned_velocity')
            if np.max(np.abs(planned_acceleration)) > config['max_command_acceleration']:
                threshold_flags.add('planned_acceleration')
            start = clock()
            last_feedback = None
            last_change = start
            for i, (t, q) in enumerate(zip(times, targets)):
                sleep(max(0., start + t - clock()))
                now = clock()
                # Host scheduling delays are logged, not robot safety faults.
                # Shift the remaining schedule rather than burst overdue targets.
                if now-(start+t) > 1/config['sample_hz']:
                    start += now-(start+t)
                s = controller.get_joint_state()
                o = controller.get_joint_cmd()
                read_end = clock()
                sv, ov = state_values(s), state_values(o)
                # Preserve offending measurements before aborting.
                row = [i, t, now, read_end, math.nan, math.nan, *q, *sv, *ov]
                rows.append(row)
                if not np.isfinite(sv).all() or not np.isfinite(ov).all():
                    raise RuntimeError("Nonfinite feedback/SDK output")
                if last_feedback is not None and s.timestamp < last_feedback:
                    raise RuntimeError("SDK feedback timestamp moved backwards")
                if last_feedback != s.timestamp:
                    last_change, last_feedback = read_end, s.timestamp
                if read_end-last_change > config["max_gap_s"]:
                    threshold_flags.add('sdk_timestamp_stale')
                if np.any(s.pos() < pose["q_min"]) or np.any(s.pos() > pose["q_max"]):
                    threshold_flags.add('feedback_outside_envelope')
                if config['max_feedback_velocity'] is not None and np.max(np.abs(s.vel())) > config["max_feedback_velocity"]:
                    threshold_flags.add('feedback_speed')
                if np.any(np.abs(s.torque()) > config["feedback_effort_abort"]):
                    threshold_flags.add('feedback_effort')
                g = controller.get_gain()
                if not np.array_equal(g.kp(), config["kp"]) or not np.array_equal(g.kd(), config["kd"]) or g.gripper_kp != gain.gripper_kp or g.gripper_kd != gain.gripper_kd:
                    raise RuntimeError("Gains changed during acquisition")
                cmd = sdk.JointState(6)
                cmd.pos()[:] = q
                cmd.vel()[:] = 0  # Match existing single-point joint command semantics.
                cmd.torque()[:] = 0  # SDK adds configured gravity compensation.
                cmd.gripper_pos = gripper
                cmd.timestamp = 0
                row[4] = clock()
                controller.set_joint_cmd(cmd)
                row[5] = clock()
                if row[5]-(start+t) > 1/config['sample_hz']:
                    start += row[5]-(start+t)
            data = np.asarray(rows)
            # q starts at 6, state at 12; state timestamp is column 33.
            duplicate = float(np.mean(np.diff(data[:, 33]) == 0))
            gap = float(np.diff(data[:, 2]).max())
            gripper_error = float(np.max(np.abs(data[:, 30]-gripper)))
            tracking_error = float(np.max(np.abs(data[:, 12:18]-data[:, 34:40])))
            qa = {"trial": name, "rows": len(rows), "duplicate_fraction": duplicate,
                  "actual_host_hz": (len(rows)-1)/(data[-1, 2]-data[0, 2]),
                  "max_host_gap_s": gap, "max_gripper_error_m": gripper_error,
                  "max_tracking_error_rad": tracking_error,
                  "tracking_error_quality_passed": tracking_error <= config['max_tracking_error'],
                  "threshold_flags": sorted(threshold_flags),
                  "start_command_error_rad": start_command_error,
                  "max_feedback_speed_rad_s": np.max(np.abs(data[:, 18:24]), axis=0).tolist(),
                  "max_feedback_effort": np.max(np.abs(data[:, 24:30]), axis=0).tolist(),
                  "max_planned_velocity_rad_s": np.max(np.abs(planned_velocity), axis=0).tolist(),
                  "max_planned_acceleration_rad_s2": np.max(np.abs(planned_acceleration), axis=0).tolist(),
                  "passed": duplicate <= config["max_duplicate_fraction"]
                            and gap <= config['max_gap_s']
                            and gripper_error <= config['gripper_tolerance']
                            and tracking_error <= config['max_tracking_error'] and not threshold_flags}
            np.savez(out / f"{trial_index:03d}_{name}.npz", samples=data,
                     planned_time=times, planned_q=targets)
            results.append(qa)
            rows = []
            if not qa["passed"]:
                print(f'数据质量提示：{name} 的跟踪误差、重复采样、时间间隔或夹爪状态需复核；已保留数据，继续流程。', flush=True)
        status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            safety_stop(error)
        except BaseException as stop_exc:
            stop_error = f"{type(stop_exc).__name__}: {stop_exc}"
        raise
    finally:
        if rows:
            np.savez(out / f"{trial_index:03d}_partial.npz", samples=np.asarray(rows))
        write_json(out / "result.json", dict(status=status, error=error,
                                             safety_stop_error=stop_error, trials=results,
                                             data_quality_passed=status == 'completed' and all(r['passed'] for r in results)))
    return out


def transition_plan(config, start, target, name):
    """Approved quintic joint-space transition plus five seconds settling."""
    h = config['home']
    lo, hi = [vector(h[k], 6, f'home {k}') for k in ('q_min', 'q_max')]
    q = vector(start, 6, 'transition start')
    goal = vector(target, 6, 'transition target')
    if h['approved'] is not True or np.any(lo >= 0) or np.any(hi <= 0):
        raise ValueError('Home path envelope must include zero and be approved')
    distance = np.max(np.abs(goal-q))
    velocity = min(h['velocity'], config['max_command_velocity'])
    acceleration = min(h['acceleration'], config['max_command_acceleration'])
    duration = max(3., 1.9*distance/velocity, np.sqrt(6*distance/acceleration))
    hz = config['sample_hz']
    duration = np.ceil(duration*hz)/hz
    t = np.arange(round((duration+5)*hz)+1)/hz
    targets = q + smooth(t/duration)[:, None]*(goal-q)
    targets[t >= duration] = goal
    return (name, t, targets)


def home_plan(config, pose):
    return transition_plan(config, pose['q_center'], np.zeros(6), 'return_zero_and_settle')


def collect_pose_and_rest(controller, config, pose_name, output_directory, *,
                          safety_stop, confirm=input, sdk=None,
                          clock=time.monotonic, sleep=time.sleep, start_q=None):
    """Collect one pose, approved return to zero, >=60 s monitored rest.

    Optional start_q enables an approved approach to the configured pose center.
    No automatic home on abort. Caller owns fault handling.
    The rest timer begins after return plus 5 s settling. Tracking error is
    recorded as QA only; no measured-speed or arrival-error abort is imposed.
    """
    import copy
    validate(config)
    pose = next(p for p in config['poses'] if p['name'] == pose_name)
    return_plan = home_plan(config, pose)
    # Validate excitation BEFORE any approach movement.
    identification_plan = list(trajectories(config, pose))
    approach_plan = None if start_q is None else transition_plan(
        config, start_q, pose['q_center'], 'approach_pose_and_settle')
    rc = controller.get_robot_config()
    h = config['home']
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    kwargs = dict(safety_stop=safety_stop, confirm=confirm, sdk=sdk, clock=clock, sleep=sleep)
    stages = []
    if approach_plan is not None:
        approach_config = copy.deepcopy(config)
        approach_config['poses'] = [dict(pose, q_center=list(start_q), q_min=h['q_min'], q_max=h['q_max'])]
        collect(controller, approach_config, pose_name, out/'approach_pose', _plan=[approach_plan], **kwargs)
        stages.append('approach_pose')
    collect(controller, config, pose_name, out/'identification', _plan=identification_plan, **kwargs)
    home_config = copy.deepcopy(config)
    hp = home_config['poses'] = [dict(pose, q_min=h['q_min'], q_max=h['q_max'])]
    try:
        collect(controller, home_config, pose_name, out/'return_zero', _plan=[return_plan], **kwargs)
        hp[0]['q_center'] = [0.]*6
        t = np.arange(round(h['rest_seconds']*config['sample_hz'])+1)/config['sample_hz']
        collect(controller, home_config, pose_name, out/'rest_zero',
                _plan=[('rest_zero', t, np.zeros((len(t), 6)))], **kwargs)
    except BaseException:
        # A preflight failure can occur before collect's stop callback. No further
        # trajectory is issued here. Let the owner apply its established stop.
        raise
    quality = {stage: json.loads((out/stage/'result.json').read_text())['data_quality_passed']
               for stage in stages+['identification', 'return_zero', 'rest_zero']}
    write_json(out/'pose_complete.json', {'status': 'completed', 'home_target': [0.]*6,
                                        'data_quality_passed': all(quality.values()), 'stage_quality': quality,
                                        'rest_seconds': h['rest_seconds'], 'next_pose': 'manual only'})
    print('At joint zero; minimum rest complete. Check temperature/state before choosing another pose.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--template", type=Path)
    group.add_argument("--plan", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.template:
        write_json(args.template, template())
        print(f"Wrote non-executable template: {args.template}; fill safety envelope and conditions")
        return
    c = json.loads(args.plan.read_text())
    validate(c)
    plans = [(pose, list(trajectories(c, pose))) for pose in c["poses"]]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
    for k, (pose, plan) in enumerate(plans):
        seconds = sum(t[-1] for _, t, _ in plan)
        print(f"{pose['name']} {pose['split']}: {len(plan)} trials, {seconds/60:.1f} minutes + operator pauses")
        if args.out:
            for i, (name, t, q) in enumerate(plan):
                np.savez(args.out / f"pose{k:02d}_{i:03d}_{name}.npz", t=t, q=q)
            write_json(args.out / f"pose{k:02d}.json", pose)
    print("OFFLINE ONLY: no SDK imported, controller created, or motor command sent")


if __name__ == "__main__":
    main()
