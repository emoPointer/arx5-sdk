"""Prepare or run a short single-arm SDK connection/state-feedback check.

Inputs: explicit --model and --interface; --seconds defaults to 2 (maximum 5).
Output: configuration and joint feedback on stdout; active --console additionally
creates a unique build/deployment/console_*.jsonl diagnostic log (never overwrites).
Prerequisites: repository .venv, configured CAN, physically confirmed arm/interface.
Dry run, safe with an energized robot (does not open CAN):
  /path/to/arx5-sdk/.venv/bin/python -I /path/to/arx5-sdk/python/examples/test_connection.py --model X5 --interface can1
Hardware test, ONLY after the operator authorizes initialization and supports the arm:
  /path/to/arx5-sdk/.venv/bin/python -I /path/to/arx5-sdk/python/examples/test_connection.py --model X5 --interface can1 --connect
--connect constructs a controller, sends initialization/control frames, and changes
motor behavior. It is NOT a read-only test and can cause physical movement.
Without --jog-deg, no home command or target trajectory is issued.
Optional --jog-deg DEG (0 < DEG <= 10) performs a joint-1 outward/return test:
  ... --model X5 --interface can1 --connect --jog-deg 10
This settles for 1 s, ramps position gains for 1.5 s, then uses 3 s per leg
and 0.5 s holds. It changes motor gains and joint targets; no homing occurs.
Abort stops further targets and uses SDK shutdown, not an automatic return. Default SDK gravity compensation
and damping remain enabled. Controller destruction enters damping, then releases
motors; the operator must support the arm. Interactive end-effector test (operator-run):
  ... --model X5 --interface can1 --connect --console
After READY, commands x/y/z <mm> translate in the base frame (max 50 mm),
home moves all six joints to zero without actuating the gripper, state prints
feedback, grip <mm> sets total gripper opening (0 to configured maximum),
Ctrl+C cancels arm/gripper motion and holds, release or Ctrl+D exits/unloads.
PACE: `pace pilot` checks J1..J6 sequentially; then `pace short train pose1`
or `pace short validation heldout`. Every completed pose returns to joint zero
and holds >=60 s. Uses live gains/gravity/SDK limits, without extra speed abort.
Before READY or on a fault, normal SDK shutdown still applies. Output is stdout;
run directly in an interactive terminal (do not pipe through tee for Ctrl+C handling).
Cartesian targets are converted by SDK IK,
validated in full before movement, then executed through the joint controller.
Console target speed cap is 5 rad/s, bounded by each native SDK joint limit;
planned minimum segment duration 1 s. PACE return-to-zero retains its slow ramp.
Optional --joint3-kp 85 changes only console J3 Kp during startup ramp;
valid range 70..100. Omit to retain SDK defaults. No files overwritten by this option.
There is no extra fixed measured-speed abort; native SDK protections remain.
Coarse-test IK acceptance is 1 mm / 1 degree, not precision pose validation.
No automatic homing occurs. Position holding ends on process/connection failure.
"""
import argparse
import json
import time
from pathlib import Path

import arx5_interface as arx5
import numpy as np


def run_jog(controller, config, control, degrees):
    delta = np.deg2rad(degrees)
    started = time.monotonic()
    last_timestamp = None
    last_fresh = started
    last_print = -1.0

    def feedback(phase, target=None, baseline=None, velocity_limit=0.3):
        nonlocal last_timestamp, last_fresh, last_print
        now = time.monotonic()
        state = controller.get_joint_state()
        q, v, tau = (x.copy() for x in (state.pos(), state.vel(), state.torque()))
        if state.timestamp != last_timestamp:
            last_timestamp, last_fresh = state.timestamp, now
        record = dict(phase=phase, elapsed_s=now-started, timestamp=state.timestamp,
                      q_rad=q.tolist(), dq_rad_s=v.tolist(), torque_nm=tau.tolist())
        if target is not None:
            record['target_rad'] = target.tolist()
        fault = None
        if not all(np.isfinite(x).all() for x in (q, v, tau)):
            fault = 'Non-finite feedback'
        elif now-last_fresh > 0.15:
            fault = 'Feedback timestamp stopped updating'
        elif np.max(np.abs(v)) > velocity_limit:
            fault = 'Measured speed exceeds phase limit'
        elif np.any(np.abs(tau) > 0.9*np.asarray(config.joint_torque_max)):
            fault = 'SDK torque feedback exceeds 90 percent of configured limit'
        elif target is not None and np.max(np.abs(q-target)) > 0.08:
            fault = 'Tracking error exceeds 0.08 rad'
        elif baseline is not None and np.max(np.abs(q[1:]-baseline[1:])) > 0.05:
            fault = 'Non-commanded joint displacement exceeds 0.05 rad'
        if now-last_print >= 0.05 or fault:
            print(json.dumps(record), flush=True)
            last_print = now
        if fault:
            raise RuntimeError(fault)
        return state

    # Initialization can move the supported arm; this phase is not pose holding.
    deadline = time.monotonic()+1.0
    while time.monotonic() < deadline:
        feedback('settle', velocity_limit=0.8)
        time.sleep(0.01)
    state = feedback('baseline', velocity_limit=0.15)
    origin = state.pos().copy()
    destination = origin.copy()
    destination[0] += delta
    if np.any(origin < config.joint_pos_min) or np.any(origin > config.joint_pos_max):
        raise RuntimeError('Starting pose outside configured limits')
    if not config.joint_pos_min[0]+0.03 < destination[0] < config.joint_pos_max[0]-0.03:
        raise RuntimeError('Destination too close to joint limit')
    gripper_pos = float(state.gripper_pos)
    print(json.dumps(dict(event='plan', origin_rad=origin.tolist(), destination_rad=destination.tolist(),
                          joint_index=0, delta_deg=degrees, leg_seconds=3)), flush=True)

    def send(target):
        cmd = arx5.JointState(config.joint_dof)
        cmd.pos()[:] = target
        cmd.gripper_pos = gripper_pos
        controller.set_joint_cmd(cmd)

    send(origin)
    ramp_start = time.monotonic()
    while True:
        alpha = min((time.monotonic()-ramp_start)/1.5, 1.0)
        feedback('gain_ramp', origin, origin)
        gain = arx5.Gain(config.joint_dof)
        gain.kp()[:] = np.asarray(control.default_kp)*alpha
        gain.kd()[:] = control.default_kd
        # Keep gripper position control disabled as in initialization.
        gain.gripper_kp = 0.0
        gain.gripper_kd = control.default_gripper_kd
        send(origin)
        controller.set_gain(gain)
        if alpha == 1.0:
            break
        time.sleep(0.01)

    endpoints = {}
    for phase, begin, end, duration in (
            ('hold_start', origin, origin, 0.5),
            ('outward', origin, destination, 3.0),
            ('hold_end', destination, destination, 0.5),
            ('return', destination, origin, 3.0),
            ('hold_return', origin, origin, 0.5)):
        phase_start = time.monotonic()
        while True:
            u = min((time.monotonic()-phase_start)/duration, 1.0)
            target = begin+(end-begin)*(3*u*u-2*u*u*u)
            feedback(phase, target, origin)
            send(target)
            if u == 1.0:
                # pybind returns a view: retain its owning JointState until copied.
                endpoint_state = controller.get_joint_state()
                endpoints[phase] = endpoint_state.pos().copy().tolist()
                break
            time.sleep(0.01)
    print(json.dumps(dict(event='jog_complete', origin_rad=origin.tolist(), endpoints_rad=endpoints)), flush=True)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def plan_console_motion(solver, config, origin, axis, distance_m):
    """Plan all joint samples before sending any; axis=None means joint zero."""
    if axis is not None and (axis not in (0, 1, 2) or not np.isfinite(distance_m) or abs(distance_m) > 0.05):
        raise ValueError('Translation must be finite and at most 50 mm per command')
    origin = np.asarray(origin).copy()
    initial_pose = solver.forward_kinematics(origin)
    waypoints = [origin.copy()]
    errors = []
    for index in range(1, 101):
        u = index / 100
        blend = 3*u*u - 2*u*u*u
        if axis is None:
            q = origin * (1-blend)
        else:
            target = initial_pose.copy()
            target[axis] += distance_m*blend
            status, q = solver.inverse_kinematics(target, waypoints[-1])
            q = np.asarray(q).copy()
            if status != 0:
                raise ValueError(f'IK failed at sample {index}: {status}')
            actual = solver.forward_kinematics(q)
            pos_error = float(np.linalg.norm(actual[:3]-target[:3]))
            angle_error = float(np.arccos(np.clip((np.trace(rpy_matrix(actual[3:]) @ rpy_matrix(target[3:]).T)-1)/2, -1, 1)))
            errors.append((pos_error, angle_error))
            # Coarse manual positional test; stricter offline regression stays unchanged.
            if pos_error > 0.001 or angle_error > np.deg2rad(1):
                raise ValueError(f'IK residual too large: {pos_error} m / {np.rad2deg(angle_error)} deg')
        if not np.isfinite(q).all() or np.any(q < config.joint_pos_min) or np.any(q > config.joint_pos_max):
            raise ValueError('Planned joint state invalid or outside limits')
        if np.max(np.abs(q-waypoints[-1])) > 0.1:
            raise ValueError('IK branch jump exceeds 0.1 rad per planning sample')
        waypoints.append(q)
    trajectory = np.array(waypoints)
    # Enforce the configured joint speed along the planned sampled path.
    interval = max(0.01, float(np.max(np.abs(np.diff(trajectory, axis=0)) / np.asarray(config.joint_vel_max))) * 1.25)
    return trajectory, interval, dict(duration_s=100*interval,
        max_ik_position_error_m=max((v[0] for v in errors), default=0.0),
        max_ik_orientation_error_deg=float(np.rad2deg(max((v[1] for v in errors), default=0.0))))


def run_console(controller, config, control, event_log=None):
    """Interactive operator loop. Ctrl+C holds only after READY is printed."""
    import select
    import signal
    import sys

    solver = arx5.Arx5Solver(config.urdf_path, config.joint_dof,
        config.joint_pos_min, config.joint_pos_max,
        config.base_link_name, config.eef_link_name, config.gravity_vector)
    stop_requested = [False]
    last_feedback_time = [None, time.monotonic()]
    target_q = [None]
    last_logged = [-1.0]
    current_command = ["initializing"]
    pace_pilot_passed = False

    def read_state(initializing=False):
        state = controller.get_joint_state()
        now = time.monotonic()
        if state.timestamp != last_feedback_time[0]:
            last_feedback_time[:] = [state.timestamp, now]
        q, vel, torque = state.pos().copy(), state.vel().copy(), state.torque().copy()
        if event_log is not None and now-last_logged[0] >= 0.02:
            event_log.write(json.dumps(dict(event='feedback', monotonic_s=now,
                sdk_timestamp=state.timestamp, command=current_command[0],
                q_rad=q.tolist(), dq_rad_s=vel.tolist(), torque_nm=torque.tolist(),
                gripper_pos_m=float(state.gripper_pos),
                target_rad=None if target_q[0] is None else target_q[0].tolist()))+'\n')
            last_logged[0] = now
        if not all(np.isfinite(x).all() for x in (q, vel, torque)):
            raise RuntimeError('Non-finite feedback')
        if now-last_feedback_time[1] > 0.2:
            raise RuntimeError('Stale feedback')
        if np.any(np.abs(torque) > 0.9*np.asarray(config.joint_torque_max)):
            raise RuntimeError('Torque feedback exceeds test bound')
        if not initializing and target_q[0] is not None and np.max(np.abs(q-target_q[0])) > 0.15:
            raise RuntimeError('Joint tracking error exceeds 0.15 rad')
        return state

    gripper = [0.0]
    def send(q):
        cmd = arx5.JointState(config.joint_dof)
        cmd.pos()[:] = q
        cmd.gripper_pos = gripper[0]
        controller.set_joint_cmd(cmd)
        target_q[0] = q.copy()

    def print_state(label):
        state = read_state()
        q = state.pos().copy()
        pose = solver.forward_kinematics(q)
        print(json.dumps(dict(event=label, q_deg=np.rad2deg(q).tolist(),
             eef_xyz_m=pose[:3].tolist(), eef_rpy_deg=np.rad2deg(pose[3:]).tolist(),
             gripper_pos_m=float(state.gripper_pos)), ensure_ascii=False), flush=True)

    deadline = time.monotonic()+1.0
    while time.monotonic() < deadline:
        read_state(initializing=True)
        time.sleep(0.01)
    state = read_state()
    origin = state.pos().copy()
    gripper[0] = float(state.gripper_pos)
    send(origin)
    ramp_start = time.monotonic()
    while True:
        alpha = min((time.monotonic()-ramp_start)/1.5, 1.0)
        read_state()
        gain = arx5.Gain(config.joint_dof)
        gain.kp()[:] = np.asarray(control.default_kp)*alpha
        gain.kd()[:] = control.default_kd
        gain.gripper_kp = 0.0
        gain.gripper_kd = control.default_gripper_kd
        send(origin)
        controller.set_gain(gain)
        if alpha == 1.0:
            break
        time.sleep(0.01)

    gripper_enabled = [False]

    def hold_current():
        state = controller.get_joint_state()
        q = state.pos().copy()
        if not np.isfinite(q).all():
            raise RuntimeError('Cannot hold non-finite state')
        if gripper_enabled[0]:
            width = float(state.gripper_pos)
            if not np.isfinite(width):
                raise RuntimeError('Cannot hold non-finite gripper state')
            gripper[0] = float(np.clip(width, 0, config.gripper_width))
        send(q)
        stop_requested[0] = False
        print('STOP: trajectory cancelled; holding measured joint position. Enter a command or release.', flush=True)

    def move_gripper(width_mm):
        target = width_mm / 1000.0
        if not np.isfinite(target) or not 0 <= target <= config.gripper_width:
            raise ValueError(f'Gripper width must be 0..{config.gripper_width*1000:g} mm')
        state = read_state()
        measured = float(state.gripper_pos)
        if not np.isfinite(measured):
            raise ValueError('Invalid gripper feedback; command rejected')
        start = float(np.clip(measured, 0, config.gripper_width))
        # Preserve the arm target while moving only the gripper.
        q_hold = target_q[0].copy()
        speed = float(config.gripper_vel_max)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('Invalid configured gripper speed')
        duration = max(1.0, abs(target-start)/speed)
        gripper[0] = start
        send(q_hold)
        gain.gripper_kp = control.default_gripper_kp
        gain.gripper_kd = control.default_gripper_kd
        controller.set_gain(gain)
        gripper_enabled[0] = True
        print(json.dumps(dict(event='gripper_plan', target_mm=width_mm, duration_s=duration)), flush=True)
        started = time.monotonic()
        while True:
            if stop_requested[0]:
                hold_current()
                return
            state = read_state()
            if not np.isfinite(float(state.gripper_pos)):
                raise RuntimeError('Non-finite gripper feedback during motion')
            fraction = min((time.monotonic()-started)/duration, 1.0)
            gripper[0] = start + fraction*(target-start)
            send(q_hold)
            if fraction == 1.0:
                break
            time.sleep(0.01)
        print_state('gripper_target_sent_holding')

    signal.signal(signal.SIGINT, lambda *_: stop_requested.__setitem__(0, True))
    current_command[0] = 'idle_hold'
    print('READY: x/y/z <mm> (max 50); home = joint zeros; grip <mm>; state; stop; release; pace pilot; pace short train NAME; pace short validation NAME.', flush=True)
    print('After READY, Ctrl+C cancels motion and holds. release or Ctrl+D unloads and exits. Not an emergency stop.', flush=True)
    print_state('ready')
    try:
        while True:
            if stop_requested[0]:
                hold_current()
            read_state()
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                print('EOF: releasing controller.', flush=True)
                return
            current_command[0] = line.strip()
            if event_log is not None:
                event_log.write(json.dumps(dict(event='command', text=line.strip()))+'\n')
            words = line.strip().split()
            if not words:
                continue
            if words == ['release']:
                return
            if words == ['stop']:
                hold_current()
                continue
            if words == ['state']:
                print_state('state')
                continue
            if words[0] == 'pace':
                import importlib.util
                module_path = Path(__file__).resolve().parents[1]/'pace_console.py'
                spec = importlib.util.spec_from_file_location('pace_console', module_path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    passed = module.run(controller, words, lambda reason: hold_current(),
                                        pilot_passed=pace_pilot_passed)
                    pace_pilot_passed = pace_pilot_passed or passed
                except (Exception, KeyboardInterrupt) as exc:
                    pace_pilot_passed = False
                    print(f'PACE stopped: {type(exc).__name__}: {exc}. No automatic home.', flush=True)
                    hold_current()
                finally:
                    holding_command = controller.get_joint_cmd()
                    target_q[0] = holding_command.pos().copy()
                    stop_requested[0] = False
                    current_command[0] = 'idle_hold'
                continue
            try:
                if len(words) == 2 and words[0] == 'grip':
                    move_gripper(float(words[1]))
                    continue
                if words == ['home']:
                    axis, distance = None, 0.0
                elif len(words) == 2 and words[0] in ('x', 'y', 'z'):
                    axis, distance = ('x', 'y', 'z').index(words[0]), float(words[1])/1000
                else:
                    raise ValueError('Use x/y/z <mm>, grip <mm>, home, state, stop, release')
                state = read_state()
                path, interval, diagnostics = plan_console_motion(solver, config, state.pos().copy(), axis, distance)
            except ValueError as exc:
                print(f'Command rejected before motion: {exc}', flush=True)
                continue
            print(json.dumps(dict(event='plan', command=line.strip(), **diagnostics)), flush=True)
            tick = time.monotonic()
            for q in path[1:]:
                if stop_requested[0]:
                    hold_current()
                    break
                read_state()
                send(q)
                tick += interval
                while time.monotonic() < tick:
                    if stop_requested[0]:
                        break
                    read_state()
                    time.sleep(min(0.01, max(0.0, tick-time.monotonic())))
            else:
                print_state('command_completed_holding')
    finally:
        # Prevent a second Ctrl+C from interrupting orderly C++ cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['X5'])
    parser.add_argument('--interface', required=True)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--jog-deg', type=float, default=0, help='Optional joint-1 excursion, maximum 10 degrees; hardware requires --connect')
    parser.add_argument('--joint3-kp', type=float, default=None, help='Console-only J3 Kp override, 70..100; applied during existing startup gain ramp')
    parser.add_argument('--console', action='store_true', help='Interactive Cartesian translation/home; Ctrl+C holds after READY')
    args = parser.parse_args()
    if args.joint3_kp is not None and (not args.console or not np.isfinite(args.joint3_kp) or not 70 <= args.joint3_kp <= 100):
        parser.error('--joint3-kp requires --console and a finite value in [70, 100]')
    if args.console and args.jog_deg:
        parser.error('--console and --jog-deg cannot be combined')
    if not np.isfinite(args.jog_deg) or not 0 <= args.jog_deg <= 10:
        parser.error('--jog-deg must be in [0, 10]')
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 5:
        parser.error('--seconds must be in (0, 5]')
    config = arx5.RobotConfigFactory.get_instance().get_config(args.model)
    # Deployment EEF contract: link6 body origin, identical to CLab.
    config.eef_link_name = 'link6'
    control = arx5.ControllerConfigFactory.get_instance().get_config('joint_controller', config.joint_dof)
    if args.joint3_kp is not None:
        kp = np.asarray(control.default_kp).copy()
        kp[2] = args.joint3_kp
        control.default_kp = kp
    config.joint_vel_max = np.minimum(config.joint_vel_max, np.full(config.joint_dof, 5.0 if args.console else (0.1 if args.jog_deg else 0.05)))
    config.gripper_vel_max = min(config.gripper_vel_max, config.gripper_width / 1.0 if args.console else 0.005)
    print(json.dumps(dict(model=args.model, interface=args.interface,
                          eef_link=config.eef_link_name,
                          jog_deg=args.jog_deg, console=args.console,
                          mode='ACTIVE_INITIALIZATION' if args.connect else 'DRY_RUN_NO_CAN',
                          joint_vel_max_rad_s=config.joint_vel_max.tolist(),
                          gripper_vel_max_m_s=config.gripper_vel_max,
                          gripper_width_m=config.gripper_width,
                          gripper_open_readout=config.gripper_open_readout,
                          kp=control.default_kp.tolist(), kd=control.default_kd.tolist(),
                          gravity_compensation=control.gravity_compensation), indent=2), flush=True)
    if not args.connect:
        print('Dry run complete. No controller constructed and no CAN frames sent.')
        return
    if not (Path('/sys/class/net') / args.interface).is_dir():
        raise RuntimeError(f'Interface not found: {args.interface}')
    print('Initializing hardware; this can move the arm. No homing will occur.', flush=True)
    console_log = None
    if args.console:
        import tempfile
        from datetime import datetime
        log_dir = Path(__file__).resolve().parents[2] / 'build' / 'deployment'
        log_dir.mkdir(parents=True, exist_ok=True)
        console_log = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', buffering=1,
            prefix='console_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_',
            suffix='.jsonl', dir=log_dir, delete=False)
        import hashlib
        console_log.write(json.dumps(dict(event='configuration', model=args.model,
            urdf_path=str(Path(config.urdf_path).resolve()),
            urdf_sha256=hashlib.sha256(Path(config.urdf_path).read_bytes()).hexdigest(),
            base_link=config.base_link_name, eef_link=config.eef_link_name,
            interface=args.interface, joint_vel_max_rad_s=config.joint_vel_max.tolist(),
            kp=control.default_kp.tolist(), kd=control.default_kd.tolist(),
            custom_measured_speed_abort=False))+'\n')
        print(f'Diagnostic log: {console_log.name}', flush=True)
    controller = None
    try:
        controller = arx5.Arx5JointController(config, control, args.interface)
        if args.console:
            run_console(controller, config, control, event_log=console_log)
            return
        if args.jog_deg:
            run_jog(controller, config, control, args.jog_deg)
            return
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            state = controller.get_joint_state()
            if not all(np.isfinite(v).all() for v in (state.pos(), state.vel(), state.torque())):
                raise RuntimeError('Non-finite joint feedback')
            print(json.dumps(dict(timestamp=state.timestamp, q_rad=state.pos().tolist(),
                                  dq_rad_s=state.vel().tolist(), torque_nm=state.torque().tolist(),
                                  gripper_pos_m=state.gripper_pos)), flush=True)
            time.sleep(0.1)
    finally:
        if controller is not None:
            print('Releasing controller: SDK damping/shutdown may move the arm. Support it.', flush=True)
            del controller
        if console_log is not None:
            console_log.close()


if __name__ == '__main__':
    main()
