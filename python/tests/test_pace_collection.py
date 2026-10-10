"""Offline: python -m pytest python/tests/test_pace_collection.py; no CAN imports."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pace_collection", ROOT / "python/pace_collection.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config():
    c = m.template()
    c['condition'] = {k: 'test fixture' for k in c['condition']}
    c['feedback_effort_abort'] = [5.] * 6
    c['poses'] = [dict(name='test', split='train', approved=True,
                       q_center=[0.]*6, q_min=[-.3]*6, q_max=[.3]*6)]
    return c


@pytest.mark.parametrize('protocol,seconds', [('pilot', 190), ('short', 150), ('extended', 850)])
def test_plan(protocol, seconds):
    c = config()
    c['protocol'] = protocol
    m.validate(c)
    plan = list(m.trajectories(c, c['poses'][0]))
    assert sum(t[-1] for _, t, _ in plan) == seconds
    for _, _, q in plan:
        np.testing.assert_allclose(q[0], 0, atol=1e-12)
        np.testing.assert_allclose(q[-1], 0, atol=1e-12)


def test_reject_template_but_envelope_threshold_does_not_abort():
    with pytest.raises(ValueError):
        m.validate(m.template())
    c = config()
    c['poses'][0]['q_max'] = [.001]*6
    plan = list(m.trajectories(c, c['poses'][0]))
    assert any(np.any(q > .001) for _, _, q in plan)


def test_validation_different_and_deterministic():
    c = config()
    c['protocol'] = 'short'
    p = c['poses'][0]
    first = list(m.trajectories(c, p))[-1][-1]
    np.testing.assert_array_equal(first, list(m.trajectories(c, p))[-1][-1])
    p['split'] = 'validation'
    assert not np.array_equal(first, list(m.trajectories(c, p))[-1][-1])


class State:
    def __init__(self, n):
        self.q, self.dq, self.tau = np.zeros(n), np.zeros(n), np.zeros(n)
        self.gripper_pos = self.gripper_vel = self.gripper_torque = self.timestamp = 0.

    def pos(self):
        return self.q

    def vel(self):
        return self.dq

    def torque(self):
        return self.tau


class Clock:
    t = 1.

    def now(self):
        return self.t

    def sleep(self, delta):
        self.t += delta


def controller(c, clock):
    rc = SimpleNamespace(robot_model='X5', joint_dof=6, joint_pos_min=np.full(6, -.3),
                         joint_pos_max=np.full(6, .3), joint_vel_max=np.ones(6),
                         joint_torque_max=np.full(6, 10), motor_id=list(range(6)),
                         motor_type=['fake']*6, gravity_vector=np.array([0, 0, -9.807]),
                         base_link_name='base', eef_link_name='eef',
                         urdf_path=str(ROOT/'models/X5.urdf'), gripper_width=.08, gripper_open_readout=5.)
    cc = SimpleNamespace(controller_type='joint_controller', controller_dt=.002,
                         gravity_compensation=True, background_send_recv=True, shutdown_to_passive=True,
                         interpolation_method='linear', default_preview_time=0., over_current_cnt_max=20)
    gain = SimpleNamespace(kp=lambda: np.array(c['kp']), kd=lambda: np.array(c['kd']), gripper_kp=5., gripper_kd=.2)

    class Controller:
        sent = 0
        q = np.zeros(6)

        def get_robot_config(self):
            return rc

        def get_controller_config(self):
            return cc

        def get_gain(self):
            return gain

        def get_joint_state(self):
            s = State(6)
            s.q = self.q.copy()
            s.timestamp = clock.now()
            return s

        get_joint_cmd = get_joint_state

        def set_joint_cmd(self, cmd):
            self.q = cmd.q.copy()
            self.sent += 1

    return Controller()


@pytest.mark.parametrize('fault', ['none', 'decline', 'speed', 'interrupt', 'stale', 'idle_velocity'])
def test_mock_collection_and_partial_preservation(tmp_path, monkeypatch, fault):
    c = config()
    c['max_feedback_velocity'] = .6
    clock = Clock()
    ctrl = controller(c, clock)
    stops = []
    count = 126 if fault == 'stale' else 6
    monkeypatch.setattr(m, 'trajectories', lambda *_: iter([('tiny', np.arange(count)*.002, np.zeros((count, 6)))]))
    old = ctrl.get_joint_state
    def read():
        s = old()
        if fault == 'idle_velocity':
            s.dq[:] = .03296661376953125
        if ctrl.sent >= 2:
            if fault == 'speed':
                s.dq[0] = 1.
            if fault == 'interrupt':
                raise KeyboardInterrupt()
        if fault == 'stale':
            s.timestamp = 1.
        return s
    ctrl.get_joint_state = read
    kwargs = dict(safety_stop=stops.append, confirm=lambda _: 'no' if fault == 'decline' else 'RUN',
                  sdk=SimpleNamespace(JointState=State, __file__=str(ROOT/'python/pace_collection.py')),
                  clock=clock.now, sleep=clock.sleep)
    if fault in ('none', 'idle_velocity', 'speed', 'stale'):
        m.collect(ctrl, c, 'test', tmp_path/'run', **kwargs)
        assert ctrl.sent == count
        assert not stops
        data = np.load(tmp_path/'run/000_tiny.npz')['samples']
        assert data.shape == (count, 56)
        assert np.isfinite(data).all()
    else:
        with pytest.raises((RuntimeError, KeyboardInterrupt)):
            m.collect(ctrl, c, 'test', tmp_path/'run', **kwargs)
        assert len(stops) == 1
    result = json.loads((tmp_path/'run/result.json').read_text())
    assert result['status'] == ('completed' if fault in ('none', 'idle_velocity', 'speed', 'stale') else 'aborted')
    if fault == 'speed':
        assert 'feedback_speed' in result['trials'][0]['threshold_flags']
    if fault == 'stale':
        assert 'sdk_timestamp_stale' in result['trials'][0]['threshold_flags']


def test_unapproved_never_commands(tmp_path):
    c = config()
    c['poses'][0]['approved'] = False
    ctrl = controller(c, Clock())
    with pytest.raises(ValueError, match='approved'):
        m.collect(ctrl, c, 'test', tmp_path/'run', safety_stop=lambda _: None)
    assert ctrl.sent == 0


def test_all_runtime_thresholds_are_diagnostic_only(tmp_path, monkeypatch):
    c = config()
    c.update(max_feedback_velocity=.6, max_planned_seconds=.001)
    clock = Clock()
    ctrl = controller(c, clock)
    count = 126
    monkeypatch.setattr(m, 'trajectories', lambda *_: iter([
        ('combined_thresholds', np.arange(count)*.002, np.zeros((count, 6)))]))

    def feedback():
        s = State(6)
        s.q[:] = .4
        s.dq[:] = 1.
        s.tau[:] = 6.
        s.gripper_pos = .01
        s.timestamp = 1.
        return s

    def command():
        s = State(6)
        s.q[:] = .02
        return s

    ctrl.get_joint_state = feedback
    ctrl.get_joint_cmd = command
    stops = []
    m.collect(ctrl, c, 'test', tmp_path/'run', safety_stop=stops.append,
              confirm=lambda _: 'RUN', sdk=SimpleNamespace(JointState=State, __file__=m.__file__),
              clock=clock.now, sleep=clock.sleep)
    result = json.loads((tmp_path/'run/result.json').read_text())
    assert ctrl.sent == count and not stops and result['status'] == 'completed'
    metadata = json.loads((tmp_path/'run/metadata.json').read_text())
    assert metadata['chirp_bands_hz'] == {'low': [.1, .5], 'high': [.5, 1.]}
    assert metadata['excitation_profile'] == 'pilot_0p1_0p5hz'
    assert not result['data_quality_passed']
    assert set(result['trials'][0]['threshold_flags']) >= {
        'start_command_error', 'sdk_timestamp_stale', 'feedback_outside_envelope',
        'feedback_speed', 'feedback_effort'}


def test_pilot_all_six_in_order_and_no_twenty_minute_cap():
    c = config()
    c['max_planned_seconds'] = 7200
    m.validate(c)
    plan = list(m.trajectories(c, c['poses'][0]))
    assert [p[0] for p in plan[1:]] == [f'pilot_j{i}' for i in range(1, 7)]
    for j, (_, _, q) in enumerate(plan[1:]):
        assert np.max(np.abs(q[:, j])) > .009
        np.testing.assert_array_equal(np.delete(q, j, axis=1), 0)
    assert c['max_feedback_velocity'] is None


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_actual_chirp_phase_stays_in_new_bands(monkeypatch, split):
    c = config()
    c.update(protocol='short', repeats=1)
    c['poses'][0]['split'] = split
    phases = []
    original_sin = np.sin

    def capture(phase):
        phases.append(phase.copy())
        return original_sin(phase)

    monkeypatch.setattr(m.np, 'sin', capture)
    plan = list(m.trajectories(c, c['poses'][0]))
    assert len(phases) == 2
    assert m.CHIRP_BANDS_HZ == {'low': (.1, .5), 'high': (.5, 1.)}
    for phase, (_, t, _), duration, band in zip(phases, plan[1:3], [40, 30], [(.1, .5), (.5, 1.)]):
        active = phase[(t >= 5) & (t <= duration+5)]
        instantaneous_hz = np.diff(active, axis=0)*c['sample_hz']/(2*np.pi)
        assert instantaneous_hz.min() >= band[0]-1e-9
        assert instantaneous_hz.max() <= band[1]+1e-9
        assert np.sign(instantaneous_hz[-1, 0]-instantaneous_hz[0, 0]) == (1 if split == 'train' else -1)


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_short_quarter_radian_amplitude_and_one_hz_speed(split):
    c = config()
    c.update(protocol='short', repeats=2)
    c['poses'][0]['split'] = split
    plan = list(m.trajectories(c, c['poses'][0]))
    assert sum(t[-1] for _, t, _ in plan) == 280
    for name, t, q in plan[1:]:
        assert np.all(np.max(np.abs(q), axis=0) > .24), name
        assert np.max(np.abs(q)) <= .25+1e-12
        velocity = np.diff(q, axis=0)*c['sample_hz']
        assert np.max(np.abs(velocity)) <= 5.
        assert np.max(np.abs(np.diff(velocity, axis=0)))*c['sample_hz'] <= 100.
        if 'high' in name:
            assert 1. < np.max(np.abs(velocity)) < 1.6
        if 'low' in name:
            assert np.max(np.abs(velocity)) < .8
        if 'holds' in name:
            np.testing.assert_allclose(np.max(q, axis=0), .25)
            np.testing.assert_allclose(np.min(q, axis=0), -.25)


def test_short_plans_entry_before_single_confirmation(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('pace_console', ROOT/'python/pace_console.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    ctrl = controller(config(), Clock())
    ctrl.get_robot_config().joint_pos_min[:] = [-3.14, -.05, -.1, -1.6, -1.57, -2]
    ctrl.get_robot_config().joint_pos_max[:] = [2.618, 3.5, 3.2, 1.55, 1.57, 2]
    ctrl.get_robot_config().joint_vel_max[:] = 5
    monkeypatch.setattr('builtins.input', lambda _: 'cancel')
    monkeypatch.setattr(adapter, 'collision_preflight', lambda *args: ROOT)
    with pytest.raises(ValueError, match='已取消'):
        adapter.run(ctrl, ['pace', 'short', 'train', 'pose1'], lambda _: None, pilot_passed=False)
    assert '自动进入采集中心' in capsys.readouterr().out
    assert ctrl.sent == 0


@pytest.mark.parametrize('later_fault', [False, True])
def test_recover_completed_pilot_but_not_after_motion_fault(tmp_path, monkeypatch, later_fault):
    spec = importlib.util.spec_from_file_location('pace_console', ROOT/'python/pace_console.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, '__file__', str(tmp_path/'pace_console.py'))
    run = tmp_path/'data/pace/001'
    (run/'identification').mkdir(parents=True)
    (run/'identification/metadata.json').write_text(json.dumps({'config': config()}))
    (run/'identification/result.json').write_text(json.dumps({'status': 'completed'}))
    (run/'pose_complete.json').write_text(json.dumps({'status': 'completed'}))
    if later_fault:
        fault = tmp_path/'data/pace/002/approach_pose'
        fault.mkdir(parents=True)
        (fault/'result.json').write_text(json.dumps({'status': 'aborted'}))
    assert adapter.completed_pilot_in_session(controller(config(), Clock())) is (not later_fault)


@pytest.mark.parametrize('approach', [False, True])
def test_zero_return_and_full_rest(tmp_path, monkeypatch, approach):
    c = config()
    c['sample_hz'] = 100
    c['home'].update(approved=True, q_min=[-.3]*6, q_max=[.3]*6)
    c['poses'][0]['q_center'] = [.1]*6
    clock = Clock()
    ctrl = controller(c, clock)
    ctrl.q = np.full(6, 0. if approach else .1)
    monkeypatch.setattr(m, 'trajectories', lambda *_: iter([('tiny', np.arange(3)*.01, np.full((3, 6), .1))]))
    m.collect_pose_and_rest(ctrl, c, 'test', tmp_path/'run', safety_stop=lambda _: None,
                           confirm=lambda _: 'RUN', sdk=SimpleNamespace(JointState=State, __file__=m.__file__),
                           clock=clock.now, sleep=clock.sleep,
                           start_q=[0.]*6 if approach else None)
    np.testing.assert_array_equal(ctrl.q, 0)
    rest = np.load(tmp_path/'run/rest_zero/000_rest_zero.npz')['samples']
    assert rest[-1, 2]-rest[0, 2] >= 60
    assert (tmp_path/'run/pose_complete.json').exists()
    if approach:
        entry = np.load(tmp_path/'run/approach_pose/000_approach_pose_and_settle.npz')['samples']
        np.testing.assert_array_equal(entry[0, 6:12], 0)
        np.testing.assert_allclose(entry[-1, 6:12], .1)
        assert 'approach_pose' in json.loads((tmp_path/'run/pose_complete.json').read_text())['stage_quality']


def test_abort_does_not_return_home(tmp_path):
    c = config()
    c['home'].update(approved=True, q_min=[-.3]*6, q_max=[.3]*6)
    clock = Clock()
    ctrl = controller(c, clock)
    stops = []
    with pytest.raises(RuntimeError, match='declined'):
        m.collect_pose_and_rest(ctrl, c, 'test', tmp_path/'run', safety_stop=stops.append,
                               confirm=lambda _: 'NO', sdk=SimpleNamespace(JointState=State, __file__=m.__file__),
                               clock=clock.now, sleep=clock.sleep)
    assert ctrl.sent == 0 and len(stops) == 1
    assert not (tmp_path/'run/return_zero').exists()


@pytest.mark.parametrize('answer', ['RUN', 'cancel'])
def test_console_only_one_confirmation(monkeypatch, tmp_path, answer):
    spec = importlib.util.spec_from_file_location('pace_console', ROOT/'python/pace_console.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter.importlib.util, 'spec_from_file_location',
                        lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None)))
    monkeypatch.setattr(adapter.importlib.util, 'module_from_spec', lambda _: m)
    (tmp_path/'request.json').write_text(json.dumps(dict(feedback=[0.]*6, width=0.)))
    monkeypatch.setattr(adapter, 'collision_preflight', lambda *args: tmp_path)
    prompts, runs = [], []

    def reply(prompt):
        prompts.append(prompt)
        return answer

    def collect(ctrl, cfg, name, output, **kwargs):
        runs.append(cfg)
        np.testing.assert_array_equal(cfg['poses'][0]['q_center'], [0, 1., 1.4, -.25, 0, 0])
        assert all(v == 'unknown' for v in cfg['condition'].values())
        for segment in ('static', 'pilot_j1', 'pilot_j6', 'return_zero', 'rest_zero'):
            assert kwargs['confirm'](f'{name}/{segment}: unused prompt') == 'RUN'

    monkeypatch.setattr('builtins.input', reply)
    monkeypatch.setattr(m, 'collect_pose_and_rest', collect)
    ctrl = controller(config(), Clock())
    ctrl.get_robot_config().joint_pos_min[:] = [-3.14, -.05, -.1, -1.6, -1.57, -2]
    ctrl.get_robot_config().joint_pos_max[:] = [2.618, 3.5, 3.2, 1.55, 1.57, 2]
    class BorrowedState(State):
        def __del__(self):
            self.q[:] = 999  # Simulate an SDK view invalidated with its owner.
    ctrl.get_joint_cmd = lambda: BorrowedState(6)
    if answer == 'RUN':
        assert adapter.run(ctrl, ['pace', 'pilot'], lambda _: None)
        assert len(runs) == 1
    else:
        with pytest.raises(ValueError, match='已取消'):
            adapter.run(ctrl, ['pace', 'pilot'], lambda _: None)
        assert not runs
    assert len(prompts) == 1
    assert ctrl.sent == 0


def test_named_pose_library_replaces_clipped_center():
    spec = importlib.util.spec_from_file_location('pace_console', ROOT/'python/pace_console.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    c = m.template()
    c.update(protocol='short', repeats=2)
    centers = [adapter.selected_pose(c, name, 'train') for name in ('pose1', 'pose2', 'pose3')]
    assert len({tuple(q) for q in centers}) == 3
    np.testing.assert_array_equal(centers[0], [0, 1., 1.4, -.25, 0, 0])
    with pytest.raises(ValueError, match='Use train'):
        adapter.selected_pose(c, 'pose1', 'validation')
    with pytest.raises(ValueError, match='Use train'):
        adapter.selected_pose(c, 'anything', 'train')
    c['seed'] += 1
    with pytest.raises(ValueError, match='configuration differs'):
        adapter.selected_pose(c, 'pose1', 'train')


@pytest.mark.parametrize('issue', ['duplicate', 'gripper', 'host_delay', 'tracking', 'effort', 'envelope'])
def test_quality_issues_recorded_without_motion_abort(tmp_path, monkeypatch, issue):
    c = config()
    clock = Clock()
    ctrl = controller(c, clock)
    times = np.arange(20)*.002
    monkeypatch.setattr(m, 'trajectories', lambda *_: iter([('tiny', times, np.zeros((20, 6)))]))
    original = ctrl.get_joint_state
    def read():
        s = original()
        if issue == 'duplicate':
            s.timestamp = 1 + (ctrl.sent//2)*.004
        if issue == 'gripper':
            s.gripper_pos = .01
        if issue == 'tracking':
            s.q[2] = .16
        if issue == 'effort':
            s.tau[2] = 6.
        if issue == 'envelope':
            s.q[2] = .4
        return s
    ctrl.get_joint_state = read
    delayed = False
    def sleep(delta):
        nonlocal delayed
        if issue == 'host_delay' and ctrl.sent == 2 and not delayed:
            delta += .25
            delayed = True
        clock.sleep(delta)
    stops = []
    m.collect(ctrl, c, 'test', tmp_path/'run', safety_stop=stops.append,
              confirm=lambda _: 'RUN', sdk=SimpleNamespace(JointState=State, __file__=m.__file__),
              clock=clock.now, sleep=sleep)
    result = json.loads((tmp_path/'run/result.json').read_text())
    assert result['status'] == 'completed' and not result['data_quality_passed']
    assert ctrl.sent == 20 and not stops
    if issue == 'tracking':
        assert result['trials'][0]['max_tracking_error_rad'] == pytest.approx(.16)
        assert result['trials'][0]['tracking_error_quality_passed'] is False
    if issue in ('effort', 'envelope'):
        flag = 'feedback_effort' if issue == 'effort' else 'feedback_outside_envelope'
        assert flag in result['trials'][0]['threshold_flags']
    if issue == 'host_delay':
        rows = np.load(tmp_path/'run/000_tiny.npz')['samples']
        assert np.min(np.diff(rows[:, 4])) >= .002-1e-9
