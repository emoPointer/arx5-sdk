"""Offline geometry regression: PYTHONPATH=python python -m pytest this_file.

Requires the isolated collision requirements plus pytest. No SDK/CAN use,
no model mutation and no hardware commands. Uses the actual full mesh assets.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('fcl')
from pace_collision import ExactCollisionModel, library_config, screen, trajectory_digest
import pace_collection

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def model():
    return ExactCollisionModel()


def test_old_commanded_collision_is_reproduced(model):
    old = [-.0089967265, .1170780996, -.0392872905,
           -.0381693592, .1884356784, -.0065514657]
    result = screen(model, [old], [0.], .01)
    assert not result['passed']
    assert result['clearance_lower_bound_m'] <= 0
    assert set(result['worst']['pair']) == {'link2_hull_0', 'link4_hull_0'}


def test_zero_and_new_centers(model):
    states = [[0.]*6] + [library_config(n)[1]['q_center'] for n in ('pose1', 'pose2', 'pose3', 'heldout')]
    result = screen(model, states, [0., .044, .088], .01)
    assert result['passed']
    with pytest.raises(ValueError):
        model.set_state([0]*6, .1)


def test_report_matches_executable_trajectories_and_sources():
    report = json.loads((ROOT/'python/pace_pose_screen.json').read_text())
    assert report['passed']
    assert len(report['results']) == 8
    for filename, expected in report['code_sha256'].items():
        assert hashlib.sha256((ROOT/'python'/filename).read_bytes()).hexdigest() == expected
    for filename, expected in report['sdk_source_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected
    for name, expected in report['trajectory_sha256'].items():
        c, pose, _ = library_config(name)
        assert trajectory_digest(list(pace_collection.trajectories(c, pose))) == expected
    assert all(x['passed'] and x['tracking_envelope_sample']['passed'] for x in report['results'])
