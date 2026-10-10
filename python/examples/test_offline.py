"""Validate SDK import, model files and FK/IK/dynamics without opening hardware.

Inputs: --model (required); URDF and parameters come from the installed SDK.
Outputs: JSON report on stdout; no files are created or overwritten.
Prerequisites: the repository .venv with arx5-interface, numpy and click installed.
Run from any directory:
  /path/to/arx5-sdk/.venv/bin/python -I /path/to/arx5-sdk/python/examples/test_offline.py --model X5
No controller is constructed; this script never opens CAN or sends motor commands.
An exception or failed numerical check exits nonzero. This is a software smoke
check, not a calibration or physical accuracy validation.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import arx5_interface as arx5
import numpy as np


def rotation_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    config = arx5.RobotConfigFactory.get_instance().get_config(args.model)
    if args.model == 'X5':
        config.eef_link_name = 'link6'
    model_path = Path(config.urdf_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    solver = arx5.Arx5Solver(str(model_path), config.joint_dof,
                             config.joint_pos_min, config.joint_pos_max,
                             config.base_link_name, config.eef_link_name,
                             config.gravity_vector)
    # Interior configuration avoids relying solely on a potentially singular home pose.
    q = 0.55 * np.asarray(config.joint_pos_min) + 0.45 * np.asarray(config.joint_pos_max)
    pose = solver.forward_kinematics(q)
    seed = np.clip(q + 0.01, config.joint_pos_min, config.joint_pos_max)
    status, recovered = solver.inverse_kinematics(pose, seed)
    if status != 0:
        raise RuntimeError(f'IK failed: {status}, {solver.get_ik_status_name(status)}')
    recovered_pose = solver.forward_kinematics(recovered)
    position_error = float(np.linalg.norm(recovered_pose[:3] - pose[:3]))
    relative_rotation = rotation_matrix(recovered_pose[3:]) @ rotation_matrix(pose[3:]).T
    angle_error = float(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
    gravity = solver.inverse_dynamics(q, np.zeros(config.joint_dof), np.zeros(config.joint_dof))
    if not all(np.isfinite(v).all() for v in (pose, recovered, recovered_pose, gravity)):
        raise RuntimeError('Non-finite FK/IK/dynamics output')
    passed = position_error <= 1e-4 and angle_error <= 1e-3
    print(json.dumps(dict(result='PASS' if passed else 'FAIL', sdk_version=importlib.metadata.version('arx5-interface'),
                         module=arx5.__file__, model=args.model, urdf=str(model_path),
                         dof=config.joint_dof, ik_status=int(status),
                         eef_link=config.eef_link_name,
                         zero_joint_pose=solver.forward_kinematics(np.zeros(config.joint_dof)).tolist(),
                         position_error_m=position_error, orientation_error_rad=angle_error,
                         gravity_torque_nm=np.asarray(gravity).tolist(), hardware_opened=False), indent=2))
    if not passed:
        raise RuntimeError(f'FK/IK mismatch: {position_error} m, {angle_error} rad')


if __name__ == '__main__':
    main()
