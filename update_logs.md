## Update (2026.10.10) — PACE branch and unified X5 EEF

- Created independent feat/pace-link6 from the visualization branch; no change
  to its branch ref or pull request. Publish collection/console/collision code,
  pose library and required portable screen manifest, optional requirements,
  offline tests and documentation. Raw samples, local logs, environments and
  build artifacts remain excluded.
- X5 source default and actual console/offline entry points use link6 origin.
  No changes to other robot models, joint limits, gains or geometry. Existing
  installed wheels require the explicit entry-point setting; no local native
  rebuild or hardware execution.
- Local offline PACE/geometry regressions: 36 passed in 2.32 s. The SDK extension
  is absent from the local visualization environment; native entry-point dry-run
  was verified on the robot host before copying these entry points. Strict
  native IK orientation check there remains failing, including with the prior
  entry point; thresholds were not relaxed. This is not motion validation.

## Update (2026.10.09)
- 2026-10-10: At user request, lowered the excitation maximum to 1 Hz and split
  low/high as 0.1-0.5 / 0.5-1.0 Hz. One canonical CHIRP_BANDS_HZ drives short and
  extended protocols, console text and recorded metadata; validation reverses
  each band. Pilot stays 0.1-0.5 Hz. Amplitude +/-0.25 rad, all durations/ramps,
  repeats, poses, gains and QA-only threshold behavior remain unchanged.
  New metadata tags `chirps_up_to_1hz`, exact bands and sweep direction; older
  up-to-3-Hz runs remain untouched and must not be silently mixed with this
  changed experimental condition. Actual complete short plans remain 280 s;
  max target speed 1.37190 rad/s, acceleration 7.59887 rad/s^2 (sampled).
  Executed full SDK+CLab mesh screening again at 500 Hz, four poses, three
  gripper widths and expanded-envelope samples: all eight combinations passed
  the diagnostic 10 mm threshold. Artifacts: ignored
  `build/deployment/pose_screen_1hz_20261010`; replaced portable report with the
  new trajectory/source digests. Command: `OPENBLAS_NUM_THREADS=1
  build/pace_collision_env/bin/python python/pace_collision.py --library
  --reference-mjcf <CLab model> --out build/deployment/pose_screen_1hz_20261010`.
  Scoped pytest command below: 36 passed, including instantaneous phase-rate
  tests for both training/validation bands and no-abort threshold regressions.
  Final deployed files backed up and byte-verified. No physical collection,
  motor command, controller restart, native-limit change, commit or push.
  A running old plan is not changed by file deployment: stop it before starting
  a new command. Every new command uses a new timestamp directory.
- 2026-10-10: User clarified ALL PACE-added numeric abort thresholds must be
  removed, superseding the narrower tracking-only change below. Converted
  feedback speed/effort/envelope, SDK update interval, initial/segment target
  deviation to QA flags; removed plan speed/acceleration/envelope/data-budget
  rejection and redundant SDK numeric-limit prechecks. Actual entry geometry
  clearance and preflight state drift now warn without blocking RUN. Removed
  forced completed-pilot prerequisite after interrupted runs. Fixed trajectories,
  slow return, 60 s rest, amplitudes, seed, gains and native SDK protections are
  unchanged. Explicit cancellation, nonfinite data, invalid model/configuration,
  changed gains/backwards timestamps and native SDK exceptions still propagate.
  34 local tests pass with the same scoped pytest command below, including a
  combined mock that exceeds start, tracking, speed, effort, envelope and stale
  thresholds simultaneously and sends all 126 samples with no stop callback.
  Four complete trajectory digests remain identical; refreshed source binding
  without claiming a new geometry campaign. Final files deployed with byte
  verification and backups; no hardware motion, SDK owner construction, restart,
  data deletion, commit or push. Prior static/low/partial high data preserved.
- 2026-10-10: User clarified removal of the extra PACE tracking-error abort.
  Inspected run 20261009T165931_201325Z: high segment stopped at 9.678 s,
  SDK-output minus feedback J3 error 0.15064433 rad, crossing the retained
  0.15 rad application threshold. This was not the effort/stale-update stop.
  Removed PACE tracking-error rejection from start, live collection and console
  preflight; `max_tracking_error` remains explicitly QA-only. Store initial and
  per-trial max error, flag data quality and continue even above the threshold.
  Applies to entry/pilot/identification/return/rest. Native SDK protections,
  effort/stale/nonfinite/envelope checks and manual-console checks are unchanged.
  Added regression with 0.16 rad error: all 20 mock samples are sent, no stop,
  run completes with failed tracking QA. 31 local tests passed using
  `OPENBLAS_NUM_THREADS=1 PYTHONPATH=python build/pace_collision_env/bin/python
  -m pytest -q python/tests/test_pace_collection.py python/tests/test_pace_collision.py`.
  Rebound collector-source hash in collision manifest only after exact trajectory
  digests for all four poses matched; no geometry/trajectory change or new full
  geometry campaign claimed. Deployed files backed up and byte-verified.
  No motion/restart, gain/SDK-limit change, commit or push. Removing the stop
  does not fix dynamic tracking; sampled +/-0.40 rad geometry is not a bound
  on actual error without tracking abort.
- 2026-10-10 final collision-pose verification: repeated the complete SDK/CLab
  screen after the pilot-entry adjustment; all eight model/pose combinations
  passed with unchanged trajectory digests. Updated the portable screen manifest;
  all 30 local collector/geometry regression tests pass and `git diff --check`
  is clean. Byte-verified final deployed files. In the robot's original Python
  environment, mocked-controller integration of BOTH short entry and relocated
  pilot passed (exit 0), using the separate geometry subprocess with no SDK
  construction/CAN access. Native-environment pytest was unavailable (not
  installed); no dependency was added there. Local test invocation without
  explicit PYTHONPATH picked up unrelated ROS plugins/missing yaml; the scoped
  `PYTHONPATH=python` invocation above passed without changing ROS dependencies.
  No commits, hardware motion, controller restarts, or gain/limit changes.
- 2026-10-10: Superseded nearest-limit-clipped PACE centers after the operator
  reported self-collision. Offline replay of the recorded commanded path
  reproduced link2/link4 mesh intersection; this does not identify the exact
  physical contact from feedback alone. Added four fixed forward/upward centers,
  full SDK triangle-mesh/FCL screening and a read-only CLab geometry/kinematics
  cross-check (`pace_collision.py`, pose library/report, isolated requirements).
  Kept +/-0.25 rad, both frequency bands, seed, repetitions, gains, effort and
  tracking protection unchanged. Full 500 Hz short/zero-entry/return samples
  passed a 10 mm modeled nonadjacent clearance threshold in both models and
  three gripper widths: 4,204,116 nominal width-state combinations, 2,672,952
  unique checks, plus 115,800 expanded-envelope samples (center +/-0.40 rad).
  Commands: `OPENBLAS_NUM_THREADS=1 build/pace_collision_env/bin/python
  python/pace_collision.py --library --reference-mjcf <CLab model> --out <new
  build/deployment directory>`; `OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
  build/pace_collision_env/bin/python -m pytest -q python/tests/test_pace_collection.py
  python/tests/test_pace_collision.py` (30 passed before the pilot adjustment;
  final results recorded below). No CLab files modified.
  Runtime rejects unknown pose/split, stale code/model/trajectory digests, failed
  actual-entry geometry, or a changed start state before RUN. Preflight evidence
  stays under ignored `python/data/pace_preflight`; each acquisition is linked.
  Initial deployment was byte-verified with recoverable file backups and a
  separate checker venv; the existing controller environment was not modified.
  Remote mocked short-entry integration passed without CAN. The same check
  rejected the old zero-centered pilot: J3=-0.01 had only 7.86 mm link2/link4
  clearance. Pilot now first approaches pose1; local screening of its full
  entry/micro-motion/return passed (130,610 nominal states, 85,567 unique,
  closed gripper). This is sampled geometry, not dynamic or physical safety:
  adjacent assemblies/finger closure are excluded, camera/cables/payload/fixtures
  absent, and 3 Hz real tracking remains unverified. No hardware motion/restart.
- 2026-10-10: Added requested automatic entry for short acquisition: nearest
  center in SDK interval inset by 0.25 rad, full excitation prevalidated before
  approach, quintic entry and 5 s settling using existing slow-return parameters.
  One RUN covers entry/collection/return/rest. No collision checker or arbitrary
  distinct named pose presets introduced. Logs approach separately, keeps amplitude.
  Recover completed pilot in same SDK session after preflight rejection, but not
  after aborted/incomplete motion or gain/model/gravity change. Fixed target owner
  handling retained. 26 offline tests pass (entry/endpoints/rest/cancel/session
  recovery/fault rejection). No hardware movement/restart executed.
- 2026-10-10: User explicitly requested 5 rad/s instead of the console's 0.5.
  Changed console cap to min(native per-joint cap, 5); restored ±0.25 rad short
  bands to 0.1–1 and 1–3 Hz. Raised trajectory acceleration check 2 ->100 rad/s²
  to accommodate the selected signal, not a claim of motor capability. No gains,
  effort/native protections changed. PACE return remains 0.1 rad/s and 0.2 rad/s².
  This console-wide cap also applies to manual console motion. Requires operator
  restart; no controller restart or physical motion executed by agent.
  23 offline tests pass, including >3 and <=5 rad/s high-band target velocity,
  <=100 rad/s² acceleration for both train/validation. py_compile/diff check pass.
- 2026-10-10 (Asia/Shanghai): User explicitly selected ±0.25 rad baseline.
  Short train/validation chirps and holds now use ±0.25 rad; pilot remains ±0.01.
  Preserve existing 0.5 rad/s console cap: bands explicitly change to 0.03–0.15
  and 0.15–0.30 Hz, envelope ramps 8 s and holds 4 s. Not paper-equivalent
  bandwidth, not validated policy coverage. Added center-envelope preflight with
  actionable joint ranges, no silent amplitude clipping or automatic reposition.
  23 offline tests pass: amplitudes, train/validation, velocity bounds, 280 s
  duration and invalid-center rejection. No hardware motion/restart by agent.
- Full follow-up audit: inspected every configuration/preflight/sample/segment/
  return/rest exception. Host scheduling delays, duplicate snapshots and gripper
  deviation are now QA warnings, not motion aborts; failed data quality remains
  explicit at trial and pose levels. Late schedules shift without catch-up bursts.
  Retained normal console checks, native bounds, unchanged-gain/command continuity
  and data/schema integrity. Documented exact residual checks in collection guide.
  20 offline tests passed, including real idle velocity, borrowed SDK owner
  lifetime, delay without bursts, gripper deviation and duplicate quality flags;
  py_compile and git diff --check passed. No hardware validation or restart.
- 2026-10-09: Corrected residual hard-coded start speed threshold 0.03 rad/s,
  inconsistent with requested normal-console operation. Adjacent idle logs show
  quantized speed readings +/-0.0329666 rad/s; exact rejected sample was not saved.
  Removed that gate and rest-only tighter 0.02 rad tracking gate; retain ordinary
  tracking/native protections. Also retain JointState owner before pos().copy()
  in adapter and deployed console: pybind uses reference, and console log had
  subnormal first two target values after failure. Console fix needs restart;
  no process or hardware changed by agent. 17 offline tests pass, including
  initial velocity 0.0329666 regression. Actual failure sample cannot be recovered.
- 2026-10-09 (Asia/Shanghai), console usability follow-up: removed operator,
  payload and temperature questionnaire and repeated segment confirmations at
  user request. Console now displays a short Chinese summary and asks RUN once
  per pose, then automatically collects, returns and rests. Unknown metadata
  stays explicitly unknown; live configuration/provenance still saved. No robot
  config or protection changed. Validation: 16 offline tests passed, including
  one-confirmation execution and cancellation before collection. No hardware run.
- 2026-10-09 23:42 (Asia/Shanghai): Follow-up removes the 20-minute cap at operator
  request; pilot is now all six joints sequentially (190 s). Added explicit
  reviewed quintic return to joint zero and >=60 s monitored hold after each
  successful pose; aborted runs never automatically home. Added pace_console.py
  adapter reading live gains/gravity/SDK limits. It retains existing console
  tracking/effort/stale checks and disables an extra measured-speed abort by
  default, per request to use normal operating configuration. Return speed and
  arrival criteria are separate from identification limits. Remote console
  integration deployed without restarting or operating hardware.
  Validation executed: 14 offline pytest cases passed; py_compile passed for
  collector, adapter and staged console. Mock checks cover six-axis sequence,
  interruption, zero return and full 60 s rest. No hardware/timing validation.
  Active viewer JSONL pauses during collection; do not use it as a safety monitor.
  Deployment: initial SSH attempt timed out before any writes; reconnect succeeded.
  Seven remote files verified by SHA256; existing overwritten files backed up in
  build/deployment/pace_backup_20261009T154411Z on the robot computer. Remote
  .venv py_compile and test_connection.py --model X5 --interface can1 --console
  (without --connect) passed. Factory configuration: 0.002 s joint controller,
  zero preview, background enabled; active PID 699888 unchanged. No CAN opened.
- 2026-10-09 (Asia/Shanghai): Add `python/pace_collection.py`, collection guide,
  and offline tests for PACE-oriented joint data. User requested script only,
  with <=20 minutes effective data; no hardware connection, remote deployment,
  gain change or motion was executed. Default 40-second pilot; suggested main
  campaign is four poses x two short repetitions = 1120 seconds, plus pilot =
  1160 seconds. Existing controller integration and a site-specific stop callback
  remain required. No safe poses or effort thresholds are invented.
  Validation: 11 offline tests passed via `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  python -m pytest -q python/tests/test_pace_collection.py`;
  ruff check passed. Initial pytest auto-plugin loading failed on unrelated ROS
  `launch_testing` missing `lark`; no dependencies changed. Python snapshot
  logging is non-atomic and not hard real time; firmware/library provenance and
  real hardware timing/safety still need validation. Raw/partial NPZ and QA
  metadata are stored outside version control under existing `python/data`.
- Add an optional read-only Viser viewer for feedback status, joint/gripper positions, and EEF pose.
- Add articulated X5 gripper visuals while preserving the original arm kinematics and EEF frame.
- Add visualization dependencies, example, documentation, tests, and wheel packaging support.

## Update (2026.01.29)
- Update library for x86_64 Ubuntu20.04. If you have trouble loading the current libraries (version GLIBC_2.32 not found), please override `lib/x86_64/libsolver_20_04.so` to `lib/x86_64/libsolver.so` and `lib/x86_64/libhardware_20_04.so` to `lib/x86_64/libhardware.so` and recompile the package.
- The default libraries are compiled on Ubuntu 22.04

## Update (2025.08.29)
- Support gripper force control (see `python/examples/test_gripper_force_compensation.py`)
- Set conda-forge::soem version to 1.4.0 in conda environment files, thanks [Haoyu Xiong](https://haoyu-x.github.io/) for pointing out.
- If you encounter error `undefined symbol: EcatError`, this is because the soem version is updated to 2.0.0, which is not compatible with the current version of the controller. Please downgrade soem version to 1.4.0 (`conda install conda-forge::soem=1.4.0`).

## Update (2024.12.05)
- Add safety checks to zmq_server
- Unify the joint interpolator in both joint controller and cartesian controller for better smoothness
- Support trajectory updating and velocity interpolation
- Fix various bugs for gravity compensation, robot initialization etc.

## Update (2024.08.22)
- Enable one-step waypoint scheduling (see `python/examples`).
- Support EtherCAT-CAN adapter (follow the instructions in [EtherCAT-CAN setup](README.md#ethercat-can-setup)).
- Support arbitrary DoF robot arm (not only 6DoF); thoguh other DoF numbers are not tested yet.
- Allow setting up robot and controller configurations as arguments (see `config.h` and `test_joint_control.py`), thanks to [Yifan Hou](https://yifan-hou.github.io/)

When updating the sdk to your codebase, please first remove the entire `build` folder and run the building process again.
