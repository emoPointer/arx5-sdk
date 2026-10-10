# PACE-oriented ARX5 data acquisition

## X5 end-effector and entry point

The EEF is exclusively the **link6 body origin**, with zero tool offset.
The X5 source default and the deployed console/offline entry points select it
explicitly. Existing vendor wheels need this explicit configuration; changing
the C++ header alone does not update an installed wheel.

With the project's existing SDK environment, check without opening hardware:

```bash
python python/examples/test_connection.py --model X5 --interface can1 --console
python python/examples/test_offline.py --model X5
```

Only adding `--connect` to the console command opens hardware and may move the arm.
Run that only after operator preflight, with no other controller owning the arm.
Check startup output `eef_link: link6`. PACE uses the owning controller's config.
The offline check keeps strict precision thresholds and can fail on SDK IK
orientation residuals; a dry-run is not a physical safety or accuracy validation.

This collector is implemented and mock-tested, **not hardware-tested**. It does
not run PACE fitting, certify trajectory collision safety, or fix servo errors.
It never constructs a controller or takes over CAN. Live acquisition must run
inside the existing joint-controller process with all other command writers
suspended. The robot console `test_connection.py` integrates `pace_console.py`.
The currently running process is not restarted automatically; use the updated
entry point only after safely exiting the previous owner.

## Data budget and experimental design

PACE reports 20–60 s chirp sequences (typically 20–40 s for full robots), not an
hour of continuous motion: https://arxiv.org/html/2509.06342v1, sections 2.1/3.1.1.
The following is our arm-specific design, not a paper reproduction requirement.

| Protocol | Excitation | Data time per pose, one repeat |
|---|---|---|
| `pilot` (default) | 10 s static + J1..J6 sequential 20 s chirps, each with 5 s holds before/after | 190 s |
| `short` | 20 s static + 40 s low chirp + 30 s high chirp (10 s padding each) + 40 s smooth bidirectional holds | 150 s |
| `extended` (explicit opt-in) | Individual joint chirps/holds and mixed sine | 850 s |

The user removed the 20-minute cap. Suggested campaign:

1. One pilot at the fixed pose1 center, `repeats=1`: 190 s. Inspect timing, currents/effort,
   speed, tracking, clearance, sound and temperature before continuing.
2. Four reviewed poses with `short`, `repeats=2`: 280 s each, total 1120 s.
3. Excitation/static data total = 1310 s = **21 min 50 s**, plus five returns,
   five minimum 60-second zero holds and operator pauses. Three main poses are `train`; one is
   `validation`. Do not tune on validation. Repeated trials vary phase/direction;
   validation reverses chirp sweep but preserves the same ±0.25 rad amplitude.

The console requires only one RUN confirmation per pose, then automatically
executes all segments, return and rest. No metadata questionnaire is shown;
unavailable condition metadata is saved explicitly as unknown. The lower-level
API still supports per-segment confirmation for custom callers. Pause between
poses depends on observed condition, not a
guaranteed cooling timer: energized holding can itself generate heat. Use the
site's support/shutdown procedure if cooling requires unloading. No temperature
telemetry is exposed by the Python state API; manual checks remain necessary.

Short-protocol low chirp is **0.1–0.5 Hz**; high chirp is **0.5–1.0 Hz**, both with
±0.25 rad target amplitude and 8 s envelope ramps. Simultaneous joint motion
needs a reviewed FULL envelope, not just independent joint limits. Smooth holds
use ±0.25 rad and 4 s ramps. Train and validation use equal amplitudes.
These are explicitly ARX5 adaptations, NOT the paper's exact excitation protocol.
At user request, the console command cap is now 5 rad/s, bounded by each native
SDK joint limit. This is not the actual sinusoid speed: at the new 1 Hz maximum,
±0.25 rad corresponds to approximately 1.57 rad/s and 9.87 rad/s² before envelope
effects. The legacy 100 rad/s² value is QA-only, not a measured motor capability.
No gain or native
effort protection changes. Adequacy and actual tracking require hardware validation.
Short acquisition now uses the four fixed centers in `pace_pose_library.json`;
the previous nearest-limit-clipped center is removed after a self-collision report.
Offline replay reproduced link2/link4 intersection in the old commanded path.
`pose1=[0,1,1.4,-.25,0,0]`, `pose2=[-.4,1.2,1.8,.15,0,0]`,
`pose3=[.4,1,1.5,-.35,0,.2]`, `heldout=[.15,1.1,1.65,-.1,0,-.2]` rad.
Names and train/validation assignments are enforced. Amplitudes are not reduced.
All full 500 Hz trajectories, zero entry/return, three gripper widths (0/.044/.088 m),
and 4,825 sampled offsets in center +/-0.40 rad per pose passed the 10 mm
nonadjacent-mesh clearance threshold in both SDK and CLab models. Exact triangle
queries resolve convex-hull false positives. The two models have slightly
different joint origins; this is a cross-check, not identical-model evidence.
Results/source digests are recorded in `pace_pose_screen.json`.

Before RUN, a separate offline process checks the actual start/entry path and
rejects stale trajectory/model hashes. Clearance below the report threshold is
diagnostic only and does not block RUN. Pilot first enters pose1, then checks
each joint with the unchanged +/-0.01 rad signal and returns to zero. Its actual
entry, micro-motions and return are screened too. A pilot centered at zero was
rejected because J3=-0.01 gives only 7.86 mm link2/link4 clearance.
The checker never imports the SDK or opens CAN.
The owning controller continues holding; state changes during screening or
while waiting for RUN are reported, not rejected. Diagnostic requests/results remain
under `python/data/pace_preflight`. The console then approaches with the existing
slow quintic planner and 5 s settling. No automatic home occurs on failure.
This is sampled geometry, NOT continuous-time or physical safety certification:
adjacent assemblies/finger closure are excluded; cameras, cables, payloads and
external fixtures are absent. A support plane at base z=0 is included. Manual
inspection remains necessary, and physical tracking is not validated by collision tests.

Install the checker in its separate environment, not the controller environment:
```bash
python3 -m venv build/pace_collision_env
build/pace_collision_env/bin/python -m pip install -r conda_environments/requirements_pace_collision.txt
OPENBLAS_NUM_THREADS=1 build/pace_collision_env/bin/python python/pace_collision.py \
  --library --reference-mjcf /path/to/CLab/mujoco_arx5/models/arx5_robot.xml \
  --out build/pose_screen_unique_run
```
The output directory must not exist. Runtime needs only SDK assets, not CLab.
An optional `build/pace_geometry/X5_gripper.urdf` holds the exact screened gripper
description if the deployment's original file has unrelated text differences.
Pilot uses each joint sequentially, amplitude 0.01 rad at 0.1–0.5 Hz. These are **candidate**
trajectories, not certified-safe limits or proof of sufficient excitation.
1 Hz excitation does not establish arbitrary 20 Hz-policy bandwidth fidelity.
Inspect signal-to-noise and held-out error before increasing bandwidth/data.

The 2026-10-10 frequency revision changes both short and extended low/high chirps
to the new bands; validation sweeps descend through the same bands. Pilot stays
at 0.1–0.5 Hz. Repetitions, holds, ramps and durations are unchanged (short=280 s
at two repeats). This is a different experimental condition from the earlier
up-to-3-Hz runs: keep those raw files, but do not silently mix campaigns.
New metadata records `excitation_profile`, `chirp_bands_hz` and sweep direction.
Each new command creates a new timestamp directory; it never resumes or overwrites
an earlier run. Stop any currently running old command before starting the new one;
hot deployment changes the next command, not a plan already running in memory.

## Prepare without hardware (existing Python environment with numpy)

From the SDK repository root:

```bash
python python/pace_collection.py --template /tmp/pace-config.json
```

Fill `condition` (serial, firmware, payload/attachments, mount, temperature notes,
operator), and `feedback_effort_abort` from reviewed limits; never guess them.
Use explicit `unknown` metadata where unavailable; unknown firmware reduces
reproducibility. The placeholder pose is deliberately NOT executable.
Add `poses` with unique name, split, `q_center`, `q_min`, `q_max` (six radian
values each) and `approved=true` ONLY after clearance review. Include room for
actual tracking error as well as commanded motion. The collector does not check
Cartesian clearance, self-collision, cables, or joint order correctness.

The center is the existing **holding target**, not necessarily feedback angle:
this preserves the steady-state error being investigated. For short acquisition,
the adapter may select a new center as described above; optional start_q enables
the logged automatic approach. After a successful
pose, `collect_pose_and_rest` follows an explicitly approved smooth joint path
to six target zeros, waits for arrival/settling, then records >=60 s holding.
An abort never triggers an automatic return. Keep payload, gains, gravity mode and gripper
opening fixed. Do not change Kp to 100 automatically.

```bash
python python/pace_collection.py --plan /tmp/pace-config.json \
  --out python/data/pace-plan-001
```

This validates all trajectories and exports NPZ for offline inspection; it does
not import SDK or connect CAN. New directories only; never overwrite data. The
plan command checks aggregate duration against `max_planned_seconds` (default 7200,
configurable; no hard 20-minute cap). Fill and approve the `home` envelope too.
For the four-pose main config use `protocol=short`, `repeats=2`,
`max_planned_seconds=7200` (pilot stored separately). Repeating runs consumes
additional budget: the operator must track cumulative actual runs, including
failed attempts; there is no cross-process session-budget database.

## Robot console startup (operator runs; no hardware validation performed)

From the robot SDK repository, after the old controller owner has safely exited:

```bash
.venv/bin/python -I -u python/examples/test_connection.py --model X5 --interface can1 --connect --console
```

Do not start two CAN owners. `release` exits the previous console and unloads
the arm; support it first. Startup initializes motors. Do not pipe through tee.
After READY and checking the current pose and clearance, enter:

```text
pace pilot
```

Review the concise Chinese workflow summary and clearance, then type RUN once.
There are no operator/payload/temperature questions or repeated segment prompts.
The pilot slowly enters pose1, checks all six joints, returns to zero, then holds
at zero for >=60 s. Its amplitude remains +/-0.01 rad.
Return and rest execute automatically after successful acquisition. To enter the
fixed screened center automatically and collect, use:

```text
pace short train pose1
```

Repeat at other approved poses using `pose2`, `pose3`; use
`pace short validation heldout` for the held-out pose. Inspect condition/heat before
each pose. Geometric screening does not establish hardware dynamic safety.
The console no longer requires a completed pilot before short acquisition.
Pilot is optional; a previous interruption does not force it to be repeated.

The adapter reads live Kp/Kd, gravity mode and SDK limits; it does not change them
or actuate the gripper. PACE tracking error above 0.15 rad is now recorded and
flagged as data quality ONLY, including entry/start, acquisition, return and rest;
it does not abort. The legacy `max_tracking_error` field is the QA threshold.
All other PACE-added numeric protection thresholds are also QA-only: feedback
speed/effort/envelope, stale-update interval, starting-target deviation, timing,
and planned speed/acceleration. No PACE-added numeric threshold abort is enabled.
SDK/firmware protections are unchanged. The console's cap was explicitly changed to 5 rad/s at user
request, bounded by native SDK limits; this also affects manual console motion.
This means the current console setup, not a separately verified policy-deployment
configuration. Optional `--joint3-kp` retains its existing meaning; do not add it
unless that is the intended inference setup.

Zero-return uses a 0.1-rad/s planned maximum and smooth acceleration, then 5 s
settling followed by the 60 s rest. Tracking error is QA only; there is no
0.03-rad/s start check or 0.02-rad arrival check.
The schema's legacy arrival_tolerance field is retained but not enforced. Holding is
energized and is not a guarantee of cooling; this API exposes no temperatures.

Outputs: `python/data/pace/<UTC timestamp>/` with optional `approach_pose`, `identification`, `return_zero`,
`rest_zero` and `pose_complete.json`. Ctrl+C cancels and invokes the existing
measured-position hold; it is not an emergency stop. Process failure can unload.

## Integration API

In the existing process, import from this repository's `python` directory as
already done for other SDK utilities. A minimal integration is:

```python
from pace_collection import collect_pose_and_rest

# config loaded from reviewed JSON; controller is the ALREADY OWNED controller.
# Suspend other motion commands. A read-only viewer can remain, but check timing.
try:
    collect_pose_and_rest(controller, config, "reviewed_pose_name", output_directory,
            safety_stop=existing_site_stop_handler)
except (Exception, KeyboardInterrupt) as exc:
    print(f"Acquisition stopped: {exc}")
    # Remain in the existing owner's safe handling loop. Do not exit/destroy it.
```

`existing_site_stop_handler(reason)` must be wired to the actual, reviewed stop
policy before use. **There is intentionally no default stop callback**: suddenly
unloading, holding a bad command, and moving to the measured pose each have
different risks. It must not silently ignore errors. The caller must retain
controller ownership on failure. SDK destruction may unload the arm.

Collector does not change gain/mode/gripper, call SDK home/calibrate/damping, or
destroy controller. The wrapper explicitly plans return/rest; basic `collect`
alone leaves the last target holding. Ctrl+C/operator cancellation, nonfinite
data, invalid configuration/model, changed gains and backwards SDK timestamps
still terminate execution as explicit cancellation or invalid data, not numeric
protection thresholds. Native SDK exceptions still propagate. Aborts
invoke the callback and preserve partial data. These software checks are NOT an
independent hardware watchdog. Per-motor stale CAN feedback cannot be detected
reliably from the SDK aggregate timestamp. An external emergency stop/operator
and the existing SDK protections remain required. Do not bypass protections to
obtain a completed dataset.

## What is saved and how to use it

`metadata.json`: complete requested config, live robot/controller settings,
gains, joint order, conditions, clock/units conventions, column map, URDF hash,
imported SDK-module hash, git HEAD/status, Python/numpy/platform versions.
`controller_model.urdf`, `collector_source.py`, `source_changes.patch`: provenance.
The SDK module hash does not hash all dynamically linked libraries: separately
archive the deployed SDK wheel/build and firmware identity before formal fitting.

Each trial NPZ contains `samples`, `planned_time`, `planned_q`. Recorded at a
requested 500 Hz (not guaranteed hard real time):

- Python submitted q, zero desired dq and zero user feedforward semantics;
- SDK output q/dq/feedforward AFTER SDK interpolation/limits/gravity, BEFORE
  motor-protocol conversion; not evidence of exact motor CAN receipt;
- measured q/dq, SDK-derived effort, gripper feedback, SDK state timestamps;
- host monotonic read begin/end and command submit begin/end times.

State and output reads are separate, not atomic; output is read BEFORE the
current Python command. Use timestamps, never match same row as simultaneous
command-response. SDK timestamps describe SDK updates, not per-motor sensor
times. Duplicate snapshots are retained, measured in QA; >10% marks data quality
as failed but does not interrupt the sequence. Host gaps >200 ms and gripper
deviation >2 mm likewise flag QA without stopping motion. Host scheduling delays
shift subsequent submission times to avoid catch-up bursts; actual times remain
recorded, so this is not guaranteed exact-frequency excitation. `result.json`
and `pose_complete.json` separate completion from `data_quality_passed`.
Native 500 Hz communication cycles may be missed by Python polling. Use a native
SDK logger if acquisition QA or delay-identification precision is insufficient.

Effort is not calibrated torque, and raw current/temperature are not available
through this API. No force sensor/external TCP measurement is collected. Keep
original irregular timestamps; do not silently deduplicate, smooth or resample.
No direct PACE `.pt` export yet: conversion requires an explicit choice of command
boundary, replay clock, gravity compensation and joint order in our sim adapter.

`result.json`: completion/abort reason, stop-handler failure (if any), per-trial
effective host rate, largest gap, duplicate ratio. Partial NPZ preserves offending
measurements; NaN send times mean the row was NOT submitted. Files are written at
trial boundaries; process kill/power loss can lose the active trial (no disk
hard-real-time logger). Fault-free completion is acquisition QA, not dynamics-fit
validation. Keep aborted runs, do not train silently on them.

Changing payload, mounting, firmware, compensation, or thermal regime may require
new data. More data today cannot guarantee no recollection in future.

## Offline tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q python/tests/test_pace_collection.py
```

Includes candidate trajectory limits/durations, six-axis pilot ordering,
train/validation separation, mock collection, operator refusal, optional speed
fault, interrupt, duplicate feedback, zero return and 60-second rest.
No real CAN/SDK control loop has been tested by these tests.

## Runtime limit audit (2026-10-09)

- Removed: unconditional 0.03 rad/s start speed; rest-only 0.02 rad tracking.
- QA only, never PACE threshold abort: tracking error, gripper deviation,
  duplicate snapshots, host gaps, stale SDK updates, feedback speed/effort,
  feedback/target envelopes, start-target deviation, planned speed/acceleration.
  Bad-quality data remains explicitly flagged. No duration-budget abort.
- Finite data/model/configuration integrity checks and native SDK protections
  remain. Manual-console motion
  outside PACE retains its own checks; the PACE path does not call that planner.
- The +/-0.40 rad collision-envelope samples are not a bound on real tracking
  error. Disabling tracking abort does not validate tracking or physical safety.
- Retained acquisition integrity checks: valid schema/model/controller mode,
  matching unchanged gains, monotonic SDK timestamps and operator confirmation.
- Trajectory generation is unchanged, including the requested slow return.
  Removing extra threshold rejection does not increase native SDK limits.
- Console measured-speed override is disabled; configuration/schema checks and
  I/O errors are not hidden. A full run can still stop on a genuine fault.
- Fixed SDK JointState lifetime when copying target angles in adapter/console.
  Restart the owning console once to load its fix; do not start a second owner.
