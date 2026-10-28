# AGENTS.md — Tremor-Stabilizing Assistive Gimbal

## Purpose and context

This repository is an initial simulation prototype for a **Robotics for Healthcare** course project about assisting people with involuntary hand tremor during daily tasks, especially eating and drinking.

The broader concept proposal considers an actively stabilized handheld tool, a passive wrist damper, and adapted utensils. **This repository initially investigates the active two-axis handheld gimbal only.** The proposal is a design reference, not a fully validated specification. Its assumed component parameters, frequency ranges, predicted suppression percentages, and control bandwidths are hypotheses to test, not established results.

The research question is not simply whether a gimbal can track an orientation. We want to understand how a tool can suppress unwanted hand motion while preserving intentional motion, and what a two-actuator mechanism can and cannot accomplish when task-relevant position and orientation objectives conflict. The estimation/intent problem is open; do not assume that a particular filter (including WFLC) correctly separates tremor from voluntary movement.

## Existing infrastructure: first working prototype

Build a small, reproducible, inspectable **MuJoCo + Python** simulation containing:

1. A kinematically driven/moving handheld base with known prescribed motion.
2. A two-revolute-joint gimbal (nominally roll followed by pitch) and a simple utensil/tool body with plausible masses/inertias.
3. Basic pose and joint-state observations, initially using simulator ground truth; later add IMU/encoder-like measurements and noise.
4. Forward kinematics and an explicitly defined task-space reference (e.g. maintain tool upright, or follow a prescribed feasible orientation).
5. A transparent reference-to-joint mapping (analytic IK for a feasible orientation task, or explicitly documented task-constrained IK) and a simple joint controller driving **torque actuators**.
6. Logging and plots of base motion, desired/actual joint angles, desired/actual tool orientation, joint torques, and tracking errors.
7. A runnable experiment demonstrating (a) tracking with a stationary base and (b) stabilization against a prescribed oscillatory base rotation.

Start with a minimal and physically understandable implementation. A single-axis test is acceptable as an intermediate step. For this first version, an ideal joint torque actuator is sufficient: **do not implement motor FOC, custom PCBs, battery, BLE, complex state machines, or clinical user studies.**

## Current goal: system benchmark phase 1

The first prototype is complete. The current research path is:

`tremor estimation -> closed-loop stabilization -> functional task outcome`

Keep the existing 2-DoF CAD topology, controller, estimator implementations, and signal-level experiment results. Build a reproducible MuJoCo transport-and-pour task with a spoon collision proxy, rigid food particles, a target container, and a prescribed smooth voluntary trajectory. Phase 1 establishes the no-compensation and perfect-oracle bounds only; do not connect a signal estimator until those bounds are separated and repeatable. Use particle retention before intentional unloading, final target transfer, tremor-band output, task-motion preservation, and controller effort as system-level outcomes.

## Technical framing

### Frames and kinematics

Define all coordinate frames, joint axes, multiplication order, and world/body conventions clearly. A nominal model is

`R_tool_world = R_handle_world * R_gimbal(q) * R_tool_mount`,

where `q = [q_roll, q_pitch]`. The exact frame conventions must agree with the implemented MJCF model; verify them numerically rather than assuming the nominal equation automatically holds.

The gimbal has only two actuated DoFs. It **cannot generally track arbitrary 3D orientation and 3D position simultaneously**. In the initial experiment, choose a task that is actually feasible. For keeping a cup/tool upright, aligning its axis with world vertical can be more appropriate than prescribing an arbitrary full SO(3) orientation. If an orientation target is unreachable, do not silently claim exact tracking; use a documented projection/optimization or declare it infeasible.

### Signal and control path

Think of the architecture as separate, replaceable stages:

`base motion / sensors -> task reference generator -> IK / reference mapping -> joint servo -> MuJoCo plant -> measurements and evaluation`

- **Task reference** may be a fixed world-frame orientation (stabilization) or a time-varying desired orientation (motion following).
- **Joint mapping** computes feasible `q_des` from the current measured/known handle pose and the target. A Jacobian maps angular velocities, not arbitrary orientations directly. Avoid confusing world angular velocity, Euler-angle rates, and joint rates.
- **Joint servo** initially uses a straightforward well-documented PD controller, optionally with gravity compensation if needed. More elaborate position–velocity–current cascades may be introduced only after the basic plant/controller is validated.
- **Actuation** initially uses MuJoCo joint torque input and models saturation explicitly when appropriate.

Numerical differentiation of noisy orientation references is not mandatory for the first prototype. If desired velocities become necessary, prefer an analytically available reference derivative, direct gyro measurement, or a documented filtered reference generator rather than naïve unfiltered finite differences.

### Tremor versus voluntary motion

For controlled simulation tests, base motion can be generated from a known intentional component plus a prescribed disturbance. Their separate ground truths may be logged **for evaluation only**; do not feed unavailable ground-truth components to an estimator presented as realistic.

Begin with a fixed-reference stabilization task, which does not require tremor classification. Later, introduce a voluntary-motion-following task in which reference generation matters. Compare straightforward filters with other estimators only when the objective, assumptions, and evaluation signals are defined. Treat single-frequency sinusoidal assumptions and WFLC as testable baselines, not privileged solutions.

### Evaluation

At minimum distinguish:
- joint reference tracking;
- tool orientation error relative to the task target;
- attenuation of prescribed base disturbances;
- controller effort and saturation;
- errors from unavailable degrees of freedom.

For the current spoon task, also evaluate transport retention, target transfer, and preservation of intentional unloading. Do not equate low tremor RMS with improved user task performance. Human-in-the-loop interaction remains future work.

## Suggested organization (adapt to the existing repository)

```text
models/         # MuJoCo MJCF, assets
motion/         # prescribed base and tremor trajectories
kinematics/     # frame transforms, FK, IK / task mapping
control/        # joint controller, later estimator interfaces
experiments/    # reproducible runs, metrics, plots
scripts/        # simple run entry points
README.md       # setup, commands, assumptions, verified results
```

This is a suggested separation of concerns, not a mandate to create empty directories or introduce unnecessary abstractions. Prefer small files, explicit units (SI internally), configurable parameters, and deterministic tests.

## Development and verification practices

- Inspect existing files and conventions before making changes; do not replace a working project structure merely to match the suggested tree.
- Favor a working minimal vertical slice over a large speculative architecture.
- Make physical assumptions and coordinate frames explicit; provide basic sanity tests for FK/IK, zero-input motion, sign conventions, joint limits, and orientation errors.
- Keep simulator ground truth distinct from simulated sensor data and from estimated states.
- If claiming an experiment works, run it and report the actual results; do not invent performance metrics.
- Preserve repeatable experiment configurations and note relevant simulation timestep, controller timestep, solver options, and random seeds.
- If a requested modification affects the scientific assumptions, control objective, mechanical topology, or requires substantial changes outside the requested scope, explain the tradeoff and seek approval first.
- Ordinary inspection, diagnostics, temporary tests, dependency installation where permitted, and non-destructive debugging are encouraged. Avoid unrequested destructive refactors, removing user data, or changing unrelated code.

## Documentation and how this file evolves

`AGENTS.md` records stable project intent, design boundaries, and collaboration expectations. It **is not automatically synchronized with code changes**.

- Update `README.md` and experiment notes when implementation details, commands, parameters, and validated results change.
- Propose an update to `AGENTS.md` when the project goal, architecture, research questions, workflow constraints, or agreed scope materially changes. Do not silently rewrite the high-level research direction.
- Small routine code changes do not require changing `AGENTS.md`.
- If the current instructions conflict with a newly approved user direction, follow the newer explicit user instruction and point out the documentation change needed.

## Initial non-goals

No claim of clinical efficacy; no full neuromuscular or patient model; no human-subject tests; no assumption that two DoFs provide full 6D tool stabilization; no requirement to implement WFLC, learning-based control, custom electronics, or a production-ready device in the first iteration.
