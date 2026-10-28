# Tremor separation baseline benchmark

This Python module is independent of the MuJoCo controller. Its input and
output are angular velocity in **deg/s**, with one scalar estimator state per
axis. The Pinheiro ET recording provides measured gyroscope waveforms; its
offline 3–12 Hz component is explicitly a **tremor-band proxy**, not a labelled
pure tremor signal. Designed voluntary velocity and any seeded noise are
stored separately from the injected proxy.

## Current implementation

- Common `create_estimator` / `reset` / `step` interface.
- Causal fixed Butterworth BPF, designed and run as second-order sections
  without a SciPy runtime dependency; order and band are configurable.
- Popović 2010 ABPF, Wang 2016 adaptive SFFT, and Veluvolu–Ang 2010 BMFLC,
  with every necessary interpretation documented in the reproduction notes.
- Four designed voluntary cases and five controlled synthetic tremor cases.
- Common causal RMSE, Welch PSD band power, diagnostic lag, and timing code.
- Standardized plots, separate `paper_validation`, `controlled_verification`
  and `project_benchmark` outputs, provenance files, and SFFT bandwidth
  sensitivity results.

Run the checks and all three modes in the existing `me5418` environment:

```bash
conda run -n me5418 python -m pytest -q analysis/tremor_baselines/tests
conda run -n me5418 python -m analysis.tremor_baselines.run_baseline_benchmark --mode all
```

Choose `--mode paper_validation`, `--mode controlled_verification`, or
`--mode project_benchmark` to run them separately. Main settings are in
`benchmark_config.json`. Raw arrays and numerical
provenance are in `experiments/results/tremor_baselines/{mode}/`; common plots
and comparison CSVs are in `reports/tremor_baselines/{mode}/`. The older
BPF-only `run_available.py` script remains available for development, but its
superseded `tremor_baselines_preliminary/` outputs have been removed.

A short Chinese guide to the checked results is in
[PROJECT_SUMMARY.md](../../reports/tremor_baselines/PROJECT_SUMMARY.md).

## Second-round diagnostics (unchanged baseline algorithms)

Run `conda run -n me5418 python -m analysis.tremor_baselines.run_next_round --part all`
from the repository root. `--part` also accepts `sfft`, `bmflc`,
`abpf`, and `replication`. This adds SFFT reconstruction snapshots and a
known-tone check, BMFLC transient-state traces, ABPF modal-offset sensitivity,
and within-subject cross-window/cross-axis replication. Outputs are separated
by topic under `reports/tremor_baselines/next_round/` and
`experiments/results/tremor_baselines/next_round/`. The Chinese
[second-round report](../../reports/tremor_baselines/next_round/第二轮诊断与复现实验报告.md)
states the conclusions and the one-subject limitation.

## Third-round cross-subject and mechanism experiments

Five Pinheiro participants' gyro tables have now been extracted. The
[third-round report](../../reports/tremor_baselines/round3/第三轮跨受试者与机制实验报告.md)
contains subject-level distributions, fixed/adaptive ABPF ablation, unchanged
ABPF response characterization, and Huber/leaky BMFLC comparisons. The
original estimators remain in their existing files; experimental variants
are isolated in `round3_variants.py` and the runner is `run_round3.py`.

```bash
MPLCONFIGDIR=/tmp/tremor_mpl conda run -n me5418 python -m analysis.tremor_baselines.run_round3 --part all
```

Results are grouped under `reports/tremor_baselines/round3/` and raw arrays
and frozen development parameters under
`experiments/results/tremor_baselines/round3/`. The primary comparison uses
one preselected complete table per person, three non-overlapping windows,
and all three gyro axes; 18 tables were audited, with one too short for the
protocol. New BMFLC parameters were selected using synthetic development
signals only. The injected proxy remains a frequency-band surrogate rather
than an independently labelled physiological tremor trace.

## Fourth-round kinematic voluntary-motion experiment

The two-stage `KinematicKFThenBMFLC` and joint
`JointKinematicBMFLCKF` estimators are isolated in `round4_estimators.py`;
`run_round4.py` runs synthetic development, controlled tests, an amplitude-ratio
sweep, and the unchanged five-subject final protocol. The joint estimator uses
a causal linear time-varying KF with a constant-angular-acceleration voluntary
state and the existing BMFLC Fourier basis. A voluntary velocity random-walk
variant is an ablation. Original baseline files and prior results are unchanged.

```bash
MPLCONFIGDIR=/tmp/tremor_mpl conda run -n me5418 python -m analysis.tremor_baselines.run_round4 --part all
```

The [Chinese report](../../reports/tremor_baselines/round4/第四轮运动学与震颤联合估计实验报告.md)
and [method notes](ROUND4_METHOD_NOTES.md) document the mixed result, frozen
synthetic-only parameter selection, units, and literature limits. Data arrays
are in `experiments/results/tremor_baselines/round4/`; plots and tables are
grouped in `reports/tremor_baselines/round4/`.

## Fifth-round recorded task-motion validation

Round 5 reads the smartwatch activity dataset in `Dataset/data3/`, converts its
three raw gyro axes from rad/s to deg/s, and fixes a subject-disjoint 15/5
development/test split. It compares causal random-walk, constant-acceleration,
and Singer prediction on the recorded task motion before injecting a resampled
Pinheiro 3–12 Hz proxy into held-out task recordings. Original BMFLC, Huber,
and ABPF are compared with the Round-4 joint models and a joint model whose
voluntary process variance comes from the development set. It leaves all
earlier results intact.

See [Round 5 method notes](ROUND5_METHOD_NOTES.md) for extraction dependencies,
units, frozen parameters, and commands. The [Chinese Round 5 report](../../reports/tremor_baselines/round5/第五轮真实主动运动先验与联合震颤估计实验报告.md)
summarizes prediction, activity-wise separation, bidirectional leakage, and
the amplitude-ratio sweep. Frozen splits and raw traces are under
`experiments/results/tremor_baselines/round5/`; figures and CSVs are under
`reports/tremor_baselines/round5/`.

## Sixth-round ProMP representation check

Round 6 uses the same frozen 15/5 smartwatch subject split as Round 5. It fits
one population ProMP per activity and gyro axis from complete recorded trials,
tests held-out representation and causal prediction against the frozen Round-5
random walk, then measures likelihood and weight-space changes after injecting
the Pinheiro band proxy. The primary spiral task fails the fitted trajectory
representation check, so the conditional ProMP-residual separation comparison
is deliberately skipped. The [Chinese report](../../reports/tremor_baselines/round6/第六轮ProMP主动运动表示验证实验报告.md)
and [method notes](ROUND6_METHOD_NOTES.md) give the results, oracle-phase
limitation, data boundaries, and run commands. Frozen models and per-sample
traces are under `experiments/results/tremor_baselines/round6/`; tables and
figures are under `reports/tremor_baselines/round6/`.

The [data5 repeat-trial check](../../reports/tremor_baselines/round6_data5/第六轮ProMP在data5重复试次上的验证报告.md)
reruns this sequence on P01's FLS peg-transfer practice, using sessions 01–02
for development and session 03 as a chronological holdout. It analyzes both
wrist and tool gyro without mixing them, and keeps its outputs under
`round6_data5/`. Full-trial ProMP reconstruction still fails the primary
right-wrist representation check, so the conditional separation stage is
skipped. See [method notes](ROUND6_DATA5_METHOD_NOTES.md) for the trial manifest,
proxy construction, unit caveat, and commands.

## Seventh-round causal prefilter and robust BMFLC ablation

Round 7 keeps the original BMFLC and Round-3 Huber recursion unchanged and
tests a causal 2 Hz Butterworth high-pass before either update law, producing
a clean raw/prefilter × LMS/Huber comparison. A 3–12 Hz band-pass and a small
high-pass cutoff sweep are separate sensitivity checks. Five Pinheiro subjects,
their fixed windows, three gyro axes, and four designed voluntary scenarios
are reused without final-data tuning. Prefiltering reduces voluntary energy
and Fourier-weight contamination but its phase response worsens causal tremor
RMSE relative to raw Huber. See the [Chinese report](../../reports/tremor_baselines/round7/第七轮预滤波与稳健BMFLC实验报告.md)
and [method notes](ROUND7_METHOD_NOTES.md). Tables and plots live under
`reports/tremor_baselines/round7/`; frozen protocol and representative traces
under `experiments/results/tremor_baselines/round7/`.

The project benchmark takes 0–15 s of the selected recording for independent
ABPF modal calibration and 20–32 s for evaluation. Its injected 3–12 Hz
recorded waveform is an **offline tremor-band proxy**, not an independent
physiological tremor label. All four estimators receive the same mixture in
each case. The paper-style synthetic checks establish behavior under stated
conditions; they do not reproduce the publications' inaccessible raw patients
or unspecified noise realizations.

Read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) and
[LITERATURE_AMBIGUITIES.md](LITERATURE_AMBIGUITIES.md) before interpreting
results.
