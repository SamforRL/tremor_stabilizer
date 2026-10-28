# Public tremor data: acquisition and evaluation plan

The earlier MuJoCo/WFLC run outputs and their separate reports were removed.
Historical numerical descriptions below document what those scripts produced;
run the referenced scripts to regenerate files. The current four-method
signal benchmark is documented in
[`analysis/tremor_baselines/README.md`](../analysis/tremor_baselines/README.md).

The user placed the complete downloads in `Dataset/data1/` (Pinheiro) and
`Dataset/data2/` (ET accelerometry). All eight files match the MD5 values
published on the two Zenodo records. `Dataset/` and derived waveforms under
`data/processed/` are ignored by Git. The download script remains available
for reproducible acquisition into `data/raw/` when terminal network access is
available:

```bash
python scripts/download_tremor_datasets.py
python scripts/download_tremor_datasets.py --verify
python scripts/inspect_tremor_mat.py data/raw/et_accelerometry/acc_signal_database.mat \
  data/raw/pinheiro_2024/p02_59_f_et_LA_20yr.mat
```

The default downloads all published files, about 148 MB. On this machine,
system Python has a compatible SciPy when user site packages are disabled;
the `me5418` environment does not currently have SciPy. Inspect the actual
files with:

```bash
PYTHONNOUSERSITE=1 python3 scripts/inspect_tremor_mat.py \
  Dataset/data2/acc_signal_database.mat Dataset/data1/p02_59_f_et_LA_20yr.mat
```

The ET MAT file has 58 single-channel hand records (29 subjects, two hands),
each with `posture`, `rest`, `Fs`, `subject`, and `hand`. `Fs=5000` Hz and
each condition lasts 50–90 s. The numerical acceleration units/calibration
are not established by the included README. The Pinheiro ET MAT file contains
two MATLAB `table` objects named `p02_1` and `p02_2`, stored as opaque MCOS
objects. SciPy alone cannot recover their table columns. We extracted them
with the optional [matloader](https://github.com/anntzer/matloader) reader;
its MCOS workspace header check needed a local change to accept version 3
as well as version 2 for this file. `scripts/extract_pinheiro_gyro.py` captures
the table's data field before an old NumPy dtype assignment drops it. This
extraction dependency is not needed to replay the already extracted
`data/processed/pinheiro_p02_2_lumbrical_gyro.npz`. MATLAB users can run
`scripts/export_pinheiro_tables.m` to inspect CSVs.

### Third-round Pinheiro extraction

All five Pinheiro MAT files were subsequently decoded using the upstream
[`matloader`](https://github.com/anntzer/matloader) source in a temporary
Python 3.8 environment. For these files, its workspace-header check needed
to accept version `3` as well as `2`; no third-party library source was added
to this repository. The reusable extractor is
`scripts/extract_pinheiro_all_gyro.py`. It checks gyro units in column names,
three aligned timestamps, finite values, and sampling rate, then writes one
ignored `data/processed/pinheiro_<table>_lumbrical_gyro.npz` per MATLAB table.
The 18-table audit is in
[`reports/tremor_baselines/round3/data_audit/tables.csv`](../reports/tremor_baselines/round3/data_audit/tables.csv).
Seventeen tables provide approximately 60.43 s of gyro data; `p01_6` has
only 3.99 s and is excluded from the common 0–15 s calibration / 12 s
evaluation protocol. Re-extracting `p02_1` and `p02_2` to temporary files
matched the older processed gyro arrays exactly. The primary third-round
comparison preselects table `_1` for each person, not the strongest-looking
table. See the
[third-round report](../reports/tremor_baselines/round3/第三轮跨受试者与机制实验报告.md).

| Dataset | Actual content | Best project use | Main limit |
| --- | --- | --- | --- |
| [ET accelerometry, Zenodo 19130599](https://zenodo.org/records/19130599) | 29 ET patients; both hands; rest and posture; hand-mounted acceleration sampled at 5 kHz, with a published 2 Hz high-pass in the acquisition chain | Characterize across-person tremor spectral peaks, intermittency, and waveform shapes; challenge synthetic disturbance models | No angular pose, tool/handle relative motion, contact, or preserved voluntary-motion ground truth; amplitude calibration must be checked |
| [Pinheiro IMU+sEMG, Zenodo 8092424](https://zenodo.org/records/8092424) | Five participants, four Parkinson's disease and one ET; raw IMU and sEMG MAT recordings. The associated paper reports IMU at 148.5 Hz and sEMG at 1925.93 Hz | Inspect gyroscope channels and timestamps; if calibrated, replay measured angular velocity as a handle disturbance and compare causal reference generators. Explore whether sEMG provides complementary timing information | One ET participant; wrist/hand sensor axes and sensor-to-handle mapping require verification; sEMG is not a direct intent label |

## Proposed offline test sequence

1. Verify hashes and inspect the MAT structure, units, timestamps, missing
   samples, and trial conditions. Preserve subject and trial IDs. Check whether
   the 5 kHz ET signal is already amplified/filtered before using its amplitude.
2. Compute per-trial acceleration or angular-velocity spectra on untouched
   recordings. Use separate subject groups for development and evaluation;
   do not split overlapping windows from one subject between them. Report
   frequencies and time-varying 3–12 Hz energy with the distribution across
   people/trials, not only an average.
3. For the ET acceleration data, replay only the **waveform shape** as a
   normalized, explicitly scaled synthetic angular disturbance (for example,
   sweep 1°, 3°, and 5° peak). This is a simulation stress test, not a measured
   reconstruction of handle angles. Do not double-integrate acceleration and
   present the result as ground-truth orientation.
4. For Pinheiro, first verify the gyroscope channel, units, timestamps, and
   hand-to-handle axis mapping. If valid, remove sensor bias, integrate angular
   velocity with a documented drift treatment, and replay it in the MuJoCo
   base. Compare the existing direct/low-pass/WFLC references and torque
   controller. Report tool-axis error against the *chosen simulated task
   target*, torque saturation, joint limits, causal delay, and 3–12 Hz output
   attenuation. Use the known injected reference only for evaluation.
5. Test voluntary-motion preservation separately by adding a known synthetic
   low-frequency intention trajectory to a recorded tremor-shaped disturbance.
   Report low-frequency tracking error/delay and high-frequency attenuation
   together. This cannot establish preservation of real user intention.
6. Keep the key/spoon/cup contact and lock experiments as synthetic tests.
   Neither dataset records tool contact, mouth/table/lock geometry, contact
   force, or spill/task success, so it cannot directly validate lock triggering
   or safe interaction forces.

## Completed ET data checks and first replay

Run the reproducible spectral summary and normalized waveform extraction:

```bash
PYTHONNOUSERSITE=1 python3 scripts/analyze_et_dataset.py
conda run -n me5418 python -m gimbal.et_replay --all
```

The spectral script uses 5 s, nonoverlapping windows and reports the median
of each recording's 3–12 Hz peak within 2–20 Hz context. Across 58 hand
recordings, the median of recording medians was 4.8 Hz (interquartile range
4.4–5.2) for posture and 4.15 Hz (3.4–4.98) for rest. The median fraction of
2–20 Hz power in 3–12 Hz was 0.825 for posture and 0.636 for rest. These
figures describe this file and preprocessing only; they are not diagnostic
thresholds.

The replay uses subject 1, left hand, posture, seconds 10–22. An offline
3–12 Hz bandpass extracts acceleration waveform shape, resampled at 1 kHz
and normalized by its 99.5th absolute percentile. **A chosen 3° angular
scale** maps this one acceleration channel to synthetic handle roll; handle
pitch tremor is zero. A separate known 0.25/0.18 Hz roll/pitch intention is
added for evaluation. The raw recording does not supply either an angle or
voluntary-motion ground truth. Noncausal filtering is used only to create the
disturbance; the online estimators receive samples causally.

In the 12 s replay, evaluated after 3 s, the synthetic hand-to-intention
tool-axis error was 0.77° RMS. Tool errors were 0.77° for direct pass-through,
0.91° for the 1.5 Hz low-pass, and 0.67° for WFLC; no torque saturation was
observed. The WFLC result is a small improvement for this **single waveform
and assumed scale**, not an across-subject claim. Results and plots are in
`experiments/results/follow_et_s01_left_posture_3deg_*.{json,csv,png}`.

The result JSON now also separates tremor-estimation and frequency-band
performance. After the first 3 s, the injected roll disturbance has 0.771°
RMS; WFLC estimates a 0.440° RMS roll tremor with 0.671° RMSE and correlation
0.499 against that *synthetic mapped* disturbance. Its final frequency output
is 3.39 Hz versus a 3.5 Hz peak in the selected 12 s waveform. The 3–12 Hz
tool-axis residual is 0.767° RMS with pass-through, 0.245° with low-pass,
and 0.667° with WFLC: 68% and 13% reduction respectively. However, the
low-pass creates 0.866° RMS of 0–1 Hz tool-axis residual against the known
voluntary target, compared with 0.019° pass-through and 0.023° WFLC. Thus
low-pass removes more high-frequency motion but degrades intentional
following; WFLC preserves slow following better but is only modestly
effective on this sample. Band metrics are FFTs of 100 Hz sampled axis-vector
errors, expressed as small-angle-equivalent degrees, not clinical outcomes.

## Completed Pinheiro gyroscope replay

The ET participant file `p02_59_f_et_LA_20yr.mat` contains two 60.4 s tables.
Each has `LLUMBRICALGyroX5IMdegsec`, `Y`, and `Z` columns, with separate `Xs12`
through `Xs14` timestamps. There are 8,954 valid IMU samples per axis; the
median timestamp interval gives 148.15 Hz. The column names explicitly state
degrees per second. The table names `p02_1` and `p02_2` do not document task
conditions, so the second table is identified only by name here.

Extract once with a `matloader` installation that reads MCOS workspace header
version 3, then run the comparison in `me5418`:

```bash
PYTHONNOUSERSITE=1 python3 scripts/extract_pinheiro_gyro.py
conda run -n me5418 python -m gimbal.pinheiro_replay --all
conda run -n me5418 python -m gimbal.pinheiro_replay --estimator wflc --view
```

The replay selects table `p02_2`, seconds 10–22, and maps sensor X/Y to
simulated handle roll/pitch. This **axis alignment is assumed**, because the
recording has no sensor-to-gimbal calibration. Gyro X/Y both peak at about
6.45 Hz in that window. A 3–12 Hz offline FFT bandpass and integration produce
a drift-free *tremor-band angular disturbance*, with about 0.135° roll and
0.091° pitch RMS after the first 3 s. This is a reconstructed proxy, not a
measured handle pose. A separate, known 0.25/0.18 Hz voluntary trajectory is
added so that intentional-motion preservation can be scored. The online
estimators receive only the resulting total simulated handle orientation.

| Reference generator | Tool error vs simulated intention | 3–12 Hz tool-axis residual | 0–1 Hz tool-axis residual |
| --- | ---: | ---: | ---: |
| Direct follow | 0.163° RMS | 0.162° RMS | 0.003° RMS |
| 1.5 Hz low-pass | 0.878° RMS | 0.075° RMS | 0.866° RMS |
| WFLC | 0.153° RMS | 0.151° RMS | 0.020° RMS |

WFLC reduced total tool error by about 6% and 3–12 Hz residual by about 7%
relative to direct follow in this one window. Its estimated disturbance RMS
was 0.051° roll / 0.037° pitch; RMSE against the reconstructed injected
disturbance was 0.127° / 0.083°. Its final tracked frequency was 3.50 Hz,
while both source gyro axes peaked around 6.45 Hz. The low-pass suppressed
about 54% of the 3–12 Hz residual, but its voluntary-motion lag overwhelmed
that benefit. No actuator saturation occurred. Results and plots are saved
under `experiments/results/follow_pinheiro_p02_2_10s_12s_*`.

This is one ET participant, one table, and one 12 s window. Neither gyro
integration nor the injected intention gives ground truth for real user
intent, tool pose, or clinical benefit.

For a broader check, run `conda run -n me5418 python
scripts/compare_pinheiro_windows.py` after extracting both `p02_1` and
`p02_2` with `scripts/extract_pinheiro_gyro.py --table p02_1` and `--table
p02_2`. It compares four nonoverlapping 12 s windows per table. The script
regenerates a summary, figure and per-run metrics under `reports/` and
`experiments/results/`. In the previous run, WFLC improved total error
in 7 of 8 windows, with median 6.4% reduction and median 8.7% attenuation in
the 3–12 Hz tool-axis residual. These eight windows still come from only one
ET participant.

A separate diagnostic script compares the default WFLC with a fixed 6.5 Hz,
higher-amplitude-gain configuration on the same eight windows:
`python scripts/diagnose_wflc_tuning.py`.
The fixed frequency uses prior knowledge of this recording and is not an
online tremor estimator; the default implementation was not changed.

The first dataset's page lists an `acc_signal_database.mat`, a small
`clinical_data.mat`, and `README.txt`; the second lists five participant MAT
files. The associated Pinheiro paper should be cited when using its data, as
requested on the [dataset record](https://zenodo.org/records/8092424).
