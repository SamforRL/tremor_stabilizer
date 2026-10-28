# Tremor estimation baseline reproduction notes

## Scope and source signal

This module benchmarks **angular velocity in deg/s**, separately for each IMU
axis. It does not feed angular velocity into an angle or displacement estimator
without an explicitly documented transform. It does not modify the MuJoCo
controller or the existing simplified WFLC in `gimbal/estimators.py`.

The selected recording is the existing Pinheiro ET participant file
`Dataset/data1/p02_59_f_et_LA_20yr.mat`. The processed `p02_1` and `p02_2`
files have `LLUMBRICALGyroX/Y/Z5IMdegsec` and separate timestamps, with
8,954 valid gyroscope samples over 60.43 s. The median timestamp interval
is 0.00675 s, giving 148.148 Hz. Sensor-to-handle alignment and trial/task
labels for the two tables are not established.

The raw gyroscope is total recorded motion, **not a labelled pure tremor**.
With the user's approval, `recorded.py` constructs an explicitly named
**recorded tremor-band proxy** by offline 3–12 Hz FFT selection, using 1 s of
context on each side of a window. This operation sees future samples and is
allowed only in input construction, never in the online estimators. It also
resamples the timestamp-rounded data to a uniform grid by linear
interpolation. The window, channels, band, context, file hash, and sampling
rate are retained as metadata. The current band follows the previous project
stress-test band and is *not* a biological label. A fixed BPF with the same
band has a construction advantage; rankings on this proxy alone will not be
interpreted as evidence of superiority on true physiological tremor.

The benchmark combines `voluntary_true + alpha * tremor_band_proxy + noise`.
The stored `tremor_true` is the **scaled injected component**. When optional
noise is nonzero, `mixed - tremor_estimate` necessarily includes remaining
noise, so voluntary reconstruction RMSE measures both estimation error and
noise. Noise-free cases are the primary separation check; noisy cases are
separate stress tests.

## Current reproduction status

| Baseline | Paper specification | Our implementation | Deviations / ambiguities |
| --- | --- | --- | --- |
| Fixed Butterworth BPF | Conventional digital Butterworth; configurable order and -3 dB band edges. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) defines `N` as prototype order, giving `N` band-pass biquads and digital order `2N`. | `butterworth.py` derives analog Butterworth poles, applies band-pass transform and frequency prewarping, bilinear mapping, conjugate pole pairing, and SOS direct-form-II transposed causal recursion. Default prototype order 4, 3–12 Hz, 148.148 Hz for this recording. Tests compare the entire streaming output and frequency response with `scipy.signal.butter(..., output='sos')` and `sosfilt`. | Pure NumPy design/runtime is used because the project's `me5418` environment currently lacks SciPy. This does not alter the standard transfer function. Default order 4 is a declared benchmark setting, not taken from the ABPF paper. Initial state is zero; startup is reported, not discarded silently. |
| Popović, Šekara & Popović 2010 ABPF, DOI [10.1016/j.cmpb.2010.03.018](https://doi.org/10.1016/j.cmpb.2010.03.018) | Supplied PDF pp. 300–302: second-order adaptive Butterworth transfer function (1), half-period `H` from zero crossings, damping and frequency limiting, `f_mod` and `Δf`; original synthetic tests use `Δf=0.25 Hz`, patient tests `0.1 Hz`, and original sampling is 1 kHz. | `abpf.py`: Tustin biquad with persistent state; interpolated zero crossings; literal ordered damping conditions; independent modal calibration in `calibration.py`. | Tustin/crossing interpolation are numerical conventions. User resolved omitted damping branches as hold and in-range limiter as pass-through. Paper synthetic and project sample rates are separate. |
| Wang, Gao, Zhu & Zhao 2016 adaptive SFFT, DOI [10.1177/1687814016654872](https://doi.org/10.1177/1687814016654872) | Supplied PDF pp. 3–5, 8: fourth-order self-convolved Blackman window, dual-bin interpolated FFT, frequency limiting, sliding band IFFT; original `f_s=100 Hz`, `L=100` samples. | `sfft.py`: 4-fold convolution window, deterministic offline ratio-to-fraction polynomial, dominant-frequency limiting, symmetric spectral mask, newest-sample causal IFFT output. | Eq. (13) is corrected by Eq. (11)'s LMS gradient; runtime uses offline least-squares coefficients so no LMS gain is used. Eq. (15)'s frequency width is exposed as `extraction_bandwidth_hz`; values 2/3/4 Hz are tested separately. |
| Veluvolu & Ang 2010 BMFLC, DOI [10.1002/rcs.340](https://doi.org/10.1002/rcs.340) | Supplied PDF pp. 337–338: Fourier basis (1)–(2), LMS (3)–(5), 6–14 Hz example band, 0.1 Hz grid and `μ=0.01`; bias extension for drift/intended motion. Original sensing system is 250 Hz and studies physiological tremor. | `bmflc_core.py`: exact sine/cosine LMS core plus constant-one intended-motion bias clarified by the [same authors' Sensors 2011 §3.2, Eqs. (17)–(19)](https://www.mdpi.com/1424-8220/11/3/3020). Tremor output excludes bias. | Zero weight initialization and same LMS gain for bias are declared implementation choices. Project 3–12 Hz band is a separate domain configuration, not the original physiological 6–14 Hz experiment. |

The 2016 SFFT paper is an **additional comparison source**, not a substitute
for the 2010 ABPF specification. Parameters it chose while comparing methods
are explicitly distinguished from the original ABPF paper's own settings.

## Resolved implementation ambiguities

See [LITERATURE_AMBIGUITIES.md](LITERATURE_AMBIGUITIES.md) for the original
page/equation, alternate interpretations, consequences and adopted decision.
These choices are user-authorized and explicitly recorded in the code and
configuration; they are not presented as unspecified original-paper facts.

## Shared software and metric conventions

`create_estimator(method, config)` creates an estimator with `reset()` and
`step(sample)`; a step takes a one-dimensional `deg/s` vector and returns an
estimated tremor vector plus a diagnostics dictionary. The scalar algorithm
is run independently on each axis; there is no cross-axis fusion.

`signals.py` stores voluntary, injected proxy/synthetic tremor, optional
seeded noise, and mixed observation separately. It includes slow smooth,
minimum-jerk point-to-point, fast minimum-jerk transition, and near-band
spectral-overlap voluntary trajectories. Constant, drifting, frequency-step,
amplitude-modulated and multi-frequency synthetic tremors are separate
verification signals, never substitutes for the recorded benchmark.

`metrics.py` computes unaligned causal tremor/voluntary RMSE, true-RMS
normalized tremor RMSE, and common 2 s Hann Welch PSD with 50% overlap,
constant detrending and one-sided density scaling. Band-power reduction is
`10 log10(P_input/P_output)`; true-tremor vs estimation-error band power is
also logged because voluntary motion may overlap the band. Cross-correlation
lag is a **separate diagnostic** and is never used to realign the RMSE.
Per-sample wall-clock timing is implementation and host dependent.
For controlled synthetic signals, methods that naturally expose frequency
are scored by frequency MAE and post-transition re-lock time, requiring a
continuous 0.5 s interval within a declared 0.5 Hz tolerance. Methods
without a frequency state have no invented frequency metric.

`plots.py` uses the same time axis and PSD settings for every method. The
fixed BPF response plot includes magnitude and group delay; its CSV stores
complex phase and group delay by frequency.

## Equation-to-code mapping and adopted decisions

### Fixed Butterworth BPF

`butterworth.py` designs standard bandpass SOS using analog Butterworth poles,
prewarping and bilinear mapping, then processes one sample per axis with
direct-form-II-transposed states. Project settings: 148.148 Hz, prototype
order 4 (digital order 8), 3–12 Hz cutoffs, zero initial states. The output
is the tremor estimate in deg/s. It uses only current/past samples. Magnitude,
phase and group delay are stored in each run's `fixed_bpf_response.csv` and
`.png`. This is a conventional reference method with no adaptive-paper
initialization or ambiguity.

### ABPF (Popović et al. 2010)

- **Paper statement:** p.301 Eq. (1) gives
  `H(s)=sqrt(2)*ωa*s/(s²+sqrt(2)*ωa*s+ωa²)`; p.300 Figs. 4–5 derive
  `f_in=1/(2H)` from consecutive filtered-output zero crossings. The p.302
  `fcn` code gives ordered `Δf`/`2Δf` damping conditions and a final
  `f_mod±1.5 Hz` limiter. Its patient protocol calibrates `f_mod` on an
  independent preceding 15 s at 1 kHz.
- **Implementation:** `abpf.py` applies Tustin to Eq. (1) as a single
  time-varying biquad per axis, preserving its two states across coefficient
  updates. Zero crossings are linearly interpolated between adjacent samples.
  The printed damping conditions run in their original order. When no
  condition fires, the previous center frequency is held; the limiter passes
  in-range updates and clips out-of-range updates. Output is the current
  bandpass tremor estimate. Diagnostics expose center frequency, half-period,
  frequency step and initialization state. No future sample is used.
- **Initialization/rates:** paper-style `f_mod=4.5 Hz`, `Δf=0.25 Hz`,
  `f_s=1000 Hz`. Project `Δf=0.1 Hz`, `f_s=148.148 Hz`; `calibration.py`
  obtains per-axis `f_mod` from the raw gyro's full-segment Hann periodogram
  peak in 3–12 Hz on 0–15 s. The checked `p02_2` X value is 6.6673 Hz.
  Evaluation is 20–32 s; proxy context is 19–33 s, so calibration never
  overlaps evaluation or its offline context. The actual calibration duration
  is 14.9985 s after sample-grid rounding.
- **Deviation class:** Tustin and interpolated crossing time are **standard
  numerical conventions** absent from the continuous Simulink specification.
  Hold-last damping and pass-through limiter are **user-approved reconstructed
  semantics** for missing pseudocode branches. The 3–12 Hz calibration search,
  project rate/split and angular-velocity task are **project assumptions**.
  Zero phase is expected only at the instantaneous filter center, not across
  the whole band. The exact original continuous solver output is unknown.

### Adaptive SFFT (Wang et al. 2016)

- **Paper statement:** pp.3–4 Eqs. (2)–(7) convolve four length-`N` periodic
  Blackman windows and append three zeroes to length `L=4N`. Eqs. (8)–(14)
  use neighboring FFT amplitudes and `β=(y₂−y₁)/(y₂+y₁)` to interpolate the
  dominant frequency. Eq. (15) limits a sliding extraction band; Eqs.
  (16)–(18) retain conjugate-symmetric spectrum bins for IFFT reconstruction.
  Paper settings include `f_s=100 Hz`, `L=100` (one second).
- **Implementation:** `sfft.py` constructs that window, computes the same
  dual-bin ratio, and deterministically calibrates `α(β)` offline using 1001
  fractional-bin offsets, checking 1000 separate offsets. Odd polynomial
  orders 3, 5, 7 are tried; the lowest with maximum error ≤0.01 bin is
  selected. Order 3 passes with <4.3e-8 bin held-out error for both tested
  windows. Coefficients and candidate errors are stored in
  `sfft_interpolation.json`. Online processing uses fixed coefficients, a
  zero-padded sliding window, a raw-window FFT for extraction, a symmetric
  mask, and the **last IFFT sample** as current tremor estimate. Diagnostics
  include peak/limited frequency and window-fill state.
- **Eq. (13) correction:** the printed recursion is dimensionally
  inconsistent. The Eq. (11) squared-error objective implies
  `w[k+1]=w[k]+2μ(α[k]−w[k]ᵀβ[k])β[k]`. The present offline least-squares
  calibration solves that same inverse-mapping objective without a runtime
  or calibration LMS gain. This is a **user-approved standard gradient
  correction**, not a claim that the paper printed this exact recursion.
- **Eq. (15) correction:** `L` is also used as sample count, so it cannot be
  added to a frequency in Hz. `extraction_bandwidth_hz=B` supplies the Hz
  width. `f_dom=clip(f_peak,f_min+B/2,f_max−B/2)`. Paper-style `B=3 Hz` is
  explicitly an **assumption**, since the paper gives no value. Paper-style
  outer limits 2–12 Hz are also an **assumption** chosen to contain its
  4–10 Hz synthetic trajectory with a centered 3 Hz extraction band, separate
  from the project's 3–12 Hz bounds. The 2/3/4 Hz sensitivity is reported
  separately without per-case selection. Project
  `f_s=148.148 Hz`, `L=148` (~0.999 s), outer range 3–12 Hz. With `B=3`,
  the allowed center is 4.5–10.5 Hz, so edge frequencies are clipped.
- **Causality/deviation:** no center-sample relabeling, future information or
  offline overlap-add is used. The latest-sample output and trailing-window
  zero placement are **documented causal conventions**. Window transition
  delay remains. Original experiment discussions include joint angle, while
  this project uses gyro deg/s; that is a **project signal-domain adaptation**.

### BMFLC (Veluvolu & Ang 2010; clarification from Sensors 2011)

- **Paper statement:** p.337 Eqs. (1)–(5) form `n` sine/cosine components,
  output `y[k]=w[k]ᵀx[k]`, error `s[k]−y[k]`, then
  `w[k+1]=w[k]+2μx[k]error`. The p.338 example is 6–14 Hz with 0.1 Hz
  spacing and `μ=0.01`; original sensing system is 250 Hz.
- **Literature-supported clarification:** the [same authors' 2011 Sensors
  extension §3.2 Eqs. (17)–(19)](https://www.mdpi.com/1424-8220/11/3/3020)
  resolves the 2010 bias notation: append constant input basis `1` and
  weight `a₀[k]`; `I[k]=a₀[k]` is intended motion, while the Fourier sum
  alone is tremor. `bmflc_core.py` keeps the **2010 LMS**, with no RLS or
  Kalman substitution. The output excludes bias.
- **States/initialization:** `2n+1` weights per axis, initialized to zero.
  At each sample, output uses pre-update weights; the current input updates
  weights for the next sample. Bias and Fourier weights share `μ=0.01` as a
  **standard LMS convention** because no separate bias gain is given.
  Diagnostics expose bias, LMS error and weights. Only present/past samples
  are used; startup depends on adaptation, not a future window.
- **Modes/deviation:** paper-style 250 Hz, 6–14 Hz, 0.1 Hz spacing; project
  148.148 Hz, **3–12 Hz**, same spacing/gain. The 3–12 Hz band and gyro
  deg/s input are **project-specific domain adaptations**, not the source's
  physiological tremor displacement/acceleration experiment.

## Verification and checked outputs

- `me5418`: 26 tests passed, 4 optional SciPy comparisons skipped. Base
  Python with SciPy 1.13.1: all 30 passed. BPF stream/response match SciPy
  to `1e-10` for orders 1, 2, 4, 6.
- ABPF tracks a clean 3→6 Hz paper-style step at 1000 Hz: the center is
  within 0.1 Hz of 3 Hz at 3–4 s and 6 Hz at 8–9 s. Its Tustin center
  response, deterministic reset and prefix causality are checked.
- SFFT at 100 Hz with `L=100` follows a 5→8 Hz step. Steady-section mean
  absolute frequency error is <0.02 Hz; post-transition pure-sinusoid
  reconstruction error is <0.01 deg/s. Its calibration and prefix causality
  are checked.
- BMFLC's one-step LMS update matches Eq. (5); its final 2 s error on the
  8 + 10 Hz two-tone form is <0.3 input units at 250 Hz. A constant input
  is carried by bias, not the tremor output.
- Paper-style checks do **not** reproduce published patient statistics:
  original patients, noise realizations, and some experiment settings are
  unavailable. See `paper_validation/` under the separate results/reports.
- Five separate controlled cases (constant/drifting/step/amplitude-modulated/
  multiple frequencies) are run with all four methods under
  `controlled_verification/`. Known frequency is scored only for the
  single-frequency cases and methods with a frequency state. The ABPF
  `f_mod±1.5 Hz` constraint remains in force even when a synthetic trajectory
  extends beyond it; the resulting error is a documented model limit.

Project arrays and provenance are in
`experiments/results/tremor_baselines/project_benchmark/`; common plots,
metrics CSV and SFFT bandwidth sensitivity are in
`reports/tremor_baselines/project_benchmark/`. All four methods see the same
saved mixture per case. In the checked `p02_2` X, 20–32 s run, the injected
proxy RMS is 6.154 deg/s. Slow-smooth tremor RMSE is 4.06 (BPF), 1.33
(ABPF), 4.49 (SFFT), and 1.42 (BMFLC) deg/s. On the deliberate fast
transition, BMFLC RMSE rises to 18.58 deg/s, illustrating this configuration's
bias/separation failure on that task. These are errors relative to a known
**injected band proxy**, never independently labelled physiological tremor.

Cross-correlation lag is a diagnostic only. Periodic signals have several
near-equal peaks a cycle apart; the metric reports the nearest lag within
0.02 normalized correlation of the largest peak. No method's causal RMSE is
post-hoc phase aligned.
