# Damped-BMFLC reproduction notes

## Paper specification

Reference: *Online learning for vibration suppression in physical robot interaction using power tools*, Autonomous Robots (2026), DOI 10.1007/s10514-026-10252-9. The paper's Eq. (8) sets a per-Fourier-coefficient step size `mu_k = eta*g_k/(1+exp(-kdmp*(w_k-xdmp)))`; Eq. (7) includes the coefficient forgetting term.

## Project implementation

The implementation is `analysis/tremor_baselines/damped_bmflc.py`. It uses eta=0.02, kdmp=350, xdmp=0.009 rad/s, lambda_ref=0.99999 at 1000 Hz, and lambda_project=0.999932501941 per sample at 148.148148 Hz. Fourier basis: 3–12 Hz, 0.1 Hz spacing; zero weights; causal update; no bias term, matching the 2026 paper. The project interface accepts/returns deg/s and the class converts to rad/s internally.

## Differences and uncertainty

The published threshold comes from a 1000 Hz robot linear-velocity error signal. The paper gives no conversion for gyro angular rate; this project applies its numeric threshold in rad/s and flags this as a cross-physical-quantity port. Lambda is converted to preserve the paper's forgetting time constant. The printed sigmoid is applied to each signed coefficient exactly; it is not replaced by the magnitude of a sine/cosine pair. The paper model has no bias, while the project's OriginalBMFLC retains its 2011 intended-motion bias. Therefore algorithm differences include both the update mechanism and bias structure.
