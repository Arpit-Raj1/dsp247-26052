# Reproduction notes: Deep ANC (Zhang & Wang, *Neural Networks* 141, 2021)

This file tracks every place this implementation deviates from the paper
because the paper doesn't specify a detail, or because a resource the
paper used isn't freely available. See `CLAUDE.md` for the paper summary
and `configs/`/`src/deepanc/` for the implementation itself.

## Data substitutions

| Paper | This repro | Why |
|---|---|---|
| TIMIT (LDC, paid license) | TIMIT | **Available** — the user has an LDC-licensed copy at `TIMIT/`, used as-is. |
| 10,000 non-speech clips, sound-ideas.com (commercial) | ESC-50 (local dev, 2000 clips minus speech-adjacent categories) / DNS-Challenge noise or MUSAN (cluster-scale, documented but not fetched in this environment) | sound-ideas.com requires a paid license. `scripts/prepare_noise.py --source {esc50,dns,musan}`; ESC-50 is what's actually been run and tested here since DNS/MUSAN (11GB+) don't fit this environment's disk/network budget. Results will not match the paper's noise diversity 1:1. |
| NOISEX-92 test noises (engine/factory/babble/SSN) | NOISEX-92 via the free SPIB (UFSC) mirror, resampled 19.98kHz→16kHz; SSN synthesized from TIMIT's long-term average spectrum since the SPIB mirror has no SSN track | `scripts/prepare_noisex.py`. Engine/factory/babble are the *real* NOISEX-92 recordings; SSN is a substitute (verified 0.998 log-spectrum correlation to the TIMIT-derived target). |

## Unspecified training/architecture choices

The paper does not give exact values for the following; each choice below
is the one this implementation makes, documented here so a discrepancy in
final numbers can be traced back to a specific assumption rather than a bug.

- **Signal levels**: TIMIT speech kept at its native scale; noise scaled to
  hit the target SNR for noisy-speech items; noise-only items are scaled to
  the RMS that noise would have at a randomly drawn SNR against a reference
  speech RMS (`REFERENCE_SPEECH_RMS = 0.0125`, measured from TIMIT) — see
  `data/datasets.py`. This matters because the SEF loudspeaker nonlinearity
  saturates around `|y| ≈ η·√(π/2)`, so signal level determines how much of
  training actually explores the nonlinear region.
- **Batch size**: 16 (not given in the paper).
- **STFT window**: periodic sqrt-Hann, chosen for exact COLA reconstruction
  at hop=n_fft/2.
- **CRN conv kernel size**: (2,3) (time, freq), stride 2 in freq — the
  paper points to Tan & Wang (2019a) for architecture details but doesn't
  reproduce the exact numbers here; not independently verified against
  that paper's released code.
- **Grouped-LSTM shuffling**: a channel-interleave shuffle between the two
  LSTM layers (Gao et al. 2018's stated "group strategy"), exact shuffle
  order is an implementation choice.
- **TIMIT utterance selection/concatenation**: 200 training speakers
  (100M/100F), 7 train + 3 test utterances/speaker; a 6-second training
  signal is built by concatenating random utterances from one speaker with
  a 0.2s gap; the paper doesn't specify how multi-segment 6s signals are
  assembled (Fig. 10 shows several speech segments per signal, supporting
  this choice, but not the exact method).
- **NOISEX file → paper's noise-type mapping**: `destroyerengine.mat` for
  "engine", `factory1.mat` for "factory" (vs. `factory2.mat`).
- **Quiet-zone T60**: fixed at 0.2s (the paper's default test T60) for all
  100 sphere-sampled training RIR pairs; not stated in Section 5.3.
- **FxLMS/THF-FxLMS secondary-path identification**: white-noise excitation
  at 0.1 amplitude through the *true* plant, least-squares FIR fit
  (Wiener-Hopf normal equations); the paper doesn't describe its own
  identification procedure.
- **THF-FxLMS**: the tanh saturation parameter `beta` is fit offline to
  match the true SEF nonlinearity by least squares; the update rule
  (filtered-reference via the estimated derivative `g'(y)`) follows the
  standard nonlinear-FxLMS construction but has **not** been checked
  against Ghasemi et al. (2016)'s original paper line-by-line — treat
  THF-FxLMS's numbers as indicative, not a certified reproduction.
- **VFxLMS/FxMLP (Section 5.5) step sizes**: not given in the paper (it
  points to Guo et al. 2018, which this repro hasn't cross-checked); tuned
  ad hoc, see `evaluate.py --mu`.
- **FxMLP filtered-error gradient**: the paper's baseline description is a
  single sentence; a full filtered-x approach (needed for a correct
  gradient through a nonlinear controller with an LTI secondary path)
  would require filtering each of the network's per-weight sensitivities
  through Ŝ, which is expensive. This implementation instead uses a
  **delay-compensated single-tap approximation**: it identifies Ŝ's
  dominant tap (index and gain), delays the backprop by that many samples,
  and scales the output-layer gradient by that tap's gain — a common
  simplification, but a deviation from a textbook-exact derivation.
- **PESQ mode**: wideband (`mode="wb"`) is used for headline numbers in
  `evaluate.py`; the paper's PESQ range (−0.5 to 4.5) suggests narrowband
  P.862 — narrowband scores (`mode="nb"`, on 8kHz-resampled signals) should
  be cross-checked if trying to match Table 2 exactly.
- **NMSE-vs-time window** (Fig. 7a): 100ms, matching the paper's caption.

## Evaluation/training fixes (2026-09-29)

The first full results (Tables 1-5, Figs. 7-10) had several pipeline
bugs, which were diagnosed and fixed here:

- **Babble was silent.** SPIB's `babble.mat` is float64 already in
  [-1, 1]; `prepare_noisex.py` divided every track by 32768, leaving
  babble at RMS 2e-6 (CRN-n "NMSE" of +38 dB was just its output
  floor against a near-zero d). The /32768 now only applies to int16 tracks.
- **Test levels didn't match training.** Noise-only test references used
  raw NOISEX amplitudes (engine RMS 0.17, SSN 1.0) while training noise is
  scaled to RMS 0.0125/10^(SNR/20) ~ 0.001-0.007, i.e. 25-800x louder than
  anything the CRN saw, and loud enough to drive the eta2=0.1 SEF deep
  into saturation (which also made FxLMS/THF-FxLMS diverge, e.g. +112 dB,
  and Fig. 7's NMSE climb to +15 dB). Noise-only test conditions now use
  the training level distribution (`data/testsets.py _level_like_training`).
  Level-matched conditions (noisy speech, `snr_change`) were already fine,
  which is why they scored -7 to -8 dB while the rest scored ~-3.5 dB.
- **Test sets weren't reproducible.** `build_testsets.py` seeded some
  conditions from `hash((noise, tag))`, which Python salts per process; it
  now uses a CRC32 of the condition name.
- **Baseline step sizes.** Every baseline ran with mu=0.05 for every noise.
  The paper tunes per noise; ours are NLMS so the paper's numbers don't
  transfer. `scripts/tune_baseline_mu.py` grid-searches mu per noise on
  held-out noise cuts (not the test signals) and the result is recorded in
  `evaluate.py DEFAULT_MU`.
- **Table 5 wasn't the Section 5.5 setup.** `evaluate.py` ignored
  `primary_model: volterra`/`secondary_model: nl`, so `crn_n_nl` (trained
  on the Volterra/tanh paths) was scored on RIR paths, and VFxLMS/FxMLP
  never saw the nonlinear plant at all. A `table5` test group with
  `setup="nl"` now routes the model through `volterra_primary`/
  `nl_secondary_path` and the baselines through the same plant
  (`plant="nl"`).
- **Tables 3/4 layouts** now follow the paper: Table 3 has the 4 noises
  (CRN-n) and engine noisy speech at 5/15/20 dB (CRN-ns) per radius; Table
  4 has a noise-only and a noisy-speech column per robustness case.
- **The checkpoints are undertrained.** Every run's validation NMSE was
  still falling at epoch 30 (CRN-n -5.8 dB, CRN-n(-2) -2.4 dB, CRN-ns
  -0.4 dB), so the models, not only the evaluation, are well short of the
  paper. One suspect: at the training level the raw MSE is ~1e-5..1e-7
  and the median gradient ~1.5e-10, below Adam's eps=1e-8. A 600-step CPU
  A/B of raw MSE vs. MSE divided by the batch's mean d^2 (same minimizer)
  was inconclusive (+5.8 vs +7.8 dB at step 600, the normalized run led at
  step 500), so `train.loss_norm: d_power` exists but is off by default.
  Note it is only the same minimizer *within* a batch: across the dataset
  it weights each batch by 1/(its d power), so quiet and loud items count
  about equally (plain MSE is dominated by the loudest, 5dB-level items) —
  closer to the NMSE metric, but a deviation from the paper's Eq. 4/6.
  The 2026-09-29 retrain of all 9 models uses `train.loss_norm=d_power`
  (old checkpoints kept in `runs_pre_fix/`).
  Longer training / an LR schedule, or the loss normalization over a full
  run, needs a GPU experiment.
- **CRN-ns checkpoint selection** used NMSE(e, d), which rewards cancelling
  the speech too; `best.pt` is now selected on validation loss (Eq. 6) for
  `mode: ns`.

A remaining, unresolved gap: at this level convention the eta2=0.1 SEF
(saturating near |y|~0.4) is almost linear for the signals involved, so
eta2 barely changes anything, unlike the paper's Table 1 where FxLMS
degrades with eta2. The paper doesn't state its signal level. The mu sweep
shows the same thing from the baseline side: THF-FxLMS scores identically
to FxLMS (its tanh derivative is ~1 at these amplitudes), and in the
"nonlinear" Section 5.5 setup the Volterra quadratic terms and the
3.3*tanh(0.3y) saturation are negligible at RMS ~0.004, so VFxLMS reaches
~-45 dB (paper: ~-15 dB). Matching the paper there needs a training/test
level convention that actually exercises the nonlinearities (and a retrain).

## Causality / latency

The CRN itself is causal (verified by `tests/test_crn.py`), but a 20ms
window with 10ms hop still gives the *analysis* stage ~1 window's latency
before a full frame is available. Combined with the frequency-domain
approach's inherent frame-length delay (Section 3.4), the **m_frames=0**
variants (`crn_n`, `crn_ns`) are not literally zero-latency in a deployed
system — **m_frames=1 or 2 are the delay-compensated, more realistic
deployable configurations** discussed in the paper's own trade-off
(Table 1/2 show m=1/2 costs a small amount of NMSE for reduced latency).

## Local smoke-test results (RTX 4050)

Single-batch overfit test (4 items, batch_size=4, `configs/exp/crn_n.yaml`,
lr=1e-3/3e-3 AMSGrad): loss and NMSE decreased **monotonically** for the
full run (no divergence, no NaNs) — from +29.9dB to -7.2dB NMSE over 1600
steps (~6.5 min) at lr=1e-3, confirming the forward simulation, loss, and
backprop are wired correctly end-to-end (STFT→CRN→iSTFT→loudspeaker→
secondary path→loss→gradients all flow). It did **not** reach the -20dB
target from the plan's smoke-test gate in this step budget. A follow-up
check forcing eta2=inf (linear loudspeaker, ruling out SEF-saturation
gradient flattening as the cause) showed a similar convergence rate, so
the slowness is not obviously loudspeaker-nonlinearity-related — it may
simply be inherent to this architecture (deep causal conv+grouped-LSTM
stack in the STFT domain) needing more steps/a tuned LR schedule to fit
even one batch tightly. This is not necessarily a problem for the real
20,000-signal/30-epoch training runs (far more steps, more diverse data),
but is worth watching on the cluster: if `crn_n`'s validation NMSE hasn't
passed roughly -8dB by epoch 10 (the plan's cluster-training gate), that
warrants a learning-rate or architecture review before running the full
9-model array.

## Known local-environment constraints

This repository was developed and smoke-tested on a single RTX 4050
laptop GPU (6GB VRAM) with a 20GB free-disk budget — sufficient for
signal-processing unit tests, a small ESC-50-based noise shard, the full
TIMIT corpus (~684MB, present locally), and short training/overfit smoke
runs, but **not** for the paper-scale 10,000-noise training set or the
full 20,000-signal × 30-epoch training runs, which are designed to run on
the SLURM cluster (`slurm/train_array.sbatch`) instead. See `CLAUDE.md`
for the exact commands to reproduce the full pipeline there.
