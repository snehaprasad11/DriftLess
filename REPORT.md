# DriftLess: Automated Cloud-Side Few-Shot Calibration for EOG Gaze Estimation

**Sneha Prasad** · Advanced Cloud Computing
Repository: https://github.com/snehaprasad11/DriftLess

---

## Abstract

Electrooculography (EOG) enables camera-free, hands-free gaze interfaces but suffers
from **baseline drift** — a slow, person-specific wander of the signal that makes a
model tuned for one user inaccurate for the next. Prior work corrects drift manually
and offline. **DriftLess** reframes the problem as a live, cloud-native service: a
single shared deep-learning model is personalised to each brand-new user in seconds
by few-shot fine-tuning of a tiny "drift adapter", and served with millisecond
latency. On subjects the model never trained on, calibration reduces near-term gaze
error by **up to 34%** (best-trained configuration) and by **~11% averaged across all
six subjects** in leave-one-subject-out evaluation, helping five of six. The system is
delivered as containerised microservices (API gateway, stateless inference, an
asynchronous calibration worker behind a Redis queue, and a versioned model registry),
runs locally with one command, is serverless-deployable with scale-to-zero, and
includes automated tests, a CI pipeline, and a live streaming demonstration.

---

## 1. Introduction

**EOG.** The eye acts as a small dipole (positive cornea, negative retina); electrodes
placed near the eyes read the voltage change as the eye rotates, so eye movement
becomes an electrical signal — no camera required. This suits accessibility (users who
cannot use their hands), AR/VR, and hands-busy contexts.

**The problem — baseline drift.** Even with a still eye, the EOG baseline drifts due to
sweat, skin, and temperature. It is (a) *person-specific* — every user drifts
differently — and (b) *time-varying* — it changes within a session. A model calibrated
for one person is therefore wrong for the next, and standard fixes are hand-tuned,
per-subject, and offline.

**The gap and contribution.** No prior system makes drift correction a *live,
automatic, scalable cloud service*. DriftLess does, contributing a **method-plus-system**:
automated cloud-side few-shot calibration of a shared EOG model to each user's
individual drift signature, with per-user model versioning and elastic serving.

## 2. System Design and Architecture

DriftLess is six single-responsibility components across three "lifecycles":

| Component | Responsibility | Lifecycle |
|---|---|---|
| Edge client | Streams EOG windows, draws predicted gaze | every use |
| API gateway | Front door; routes predictions, enqueues calibration | every use |
| Inference service | Loads a user's model, predicts gaze fast (stateless) | every use |
| Calibration worker | Few-shot fine-tunes a personal model (background) | per new user |
| Model registry | Stores base + per-user models, versioned | all |
| Trainer + CI/CD | Trains the base model; tests and builds on push | build time |

**Design principles:** stateless inference (so it clones and scales to zero); async
calibration via a Redis queue (the gateway never blocks on the ~seconds job); one
versioned source of truth for models; and vendor-portable free-tier components.

## 3. Machine-Learning Method

**Data pipeline.** Continuous recordings are cut into fixed 320-sample windows around
each saccade/blink. Each window is labelled with the saccade's (x, y) displacement in
degrees and a blink flag. Signals are scaled with fixed statistics that **preserve the
drift** (rather than per-window recentring, which would erase it), because handling
drift is precisely what calibration must learn.

**Model — GazeNet.** A shared **1D-CNN** extracts local shapes (saccade steps, blink
spikes); a **BiLSTM** models their temporal evolution; a small **drift adapter** (a
residual bottleneck, ~8.4k parameters, 6% of the model) is the only per-user part; two
heads predict gaze (regression) and blink (classification). Training minimises squared
error on gaze plus cross-entropy on blink, by gradient descent (Adam), on a free Colab
GPU.

**Per-user calibration.** For a new user, the backbone **and heads are frozen** and only
the drift adapter is fine-tuned on ~10 of their trials, for a few steps, with mild
weight decay. This combines **transfer learning** (reuse general knowledge),
**few-shot learning** (~10 examples), and **parameter-efficient fine-tuning** (move a
tiny adapter). Restricting to the adapter and regularising it prevents the overfitting
that occurs when the heads are also trained on so few examples.

## 4. Implementation

Each service is a FastAPI or worker process in its own Docker image; `docker compose up`
runs the full stack (gateway :8000, inference :8001, Redis, worker). The gateway is
deliberately light (no PyTorch); inference and worker carry the model. The model
registry is a filesystem abstraction locally and swaps to S3-compatible object storage
in the cloud behind the same interface. A `pytest` suite (16 tests, fully synthetic so
it needs no dataset) covers windowing, the network, calibration, the registry, and both
service APIs; GitHub Actions runs tests then builds the images on every push. A
single-page web client streams a subject's EOG over a WebSocket and draws the predicted
gaze live with a latency read-out.

## 5. Evaluation and Results

All evaluation is on subjects **the model never trained on** (by-subject split, no data
leakage). Gaze error is mean Euclidean distance (degrees).

**Base model (before personalisation).** Trained on four subjects, tested on two
never-seen: **10.18°** gaze error and **85.4%** blink accuracy on new users, versus
3.34° / 99.6% on training subjects. The ~6.8° gap is the person-specific drift.

**Per-user calibration (fixed split).** Near-term (right after calibration): mean
**6.4° → 4.2° (−34%)** (S5 5.5→3.8, S6 7.2→4.5). Verified live through the deployed
stack with ~6 ms prediction latency.

**Leave-one-subject-out (all six subjects, fair 80-epoch training).** Rotating every
subject as the new user, mean near-term error **7.09° → 6.30° (−11%)**, helping **five
of six**:

| Held-out user | Before | After | Change |
|---|---|---|---|
| S4 | 8.11° | 5.54° | −32% |
| S3 | 8.52° | 6.47° | −24% |
| S1 | 10.65° | 9.98° | −6% |
| S6 | 4.64° | 4.50° | −3% |
| S2 | 5.52° | 5.40° | −2% |
| S5 | 5.12° | 5.88° | +15% |

**Systems.** Warm prediction latency ~6 ms; calibration a few seconds as a background
job; inference stateless and scale-to-zero; the worker scales horizontally (Compose
`--scale`, or a Kubernetes HorizontalPodAutoscaler).

## 6. Discussion and Limitations

The mechanism clearly works — large, consistent gains on subjects with real drift to
correct (S3, S4) — but its benefit is **not uniform**. Two honest reasons: (1) with only
**six subjects**, one atypical subject (S5, already accurate, so calibration overfits
and can hurt) swings the mean; and (2) outcomes are **base-dependent** — S5 improved
under the fixed split but worsened in leave-one-subject-out, purely because the two runs
trained on different subjects. Full-session accuracy also degrades for several subjects
because drift is **time-varying**: a one-time calibration slowly goes stale. These are
the small-sample and drift limitations anticipated at the outset, and reporting them —
rather than only the best split — is what makes the evaluation trustworthy.

## 7. Deployment

The inference service is stateless and binds the platform-injected `$PORT`, so it
deploys to Google Cloud Run or Render with min-instances 0 (scale-to-zero); a
`render.yaml` blueprint makes this a repo connection. The optional Kubernetes path
(`k8s/`) runs the stack on a local cluster and autoscales the worker via an HPA under
load. CI can auto-deploy on green `main`.

## 8. Conclusion and Future Work

DriftLess turns EOG baseline drift into an automatic, cloud-native service that makes a
shared model accurate for a brand-new user in seconds, served with millisecond latency
and evaluated honestly and leak-free. Future work: **periodic / continual
re-calibration** to track time-varying drift; a **larger subject cohort** for a firmer
efficacy claim; **live EOG hardware** in place of dataset replay; and **federated /
on-device** personalisation for privacy.
