# DriftLess — Evaluation (Phase 7)

All numbers are on **held-out subjects** (leave-one-subject-out spirit): the base
model was trained on S1–S4 and every metric below is measured on **S5/S6, whom
the model never trained on** — the honest "does this help a genuinely new user?"
test the report demands.

## 1. The ML result

**Base model (shared, before any calibration):**

| Set | Gaze mean error | Blink accuracy |
|-----|-----------------|----------------|
| Train subjects (reference) | 3.34° | 99.6% |
| **Never-seen (S5+S6)** | **10.18°** (full session) | **85.4%** |

The ~6.8° train-vs-new gap is the person-specific baseline drift the shared model
can't know in advance — the problem calibration exists to close.

**After per-user calibration (~10 trials, only the 8.7k-param adapter+heads, backbone frozen):**

| Metric (never-seen) | Before | After | Change |
|---------------------|--------|-------|--------|
| Gaze error, right after calibration (realistic) | 6.4° | **4.5°** | **−30%**, both users |
| — S5 | 5.5° | 4.1° | −26% |
| — S6 | 7.2° | 4.9° | −33% |
| Gaze error, full ~20-min session (conservative) | 10.3° | 9.0° | −12% |

Calibration reliably cuts new-user error ~30% in the near term. Confirmed **live**
through the deployed stack (S5, streamed): **4.94° → 4.11°**.

## 2. The cloud result

| Property | Result |
|----------|--------|
| Inference latency (warm) | ~6–16 ms round-trip (gateway → inference → back) |
| Inference latency (cold start) | higher on the first request after idle (model load) — the scale-to-zero trade-off |
| Serverless | inference is stateless + binds `$PORT` → deploys to Cloud Run/Render with **min-instances 0** (scale-to-zero) |
| Async calibration | gateway never blocks: jobs go on a **Redis** queue; the worker consumes them (~seconds warm) |
| Horizontal scaling | add workers under load — `docker compose up --scale worker=2`, or a Kubernetes **HPA** (see `DEPLOY.md`) |
| Reproducibility | pinned deps, `pytest` suite (16 tests), CI builds+tests every push |

## 3. Honest limitations

- **Time-varying drift.** A one-time calibration decays over a long session — clearly
  for S5 (error climbs from ~4.8° right after calibration to ~14° by 20 min), mildly
  for S6. This is the report's own listed limitation, caught in our data, and it
  motivates the future work (periodic / continual re-calibration).
- **Few subjects (6).** Public EOG datasets are small, so per-user gains are noisy;
  leave-one-subject-out keeps the evaluation honest but the sample is small.
- **Lab data.** Recorded with a chin rest and clean electrodes; a real wearable is
  noisier and moves. This project is the algorithm-and-system proof; hardware is future work.

## 4. What proves each half of the grade
- **ML:** the before→after calibration drop on never-seen users (§1), leak-free by-subject split.
- **Cloud:** stateless serverless inference + scale-to-zero, async queue + scalable worker,
  one-command local stack (`docker compose up`), CI/CD, and a live streaming demo.
