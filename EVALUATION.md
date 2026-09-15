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

**After per-user calibration (~10 trials, only the 8.4k-param drift adapter — backbone and heads frozen, with mild weight decay to prevent overfitting):**

| Metric (never-seen) | Before | After | Change |
|---------------------|--------|-------|--------|
| Gaze error, right after calibration (realistic) | 6.4° | **4.2°** | **−34%**, both users |
| — S5 | 5.5° | 3.8° | −30% |
| — S6 | 7.2° | 4.5° | −38% |
| Gaze error, full ~20-min session (conservative) | 10.3° | 8.7° | −16% |

Confirmed **live** through the deployed stack (a never-seen subject streamed over
WebSocket) with ~6 ms prediction latency.

**Leave-one-subject-out (all 6 subjects, fair 80-epoch training):** rotating every
subject as the never-seen new user, near-term error improves **7.09° → 6.30° (−11%)**,
helping **5 of 6** subjects. This is more modest than the single fixed split (−34%)
because with only 6 subjects one atypical subject swings the mean, and outcomes are
base-dependent. The mechanism clearly works (large, consistent wins on subjects with
real drift, e.g. S3/S4 −24% to −32%), but a firm *uniform* claim needs more subjects.
See `RESULTS.md` §4.

## 2. The cloud result

| Property | Result |
|----------|--------|
| Inference latency (warm) | ~6–16 ms round-trip (gateway → inference → back) |
| Inference latency (cold start) | higher on the first request after idle (model load) — the scale-to-zero trade-off |
| Serverless | inference is stateless + binds `$PORT` → deploys to Cloud Run/Render with **min-instances 0** (scale-to-zero) |
| Async calibration | gateway never blocks: jobs go on a **Redis** queue; the worker consumes them (~seconds warm) |
| Horizontal scaling | add workers under load — `docker compose up --scale worker=2`, or a Kubernetes **HPA**. **Verified live on Kubernetes:** 400 queued jobs drove worker CPU to 333%, and the HorizontalPodAutoscaler scaled the worker **1 → 4 pods** automatically (`k8s/hpa_demo_output.txt`) |
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
