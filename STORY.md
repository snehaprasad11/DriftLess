# DriftLess — the whole project in plain language

A simple, story-style walkthrough for explaining the project (e.g. in a demo or
viva). No jargon. For the technical detail, see `REPORT.md`, `EVALUATION.md`,
and `RESULTS.md`.

---

## 1. The problem (in one line)

We want to control things with your **eyes** — no camera, no hands. We read eye
movement from **EOG** (tiny electrical signals from electrodes near the eyes).
The catch: EOG has **baseline drift** — even when the eye is still, the signal
slowly wanders, and it wanders **differently for every person**. So a model
tuned for one person is wrong for the next.

## 2. The data

The **UM Malta EyeCon EOG dataset** — a public research dataset with real
recordings from **6 people**. Electrodes captured horizontal + vertical eye
movement at 256 Hz; each person did 300 trials (a saccade out, a saccade back,
and a blink) with the true gaze angles as labels.

URL: https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/

## 3. What we did, step by step

1. **Explore + window the data** — cut the long recordings into short overlapping
   windows the model can read.
2. **Train ONE shared model** (the "base model") on several people. It learns eye
   movement *in general*. Decent for anyone, but not tuned to any one person.
3. **Personalise per user** — when a brand-new person arrives, we **freeze** the
   base model and fine-tune only a **tiny 8,000-number "drift adapter"** using
   ~10 of *their* eye movements. Takes a few seconds. This is the core idea.
4. **Wrap it as a cloud system** — split into small services, containerise,
   deploy live, and prove it autoscales.

## 4. Base model vs. personal model (the part people ask about)

**We trained only ONE model — the base.**

- **Base model** = the one model we trained, on 4 people (S1–S4). Used for
  everyone by default. A bit off on a new person (~5° error) because it doesn't
  know their drift yet. In the demo the badge shows **`base`**.
- **Personal model** = the **same base model**, with only its tiny adapter nudged
  to fit **one specific person**. It is **not** a second model trained from
  scratch. We make it on the fly, per user, in seconds. Badge shows **`personal`**;
  error drops (~4°) and the dot lands on the target.

> **Glasses analogy:** the base is one-size-fits-all glasses (okay for most). The
> personal model is those *same* glasses with a 10-second tweak to your exact eyes.
> We didn't make a new pair — we nudged the one pair to fit each person.

Why only tweak a tiny adapter instead of retraining everything? Retraining needs
thousands of samples and minutes of GPU. Adjusting the tiny adapter needs ~10
samples and a few seconds, and it can't overfit — which is exactly why it can run
live, per user, in the cloud.

## 5. How we used the cloud (the ACC part)

Instead of one big program, DriftLess is split into **small independent services**
(microservices) that each do one job. Like a restaurant:

| Service | Restaurant role | What it does |
|---|---|---|
| **Gateway** | Waiter / front door | Takes requests from the web page; forwards predictions to inference; drops calibration jobs onto the queue. |
| **Inference** | Line cook | Loads the model and answers gaze predictions **fast** (~6 ms). |
| **Redis queue** | Order rail | A waiting line for calibration jobs, so nobody waits in real time. |
| **Worker** | Prep chef in back | Picks jobs off the queue and does the slow calibration in the **background**. |
| **Model registry** | Pantry / storage | Stores the base model **and** each user's personal model. |

**Why split it this way?** Prediction is fast, tiny, and constant; calibration is
slow, heavy, and rare (bursty). If they shared one program, a heavy calibration
would freeze predictions. Putting calibration **behind a queue with its own
worker** keeps predictions instant while personalisation happens quietly in the
background.

**The flow:**
1. Web page streams eye signal → **Gateway**.
2. Gateway → **Inference** → predicts gaze → dot appears on the page.
3. User clicks Calibrate → Gateway drops a job on the **Redis queue**.
4. **Worker** grabs it, fine-tunes that user's adapter, saves the personal model
   to the **registry**.
5. Next prediction loads the **personal** model → badge flips to `personal`, error
   drops.

**The three cloud things that make it "advanced":**

1. **Containers (Docker).** Every service is packaged in its own container. One
   command — `docker compose up` — runs the whole system identically anywhere.
2. **Stateless + autoscaling.** Inference holds no user data itself (it reads
   models from the registry), so it can run many copies when busy and **zero when
   idle**. We proved autoscaling with **Kubernetes HPA**: the calibration worker
   scaled **1 → 4 pods** under load (`k8s/hpa_demo_output.txt`).
3. **Deployed live + CI/CD.** It runs on **Render**
   (https://driftless-inference.onrender.com), with a **GitHub Actions** pipeline
   that tests and rebuilds on every push.

## 6. What the final website shows

- **Orange ring** = where the person is *really* looking (the true target).
- **Teal dot** = the AI's guess (predicted gaze).
- **Badge** = which model is answering: `base` (shared) or `personal` (tuned to
  this user).
- **Gaze error (°)** = distance between dot and ring — lower is better.
- Click **Start** to stream a never-seen person; click **Calibrate** to build
  their personal model; watch the error drop and the badge flip.

## 7. One-line pitch

> *DriftLess turns the hardest problem in eye-signal tracking — per-person drift —
> into an automatic, cloud-hosted service that tunes itself to any new user in
> seconds.*
