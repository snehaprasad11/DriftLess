# DriftLess

**Eye-controlled gaze tracking that personalises itself to each new user in seconds — running as a small cloud system.**

> 🟢 **Live API:** https://driftless-inference.onrender.com — try [`/health`](https://driftless-inference.onrender.com/health) or `POST /predict`. *(Free tier: the first request after ~15 min idle wakes it in ~30–60 s; then it's fast.)*

DriftLess reads eye movement from **EOG** (electrooculography — tiny voltage
changes picked up by electrodes near the eyes, no camera needed) and predicts
**where the eye looked**. It's built for hands-free interfaces (accessibility,
AR/VR, situations where your hands are busy).

---

## The problem, in one line

EOG has a stubborn flaw called **baseline drift**: even when the eye is perfectly
still, the signal slowly wanders — and it wanders *differently for every person*.
So a model tuned for one person is wrong for the next.

## What DriftLess does about it

1. **Learn once** — train one shared deep-learning model on several people. It's
   already decent for anyone.
2. **Adapt fast** — when a brand-new user connects, freeze the shared model and
   fine-tune only a tiny "drift adapter" on about **10 quick eye movements**. This
   takes a few seconds and locks onto *that person's* drift.
3. **Serve in the cloud** — store each person's small personal model and serve
   fast predictions. Calibration runs in the background so nobody waits.

**Result:** on people the model has never seen, gaze error drops about **30%**
right after this quick calibration.

---

## How it works (the flow)

```
        your eyes                   the cloud
   ┌──────────────┐          ┌───────────────────────────────────────┐
   │ EOG signal   │  stream  │  API gateway ──▶ inference service      │
   │ (web client) │ ───────▶ │      │          (loads your model,      │
   └──────────────┘          │      │           predicts gaze fast)    │
          ▲                  │      │                                  │
          │ predicted gaze   │      └─▶ Redis queue ─▶ calibration      │
          └──────────────────│          worker (fine-tunes your        │
                             │           personal model, once)         │
                             │                 ▲                        │
                             │           model registry (base + your   │
                             │           personal model, stored)        │
                             └───────────────────────────────────────┘
```

The model itself: **1D-CNN → BiLSTM → drift adapter → two outputs** (gaze x/y, and
a blink yes/no). Only the little drift adapter is personalised per user.

---

## What's in this repo

```
DriftLess/
├── model/        the shared model, windowing, calibration, registry (the ML core)
├── notebooks/    step-by-step: explore data → make windows → train → calibrate
├── gateway/      API gateway (the front door) + the live web demo
├── inference/    fast gaze-prediction service
├── worker/       background calibration worker
├── client/       the streaming demo web page
├── tests/        automated tests (pytest)
├── docker-compose.yml   runs the whole system locally
├── DEPLOY.md     how to put it on the cloud (Cloud Run / Render) + CI/CD
└── EVALUATION.md the results and honest limitations
```

---

## What you need

- **Docker Desktop** (to run the whole system with one command), and/or
- **Python 3.12** (to run the notebooks / tests). Not 3.13+ — PyTorch needs 3.12.

---

## How to run it

### Option A — the live demo (easiest, needs Docker)

From the project folder:

```bash
docker compose up --build
```

Then open **http://localhost:8000/demo/** in your browser:

- Click **Start streaming** — a never-seen person's eye movements replay through
  the system, and you'll see the predicted gaze (teal dot) vs. the true target
  (orange ring). Error starts a bit high.
- Click **Calibrate this user** — after a few seconds the personal model kicks in
  and the dot snaps onto the target: error drops and latency stays a few
  milliseconds.

Stop everything with `docker compose down`.

> The repo already includes a small trained base model (`models/`) and a sample
> replay (`client/demo_windows.json`) so the demo works out of the box.

### Option B — the notebooks (to see the ML end-to-end)

1. Create a Python 3.12 environment and install dependencies:
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
2. Download the EOG dataset (see `data/README.md`) — it's **not** in this repo
   (biometric + large). Put it in `data/dataset1/`.
3. Run the notebooks in order (train `03` on Google Colab's free GPU):
   `01_explore` → `02_windowing` → `03_train_base` → `04_calibrate`.

### Run the tests

```bash
.venv\Scripts\python -m pytest tests/ -q
```

The tests are self-contained (they make their own fake data), so they need
neither the dataset nor a trained model.

---

## The API (only two calls)

```
POST /predict    { "user_id": "...", "window": [[...],[...]] }
                 → { "gaze": {"x": 12.4, "y": -3.1}, "blink": false, "model": "base" }

POST /calibrate  { "user_id": "...", "trials": [ ...~10 labelled windows... ] }
                 → { "status": "queued", "job_id": "cal_8f2a", "n_trials": 10 }
```

---

## Results (measured on people the model never trained on)

- Gaze error **~10° → ~4.5°** right after ~10-trial calibration (about **30% lower**).
- Inference latency **~6 ms** when warm.
- See **`EVALUATION.md`** for the full breakdown and honest limitations (the main
  one: drift keeps changing, so a one-time calibration slowly goes stale — future
  work is periodic re-calibration).

---

## Deploying to the cloud

See **`DEPLOY.md`** — the inference service is stateless and scales to zero, so it
runs on free serverless tiers (Google Cloud Run / Render), with a GitHub Actions
pipeline that tests and builds on every push.

---

*Dataset: **UM Malta EyeCon EOG** (University of Malta, Centre for Biomedical
Cybernetics) — 6 subjects, 256 Hz, horizontal + vertical EOG, saccades + blinks
with target gaze angles. https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/
Built for the Advanced Cloud Computing course.*
