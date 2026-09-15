# DriftLess — Demo Script (for presenting to your teacher)

**Plan:** do the live clicking on the **local** version (`docker compose`, fast and
reliable — calibration in ~3 s). Only *point at* the public URL to show it's
deployed; don't calibrate live there (free 0.1-CPU tier is slow). Total ≈ 7–8 min.

---

## Before the demo (2-min setup)

1. Start the system:
   ```bash
   cd "...\Semester 5\Project_College\ACC Project\DriftLess"
   docker compose up -d
   ```
   Open **http://localhost:8000/demo/** and confirm it loads.
2. Open tabs ready to switch to: the **GitHub repo**, **`RESULTS.md`**, the **LOSO
   chart** (`notebooks/loso_before_after.png`), and the public URL
   (driftless-demo.onrender.com — just to point at).

Each run: **Start → Calibrate**; press **↻ New user** to reset and repeat.

---

## The demonstration (7–8 min)

### 1. The problem (30 s) — talk
> "DriftLess does hands-free eye control. It reads eye movement from EOG — tiny
> electrical signals from electrodes near the eyes, no camera. The hard part is
> 'baseline drift': the signal drifts differently for every person, so one model
> is wrong for the next. DriftLess makes the system **auto-tune to each new user
> in seconds**, as a cloud service."

### 2. The data (30 s) — talk
> "The data is the **UM Malta EyeCon EOG dataset** — a public research dataset from
> the University of Malta's Centre for Biomedical Cybernetics. Real recordings from
> **6 people**: electrodes captured horizontal and vertical eye movement at 256 Hz.
> Each did 300 trials — each trial a forward saccade, a return saccade, and a blink —
> with the true gaze angles as labels. I used their cleanest set (zero-centred
> H+V EOG)."
- **URL:** https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/
- Facts: 6 subjects (2M/4F, ~24.7 yrs), 256 Hz, 2 channels (H = V3−V4, V = V1−V2),
  band-pass filtered, 300 trials/subject. Ref paper: Barbara et al., *"A comparison
  of EOG baseline drift mitigation techniques,"* Biomed. Signal Proc. & Control, 2020.

### 3. Live demo — localhost (2 min) — the highlight
- `localhost:8000/demo/` → **▶ Start streaming**.
  > "A person the model has never seen. Orange ring = where they're really looking;
  > dot = the AI's guess. Close but ~5° off — it doesn't know this person's drift yet."
- **✦ Calibrate this user** → wait ~3 s.
  > "It takes ~10 of their eye movements and fine-tunes a tiny personal model.
  > There — it flipped to 'personal', error dropped ~5° → ~3.5°, about 25% better.
  > That automatic few-second personalisation is the core contribution."
- **↻ New user** to reset and show it again.

### 4. How it works (1.5 min) — show GitHub
> "It's a small neural net — a 1D-CNN and BiLSTM read the signal, and a tiny 'drift
> adapter' is the only part personalised per user. And it's proper cloud
> microservices:"
- Point at folders: `gateway/` (front door), `inference/` (fast, scales to zero),
  `worker/` (calibration in the background off a Redis queue), `model/` (shared
  model + registry). "One command — `docker compose up` — runs it all."

### 5. Engineering quality (1 min) — show GitHub
> "Automated tests + a CI pipeline on every push, and I verified Kubernetes
> autoscaling — the calibration worker scaled 1→4 pods under load."
- Show `k8s/hpa_demo_output.txt` and the CI file.

### 6. Honest results (1.5 min) — show RESULTS.md + LOSO chart
> "Evaluated only on people the model never trained on — leave-one-subject-out, no
> cheating. Calibration cuts error up to 34% in the best case, ~11% averaged across
> all six subjects (helps 5 of 6). Not uniform because there are only 6 subjects and
> drift is time-varying — my honest limitation and future work."

### 7. It's really deployed (30 s)
> "It's also live on the internet on a free cloud tier." — point at
> driftless-demo.onrender.com (don't calibrate live there).

### 8. Close (15 s)
> "DriftLess turns the hardest problem in eye-signal tracking — per-person drift —
> into an automatic, cloud-hosted service that tunes itself to any new user in
> seconds. All on GitHub, tested, and deployed."

---

## Likely questions → quick answers
- **"Real person?"** → "Real recorded human EOG from that published dataset; live
  capture just needs an electrode headset — my documented next step."
- **"Tested fairly?"** → "Leave-one-subject-out; each subject held out, never trained
  on. No data leakage."
- **"Why cloud?"** → "EOG is biometric and calibration is short and bursty — ideal for
  elastic, scale-to-zero cloud workers."
- **"What's new?"** → "Prior work fixes drift by hand, offline. Mine is automatic,
  few-shot, and a live cloud service — a method *and* a system."

## One-line pitch
> *"DriftLess turns per-person eye-signal drift into an automatic, cloud-hosted
> service that tunes itself to any new user in seconds."*
