# DriftLess — Demo Script (for presenting to your teacher)

**Key idea:** do the live clicking on the **local** version (fast, reliable —
calibration in ~3 s), and show the **public URL** to prove it's really deployed.
The free public tier is 0.1 CPU, so calibration there takes ~30–45 s.

---

## Before the demo (2 minutes of prep)

1. **Start the local stack** (fast, reliable):
   ```bash
   cd "...\Semester 5\Project_College\ACC Project\DriftLess"
   docker compose up -d
   ```
   Open **http://localhost:8000/demo/** and confirm it loads.
2. **Wake the public site** so it isn't cold mid-demo: open
   **https://driftless-demo.onrender.com/demo/** once and leave it.
3. **Open tabs** ready: the GitHub repo, and `REPORT.md` / `EVALUATION.md`.

Each demo run: click **Start → Calibrate**; hit **↻ New user** to reset and run again.

---

## The demo (~6 minutes)

### 1. The problem (30 s)
> "EOG reads eye movement as an electrical signal — no camera. But it has
> 'baseline drift' that's different for every person, so one model is wrong for
> the next. DriftLess makes the system auto-personalise to each new user in
> seconds, as a cloud service."

### 2. Live demo — localhost (2 min)
- `localhost:8000/demo/` → **▶ Start streaming**.
  > "A person the model has never seen. Orange ring = where they're really
  > looking; dot = the AI's guess. Roughly right but off (~5°) — it doesn't know
  > this person's drift yet."
- **✦ Calibrate this user** → wait ~3 s.
  > "It takes ~10 of their eye movements and fine-tunes a tiny personal model.
  > There — badge flipped to 'personal', dot snapped onto the target, error
  > dropped. That few-second personalisation is the whole contribution."
- **↻ New user** to reset and show it again if asked.

### 3. It's really deployed (1 min)
- Switch to **https://driftless-demo.onrender.com/demo/**.
  > "It's not just on my laptop — it's live on the internet on a free cloud tier.
  > Same system, running publicly." (Slower here — free tier is 0.1 CPU.)
- Optional: **https://driftless-inference.onrender.com** — the raw API.

### 4. The engineering (1 min) — show GitHub
> "Built as proper cloud microservices — API gateway, a stateless inference
> service that scales to zero, a calibration worker behind a Redis queue, and a
> model registry — all containerised, one command to run, with automated tests
> and CI. I verified Kubernetes autoscaling too: the worker scaled 1→4 pods under
> load." (Show `k8s/hpa_demo_output.txt`.)

### 5. Honest results (1 min) — show EVALUATION.md / RESULTS.md
> "Evaluated only on people the model never trained on. Calibration cuts error
> ~34% in the best case, and ~11% averaged across all six subjects in
> leave-one-subject-out (helps 5 of 6). It's not uniform because there are only 6
> subjects and drift is time-varying — my honest limitation and future work."

---

## Likely questions → quick answers

- **"Is it a real person?"** → "Real recorded human EOG from a published research
  dataset; live capture just needs electrode hardware — the documented next step."
- **"Did you test fairly?"** → "Leave-one-subject-out: each subject is held out and
  tested, never trained on. No data leakage."
- **"Why the cloud?"** → "EOG is biometric, and calibration is a short, bursty
  compute job — ideal for elastic, scale-to-zero cloud workers."
- **"What's novel?"** → "Prior work fixes drift per-person, offline, by hand. This
  is automatic, few-shot, and a live cloud service — method + system."

---

## One-line pitch (open or close with this)
> *"DriftLess turns the hardest problem in eye-signal tracking — per-person drift —
> into an automatic, cloud-hosted service that tunes itself to any new user in
> seconds."*
