---
title: DriftLess Inference
emoji: 👁️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DriftLess — Inference API

Live EOG → gaze inference for the [DriftLess](https://github.com/snehaprasad11/DriftLess)
project. The shared model is served here; per-user calibration produces personal
models that this service loads.

**Endpoints**
- `GET /health` — service status
- `POST /predict` — body `{ "user_id": "...", "window": [[...],[...]] }` (a 2×320 EOG window) → `{ "gaze": {"x":..,"y":..}, "blink": .., "model": "base" }`
