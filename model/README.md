# model/

The shared model definition and the calibration logic — the machine-learning
heart that every service imports.

Built in later phases:

- **Phase 3** — `GazeNet`: a shared backbone (1D-CNN → BiLSTM) feeding two heads
  (gaze = x,y displacement; blink = yes/no), plus a small **drift adapter** block
  that is the only per-user part.
- **Phase 4** — `calibrate(...)`: freeze the backbone and few-shot fine-tune only
  the drift adapter on ~10 trials from a new user. This is the novel core.

Keeping this as one importable package means the trainer, the inference service,
and the calibration worker all use the exact same model code — one source of truth.
