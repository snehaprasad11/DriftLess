# worker/  (Phase 5)

**The novel core, as a background job.** A worker that:

1. reads a calibration job from the **Redis queue**,
2. runs `model.calibrate(...)` — freezes the shared backbone and few-shot
   fine-tunes only the tiny drift adapter on the new user's ~10 trials,
3. writes the finished **personal model** to the registry, tagged to `user_id`.

**Why a worker + queue:** calibration is slow (~20 s), so making it a background
job means no user ever waits. And running a *second* worker when the queue grows
is the live horizontal-scaling demo (Phase 6 autoscaler).
