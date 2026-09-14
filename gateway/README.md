# gateway/  (Phase 5)

**The single front door.** A FastAPI app that receives EOG windows, identifies
the user, and routes traffic to the right place:

- **known user** → forward the window to the **inference service** for a fast
  gaze/blink prediction.
- **new user (no model yet)** → drop a calibration job on the **Redis queue** and
  answer immediately. It never waits for the ~20-second calibration.

**Why it never waits:** if the gateway blocked on calibration, a new user's first
request would hang ~20 s and a burst of new users could freeze the service. The
queue decouples them — this async decoupling is the central cloud lesson.

Two API calls cover the whole system:

```
POST /predict    { user_id, window }         → { gaze:{x,y}, blink }
POST /calibrate  { user_id, trials:[...~10] } → { status:"queued", job_id }
```
