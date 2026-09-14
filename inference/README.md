# inference/  (Phase 5)

**Fast, stateless gaze prediction.** A FastAPI app that, given a window +
`user_id`, loads that user's model from the registry (cached in memory) and
returns the predicted gaze displacement and blink flag.

Key property: **stateless** — it keeps nothing between requests; the model always
comes from the registry. That's what lets the cloud run many identical copies and
**scale to zero** when idle (Phase 6). This is the service that must stay fast, so
it's on the "every use" hot path (milliseconds, constant).
