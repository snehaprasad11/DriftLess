# client/  (Phase 7)

**The streaming web client — stands in for a wearable.** A simple web page that:

- replays a chosen subject's EOG recording over a **WebSocket**,
- draws the predicted gaze live as a moving cursor,
- shows a latency read-out (so you can demo cold-start vs warm behaviour).

**Why a live demo:** a moving demo convinces more than static numbers. When you
connect as a new user, predictions start rough, the calibration job runs, and
predictions visibly snap into accuracy — the whole thesis, on screen.

In a real deployment this is where the EOG headset would plug in; for the demo it
replays dataset recordings instead.
