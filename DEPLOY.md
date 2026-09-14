# DriftLess — Cloud Deployment (Phase 6)

The stack runs locally with `docker compose up`. This guide takes the **inference
service** serverless (the headline advanced-cloud feature: scale-to-zero,
wake-on-request, clone-under-load), wires **CI/CD**, and sketches the optional
**Kubernetes** autoscaling demo.

The services are already cloud-ready: they're **stateless**, each is its own
container, `/health` exists, and they bind to the platform-injected **`$PORT`**.

---

## 0. One decision first: where does the model live?

Serverless containers have no shared volume, so the base model must be reachable
at start-up. Two options:

- **Bake it into the image (simplest for the demo).** Temporarily allow the
  model into the build context by removing the `models/` line from
  `.dockerignore`, add `COPY models/ ./models/` to `inference/Dockerfile`, and
  build. The image then carries `base_model.pt` + `base_norm.npz`.
- **Object storage (production form).** Store models in S3-compatible storage
  (Cloudflare R2 / Backblaze B2) and swap `model/registry.py` for the S3 backend
  (same interface). This is the real "model registry" and what Phase 6 argues for.

For a first deploy, bake it in.

---

## 1. Deploy inference to Google Cloud Run (free tier, scale-to-zero)

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# build + push the image (Cloud Build), then deploy
gcloud run deploy driftless-inference \
  --source . \
  --dockerfile inference/Dockerfile \
  --region asia-south1 \
  --allow-unauthenticated \
  --min-instances 0 \          # scale to ZERO when idle (costs nothing)
  --max-instances 5 \          # clone under load
  --cpu 1 --memory 1Gi
```

Cloud Run injects `$PORT` (the container already honours it) and routes to
`/`; use `/health` as the health check. `--min-instances 0` is the scale-to-zero
you screenshot for the report; the first request after idle is the **cold start**
(measure and report cold-vs-warm latency — the report explicitly asks for this).

## 1b. Or Render (also free, also scale-to-zero)

- New → **Web Service** → connect the GitHub repo.
- Runtime **Docker**, Dockerfile path `inference/Dockerfile`, health check `/health`.
- Free instance type sleeps when idle (Render's scale-to-zero equivalent).
- Copy the **Deploy Hook** URL into a repo secret `RENDER_DEPLOY_HOOK` to enable
  the CI deploy step (see below).

---

## 2. CI/CD (`.github/workflows/ci.yml`)

Already wired: on every push it **runs the tests**, then **rebuilds all three
service images**. The final `deploy` job is commented out — uncomment it and add
your `RENDER_DEPLOY_HOOK` secret (or a `gcloud run deploy` step with a service
account key) to make green-main auto-deploy. That's the full
**build → test → deploy** MLOps loop.

---

## 3. (Optional, high marks) Kubernetes autoscaling — Phase 6.2

Run a free local cluster (`kind create cluster` or k3s) and give the **worker** a
Horizontal Pod Autoscaler so extra calibration workers spawn when the queue grows
— the textbook orchestration demo:

```yaml
# k8s/worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: driftless-worker }
spec:
  replicas: 1
  selector: { matchLabels: { app: driftless-worker } }
  template:
    metadata: { labels: { app: driftless-worker } }
    spec:
      containers:
        - name: worker
          image: driftless-worker:latest
          env:
            - { name: REDIS_URL, value: "redis://redis:6379/0" }
            - { name: INFERENCE_URL, value: "http://inference:8000" }
          resources:
            requests: { cpu: "250m", memory: "512Mi" }
            limits:   { cpu: "1",    memory: "1Gi" }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: driftless-worker-hpa }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: driftless-worker }
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
```

Load the queue, watch `kubectl get hpa -w` scale the worker up, then back to 1 when
idle — screenshot that for the orchestration section.

---

## What to capture for the report
- Cloud Run/Render URL serving `/predict`, and a **scale-to-zero** screenshot.
- **Cold vs warm latency** numbers (first request after idle vs subsequent).
- A green **CI run** (tests + build).
- (If done) the **HPA** scaling the worker under load.
