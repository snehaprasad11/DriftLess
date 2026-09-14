# Kubernetes deployment (Phase 6.2 — optional orchestration demo)

Runs the whole stack on a **free local cluster** and lets the calibration
**worker autoscale** under load — the textbook orchestration demo.

## Run it

1. Create a local cluster (either works):
   ```bash
   kind create cluster            # needs Docker
   # or: k3d cluster create driftless
   ```
2. Build the images and load them into the cluster (kind can't pull local images):
   ```bash
   docker build -f inference/Dockerfile -t driftless-inference:latest .
   docker build -f gateway/Dockerfile   -t driftless-gateway:latest .
   docker build -f worker/Dockerfile    -t driftless-worker:latest .
   kind load docker-image driftless-inference:latest driftless-gateway:latest driftless-worker:latest
   ```
3. Deploy, and (for the HPA) install the metrics-server:
   ```bash
   kubectl apply -f k8s/driftless.yaml
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   # kind only: patch metrics-server to skip TLS verify
   kubectl -n kube-system patch deployment metrics-server --type=json \
     -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
   ```
4. Watch the worker scale as the queue drives CPU up:
   ```bash
   kubectl get hpa -w          # replicas climb 1 -> up to 4 under load, then back to 1
   kubectl get pods -w
   ```

The gateway is reachable at `http://localhost:30080` (NodePort).

> Models: bake them into the inference/worker images (add `COPY models/ ./models/`
> to those Dockerfiles) or mount a PersistentVolume — see the note in
> `driftless.yaml`. The autoscaling demo itself doesn't depend on this.
