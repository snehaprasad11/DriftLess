# Calibration worker — build context is the repo root
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt

COPY model/ model/
COPY worker/ worker/
COPY models/ models/          # bake the base model in (no shared volume in serverless/k8s)

ENV MODELS_DIR=/app/models
CMD ["python", "worker/worker.py"]
