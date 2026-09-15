# Inference service — build context is the repo root (see docker-compose.yml)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# CPU-only torch (small) from the PyTorch index, then the rest from PyPI
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY inference/requirements.txt inference/requirements.txt
RUN pip install --no-cache-dir -r inference/requirements.txt

COPY model/ model/
COPY inference/ inference/
# bake the base model in (serverless pods have no shared volume)
COPY models/ models/

ENV MODELS_DIR=/app/models
EXPOSE 8000
# shell form so a cloud platform's injected $PORT is honoured (default 8000 locally)
CMD uvicorn inference.app:app --host 0.0.0.0 --port ${PORT:-8000}
