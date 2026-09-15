# Hugging Face Space (Docker SDK) for the DriftLess inference service.
# It clones the public repo and runs the inference API on port 7860 (HF's default).
# Put THIS file (named exactly "Dockerfile") at the ROOT of your Hugging Face Space.
FROM python:3.12-slim
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# pull the project (public repo — includes model/, inference/, and the base model)
RUN git clone --depth 1 https://github.com/snehaprasad11/DriftLess.git .

# CPU-only torch, then the inference deps
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r inference/requirements.txt

ENV MODELS_DIR=/app/models
EXPOSE 7860
CMD uvicorn inference.app:app --host 0.0.0.0 --port 7860
