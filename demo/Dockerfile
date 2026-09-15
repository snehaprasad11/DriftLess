# Self-contained visual demo — build context is the repo root.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY model/ model/
COPY demo/ demo/
COPY client/ client/
# bake the base model in (single-container demo, no shared volume)
COPY models/ models/

ENV MODELS_DIR=/app/models
EXPOSE 8000
CMD uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-8000}
