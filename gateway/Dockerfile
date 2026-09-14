# API gateway — light (no torch); build context is the repo root
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

COPY gateway/requirements.txt gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

COPY model/ model/
COPY gateway/ gateway/
COPY client/ client/

EXPOSE 8000
# shell form so a cloud platform's injected $PORT is honoured (default 8000 locally)
CMD uvicorn gateway.app:app --host 0.0.0.0 --port ${PORT:-8000}
