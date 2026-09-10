# HF Space image. CPU-only by design: the free tier has no GPU, and the text
# baseline does not need one.
FROM python:3.12-slim

# HF Spaces runs as uid 1000 and mounts nothing writable at /; keep everything
# under a home the app owns.
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=1337 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    SPHERE_ARTIFACTS_DIR=/home/app/app/artifacts \
    SPHERE_REPORTS_DIR=/home/app/app/reports

WORKDIR /home/app/app

# Dependency layer first, so source edits do not reinstall the world.
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app src ./src
RUN pip install --no-cache-dir --user '.[demo]'

COPY --chown=app:app apps ./apps
COPY --chown=app:app configs ./configs
COPY --chown=app:app manifests ./manifests
# artifacts/ and reports/ are produced by training and mounted or fetched at
# deploy time; the image deliberately ships no weights.

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/')"

CMD ["python", "apps/space/app.py"]
