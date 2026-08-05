# Coloratura — Stage A web MVP, containerized for cloud deployment (built
# for Render; also works on Hugging Face Spaces / Cloud Run / anywhere else
# that speaks plain Docker). Runs the same webapp.py FastAPI app used
# locally; only Stage A (free local compute — CV + CLIP + our own harmony
# engine + synthesizer) is exposed here, consistent with webapp.py's own
# docstring. HTTP Basic Auth (see webapp.py) is what actually protects this
# once it's reachable on the public internet — set WEBAPP_USER /
# WEBAPP_PASSWORD as environment variables/secrets in the host's dashboard,
# not baked into the image.
#
# Listens on $PORT if the platform sets one (Render defaults this to
# 10000), falling back to 7860 (Hugging Face Spaces' convention) otherwise.

FROM python:3.12-slim

# CPU-only torch explicitly, from PyTorch's own CPU wheel index — plain
# `pip install torch` resolves CUDA wheels (multiple GB, irrelevant on a
# free CPU Space and slow enough to risk a build timeout).
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOME="/home/user"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src

EXPOSE 7860

# shell form (not exec-form JSON array) so ${PORT:-7860} actually expands
CMD python -m uvicorn webapp:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir src
