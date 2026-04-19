# ── Stage: Base Python image ──────────────────────────────────────────────────
FROM python:3.11-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: prevents .pyc files (cleaner container)
# PYTHONUNBUFFERED: ensures logs stream to stdout immediately (visible in Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker's layer cache —
# dependencies won't reinstall unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
# NOTE: model.pkl must be generated locally (python model.py) before building.
# It is copied into the image here so the container can serve predictions
# without needing the training data or scikit-learn training step at runtime.
COPY app.py .
COPY model.pkl .
COPY templates/ templates/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Run with Gunicorn (production WSGI server) ────────────────────────────────
# Flask's built-in dev server is single-threaded and not safe for production.
# Gunicorn runs 4 worker processes for better concurrency.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]