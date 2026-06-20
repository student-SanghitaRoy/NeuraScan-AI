FROM python:3.11-slim

# Hugging Face Spaces convention: run as a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies first (better Docker layer caching -
# this layer only rebuilds when requirements.txt actually changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the project
COPY --chown=user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["gunicorn", "--chdir", "backend", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--max-requests", "5", "--max-requests-jitter", "2"]
