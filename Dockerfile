# Multi-stage build for Data Engineering Agents
# -----------------------------------------------

FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# -----------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code.
COPY src/ src/
COPY templates/ templates/
COPY config/ config/
COPY main.py .

# Default: run the ingestion agent (overridden per Cloud Run service).
ENV AGENT="ingestion"

EXPOSE 8080

ENTRYPOINT ["python", "main.py"]
CMD ["--agent", "ingestion"]
