# Multi-stage Dockerfile for the Gradio application
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    gradio httpx pydantic mcp chromadb sentence-transformers

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY app.py .
COPY agent/ ./agent/
COPY mcp_server/ ./mcp_server/

RUN mkdir -p logs vector-db corpus

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/', timeout=5)"

CMD ["python", "app.py"]
