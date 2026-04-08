FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir mcp chromadb sentence-transformers
COPY mcp_server/ ./mcp_server/
RUN mkdir -p vector-db corpus
CMD ["python", "mcp_server/server.py"]
