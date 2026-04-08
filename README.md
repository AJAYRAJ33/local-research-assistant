# Local Research Assistant

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/local-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/local-research-assistant/actions/workflows/ci.yml)

A fully local AI pipeline: fine-tuned Phi-3-mini → Ollama inference → MCP tool server → ReAct agent → Gradio UI. Runs entirely on your laptop with no cloud dependencies.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free on HDD | 20 GB |
| OS | Ubuntu 22.04+ / macOS 13+ | Ubuntu 24.04 |
| Python | 3.11 | 3.11 |
| GPU | None (CPU-only supported) | NVIDIA 8GB VRAM |

> **8GB RAM users**: Store all models on the HDD (not SSD). Close browsers and other apps during training. Inference will run at 2–5 tokens/sec — this is expected.

---

## One-Command Setup

```bash
git clone https://github.com/YOUR_USERNAME/local-research-assistant
cd local-research-assistant
chmod +x setup.sh && ./setup.sh
```

Or manually follow the steps below.

---

## Manual Setup

### 1. Install system dependencies

```bash
# Python 3.11
sudo apt update && sudo apt install python3.11 python3.11-venv git curl -y

# Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Docker (for containerised deployment)
curl -fsSL https://get.docker.com | sh
```

### 2. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure storage (8GB RAM users — IMPORTANT)

```bash
# Find your 1TB HDD mount point
lsblk

# Set Ollama to store models on the HDD (replace path with yours)
echo 'export OLLAMA_MODELS=/media/yourname/data/ollama-models' >> ~/.bashrc
source ~/.bashrc
mkdir -p /media/yourname/data/ollama-models
```

### 4. Prepare dataset

```bash
python data/prepare_dataset.py
# Output: data/train/ and data/val/ (210 samples, 90/10 split)
```

### 5. Fine-tune the model (run overnight)

```bash
python train.py
# Takes 4–8 hours on CPU. LoRA adapter saved to ./lora-adapter/
```

### 6. Convert to GGUF and load into Ollama

```bash
# Merge adapter into base model
python merge_adapter.py

# Clone llama.cpp and convert
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../merged-model \
    --outtype q4_k_m \
    --outfile ../phi3-devops-q4.gguf
cd ..

# Load into Ollama
ollama create phi3-devops -f Modelfile

# Test it works
ollama run phi3-devops "What is a Kubernetes pod?"
```

### 7. Index documents into vector store

```bash
python mcp_server/index_docs.py
# Creates ./corpus/ with sample docs and indexes into ./vector-db/
```

### 8. Start everything

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Inference server
uvicorn inference_server.main:app --host 0.0.0.0 --port 8000

# Terminal 3: Launch the app (starts MCP server + Gradio UI)
python app.py
```

Open http://localhost:7860 in your browser.

---

## Docker (single command)

```bash
cp .env.example .env
# Edit .env: set OLLAMA_MODELS_PATH to your HDD path

docker-compose up -d
```

Open http://localhost:7860

---

## Project Structure

```
local-research-assistant/
├── data/
│   └── prepare_dataset.py      # Task 01: dataset creation
├── train.py                    # Task 01: QLoRA fine-tuning
├── merge_adapter.py            # Task 01: merge + GGUF export
├── evaluate.py                 # Task 01: ROUGE-L evaluation
├── Modelfile                   # Task 02: Ollama model config
├── inference_server/
│   ├── main.py                 # Task 02: FastAPI wrapper
│   ├── benchmark.py            # Task 02: quantization benchmarks
│   └── Dockerfile
├── mcp_server/
│   ├── server.py               # Task 03: MCP server (3 tools)
│   └── index_docs.py           # Task 03: document indexer
├── agent/
│   └── agent.py                # Task 04: ReAct agent
├── app.py                      # Task 05: Gradio UI + orchestration
├── docker-compose.yml          # Task 02+05: full stack
├── .github/workflows/ci.yml    # Task 06: CI/CD pipeline
├── tests/
│   └── test_mcp_tools.py       # Task 03: tool tests
├── NOTES.md                    # Written explanations
├── pyproject.toml              # ruff + mypy config
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running Evaluation

```bash
# Requires Ollama running with phi3-devops loaded
python evaluate.py

# CI mode (fails if ROUGE-L < threshold)
python evaluate.py --threshold 0.20
```

Results saved to `eval_results.json`.

---

## Benchmark Quantization

```bash
# Requires Ollama running
python inference_server/benchmark.py
```

---

## Required GitHub Actions Secrets

Set these in your repo under **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (create at hub.docker.com/settings/security) |
| `DEPLOY_HOST` | IP or hostname of deployment server |
| `DEPLOY_USER` | SSH username on server |
| `DEPLOY_SSH_KEY` | Full PEM content of your private SSH key |

---

## Model Weights

- **LoRA adapter**: [HuggingFace Hub link — upload after training]
- **Merged GGUF (Q4_K_M)**: [Google Drive link — upload after conversion]

---

## Known Limitations

1. **Inference speed**: 2–5 tokens/sec on CPU. Answers take 10–30 seconds. This is a hardware constraint, not a bug.
2. **Context window**: 2048 tokens. Long conversations truncate older history.
3. **Training data**: 210 samples is small. The model improves format adherence more than factual knowledge.
4. **No GPU**: bitsandbytes 4-bit training on pure CPU is significantly slower than GPU. CUDA support dramatically speeds this up.
5. **MCP sandbox**: The `run_python` sandbox blocks `os`, `sys`, and `subprocess` but is not a true sandbox (no seccomp, no containerisation). Do not run untrusted code in production.

## Improvements Given More Time

1. **OpenTelemetry tracing** — per-request spans across all services for true distributed observability
2. **Vector memory** — store conversation summaries in ChromaDB so the agent recalls past sessions
3. **Blue/green deployment** — proper atomic traffic switching instead of rolling restart
4. **Larger dataset** — scrape Stack Overflow DevOps tags for 2000+ samples
5. **Integration tests in CI** — spin up the full docker-compose stack and run end-to-end query tests
