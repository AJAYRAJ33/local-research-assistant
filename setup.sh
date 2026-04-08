#!/usr/bin/env bash
# setup.sh — one-command setup for Local Research Assistant
# Usage: chmod +x setup.sh && ./setup.sh

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "=================================================="
echo "   Local Research Assistant — Setup"
echo "=================================================="
echo ""

# ── Check Python ───────────────────────────────────────
if ! command -v python3.11 &>/dev/null; then
    warn "Python 3.11 not found. Installing..."
    sudo apt update && sudo apt install python3.11 python3.11-venv -y
fi
info "Python: $(python3.11 --version)"

# ── Virtual environment ────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    python3.11 -m venv .venv
fi
source .venv/bin/activate
info "Virtual environment activated"

# ── Python dependencies ────────────────────────────────
info "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
info "Dependencies installed"

# ── Ollama ─────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    info "Ollama already installed: $(ollama --version)"
fi

# ── Prepare dataset ────────────────────────────────────
if [ ! -d "data/train" ]; then
    info "Preparing dataset..."
    python data/prepare_dataset.py
else
    info "Dataset already exists (data/train/)"
fi

# ── Index documents ────────────────────────────────────
if [ ! -d "vector-db" ] || [ -z "$(ls -A vector-db 2>/dev/null)" ]; then
    info "Indexing documents into vector store..."
    python mcp_server/index_docs.py
else
    info "Vector store already exists"
fi

# ── Check for GGUF model ───────────────────────────────
if [ ! -f "phi3-devops-q4.gguf" ]; then
    warn "No GGUF model found (phi3-devops-q4.gguf)"
    echo ""
    echo "You have two options:"
    echo "  A) Run fine-tuning yourself (takes 4-8 hours):"
    echo "     python train.py"
    echo "     python merge_adapter.py"
    echo "     # then convert with llama.cpp (see README)"
    echo ""
    echo "  B) Download pre-built model (if you have the HuggingFace link):"
    echo "     # See model weights section in README.md"
    echo ""
else
    info "GGUF model found. Loading into Ollama..."
    ollama create phi3-devops -f Modelfile
    info "Model loaded: phi3-devops"
fi

# ── Copy env file ──────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn "Created .env from .env.example — edit OLLAMA_MODELS_PATH if needed"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. If you haven't fine-tuned yet: python train.py (run overnight)"
echo "  2. Start Ollama:          ollama serve"
echo "  3. Start inference API:   uvicorn inference_server.main:app --port 8000"
echo "  4. Launch the app:        python app.py"
echo "  5. Open browser:          http://localhost:7860"
echo ""
