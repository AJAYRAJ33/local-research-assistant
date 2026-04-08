"""
Task 02 - Inference Server
Wraps Ollama with a FastAPI REST API exposing:
  POST /generate   — streaming text generation
  GET  /health     — readiness check

Start: uvicorn inference_server.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import time
import logging
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local Research Assistant — Inference Server",
    description="Wraps Ollama to serve the fine-tuned Phi-3-mini DevOps model.",
    version="1.0.0",
)

OLLAMA_URL = "http://localhost:11434"  # docker-compose service name
MODEL_NAME = "llama3.2:1b"


# ─── Request / Response models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt to send to the model")
    stream: bool = Field(True, description="Whether to stream the response")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=2048)


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_reachable: bool
    latency_ms: float | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness check — verifies Ollama is reachable and model is loaded."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            loaded = any(MODEL_NAME in m for m in models)
        latency = round((time.time() - start) * 1000, 1)
        return HealthResponse(
            status="ok" if loaded else "model_not_loaded",
            model=MODEL_NAME,
            ollama_reachable=True,
            latency_ms=latency,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            model=MODEL_NAME,
            ollama_reachable=False,
        )


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate text from the fine-tuned model.
    Returns a streaming or non-streaming response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    if req.stream:
        return StreamingResponse(
            _stream_response(payload),
            media_type="text/plain",
        )
    else:
        return await _full_response(payload)


async def _stream_response(payload: dict) -> AsyncGenerator[str, None]:
    """Streams tokens from Ollama as plain text."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/generate", json=payload
            ) as response:
                if response.status_code != 200:
                    yield f"[ERROR {response.status_code}]"
                    return
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
    except httpx.TimeoutException:
        yield "\n[ERROR: Generation timed out]"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"\n[ERROR: {e}]"


async def _full_response(payload: dict) -> dict:
    """Returns the complete response as JSON (non-streaming)."""
    try:
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=120) as client:
            logger.info(f"Sending payload to Ollama: {payload}")
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            logger.info(f"Ollama status: {r.status_code}")
            data = r.json()
            logger.info(f"Ollama response: {data}")
            
            response_text = data.get("response", "")
            if not response_text:
                logger.warning(f"Empty response from Ollama. Full data: {data}")
            
            return {
                "response": response_text,
                "model": data.get("model"),
                "total_duration_ms": round(data.get("total_duration", 0) / 1e6, 1),
                "tokens_per_second": round(
                    data.get("eval_count", 0)
                    / max(data.get("eval_duration", 1), 1)
                    * 1e9,
                    2,
                ),
            }
    except Exception as e:
        logger.error(f"Full response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Inference server running. POST /generate or GET /health"}
