"""
Task 05 - Orchestrated Application Entry Point
Starts all services and launches the Gradio chat UI.

Usage: python app.py
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from agent.agent import ReActAgent
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# ─── Logging setup ────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)


# Structured JSON logger
json_logger = logging.getLogger("structured")
json_handler = logging.FileHandler("logs/requests.jsonl")
json_handler.setFormatter(logging.Formatter("%(message)s"))
json_logger.addHandler(json_handler)
json_logger.setLevel(logging.INFO)

# Console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Global agent (maintains session memory) ─────────────────────────────────
agent = ReActAgent(max_steps=8)


def log_request(query: str, response: str, latency_ms: int, tools_used: list[str]):
    """Write a structured JSON log entry per request."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "response_preview": response[:200],
        "tools_invoked": tools_used,
        "latency_ms": latency_ms,
        "token_estimate": {
            "input": len(query.split()),
            "output": len(response.split()),
        },
    }
    json_logger.info(json.dumps(entry))


def extract_tools_used(trace) -> list[str]:
    """Extract list of tools called from agent trace."""
    return [
        step.action
        for step in trace
        if step.action is not None
    ]


# ─── Gradio chat handler ──────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """
    Main chat handler called by Gradio.
    Runs the ReAct agent and returns the final answer.
    """
    if not message.strip():
        return "Please enter a question."

    start = time.time()
    logger.info(f"Query: {message[:80]}...")

    try:
        answer = asyncio.run(agent.run(message))
    except Exception as e:
        logger.error(f"Agent error: {e}")
        answer = f"Sorry, I encountered an error: {e}"

    latency_ms = int((time.time() - start) * 1000)
    tools_used = extract_tools_used(agent.trace)

    log_request(message, answer, latency_ms, tools_used)

    logger.info(f"Answered in {latency_ms}ms | Tools: {tools_used}")

    # Add trace to response if tools were used
    if tools_used:
        trace_summary = f"\n\n---\n*Tools used: {', '.join(tools_used)} | {latency_ms}ms*"
        return answer + trace_summary

    return answer


def reset_session():
    """Clear agent memory for a fresh session."""
    agent.reset_session()
    return [], "Session cleared. Starting fresh conversation."


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Local Research Assistant",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Local Research Assistant
            AI-powered DevOps knowledge base running entirely on your laptop.
            Ask questions about Kubernetes, Docker, CI/CD, Terraform, and more.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.ChatInterface(
                    fn=chat,
                    examples=[
                        "What is a Kubernetes Deployment?",
                        "How do I scale a deployment to 5 replicas?",
                        "What is the difference between a ConfigMap and a Secret?",
                        "Search for information about Helm and summarise the main commands.",
                        "Find docs about Prometheus, then calculate how many minutes of downtime 99.9% uptime allows per month.",
                    ],
                    title="",
                    description="Ask any DevOps question. The assistant searches documentation and can run Python code.",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Session Controls")
                reset_btn = gr.Button("Reset Memory", variant="secondary")
                reset_status = gr.Textbox(label="Status", interactive=False)
                reset_btn.click(
                    fn=reset_session,
                    inputs=[],
                    outputs=[gr.State([]), reset_status],
                )

                gr.Markdown("### About")
                gr.Markdown(
                    """
                    **Model**: llama3.2:1b (fine-tuned)
                    **Tools**: search_documents, run_python, get_date_context
                    **Logs**: `logs/requests.jsonl`
                    """
                )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Local Research Assistant...")
    logger.info("Make sure Ollama is running: ollama serve")
    logger.info("Make sure model is loaded: ollama run llama3.2:1b")
    logger.info("Make sure docs are indexed: python mcp_server/index_docs.py")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
