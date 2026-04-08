"""
Task 03 - MCP Server
Exposes 3 tools via the official MCP Python SDK:
  1. search_documents(query, top_k)  — semantic search over ChromaDB
  2. run_python(code)                — sandboxed Python execution
  3. get_date_context()              — current date/time context

Start: python mcp_server/server.py
"""

import asyncio
import subprocess
import datetime
import logging
from typing import Any

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── ChromaDB setup ───────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./vector-db")
collection = chroma_client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)

# ─── MCP Server ───────────────────────────────────────────────────────────────
app = Server("local-research-assistant")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description=(
                "Performs semantic search over a local vector store containing "
                "DevOps and Kubernetes documentation. Returns the top-k most "
                "relevant document chunks for the given query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="run_python",
            description=(
                "Executes a Python code snippet in a sandboxed subprocess with a "
                "10-second timeout. Returns stdout, stderr, and exit code. "
                "Use for calculations, data parsing, or verifying logic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute",
                    },
                },
                "required": ["code"],
            },
        ),
        types.Tool(
            name="get_date_context",
            description=(
                "Returns the current date, time, day of week, and ISO week number. "
                "Use when the user's query involves scheduling, timelines, or "
                "time-relative reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:

    if name == "search_documents":
        return await _search_documents(arguments)
    elif name == "run_python":
        return await _run_python(arguments)
    elif name == "get_date_context":
        return _get_date_context()
    else:
        return [types.TextContent(
            type="text",
            text=f'{{"error": "Unknown tool: {name}"}}'
        )]


# ─── Tool implementations ─────────────────────────────────────────────────────

async def _search_documents(args: dict) -> list[types.TextContent]:
    query = args.get("query", "").strip()
    top_k = max(1, min(10, int(args.get("top_k", 3))))

    if not query:
        return [types.TextContent(
            type="text",
            text='{"error": "query must be a non-empty string"}'
        )]

    try:
        count = collection.count()
        if count == 0:
            return [types.TextContent(
                type="text",
                text='{"error": "Vector store is empty. Run index_docs.py first."}'
            )]

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = {
            "query": query,
            "top_k": top_k,
            "results": [
                {
                    "rank": i + 1,
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "similarity": round(1 - dist, 4),
                }
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances))
            ],
        }
        import json
        return [types.TextContent(type="text", text=json.dumps(output, indent=2))]

    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return [types.TextContent(
            type="text",
            text=f'{{"error": "Search failed: {str(e)}"}}'
        )]


async def _run_python(args: dict) -> list[types.TextContent]:
    code = args.get("code", "").strip()

    if not code:
        return [types.TextContent(
            type="text",
            text='{"error": "code must be a non-empty string"}'
        )]

    # Basic safety: block obvious dangerous patterns
    BLOCKED = ["import os", "import sys", "subprocess", "open(", "__import__",
               "exec(", "eval(", "compile(", "globals(", "locals("]
    for pattern in BLOCKED:
        if pattern in code:
            return [types.TextContent(
                type="text",
                text=f'{{"error": "Blocked pattern detected: {pattern}. '
                     f'Only pure computation is allowed."}}'
            )]

    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin"},  # minimal env
        )
        import json
        result = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except subprocess.TimeoutExpired:
        return [types.TextContent(
            type="text",
            text='{"error": "Execution timed out after 10 seconds", "exit_code": -1}'
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f'{{"error": "Execution failed: {str(e)}", "exit_code": -1}}'
        )]


def _get_date_context() -> list[types.TextContent]:
    import json
    now = datetime.datetime.now()
    result = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "iso_week": now.isocalendar().week,
        "timezone": "local",
        "unix_timestamp": int(now.timestamp()),
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ─── Entry point ──────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
