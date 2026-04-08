"""
Task 04 - ReAct Agent
Implements a ReAct (Reason + Act) loop connecting:
  - Local inference server (FastAPI → Ollama → Phi-3-mini)
  - MCP tool server (search_documents, run_python, get_date_context)

The agent:
  1. Thinks (generates a Thought)
  2. Acts (calls a tool if needed)
  3. Observes (gets tool result)
  4. Repeats until FINAL ANSWER or max_steps reached
  5. Maintains full conversation history across turns
"""

import asyncio
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INFERENCE_URL = "http://localhost:8000"
MCP_SERVER_CMD = ["python", "mcp_server/server.py"]


@dataclass
class Turn:
    role: str   # "user" | "assistant" | "tool"
    content: str


@dataclass
class AgentStep:
    step: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None


class ReActAgent:
    def __init__(
        self,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ):
        self.max_steps = max_steps
        self.conversation_history: list[Turn] = []
        self.trace: list[AgentStep] = []
        self.tool_registry: dict[str, Any] = {}

        self.system_prompt = system_prompt or (
            "You are a helpful DevOps research assistant. "
            "You have access to tools to search documentation and run Python code. "
            "Think step by step. Use tools when you need specific information or computations. "
            "Always provide accurate, practical answers."
        )

    # ─── LLM call ─────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{INFERENCE_URL}/generate",
                    json={"prompt": prompt, "stream": False, "max_tokens": 512},
                )
                if r.status_code != 200:
                    return f"[LLM ERROR {r.status_code}]"
                data = r.json()
                response_text = data.get("response", "").strip()
                if not response_text:
                    logger.warning(f"Empty response from inference server: {data}")
                    return "[LLM ERROR: Empty response from model]"
                return response_text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM ERROR: {e}]"

    # ─── Tool call ────────────────────────────────────────────────────────────

    async def _call_tool(
        self, session: ClientSession, name: str, args: dict
    ) -> str:
        try:
            result = await session.call_tool(name, args)
            if result.content:
                return result.content[0].text
            return '{"result": "no output"}'
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return json.dumps({"error": str(e)})

    # ─── Parse LLM output ─────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> tuple[str, str | None, dict | None]:
        """
        Parse LLM output for:
          Thought: ...
          Action: tool_name
          Action Input: {...}
          Final Answer: ...
        """
        thought = ""
        action = None
        action_input = None

        # Extract Thought
        t_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        if t_match:
            thought = t_match.group(1).strip()

        # Check for Final Answer
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if fa_match:
            return fa_match.group(1).strip(), None, None

        # Check for Action
        a_match = re.search(r"Action:\s*(\w+)", text)
        ai_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)

        if a_match:
            action = a_match.group(1).strip()
            if ai_match:
                try:
                    action_input = json.loads(ai_match.group(1))
                except json.JSONDecodeError:
                    # Try to extract key=value pairs
                    action_input = {}
                    for kv in re.finditer(r'"(\w+)":\s*"?([^,"}\n]+)"?', ai_match.group(1)):
                        action_input[kv.group(1)] = kv.group(2).strip('"')

        if not thought:
            thought = text[:200]

        return thought, action, action_input

    # ─── Build prompt ─────────────────────────────────────────────────────────

    def _build_react_prompt(self, query: str, tool_names: list[str]) -> str:
        history_str = ""
        for turn in self.conversation_history[-6:]:  # last 3 exchanges
            history_str += f"{turn.role.upper()}: {turn.content}\n"

        tools_desc = ", ".join(tool_names) if tool_names else "none"

        return f"""### System:
{self.system_prompt}

Available tools: {tools_desc}

Instructions:
- Think step by step before acting.
- Use this exact format:
  Thought: <your reasoning>
  Action: <tool_name>
  Action Input: {{"key": "value"}}
- Or if you have the answer:
  Thought: <reasoning>
  Final Answer: <your complete answer>

{history_str}

### Current Question:
{query}

### Scratchpad:
"""

    # ─── Main run loop ────────────────────────────────────────────────────────

    async def run(self, user_query: str) -> str:
        self.conversation_history.append(Turn("user", user_query))
        self.trace = []
        final_answer = "I was unable to find a complete answer."

        async with stdio_client(
            StdioServerParameters(command=MCP_SERVER_CMD[0], args=MCP_SERVER_CMD[1:])
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Discover tools
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                self.tool_registry = {t.name: t for t in tools_response.tools}
                logger.info(f"Discovered tools: {tool_names}")

                scratchpad = ""
                for step_num in range(1, self.max_steps + 1):
                    prompt = self._build_react_prompt(user_query, tool_names)
                    prompt += scratchpad

                    logger.info(f"Step {step_num}/{self.max_steps}")
                    raw = await self._call_llm(prompt)

                    thought, action, action_input = self._parse_response(raw)

                    agent_step = AgentStep(step=step_num, thought=thought)

                    # Final answer reached
                    if action is None and not any(
                        kw in raw.lower() for kw in ["action:", "action input:"]
                    ):
                        final_answer = thought
                        agent_step.thought = thought
                        self.trace.append(agent_step)
                        logger.info(f"Final answer reached at step {step_num}")
                        break

                    # Execute tool
                    if action and action in self.tool_registry:
                        agent_step.action = action
                        agent_step.action_input = action_input or {}

                        logger.info(f"Calling tool: {action}({action_input})")
                        observation = await self._call_tool(
                            session, action, action_input or {}
                        )
                        agent_step.observation = observation

                        scratchpad += (
                            f"\nThought: {thought}"
                            f"\nAction: {action}"
                            f"\nAction Input: {json.dumps(action_input or {})}"
                            f"\nObservation: {observation[:500]}"
                        )
                    elif action:
                        observation = f'{{"error": "Unknown tool: {action}. Available: {tool_names}"}}'
                        agent_step.observation = observation
                        scratchpad += f"\nObservation: {observation}"
                    else:
                        # No clear action — take thought as partial answer
                        final_answer = thought
                        self.trace.append(agent_step)
                        break

                    self.trace.append(agent_step)

        # Update conversation memory
        self.conversation_history.append(Turn("assistant", final_answer))
        return final_answer

    def get_trace_text(self) -> str:
        """Return human-readable trace of the last run."""
        lines = []
        for step in self.trace:
            lines.append(f"\n--- Step {step.step} ---")
            lines.append(f"Thought    : {step.thought}")
            if step.action:
                lines.append(f"Action     : {step.action}")
                lines.append(f"Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                obs_preview = step.observation[:300] + "..." if len(step.observation) > 300 else step.observation
                lines.append(f"Observation: {obs_preview}")
        return "\n".join(lines)

    def reset_session(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []
        self.trace = []


# ─── Demo queries ──────────────────────────────────────────────────────────────

async def demo():
    agent = ReActAgent(max_steps=8)

    queries = [
        "What is a Kubernetes Deployment and how do I scale it?",
        "Search for information about Helm charts and tell me the main commands.",
        "Find docs about Prometheus metrics and then calculate: if my service has 99.5% uptime, how many minutes of downtime is that per month? (30 days)",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("="*60)

        answer = await agent.run(query)
        print(agent.get_trace_text())
        print(f"\nFinal Answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(demo())
