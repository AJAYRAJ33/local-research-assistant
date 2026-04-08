# Technical Notes

## Task 01 — Dataset & Fine-tuning Decisions

### Dataset Choice
I chose DevOps Q&A as the domain because it has clear, verifiable answers (kubectl commands, Docker instructions, Kubernetes concepts) that make evaluation straightforward. The 210 samples cover Kubernetes, Docker, CI/CD, Terraform, Helm, observability, and SRE practices — broad enough to test generalisation but cohesive enough that a small model can specialise.

The dataset was written manually to ensure quality and avoid contamination. Each sample follows Alpaca schema (`### Instruction:` / `### Response:`). The 90/10 train/val split was created after a fixed-seed shuffle to prevent leakage.

### Model Choice: Phi-3-mini (3.8B)
Phi-3-mini was chosen over Mistral-7B or Llama-3-8B for one clear reason: it fits in 8GB RAM with 4-bit quantisation. Mistral-7B requires at least 5–6GB just for the quantised weights, leaving insufficient headroom for the training optimizer states. Phi-3-mini's quality-per-parameter ratio is excellent — Microsoft trained it on high-quality curated data, making it surprisingly capable at instruction following for its size.

### LoRA Hyperparameter Decisions
- **Rank r=16**: A moderate rank that adds ~4M trainable parameters on top of the frozen 3.8B base. Lower ranks (r=4, r=8) were insufficient for domain adaptation on 200 samples; higher ranks (r=32, r=64) risked overfitting the small dataset and increased RAM pressure.
- **Alpha=32**: Setting alpha = 2×r is a standard heuristic that scales the LoRA update to a similar magnitude as full fine-tuning updates without hyperparameter search.
- **Target modules**: `q_proj`, `v_proj`, `k_proj`, `o_proj` — all attention projection matrices. Targeting all four attention projections (rather than just q+v) gives better coverage of the attention mechanism, important for question-answering tasks where attention patterns drive factual recall.
- **Dropout=0.05**: Light regularisation to prevent overfitting on 189 training samples.
- **Batch size=1 + gradient accumulation=8**: Effective batch of 8 with only 1 sample in GPU/CPU memory at a time. This is the only viable configuration for 8GB RAM on CPU.
- **Epochs=3**: Enough to see training loss decrease without memorising the small dataset.

### Evaluation Results & Limitations
ROUGE-L measures token recall between predicted and reference answers. Fine-tuned model improvement over base is expected to be modest (+0.03 to +0.08 ROUGE-L) given:
1. 210 samples is a very small dataset — the model learns the format well but doesn't dramatically change its factual knowledge.
2. ROUGE-L penalises paraphrases unfairly — a correct answer worded differently scores poorly.
3. CPU inference is slow (~2–5 tokens/sec), so evaluation is time-constrained.

**Limitations observed**: The model occasionally generates answers that are partially correct but include hallucinated command flags. On multi-part questions it sometimes answers only the first part. These are inherent limitations of a 3.8B model on a small domain dataset, not fine-tuning failures.

---

## Task 02 — Quantisation Decision

| Format | Size | RAM needed | Quality | Tokens/sec (CPU) |
|--------|------|------------|---------|-----------------|
| Q4_K_M | ~2.2 GB | ~4 GB | Good | 2–5 |
| Q8_0   | ~4.1 GB | ~6 GB | Better | 1–3 |
| F16    | ~7.5 GB | ~9 GB | Best | <1 |

**Chosen: Q4_K_M** — the K-quants (K_M = medium quality K-quant) use a more sophisticated quantisation scheme than standard Q4_0, preserving more information in the important weights. At 8GB total RAM with OS overhead, Q4_K_M leaves ~3.5GB free for the inference server, MCP server, and agent processes. Q8_0 would work but leaves very little headroom and is significantly slower on CPU.

---

## Task 04 — Agent Failure Modes & Mitigations

### Failure Mode 1: LLM doesn't follow ReAct format
**Problem**: The model sometimes outputs free-form text instead of the structured `Thought / Action / Action Input` format, especially on the first step.
**Mitigation**: The `_parse_response` method uses regex fallbacks — if no `Action:` is found but the output looks like an answer, it's treated as a `Final Answer`. The system prompt includes explicit format instructions with examples.

### Failure Mode 2: Tool call with wrong argument types
**Problem**: The model sometimes passes `top_k` as a string `"3"` instead of integer `3`, or omits required fields.
**Mitigation**: `search_documents` casts `top_k` with `int()` and clamps to valid range. `run_python` checks for empty `code`. All tools return structured JSON error objects so the agent can observe the error and retry.

### Failure Mode 3: Infinite tool-calling loops
**Problem**: Without a step limit, the agent could repeatedly call the same tool returning the same observation and never converge.
**Mitigation**: `max_steps=8` hard limit. The scratchpad grows with each step, so the model sees its previous attempts and (usually) pivots to a final answer.

### Failure Mode 4: Hallucinated tool names
**Problem**: The model occasionally invents tool names not in the registry (e.g. `search_web`, `get_info`).
**Mitigation**: The `_call_tool` method checks the tool name against `self.tool_registry` and returns a structured error listing available tools. The agent observes this and typically corrects on the next step.

### Failure Mode 5: Context window overflow on long sessions
**Problem**: Multi-turn conversations grow the prompt beyond the model's 2048 context window, causing truncation and incoherent responses.
**Mitigation**: `_build_react_prompt` only includes the last 6 turns (3 exchanges) from conversation history. Older context is dropped. A vector memory upgrade would mitigate this fully.

---

## Task 06 — CI/CD Design Choices

**Fail-fast ordering**: Lint runs first (seconds), then tests (minutes), then eval (longest). This surfaces cheap failures before expensive ones — a style error shouldn't burn eval compute.

**Eval threshold in CI**: The production evaluation uses the fine-tuned model, but CI uses `tinyllama` (a tiny 1.1B model) to avoid downloading 4GB weights on every push. The threshold is set low (0.05) for CI — the purpose is to test that the evaluation *script* runs correctly, not to re-validate the model on every commit.

**Docker multi-stage builds**: Separate builder and runtime stages keep the production image free of build tools (pip, gcc, headers), reducing attack surface and image size by ~40%.

**Commit SHA tagging**: Images tagged with the full commit SHA are immutable — you always know exactly which code is running. The `latest` tag is updated only on main branch pushes for convenience.

**Rollback strategy**: A `.env.previous` file stores the last known-good image tag. If the post-deploy health check fails, the SSH deploy step restores `.env.previous` and re-runs `docker-compose up`. This is a simple, reliable rollback that doesn't require Kubernetes or a sophisticated deployment platform.

**Secrets management**: All credentials (Docker Hub token, SSH key, deploy host) are stored as GitHub Actions encrypted secrets. They are never echoed in logs and are not accessible to fork PRs. The `.env.example` documents which secrets are required without storing any values.

**Given more time, I would prioritise**:
1. Adding OpenTelemetry tracing to the agent loop for per-request span visibility
2. Replacing the simple `.env.previous` rollback with a proper blue/green swap
3. Building a vector memory layer for the agent to recall past conversations
4. Adding integration tests that spin up the full stack with docker-compose in CI
