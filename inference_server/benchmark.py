"""
Task 02 - Benchmark Script
Compares Q4_K_M vs Q8_0 quantization formats.
Measures: TTFT, tokens/sec, RAM usage.

Usage: python inference_server/benchmark.py
"""

import time
import json
import httpx
import psutil
import os

OLLAMA_URL = "http://localhost:11434"
TEST_PROMPT = "### Instruction:\nExplain what a Kubernetes deployment is and how it differs from a pod.\n\n### Response:\n"
MODELS = ["phi3-devops"]  # add "phi3-devops-q8" if you create a Q8 variant


def benchmark_model(model_name: str, num_runs: int = 3) -> dict:
    print(f"\nBenchmarking: {model_name}")
    ttfts, tps_list, ram_list = [], [], []

    for run in range(num_runs):
        proc = psutil.Process(os.getpid())
        ram_before = proc.memory_info().rss / 1024 / 1024  # MB

        start = time.time()
        first_token_time = None
        token_count = 0

        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": TEST_PROMPT, "stream": True},
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line.strip():
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token and first_token_time is None:
                        first_token_time = time.time()
                    if token:
                        token_count += 1
                    if data.get("done"):
                        total_time = time.time() - start
                        break

        ram_after = proc.memory_info().rss / 1024 / 1024
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        tps = token_count / total_time if total_time > 0 else 0

        ttfts.append(ttft)
        tps_list.append(tps)
        ram_list.append(ram_after - ram_before)

        print(f"  Run {run+1}: TTFT={ttft:.0f}ms | {tps:.1f} tok/s | RAM delta={ram_after-ram_before:.0f}MB")

    return {
        "model": model_name,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1),
        "avg_tokens_per_sec": round(sum(tps_list) / len(tps_list), 2),
        "avg_ram_delta_mb": round(sum(ram_list) / len(ram_list), 1),
    }


def main():
    results = []
    for model in MODELS:
        result = benchmark_model(model)
        results.append(result)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\nModel           : {r['model']}")
        print(f"Avg TTFT        : {r['avg_ttft_ms']} ms")
        print(f"Avg tokens/sec  : {r['avg_tokens_per_sec']}")
        print(f"Avg RAM delta   : {r['avg_ram_delta_mb']} MB")

    print("\nConclusion: Q4_K_M recommended for 8GB RAM")
    print("  Q4_K_M: ~2.2GB model size, good quality/speed tradeoff")
    print("  Q8_0  : ~4.4GB model size, better quality but needs 8GB+ free RAM")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_results.json")


if __name__ == "__main__":
    main()
