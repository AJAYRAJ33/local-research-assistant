"""
Task 01 - Evaluation Script
Compares base Phi-3-mini vs fine-tuned model on 20 held-out samples.
Metric: ROUGE-L (measures recall of n-grams between prediction and reference).
Writes results to eval_results.json (used by CI/CD pipeline).

Usage:
  python evaluate.py
  python evaluate.py --threshold 0.25   # CI mode — fails if below threshold
"""

import json
import argparse
import time
from pathlib import Path

import httpx
from rouge_score import rouge_scorer

# 20 held-out evaluation samples (NOT in training data)
EVAL_SAMPLES = [
    {"instruction": "What is kubectl apply used for?",
     "reference": "kubectl apply creates or updates Kubernetes resources declaratively from YAML/JSON files. It merges changes rather than replacing the entire resource, making it safe to run repeatedly."},
    {"instruction": "What is a Kubernetes deployment?",
     "reference": "A Deployment manages a set of identical pods, ensuring the specified number of replicas are running. It supports rolling updates, rollbacks, and scaling."},
    {"instruction": "How do you create a Docker image?",
     "reference": "Write a Dockerfile defining the base image, dependencies, and startup command. Then run: docker build -t image-name:tag . The dot refers to the current directory as the build context."},
    {"instruction": "What is continuous integration?",
     "reference": "CI automatically builds and tests code on every commit to catch bugs early. Developers integrate changes frequently (at least daily) to a shared branch, reducing integration conflicts."},
    {"instruction": "What is Kubernetes service?",
     "reference": "A Service provides a stable network endpoint to a set of pods. It load balances traffic and gives pods a fixed DNS name and IP, decoupling clients from individual pod IPs that change."},
    {"instruction": "What is environment variable injection in Docker?",
     "reference": "Pass env vars at runtime with docker run -e KEY=value, or define them in docker-compose.yml under environment. The app reads them via os.environ. Never bake secrets into images."},
    {"instruction": "What is a Kubernetes ReplicaSet?",
     "reference": "A ReplicaSet ensures a specified number of pod replicas are running at all times. If a pod fails, the ReplicaSet creates a replacement. Deployments manage ReplicaSets automatically."},
    {"instruction": "What does the FROM instruction do in a Dockerfile?",
     "reference": "FROM sets the base image for subsequent instructions. Every Dockerfile must start with FROM. Example: FROM python:3.11-slim starts from an official slim Python image."},
    {"instruction": "What is a linter in CI/CD?",
     "reference": "A linter analyzes code for style errors, bugs, and anti-patterns without running it. In CI, linters run on every commit. Python linters: ruff, flake8, pylint."},
    {"instruction": "What is Kubernetes pod scheduling?",
     "reference": "The Kubernetes scheduler assigns pods to nodes based on resource requests, taints/tolerations, affinity rules, and available capacity. Unschedulable pods wait in Pending state."},
    {"instruction": "What is a Docker RUN instruction?",
     "reference": "RUN executes a command during the image build process and commits the result as a new layer. Used to install packages: RUN pip install -r requirements.txt. Chain commands with && to reduce layers."},
    {"instruction": "What is automated testing in DevOps?",
     "reference": "Automated tests run on every commit to verify code correctness. Types: unit (fast, isolated), integration (multiple components), end-to-end (full system). Run in CI to catch regressions."},
    {"instruction": "What is a Kubernetes node?",
     "reference": "A node is a worker machine (VM or physical) in a Kubernetes cluster. Each node runs the kubelet (agent), kube-proxy (networking), and a container runtime like containerd."},
    {"instruction": "What is Docker Compose used for?",
     "reference": "Docker Compose defines and runs multi-container applications with a single YAML file. docker-compose up starts all services, networks, and volumes defined. Ideal for local development."},
    {"instruction": "What is a GitHub Actions job?",
     "reference": "A job is a set of steps that run sequentially on a runner. Jobs in the same workflow run in parallel by default unless you add needs: to create dependencies between them."},
    {"instruction": "What is Kubernetes pod lifecycle?",
     "reference": "Pods go through: Pending (waiting to be scheduled), Running (at least one container running), Succeeded (all containers exited 0), Failed (at least one container exited non-zero), Unknown."},
    {"instruction": "What is the purpose of a .gitignore file?",
     "reference": "It lists files and directories Git should not track: build artifacts, secrets, virtual environments, IDE configs. Prevents accidental commits of sensitive data and keeps repos clean."},
    {"instruction": "What is a container entrypoint?",
     "reference": "The entrypoint is the command that runs when a container starts. Set via ENTRYPOINT in Dockerfile or entrypoint in docker-compose. CMD provides default arguments to the entrypoint."},
    {"instruction": "What is a Kubernetes cluster?",
     "reference": "A cluster is a set of machines (nodes) that run containerized workloads managed by Kubernetes. It consists of a control plane (API server, etcd, scheduler) and worker nodes running pods."},
    {"instruction": "What is infrastructure monitoring?",
     "reference": "Infrastructure monitoring tracks the health and performance of servers, containers, and networks: CPU, memory, disk I/O, network throughput, and error rates. Tools: Prometheus, Datadog, CloudWatch."},
]


def generate_from_ollama(prompt: str, model: str = "phi3-devops") -> str:
    """Call local Ollama to generate a response."""
    try:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def generate_from_base(prompt: str) -> str:
    """Call Ollama with the base (untuned) model."""
    return generate_from_ollama(prompt, model="phi3-mini")


def evaluate(threshold: float = 0.20):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    finetuned_scores = []
    base_scores = []

    print(f"\nEvaluating on {len(EVAL_SAMPLES)} samples...\n")
    print(f"{'#':<3} {'ROUGE-L (ft)':>13} {'ROUGE-L (base)':>15}")
    print("-" * 35)

    for i, sample in enumerate(EVAL_SAMPLES):
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
        ref = sample["reference"]

        ft_pred = generate_from_ollama(prompt, model="phi3-devops")
        base_pred = generate_from_base(prompt)

        ft_score = scorer.score(ref, ft_pred)["rougeL"].fmeasure
        base_score = scorer.score(ref, base_pred)["rougeL"].fmeasure

        finetuned_scores.append(ft_score)
        base_scores.append(base_score)

        print(f"{i+1:<3} {ft_score:>13.4f} {base_score:>15.4f}")

        results.append({
            "instruction": sample["instruction"],
            "reference": ref,
            "finetuned_prediction": ft_pred,
            "base_prediction": base_pred,
            "finetuned_rougeL": ft_score,
            "base_rougeL": base_score,
        })

        time.sleep(0.5)  # be gentle with local server

    avg_ft = sum(finetuned_scores) / len(finetuned_scores)
    avg_base = sum(base_scores) / len(base_scores)
    improvement = avg_ft - avg_base

    summary = {
        "avg_finetuned_rougeL": avg_ft,
        "avg_base_rougeL": avg_base,
        "improvement": improvement,
        "threshold": threshold,
        "passed": avg_ft >= threshold,
        "samples": results,
    }

    Path("eval_results.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 35)
    print(f"Fine-tuned avg ROUGE-L : {avg_ft:.4f}")
    print(f"Base model  avg ROUGE-L : {avg_base:.4f}")
    print(f"Improvement             : {improvement:+.4f}")
    print(f"Threshold               : {threshold}")
    print(f"Result                  : {'PASSED' if summary['passed'] else 'FAILED'}")
    print("\nFull results saved to eval_results.json")

    if not summary["passed"]:
        raise SystemExit(
            f"Evaluation FAILED: {avg_ft:.4f} < threshold {threshold}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Minimum ROUGE-L score to pass CI")
    args = parser.parse_args()
    evaluate(threshold=args.threshold)
