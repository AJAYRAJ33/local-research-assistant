"""
Task 03 - Index Documents into ChromaDB
Reads .txt files from ./corpus/ and loads them into the vector store.
Run once before starting the MCP server.

Usage: python mcp_server/index_docs.py
"""

import os
import glob
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CORPUS_DIR = "./corpus"
VECTOR_DB_DIR = "./vector-db"
CHUNK_SIZE = 400   # characters per chunk
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def create_corpus():
    """Create a sample corpus if ./corpus/ is empty."""
    Path(CORPUS_DIR).mkdir(exist_ok=True)
    sample_docs = {
        "kubernetes_basics.txt": """
Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications. It was originally designed by Google and is now maintained by the Cloud Native Computing Foundation (CNCF).

A Kubernetes cluster consists of a control plane and worker nodes. The control plane manages the cluster state using components like the API server, etcd, scheduler, and controller manager. Worker nodes run the actual application workloads as pods.

Pods are the smallest deployable units in Kubernetes. A pod can contain one or more containers that share the same network namespace and storage volumes. Pods are ephemeral — they can be created, destroyed, and replaced at any time.

Deployments manage replica sets of pods and handle rolling updates. When you update a deployment, Kubernetes gradually replaces old pods with new ones, ensuring zero downtime. You can roll back to previous versions using kubectl rollout undo.

Services provide stable network endpoints to pods. Since pod IPs change when pods restart, Services give a fixed IP and DNS name. ClusterIP is internal-only; NodePort exposes a port on every node; LoadBalancer provisions a cloud load balancer.

ConfigMaps store configuration data as key-value pairs that pods can consume as environment variables or mounted files. Secrets store sensitive data like passwords and API tokens, also in key-value form but with additional access controls.

Namespaces provide virtual clusters within a physical cluster, allowing teams to share resources while maintaining isolation. Resource quotas and LimitRanges can be applied per namespace to control resource consumption.
        """.strip(),

        "docker_fundamentals.txt": """
Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies into a single, portable unit that runs consistently across environments.

A Docker image is a read-only template used to create containers. Images are built from a Dockerfile, which contains a series of instructions. Each instruction creates a new layer in the image. Layers are cached, making subsequent builds faster.

The Dockerfile FROM instruction specifies the base image. RUN executes commands during the build. COPY transfers files from the host. ENV sets environment variables. EXPOSE documents listening ports. CMD or ENTRYPOINT defines the startup command.

Multi-stage builds use multiple FROM statements in one Dockerfile. Earlier stages compile code or install build tools. The final stage copies only the required artifacts, producing a smaller, more secure image without build tools.

Docker Compose orchestrates multi-container applications using a docker-compose.yml file. It defines services, networks, and volumes. A single docker-compose up -d command starts the entire stack.

Container networking allows containers to communicate. The bridge network (default) isolates containers on the same host. Overlay networks span multiple Docker hosts for Swarm mode. Host networking shares the host's network stack directly.

Docker volumes persist data beyond a container's lifecycle. Named volumes are managed by Docker and stored in /var/lib/docker/volumes. Bind mounts link host directories directly into containers, useful for development hot-reload.
        """.strip(),

        "cicd_pipelines.txt": """
Continuous Integration and Continuous Delivery (CI/CD) automates the software delivery process from code commit to production deployment. CI runs automated builds and tests on every commit. CD automates deployment to staging and production environments.

GitHub Actions is a popular CI/CD platform integrated into GitHub. Workflows are defined in YAML files stored in .github/workflows/. Each workflow contains jobs; each job contains steps that run sequentially on a runner machine.

A typical CI pipeline includes: code checkout, dependency installation, linting (code style), type checking, unit tests, integration tests, security scanning, and coverage reporting. Failing any stage stops the pipeline immediately (fail fast).

Docker multi-stage builds in CI produce optimized images tagged with the commit SHA for immutability. Images are pushed to a container registry (Docker Hub, GHCR, ECR) using credentials stored as GitHub Actions secrets.

The CD stage pulls the new image, deploys it to the target environment, waits for health checks to pass, and rolls back to the previous image if health checks fail within a timeout window.

GitHub Actions reusable workflows allow teams to share common CI/CD logic across repositories. Matrix strategies run the same job with different parameters (e.g., multiple Python versions) in parallel, reducing feedback time.

Build caching stores dependency installations (pip packages, npm modules) between runs using actions/cache. This dramatically reduces build times from minutes to seconds on cached dependencies.
        """.strip(),

        "gitops_devops.txt": """
GitOps is a practice where the entire system state is described declaratively in Git, and automated tooling continuously reconciles the running state to match. Git is the single source of truth for both infrastructure and application configuration.

ArgoCD is a declarative GitOps continuous delivery tool for Kubernetes. It watches a Git repository for changes and automatically applies them to the cluster. It provides a web UI to visualize application health and sync status.

Infrastructure as Code (IaC) manages cloud infrastructure through machine-readable files rather than manual processes. Terraform uses HCL to define resources across cloud providers. Running terraform plan previews changes; terraform apply executes them.

Monitoring and observability are essential DevOps practices. Prometheus scrapes metrics from application endpoints and stores them as time-series data. Grafana visualizes these metrics in dashboards. Alertmanager routes alerts to teams.

The three pillars of observability are: metrics (quantitative measurements over time), logs (timestamped records of events), and traces (records of requests flowing through distributed systems). All three are needed for effective debugging.

Site Reliability Engineering (SRE) applies software engineering to infrastructure and operations. SREs define Service Level Indicators (SLIs), set Service Level Objectives (SLOs), and manage error budgets to balance reliability with development velocity.

Security in DevOps (DevSecOps) integrates security practices into every phase of the development lifecycle: scanning dependencies for vulnerabilities, signing container images, enforcing admission policies, and managing secrets with tools like Vault.
        """.strip(),

        "networking_kubernetes.txt": """
Kubernetes networking follows four fundamental rules: all pods can communicate with all other pods without NAT, all nodes can communicate with all pods without NAT, the IP a pod sees itself as is the same IP others see it as, and services can reach pods via stable virtual IPs.

Container Network Interface (CNI) plugins implement the Kubernetes network model. Popular choices include Calico (policy-rich), Flannel (simple overlay), Cilium (eBPF-based, high performance), and Weave Net. Each implements pod-to-pod networking differently.

Ingress controllers manage external HTTP/HTTPS traffic routing into the cluster. NGINX Ingress Controller is the most widely used. It reads Ingress resources and configures NGINX accordingly, supporting TLS termination, path-based routing, and virtual hosting.

Network Policies are Kubernetes-native firewall rules for pods. By default, all pod-to-pod communication is allowed. A NetworkPolicy selects pods with label selectors and defines which ingress and egress traffic is permitted. Policies are additive.

Service meshes like Istio and Linkerd add a layer of infrastructure for service-to-service communication. They inject sidecar proxies (Envoy) alongside each pod, handling mTLS encryption, traffic management, circuit breaking, and distributed tracing transparently.

DNS in Kubernetes is provided by CoreDNS. Each Service gets an A record: service-name.namespace.svc.cluster.local. Pods can also get DNS entries. Applications use short names (just the service name within the same namespace) or FQDNs across namespaces.

Load balancing in Kubernetes Services uses iptables or IPVS rules maintained by kube-proxy on each node. IPVS mode is more performant for large clusters. The default algorithm is round-robin; IPVS supports more algorithms including least connections.
        """.strip(),

        "security_kubernetes.txt": """
Kubernetes security operates at multiple layers: cluster infrastructure, cluster network, workloads (pods/containers), and supply chain (images). Each layer requires different controls and tools.

Role-Based Access Control (RBAC) is the primary authorization mechanism in Kubernetes. Roles define permissions (verbs on resources). RoleBindings associate roles with users, groups, or service accounts. Always follow the principle of least privilege.

Pod Security Standards (PSS) define three profiles: Privileged (no restrictions), Baseline (minimal restrictions), and Restricted (heavily hardened). They're enforced via Pod Security Admission or policy engines like OPA/Gatekeeper or Kyverno.

Image security involves scanning images for CVEs before deployment. Tools like Trivy, Snyk, and Clair integrate into CI pipelines. Image signing with Cosign (sigstore) ensures images come from trusted sources and haven't been tampered with.

Secrets management: Kubernetes Secrets are base64-encoded (not encrypted) by default. For production, use envelope encryption at rest, or an external secrets manager (HashiCorp Vault, AWS Secrets Manager) synced into the cluster via External Secrets Operator.

Network security uses NetworkPolicies to restrict pod communication, mTLS via service meshes for encrypted service-to-service traffic, and Ingress TLS for external traffic. Never expose the Kubernetes API server publicly without strict authentication.

Audit logging records all API server requests: who requested what, when, and the outcome. Combined with tools like Falco for runtime threat detection, audit logs provide the forensic trail needed to investigate security incidents.
        """.strip(),

        "observability.txt": """
Observability is the ability to understand a system's internal state from its external outputs — metrics, logs, and traces. Unlike monitoring (which alerts on known failure modes), observability helps you debug unknown failures.

Prometheus is the de-facto metrics system for Kubernetes. It scrapes /metrics endpoints exposed by applications and Kubernetes components. PromQL is its powerful query language. It stores data as time-series with labels for multi-dimensional queries.

Grafana connects to Prometheus and other data sources to build dashboards. Pre-built Kubernetes dashboards from the Grafana community cover cluster health, namespace resource usage, pod performance, and more. Alerting rules trigger notifications via Slack, PagerDuty, etc.

Structured logging outputs logs as JSON with consistent fields: timestamp, level, service, trace_id, message, and context-specific fields. Structured logs are machine-parseable and easily filtered in tools like Kibana, Loki, or CloudWatch Insights.

Distributed tracing follows a request as it flows through multiple services, assigning a trace ID that spans all hops. Each service records spans with timing data. OpenTelemetry is the standard instrumentation library. Jaeger and Tempo store and visualize traces.

The USE method (Utilization, Saturation, Errors) guides infrastructure analysis: for every resource, check utilization (how busy), saturation (how much queued work), and errors. The RED method (Rate, Errors, Duration) applies to services: request rate, error rate, latency.

Service Level Indicators (SLIs) are quantitative measures of service behavior: availability (% successful requests), latency (p50/p95/p99), throughput (req/sec), and error rate (% failed). SLOs set targets for SLIs; exceeding them burns the error budget.
        """.strip(),

        "helm_packaging.txt": """
Helm is the package manager for Kubernetes. Charts are collections of YAML templates and a values.yaml file. helm install deploys a chart; helm upgrade updates it; helm rollback reverts to a previous release.

A Helm chart directory structure: Chart.yaml (metadata), values.yaml (defaults), templates/ (Kubernetes manifests with Go templating). The {{ .Values.image.tag }} syntax references values, making charts reusable across environments.

Helm repositories store and distribute charts. Public repos: Artifact Hub (aggregates community charts), Bitnami (production-grade charts). Private repos: Harbor, Nexus, or an S3 bucket. Use helm repo add to register repos and helm search to find charts.

Helm hooks run Jobs at specific lifecycle points: pre-install, post-install, pre-upgrade, post-upgrade, pre-delete. Use them for database migrations before deployment, or for sending notifications after deployment completes.

Helmfile manages multiple Helm releases across environments. A helmfile.yaml declares all releases, their charts, values files, and dependencies. helmfile sync applies all releases; helmfile diff shows pending changes before applying.

Chart testing with helm lint validates chart syntax and structure. Helm unit testing with the helm-unittest plugin lets you write assertions on rendered templates. CI pipelines should lint and test charts before publishing to a repo.

Kustomize is an alternative to Helm that uses pure YAML overlays without templating. You define a base configuration and environment-specific overlays (patches). kubectl apply -k applies a Kustomization. Some teams use both: Helm for third-party apps, Kustomize for their own.
        """.strip(),

        "terraform_iac.txt": """
Terraform by HashiCorp is the leading Infrastructure as Code tool. It supports 300+ providers (AWS, GCP, Azure, Kubernetes, GitHub) through a provider plugin architecture. Configuration is written in HCL (HashiCorp Configuration Language).

The Terraform workflow: terraform init (download providers), terraform plan (preview changes), terraform apply (apply changes), terraform destroy (tear down). Always review the plan before applying — it shows exactly what will change.

Terraform state tracks real-world resources and their configuration. State is stored locally in terraform.tfstate by default. For teams, use remote state backends (S3, GCS, Terraform Cloud) with state locking to prevent concurrent modifications.

Modules in Terraform are reusable, composable units of configuration. A module encapsulates resources for a specific purpose (e.g., a VPC module, an EKS cluster module). Use published modules from the Terraform Registry to avoid reinventing the wheel.

Terraform workspaces allow multiple state instances from the same configuration — useful for dev/staging/prod environments. Each workspace has its own state file. terraform workspace new staging creates a new workspace.

Remote backends like Terraform Cloud provide collaboration features: remote execution, state management, policy as code (Sentinel), and team access controls. Plans run in Terraform Cloud's infrastructure rather than your local machine.

Best practices: pin provider versions, use remote state, enable state locking, review plans before applying, use modules for repeated patterns, tag all resources consistently, and never store secrets in Terraform files — use environment variables or Vault provider instead.
        """.strip(),

        "sre_practices.txt": """
Site Reliability Engineering (SRE) was developed at Google to apply software engineering to operations. SREs write code to automate toil (repetitive manual work), improve reliability, and build self-healing systems.

Error budgets quantify acceptable unreliability. If an SLO is 99.9% availability, the error budget is 0.1% downtime — about 43 minutes per month. When the budget is exhausted, feature development slows down until reliability improves. This aligns incentives between product and reliability teams.

Toil is manual, repetitive, tactical work that scales linearly with service growth and provides no enduring value. Examples: manually restarting services, responding to the same alert repeatedly, manually provisioning servers. SREs aim to keep toil below 50% of their time.

Incident management follows a structured process: detection (alert fires), triage (severity assessment), mitigation (restore service), investigation (find root cause), postmortem (document and prevent recurrence). The postmortem is blameless — focus on systems, not people.

Postmortems document what happened, the timeline, contributing factors, impact, and action items. Blameless culture is essential: people must feel safe reporting mistakes for the organization to learn from them. Action items have owners and deadlines.

Chaos engineering deliberately injects failures into systems to verify resilience. Netflix's Chaos Monkey randomly terminates EC2 instances. GameDays simulate large-scale failures in controlled conditions. The goal is to find weaknesses before they cause real outages.

On-call rotation ensures someone is always responsible for responding to incidents. Effective on-call requires: runbooks for common alerts, dashboards for quick diagnosis, escalation paths for critical issues, and compensation/time-off to prevent burnout.
        """.strip(),
    }

    for filename, content in sample_docs.items():
        filepath = Path(CORPUS_DIR) / filename
        if not filepath.exists():
            filepath.write_text(content)
            print(f"Created sample: {filename}")


def index_documents():
    create_corpus()

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Use the default embedding function (all-MiniLM-L6-v2 via sentence-transformers)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col = client.get_or_create_collection(
        name="docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Clear existing
    existing = col.count()
    if existing > 0:
        print(f"Clearing {existing} existing documents...")
        col.delete(where={"source": {"$ne": ""}})

    txt_files = glob.glob(f"{CORPUS_DIR}/*.txt")
    if not txt_files:
        print(f"No .txt files found in {CORPUS_DIR}/")
        return

    total_chunks = 0
    for filepath in txt_files:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        source = os.path.basename(filepath)

        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]

        col.add(documents=chunks, metadatas=metadatas, ids=ids)
        total_chunks += len(chunks)
        print(f"  Indexed {source}: {len(chunks)} chunks")

    print(f"\nTotal: {total_chunks} chunks from {len(txt_files)} files")
    print(f"Vector DB saved to {VECTOR_DB_DIR}/")


if __name__ == "__main__":
    index_documents()
