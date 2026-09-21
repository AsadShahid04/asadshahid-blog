---
title: "The Full-Stack View of AI: Why Infrastructure Determines Intelligence"
date: 2026-03-22
tags:
  - ai-infrastructure
  - scalable-ai
  - berkeley
  - distributed-systems
  - llm-serving
authors:
  - name: Asad Shahid
---

**If you remember one thing from this post:** in 2026, "a training run" is not a system. Large-scale AI is end-to-end engineering, and quality, cost, and reliability need co-design across stages.

<!--more-->

## Why I'm Taking This Course

I'm auditing Berkeley's [Scalable AI: Bridging Theory, Understanding, and Practice](https://scalable-ai.eecs.berkeley.edu/) this semester — a course taught by professors who are also NVIDIA scientists. The timing feels right. My work on [Dynamo](https://github.com/ai-dynamo/dynamo) has given me hands-on experience with one narrow slice of the AI stack (inference serving, specifically agentic tool-call streaming), but I've been operating without the full mental model.

Lecture 1 provided that model. It's the kind of lecture that reorganizes how you think about the entire field — not by teaching new algorithms, but by showing you the **coordinate system** you need to navigate the space.

This post is my synthesis of that lecture: what the full-stack view of AI actually means, why upstream mistakes compound into downstream failures, and how the lifecycle and stack maps help you avoid getting lost.

## The Central Thesis: A Training Run Is Not a System

Modern AI development is not just about loss curves and benchmarks. It's about shipping a system that works under real constraints: latency SLOs, cost budgets, memory ceilings, safety requirements, and operational reliability.

The model-centric view — "train until metrics stabilize, then figure out deployment" — is insufficient. You end up with weights that look great in isolation but can't hit production SLOs. Or you build an efficient serving stack that can't learn enough to be useful.

The systems-centric view — "optimize for throughput and latency first" — is equally dangerous. You end up with a model that's fast but wrong, or a deployment that's reliable but useless.

**Both views must be held simultaneously.** That's the discipline.

## Two Maps You Need: Lifecycle and Stack

To avoid getting lost, you need two coordinate systems:

1. **A time map (lifecycle)** — what happens next in the development process
2. **A layer map (stack)** — where in the software/hardware tower a problem lives

When something is slow, expensive, or unreliable, these maps tell you where to look.

### The Lifecycle: Stages 0–6

The lifecycle is the temporal sequence of artifacts you produce when building an AI system. Each stage has a clear goal and emits something you can point to, measure, and version.

{{< figure src="ai-lifecycle.svg" alt="AI lifecycle showing stages 0-6: Targets, Architecture, Pre-train, Post-train, Inference, Apps, Research" caption="The AI lifecycle: six stages from targets to research iteration. Every stage emits artifacts you can measure and version. If you can't name the artifact, you're not done with the stage." >}}

**Stage 0: Targets & Architecture** — Before burning GPU-hours, get concrete about success criteria. What distribution do you want to be good at (chat, code, math, tool use)? What's your quality definition (benchmarks, human feedback, app metrics)? What's your cost envelope (training budget, throughput, inference SLOs)? You get what you measure. If you don't measure a capability, expect it to quietly regress.

Architecture choices — attention structure, parameter allocation (dense vs. MoE), context handling, KV cache strategy — set both training dynamics and serving economics. A randomly initialized model definition is your artifact.

**Stage 2: Pre-training** — This is industrial-scale self-supervised learning. Three coupled problems: data (acquisition, filtering, deduplication, decontamination), training (loss, schedule, stability, throughput), and systems (distributed strategy, memory management, fault tolerance).

Your artifacts: a tokenizer + data format (stable and versioned), many checkpoints with a rational selection strategy (not vibes), training telemetry (loss, stability, throughput, failure modes), and a base-model evaluation methodology that's contamination-aware.

**Stage 3: Post-training** — Convert "capable" into "controllable" and "useful." This is where supervised fine-tuning (instruction following, format discipline, tool schemas) and preference optimization / RL (helpfulness, safety, task success) turn weights into behavior.

Evaluation stops being "just perplexity" and becomes behavior-centric: task success, binary correctness, refusal for harmful requests, tool call validity, schema adherence. You aim for a serving-ready checkpoint.

**Stage 4: Efficient Inference** — Serve the model under real traffic while meeting SLOs. The questions change: What's the cost per token? What's the latency per request (including tail latency)? How do we batch without breaking UX? What happens when context length grows and KV cache pressure builds?

The tooling is different too. Compilation and graph capture (Dynamo), quantization and reduced precision (often paired with TensorRT-LLM), deployment-prep efficiency work (pruning, NAS), and high-performance decoding engines with continuous batching and scheduling (vLLM, SGLang). Your artifact: a production serving configuration, often with quantized variants.

**Stage 5: Applications** — Build a system that solves a task reliably for real users, not just curated prompts. Context engineering (retrieval, reranking, memory, compression), tool use (function calling loops, agents, planners, verifiers), and reliability patterns (schema-constrained outputs, retries, validation, guardrails).

A single weak link upstream can make the model effectively unusable in production, even if the weights look great in isolation. Your artifact: an end-to-end product you can measure and ship to customers.

**Stage 6: Research** — Make a credible claim that tightens the next lifecycle iteration. Which bottleneck is fundamental vs. contingent on current hardware? Which architectural changes reduce total cost without breaking quality? What objectives unlock better reasoning, tool use, or robustness? Your artifact: evidence (positive or negative) that improves the next cycle.

### The Stack: Where Things Run

The lifecycle tells you *when* things happen. The stack tells you *where* they happen — which layer of software and hardware is responsible.

{{< figure src="ai-stack.svg" alt="Modern AI stack with five layers: Workloads, Frameworks/Engines, Distributed Compute, Orchestration, Compute Substrate" caption="The modern AI stack. When something fails, this tells you which layer to inspect. Workloads sit on top of frameworks/engines (PyTorch, NeMo AutoModel, Dynamo, vLLM), which run on distributed compute (Ray, Spark), orchestrated by Kubernetes/SLURM, on top of GPU clusters." >}}

**Workloads** — Data, training/post-training, evaluation, serving, monitoring, safety. This is the "what" — the actual task you're trying to accomplish.

**Frameworks & Engines** — PyTorch, JAX, NeMo AutoModel, Megatron, DeepSpeed, Dynamo, TensorRT-LLM, vLLM, SGLang. This is where the critical split happens: training frameworks optimize for throughput and numerical stability, while inference engines optimize for latency, concurrency, and KV cache management.

**Distributed Compute** — Ray, Spark, custom distributed services. This layer handles multi-node coordination, task scheduling, and failure recovery.

**Orchestration** — Kubernetes, SLURM, VM-based deployments. Resource allocation, job scheduling, and cluster management live here.

**Compute Substrate** — GPUs, networking (NVLink, InfiniBand), storage, cloud providers, on-prem clusters. The physical hardware that everything runs on.

When something breaks, these maps tell you where to look. Slow inference? Could be framework-level batching (Frameworks layer), inter-node communication (Distributed Compute), or GPU memory bandwidth (Compute Substrate). The stack narrows your search space.

## Model View vs. Systems View: Hold Both at Once

Good teams learn to switch cleanly between two modes of reasoning — and to be explicit about which mode they're in.

{{< figure src="model-vs-systems.svg" alt="Model View optimizes quality; Systems View optimizes constraints" caption="The Model View optimizes for quality (loss, benchmarks, behavior). The Systems View optimizes for constraints (latency, throughput, memory, cost, reliability). Only one view → either great weights that can't hit SLOs, or an efficient model that can't learn enough." >}}

**Model View: Optimize for Quality**

- Choose architectures with good scaling on the target distribution
- Curate data to drive specific capabilities (not just "more tokens")
- Train until metrics stabilize; pick checkpoints with defensible evaluation
- Iterate with measurement and ablation (don't guess)

If this is your only view: you end up with great weights that can't hit latency, memory, or cost budgets.

**Systems View: Optimize for Constraints**

- Start from serving constraints (SLOs, budget, memory ceilings)
- Design architecture + deployment that can actually satisfy them
- Train/post-train to reach the quality bar inside that envelope
- Measure relentlessly (profiling, latency percentiles, memory pressure, cost per request)

If this is your only view: you build an efficient model that can't learn enough to be useful.

The discipline is holding both views simultaneously and making explicit trade-offs between them.

## Why Upstream Mistakes Compound

In modern LLM development, problems rarely stay contained to one stage. Small decisions early in the lifecycle create constraints that ripple forward.

**Architecture issues → unstable training or brutal serving costs.** Pick a sparse attention pattern that doesn't parallelize well, and you'll fight GPU utilization for months. Choose a KV cache strategy that doesn't compress, and inference becomes prohibitively expensive.

**Pre-training issues → no headroom for post-training to "fix" things.** If your base model doesn't learn reasoning primitives during pre-training, no amount of RLHF will teach it to reason reliably. You can't SFT your way out of a bad pre-training run.

**Post-training issues → behavior that demos well and fails in real apps.** Optimize for benchmark performance without testing tool-call streaming reliability, and you ship a model that looks great in evals but silently drops function calls in production.

**Inference issues → latency/cost kills the user experience.** Serve a model without continuous batching or KV cache optimization, and tail latencies blow out. Users bounce even if the weights are excellent.

**Application issues → users bounce even if the weights are strong.** Build a RAG system that doesn't validate retrieved context or handle schema errors, and the model's capabilities become irrelevant. Reliability at the application layer determines whether people actually use your system.

The lesson: **don't design stages independently.** Design them as a coupled system, then measure relentlessly.

## Training vs. Inference: Same Weights, Different Physics

One of the clearest insights from Lecture 1: training frameworks and inference engines solve fundamentally different problems, even though they run the same weights.

**Training Frameworks** (NeMo AutoModel, Megatron, DeepSpeed, FSDP)

- Optimize for throughput (eat tokens fast)
- Numerical stability under backpropagation
- Large batches; gradient synchronization
- Distributed optimizer + activation memory strategies

**Inference Engines** (Dynamo, TensorRT-LLM, vLLM, SGLang)

- Optimize for latency and high concurrency
- Continuous batching and request scheduling
- KV cache management (memory is the boss here)
- Production concerns: timeouts, retries, backpressure

If you've only ever trained models, inference will surprise you. If you've only ever served models, training will surprise you.

My work on Dynamo has been squarely in the inference camp — specifically the part where tool-call parsing has to happen **while the model is still generating tokens**. That's a serving problem, not a training problem. The streaming pipeline (preprocessor → jail → parser → aggregator) exists because inference is latency-critical and stateful in ways training never is.

Understanding this split clarified something for me: when people talk about "model serving," they're often conflating two very different phases (prefill vs. decode) with very different bottlenecks (compute-bound vs. memory-bound). Dynamo's continuous batching scheduler exists because those phases interfere with each other when they share GPU cycles. [AIConfigurator](https://asadshahid.com/blog/aiconfigurator/), which I wrote about recently, models that interference analytically to find optimal configurations.

## The Scaling Walls: Where Bottlenecks Come From

Across layers, you're always trading off among correctness (the computation matches the mathematical intent), performance (tokens-per-second at acceptable cost), and reliability (reproducibility, fault tolerance, stable operation).

The classic scaling walls you'll hit:

**Compute (FLOPs)** — Can your GPUs deliver enough floating-point operations to process the workload? For large models, this often means choosing the right parallelism strategy (tensor, pipeline, expert) to keep utilization high.

**Memory Capacity** — Parameters + activations + optimizer state + KV cache. This is why quantization matters, why activation checkpointing exists, and why KV cache eviction strategies are critical for long-context serving.

**Memory Bandwidth** — Feeding the compute units fast enough. For inference, this is often the bottleneck: decoding is memory-bound because you're moving weights repeatedly while generating one token at a time.

**Communication** — All-reduces, gradient synchronization, KV cache transfers in disaggregated serving. When you scale to multiple nodes, network bandwidth and topology (NVLink vs. InfiniBand) determine whether your model trains efficiently or spends most of its time waiting.

**Data** — Feeding high-quality tokens efficiently. For training, this means disk I/O, deduplication, and streaming pipelines that don't starve the GPUs. For inference, this means request batching and scheduling to keep throughput high.

In this course, we'll learn to say which wall we're hitting **from measurements, not vibes**.

## Why This Matters: The Economics of Compound Failures

At scale, configuration and lifecycle mistakes aren't just engineering oversights — they're strategic failures.

A 40% throughput loss from suboptimal serving configurations ([AIConfigurator](https://asadshahid.com/blog/aiconfigurator/) found this was common) means 40% higher inference costs. For an organization spending millions annually on inference, that's hundreds of thousands of dollars left on the table.

Training a model without clear success criteria (Stage 0) means burning GPU-hours on runs that can't meet production SLOs, then discovering too late that you need to start over.

Shipping a model without guardrails or tool-call validation (Stage 5) means users encounter failures in production that should have been caught in post-training eval (Stage 3).

The full-stack view forces you to think about these dependencies upfront. It won't prevent all failures — research is stochastic, and hardware is adversarial — but it makes the failure modes explicit and measurable.

## What's Next

The course follows the lifecycle: architecture, pre-training, post-training, efficient inference, applications, research. Each module maps to a stage, and the tooling we'll use is production-grade (NeMo AutoModel, NeMo Curator, Dynamo, TensorRT-LLM, vLLM, SGLang, NeMo Guardrails).

I'll be writing lecture-by-lecture breakdowns as I go, grounding the academic concepts in systems I've worked with. The goal is translation: taking the theory and showing how it shows up in real infrastructure.

For now, the takeaway is this: **large-scale AI is full-stack engineering.** You can't optimize one stage in isolation. You can't only think about quality or only think about cost. You can't treat training and inference as the same problem with different scripts.

The lifecycle and stack maps keep you honest. Use them.

---

**Resources:**

- **Scalable AI Course:** https://scalable-ai.eecs.berkeley.edu/
- **Lecture 1 Slides (PDF):** [Course Overview and the Modern AI Stack](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture1.pdf)
- **Recommended Readings:**
  - "The Bitter Lesson" — Rich Sutton ([link](http://www.incompleteideas.net/IncIdeas/BitterLesson.html))
  - "The Hardware Lottery" — Sara Hooker ([link](https://arxiv.org/abs/2009.06489))
  - "Training Compute-Optimal Large Language Models" (Chinchilla) — Hoffmann et al. ([link](https://arxiv.org/abs/2203.15556))
  - "An introduction to transformers" — Richard E. Turner ([link](https://arxiv.org/abs/2304.10557))
