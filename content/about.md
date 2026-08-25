---
title: About
toc: true
---

## Who I Am

I'm **Asad Shahid**, a senior at the University of California, Berkeley studying **Statistics & Data Science** with a Certificate in **Entrepreneurship & Technology** (SCET). I graduate in December 2026.

My coursework includes Data Structures & Algorithms, Linear Algebra, Data Science, and Probability Theory. I've been recognized with the CIF Scholar-Athlete Award, Avi Raina Scholarship, Las Positas Engineering Scholarship, and NSLS.

I'm passionate about the infrastructure that makes AI work at scale — the distributed systems, serving frameworks, and optimization techniques that turn research models into production-ready services.

## What I'm Doing Now

I just finished my summer internship at **Tesla** (Aug 2026), where I worked on firmware build infrastructure. I'm now a senior graduating in December 2026, contributing to open source projects at [**NVIDIA Dynamo**](https://github.com/ai-dynamo/dynamo) and [**NVIDIA Grove**](https://github.com/ai-dynamo/grove), as well as the [**vLLM Project**](https://github.com/vllm-project). I'm also auditing Berkeley's **Scalable AI** course (EE 290/194) and maintaining this blog.

**Open Source Contributions:**
- **NVIDIA Dynamo** — Focus area: agentic inference and serving correctness. Fixed tool_choice=required bypassing format-specific parsers (100% failure for 7+ non-JSON formats), contributed NIXL memory type canonical names, TRT-LLM arg_map preservation, planner step_size improvements, and Responses API input_tokens. Built a Rust-based K9s-style TUI for cluster debugging and rebuilt benchmarking tools.
- **NVIDIA Grove** — Reduced cascade-delete log noise at 5,000-replica scale and added topology-sync exponential backoff (~63s budget) to prevent crash-loops from transient API errors.
- **llm-compressor** — Added AWQ/SmoothQuant model-registry mappings for Qwen2.5-VL, Qwen2.5-Omni, SeedOss, and Ernie4.5-MoE.

## Previous Experience

**Tesla** — Software Engineer Intern (May 2026 – Aug 2026)
- Built firmware build cache reducing median CI time from 22 min to 5 min, saving 3,000 compute-hours/week ($25K/mo) across 600+ SLURM jobs for Model S/3/X/Y, Semi, and Optimus
- Developed firmware build pipeline in Go with 88% cache hit rate, cutting prebuild time from 12 min to 45s on 200+ daily jobs
- Created build-node download coordinator as a Go daemon with SSE fan-out and disk completion markers

**SanDisk (Western Digital)** — Software Engineer Intern (Jan 2026 – May 2026)
- Built Codesigner, an AI ASIC design assistant on FastMCP with hybrid BM25 + vector retrieval across 10,000+ docs
- Achieved sub-500ms query latency using Azure AI Search and OpenAI, reducing debugging time by 30% for 50+ engineers

**Hewlett Packard Enterprise** — Software Engineer Intern (May 2025 – Aug 2025)
- Deployed NVIDIA Container Security Blueprint Helm chart on Kubernetes, achieving 90% faster CVE triage and 50% fewer false positives
- Self-hosted Llama 3.1 8B + embedding model via NVIDIA NIM on L40/A100 GPUs to analyze 300+ Docker images
- Built Go API gateway + NGINX cache with 2 Morpheus replicas, improving throughput by 3.2x and cutting scan time by 70%

**Genentech** — Software Engineer Intern (Summer 2024)
- Built a full-stack RAG application with BioBERT, MilvusDB, Vue.js, and FastAPI
- Engineered a document revisioning platform reducing page load time by 40%

## What This Blog Is About

I'm auditing Berkeley's **Scalable AI** course (EE 290/194) — taught by professors who are also NVIDIA scientists — and using it to deepen my understanding of the full AI model lifecycle. This blog bridges that academic learning with hands-on open source work:

- **Dynamo contributions**: Technical walkthroughs of PRs and the systems context behind them
- **Scalable AI learnings**: Deep dives into architecture, training, inference, and applications
- **Research notes**: Findings from independent research on agentic inference efficiency
- **"Paper to Practice"**: Taking academic papers and showing how they manifest in real systems

## Skills

**Languages:** Python, Golang, Rust, C++, JavaScript, Bash

**Frameworks & Tools:** Kubernetes, Docker, Buck2, Bazel, SLURM, NGINX, Triton Inference Server, Azure, FastAPI, Vue.js, PyTorch, vLLM, TensorRT-LLM, SGLang

## Get in Touch

- **Email:** asad.shahid@berkeley.edu
- **GitHub:** [AsadShahid04](https://github.com/AsadShahid04)
- **LinkedIn:** [asadshahid04](https://linkedin.com/in/asadshahid04)

## Interests

When I'm not debugging distributed systems: tennis, traveling, aviation, and cars.
