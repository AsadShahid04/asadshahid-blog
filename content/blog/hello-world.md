---
title: "Hello World — Why I'm Writing About AI Infrastructure"
date: 2026-03-25
tags:
  - personal
  - ai-infrastructure
  - dynamo
authors:
  - name: Asad Shahid
---

This is the first post on my blog. Let me tell you why I'm here and what to expect.

<!--more-->

## Who I Am

I'm Asad — a senior at UC Berkeley studying Statistics & Data Science. By day, I'm a software engineering intern at SanDisk building enterprise AI search systems. By night (and weekends), I'm an open source contributor to [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), the distributed LLM serving platform that powers companies like Perplexity, Together AI, Voyage AI, and Groq.

Before this, I spent a summer at HPE building vulnerability triage systems using NVIDIA's Container Security Blueprint on Kubernetes, and a summer at Genentech building RAG applications for clinical protocol authoring.

## Why AI Infrastructure

Here's what I've learned: the gap between "a model works in a notebook" and "a model serves millions of requests reliably" is enormous. That gap is infrastructure.

When I started contributing to Dynamo, I was drawn to a specific corner of this problem: **agentic inference**. When an AI model needs to call tools — search the web, run code, query a database — the serving framework has to parse those tool calls from the model's output stream in real-time, often while the model is still generating tokens. Get it wrong, and the tool call silently fails. Get it right, and you unlock an entirely new class of AI applications.

My first contribution ([#6821](https://github.com/ai-dynamo/dynamo/issues/6821)) fixed a bug where `tool_choice: "required"` caused tool calls to leak as raw XML instead of being parsed into structured function calls. A 100% failure rate for non-JSON parsers. The fix was surgical — prioritize format-aware parsers over the default JSON mode when a parser is configured — but understanding *why* the bug existed required tracing through the entire streaming pipeline: preprocessor → jail → parser → aggregator.

That experience taught me something: **you don't really understand a system until you've fixed a bug in it.**

## What This Blog Will Cover

I'm currently auditing Berkeley's **Scalable AI** course (EE 290/194), taught by professors who are also NVIDIA scientists. The course follows the full AI model lifecycle:

> Architecture → Pre-training → Post-training → Efficient Inference → Applications

This blog bridges that academic learning with hands-on infrastructure work:

1. **Dynamo Contributions** — Technical walkthroughs of my PRs. Not just "what I changed" but "why the system works this way" and "what I learned about distributed inference."

2. **Scalable AI Deep Dives** — Lecture-by-lecture breakdowns from the course, translated into practitioner terms. Roofline models, parallelism strategies, optimizer fundamentals, and more.

3. **Research Notes** — I'm working on improving agentic inference efficiency in Dynamo: tool-call parsing correctness, streaming performance, and request scheduling for tool-augmented workloads.

4. **Paper to Practice** — Taking academic papers from the course reading list and showing how their ideas show up in real systems.

## The Thesis

By combining structured learning with open source contributions, you create a compound feedback loop. Understanding *why* systems work the way they do makes you a better contributor. Contributing makes you a better learner. Writing about both forces clarity.

That's what this blog is — my attempt to learn in public and share what I find.

Let's go.

---

*Next up: "The Full-Stack View of AI: Why Infrastructure Determines Intelligence" — a deep dive into Lecture 1 of Scalable AI.*
