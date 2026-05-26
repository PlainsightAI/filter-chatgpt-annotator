---
title: ChatTag
sidebar_label: Overview
sidebar_position: 1
---

`FilterChatTag` is a LangChain-backed OpenFilter for multimodal vision annotation. It sends each video/image frame to a configurable chat model (OpenAI, Gemini, Claude, Ollama, or any other LangChain-compatible vision model) and attaches structured `{present, confidence}` annotations to the frame metadata.

### Features

- **Multi-provider via LangChain** — switch providers by changing one env var (`FILTER_CHATTAG_MODEL`).
- **Structured output enforced by Pydantic** — every provider returns the same shape, no manual JSON parsing.
- **Versioned output contract** (`schema_version`) on both the stream metadata and the `labels.jsonl` lines.
- **Dataset generators** — binary classification, balanced variant, and COCO multilabel export produced on shutdown.

### Use cases

- Auto-labeling image / video datasets for downstream ML training.
- Real-time per-frame classification in a pipeline (e.g. ingredient detection, defect inspection, pet detection).
- Quick spike of any vision-classification task with a prompt and a JSON schema — no model code required.
