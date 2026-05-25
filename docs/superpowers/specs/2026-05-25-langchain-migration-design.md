# LangChain Migration & Rebrand to `FilterChatTag`

**Status:** Approved
**Date:** 2026-05-25
**Author:** Leandro Bezerra Marinho

## Goal

Replace the direct `openai` SDK usage with [LangChain](https://python.langchain.com/) so the filter supports OpenAI, Google Gemini, Anthropic Claude, and Ollama via the same interface. Rebrand the filter to `FilterChatTag` (aligning with the existing `ChatTag` JSONL schema) and drop the `chatgpt`/`chatgpt-annotator` naming everywhere.

This is a **breaking change** (major version bump). No backward-compat shims.

## Rename map

| Surface | Before | After |
| --- | --- | --- |
| Class | `FilterChatgptAnnotator` | `FilterChatTag` |
| Config class | `FilterChatgptAnnotatorConfig` | `FilterChatTagConfig` |
| Package | `filter_chatgpt_annotator` | `filter_chattag` |
| PyPI package | `filter-chatgpt-annotator` | `filter-chattag` |
| Docker image | `plainsightai/openfilter-chatgpt-annotator` | `plainsightai/openfilter-chattag` |
| GitHub repo | `PlainsightAI/filter-chatgpt-annotator` | unchanged |
| Frame metadata key | `chatgpt_annotator` | `chattag` |
| Model field | `chatgpt_model` | `chattag_model` |
| API key field | `chatgpt_api_key` | *(removed — use provider native env)* |
| Env: model | `FILTER_CHATGPT_MODEL` | `FILTER_CHATTAG_MODEL` |
| Env: API key | `FILTER_CHATGPT_API_KEY` | *(removed)* |
| Env (others) | `FILTER_PROMPT`, `FILTER_MAX_TOKENS`, … | unchanged |

## Configuration shape

```
FILTER_CHATTAG_MODEL=openai:gpt-4o-mini       # provider:model string (LangChain init_chat_model)
# OR: google_genai:gemini-2.0-flash
# OR: anthropic:claude-3-5-sonnet-latest
# OR: ollama:llava

# Provider native env vars (set whichever you use):
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
OLLAMA_HOST=http://localhost:11434
```

Default model: `openai:gpt-4o-mini` (preserves current behavior).

## Architecture

`filter_chattag/filter.py` keeps the same overall structure but swaps three pieces:

1. **Model construction** (`setup`): `init_chat_model(config.chattag_model, max_tokens=..., temperature=...)` → returns a provider-agnostic `BaseChatModel`. Then `.with_structured_output(schema)` to enforce JSON via each provider's native mechanism (tool-calling on OpenAI/Anthropic/Gemini, JSON-mode on Ollama).

2. **Pydantic schema builder** (`_build_schema`): walks `FILTER_OUTPUT_SCHEMA` dict and dynamically creates a Pydantic model where each top-level key becomes a `LabelAnnotation` field (`present: bool`, `confidence: float`). Same user-facing config; far more reliable parsing across providers.

3. **Image call** (`_analyze_image`): same BGR→base64 conversion as today, but call becomes `model.invoke([HumanMessage(content=[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}])])`. This multimodal content format works for all 4 providers in LangChain ≥ 0.3.

Everything else (frame loop, save_frames, JSONL output, dataset generators, topic filtering, no_ops, forward_main, masked-config logging) stays the same — just renamed.

## Dependencies

`pyproject.toml`:
- **Remove:** `openai>=1.0.0`
- **Add:** `langchain>=0.3,<0.4`, `langchain-openai>=0.2,<0.3`, `langchain-google-genai>=2.0,<3.0`, `langchain-anthropic>=0.3,<0.4`, `langchain-ollama>=0.2,<0.3`, `pydantic>=2.0,<3.0`

All 4 provider packages installed by default so `FILTER_CHATTAG_MODEL` can switch without re-installing.

## Tests

- Existing unit tests rewritten against new names (`FilterChatTag`, `chattag` metadata key, `chattag_model` field).
- New tests for `_build_schema` (Pydantic generation from dict) and that `_build_model` returns the right `BaseChatModel` subclass per `provider:` prefix.
- Integration tests against real providers are skipped unless the matching env var is set — local `pytest` runs only the offline suite.

## Docs (must explicitly mention LangChain)

- `README.md`: top-of-file note "Powered by LangChain — multi-provider (OpenAI / Gemini / Claude / Ollama)". Replace all "ChatGPT"/"OpenAI" prose where it refers to the architecture (keep where it refers to a specific provider example).
- `docs/adding_other_llms.md`: rewrite as "Adding more providers" — point to LangChain's provider list and explain that any LangChain-supported chat model with vision can be plugged in by changing one env var.
- `docs/filter_usage_guide.md`, `docs/usage_examples.md`, `docs/output_contract.md`, `docs/architecture_diagram.txt`: rebrand + reflect LangChain backend.
- `MIGRATION.md` (new): table of old→new env vars + class names + the architectural shift (cliente `openai` → LangChain) and why.
- `RELEASE.md`: log the major version with a one-line summary.

## Out of scope (YAGNI)

- Retry logic (LangChain exposes `.with_retry()` if needed later).
- LangSmith / tracing (env-var addable later, no code change).
- Response caching.
- Streaming (not useful for one-shot per-frame annotation).
- Backward-compat aliases for old env vars / class names.

## Repo rename

Decided against — the GitHub repository keeps its current name `PlainsightAI/filter-chatgpt-annotator`. Only the published artifact names (PyPI package `filter-chattag`, Docker image `plainsightai/openfilter-chattag`) and in-tree identifiers (package, class, metadata key, env vars) change.
