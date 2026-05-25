# v1.0.0 migration: LangChain backend, `FilterChatTag`

In v1.0.0 the filter's OpenAI-only backend was replaced with [LangChain](https://python.langchain.com/) so the same code now works with OpenAI, Google Gemini, Anthropic Claude, Ollama, and any other LangChain-compatible vision model. The Python class, import path, frame metadata key, env vars, and Docker image were rebranded as `ChatTag` (aligning with the existing `ChatTag` JSONL schema).

The **PyPI distribution name** (`filter-chatgpt-annotator`) and **GitHub repo URL** are unchanged in this release, so `pip install filter-chatgpt-annotator==1.0.0` keeps working. Only the import path changes (install one name, import another — same pattern as `pillow → PIL`, `beautifulsoup4 → bs4`).

This is a breaking change with no backward-compat shims. The mappings below cover everything you need to update.

## Why LangChain?

Coupling directly to the `openai` SDK meant adding a second provider required a custom abstraction, per-provider request/response adapters, and a parallel test matrix. LangChain already solves that — one model string switches providers, multimodal image input is normalized, and `with_structured_output(Pydantic)` enforces the output shape on every provider. We get multi-provider support, more reliable JSON parsing, and less code we have to maintain.

## Rename map

| Old | New |
| --- | --- |
| `pip install filter-chatgpt-annotator` | unchanged — still `pip install filter-chatgpt-annotator` |
| `from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig` | `from filter_chattag.filter import FilterChatTag, FilterChatTagConfig` |
| `docker pull plainsightai/openfilter-chatgpt-annotator` | `docker pull plainsightai/openfilter-chattag` |
| Frame metadata: `frame.data["meta"]["chatgpt_annotator"]` | `frame.data["meta"]["chattag"]` |

The GitHub repo URL and the PyPI distribution name both stay the same in v1.0.0. Only the import path, class, frame metadata key, env vars, and Docker image were rebranded.

## Env var map

| Old | New |
| --- | --- |
| `FILTER_CHATGPT_API_KEY=sk-...` | *(removed)* set the provider's native env var: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, or `OLLAMA_HOST` |
| `FILTER_CHATGPT_MODEL=gpt-4o-mini` | `FILTER_CHATTAG_MODEL=openai:gpt-4o-mini` (LangChain `provider:model` string) |
| `FILTER_PROMPT`, `FILTER_MAX_TOKENS`, `FILTER_TEMPERATURE`, `FILTER_OUTPUT_SCHEMA`, `FILTER_SAVE_FRAMES`, `FILTER_OUTPUT_DIR`, `FILTER_TOPIC_PATTERN`, `FILTER_EXCLUDE_TOPICS`, `FILTER_FORWARD_MAIN`, `FILTER_NO_OPS`, `FILTER_MAX_IMAGE_SIZE`, `FILTER_IMAGE_QUALITY`, `FILTER_PRESERVE_ORIGINAL_FORMAT`, `FILTER_DEBUG_METADATA`, `FILTER_CONFIDENCE_THRESHOLD` | **unchanged** |

### Pick the right model string

```bash
# OpenAI (same default behavior as before)
FILTER_CHATTAG_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=sk-...

# Google Gemini
FILTER_CHATTAG_MODEL=google_genai:gemini-2.0-flash
GOOGLE_API_KEY=...

# Anthropic Claude
FILTER_CHATTAG_MODEL=anthropic:claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local)
FILTER_CHATTAG_MODEL=ollama:llava
OLLAMA_HOST=http://localhost:11434
```

## Code map

### Filter construction

```python
# Before
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

FilterChatgptAnnotatorConfig(
    chatgpt_model="gpt-4o-mini",
    chatgpt_api_key=os.getenv("OPENAI_API_KEY"),
    prompt="./prompts/food.txt",
    output_schema={"avocado": {"present": False, "confidence": 0.0}},
)

# After
from filter_chattag.filter import FilterChatTag, FilterChatTagConfig

FilterChatTagConfig(
    chattag_model="openai:gpt-4o-mini",          # provider:model
    # no api_key field — OPENAI_API_KEY is read by LangChain natively
    prompt="./prompts/food.txt",
    output_schema={"avocado": {"present": False, "confidence": 0.0}},
)
```

### Reading the output

```python
# Before
results = frame.data["meta"]["chatgpt_annotator"]

# After
results = frame.data["meta"]["chattag"]
```

The structure inside that dict (`schema_version`, `annotations`, `usage`, `processing_time`, `timestamp`, `model`, `frame_id`, optional `error`) is **unchanged**, and the `labels.jsonl` line format is **unchanged** — only the metadata key was renamed. Downstream consumers that only read `labels.jsonl` need no code changes.

## Behavioral changes

- **Output JSON is enforced by Pydantic** via LangChain's `with_structured_output`. The filter no longer parses raw text with `json.loads`; the provider is asked to return structured output natively (tool-calling for OpenAI / Anthropic / Gemini, JSON-mode for Ollama). Prompts that previously begged the model to "return only valid JSON" still work but are less critical.
- **`model` field in output metadata** now contains the LangChain string (`openai:gpt-4o-mini`) instead of just the model name (`gpt-4o-mini`). Downstream consumers that filter by model name should look for substring match.
- **Default model is unchanged** (`openai:gpt-4o-mini`), so an existing `.env` with `OPENAI_API_KEY` set will continue to work after renaming `FILTER_CHATGPT_MODEL` → `FILTER_CHATTAG_MODEL`.

## Migration checklist

1. Bump version: `pip install --upgrade filter-chatgpt-annotator==1.0.0` (install name unchanged). Switch Docker pulls to `plainsightai/openfilter-chattag`.
2. Rename imports: `filter_chatgpt_annotator` → `filter_chattag`, `FilterChatgptAnnotator{,Config}` → `FilterChatTag{,Config}`.
3. Rename env var `FILTER_CHATGPT_MODEL` → `FILTER_CHATTAG_MODEL` and prefix the value with `openai:` (or another provider).
4. Drop `FILTER_CHATGPT_API_KEY` — set `OPENAI_API_KEY` (or another provider's env var) instead.
5. Rename downstream consumers reading `meta["chatgpt_annotator"]` → `meta["chattag"]`.
6. If you have `chatgpt_api_key=...` or `chatgpt_model=...` kwargs in `FilterChatTagConfig(...)`, rename / remove them.
