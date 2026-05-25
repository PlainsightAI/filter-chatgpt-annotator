# FilterChatTag

[![PyPI version](https://img.shields.io/pypi/v/filter-chatgpt-annotator.svg?style=flat-square)](https://pypi.org/project/filter-chatgpt-annotator/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-chattag?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-chattag)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-chatgpt-annotator/blob/main/LICENSE)

> **Powered by [LangChain](https://python.langchain.com/) — works with OpenAI, Google Gemini, Anthropic Claude, and Ollama. Pick the provider by changing one env var.**

`FilterChatTag` is an OpenFilter that sends each video/image frame to a multimodal chat model and attaches structured annotations (`{present, confidence}` per label) to the frame metadata. Built on top of LangChain's `init_chat_model`, so any LangChain-supported chat model with vision can be plugged in.

> **Breaking change in v1.0.0** — this filter was previously published as `filter-chatgpt-annotator` / `FilterChatgptAnnotator`. See [MIGRATION.md](MIGRATION.md) for the rename map.

## Features

- **Multi-provider** — OpenAI, Gemini, Claude, Ollama via LangChain. Same code, change `FILTER_CHATTAG_MODEL`.
- **Structured output enforcement** — annotations are validated by a Pydantic schema generated from `FILTER_OUTPUT_SCHEMA`; LangChain picks the best mechanism per provider (tool-calling, JSON-mode).
- **Standardized output contract** — versioned JSON payload on the frame (`meta.chattag`) and per-line in `labels.jsonl`. See [docs/output_contract.md](docs/output_contract.md).
- **Image optimization** — optional resize/quality settings to control cost.
- **Dataset generators on shutdown** — binary classification datasets (balanced + unbalanced) and COCO multilabel export when `output_schema` has more than one label.
- **Topic filtering / forwarding**, **no-ops mode**, **frame persistence** — pipeline-friendly.

## Quick start

```bash
make install
cp env.example .env
# Edit .env: pick a provider + set the credential
make run
```

### Pick a provider

Set `FILTER_CHATTAG_MODEL` to a LangChain `provider:model` string. All four providers are installed by default — no extra install step.

| Provider | `FILTER_CHATTAG_MODEL` example | Credential env var |
| --- | --- | --- |
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Google Gemini | `google_genai:gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic:claude-3-5-sonnet-latest` | `ANTHROPIC_API_KEY` |
| Ollama (local) | `ollama:llava` | `OLLAMA_HOST` |

Any other LangChain-supported chat model with vision works too — just install the matching `langchain-*` package and use its provider prefix.

## Configuration

```bash
# Required
FILTER_CHATTAG_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=sk-...
FILTER_PROMPT=./prompts/food_annotation_prompt.txt

# Optional — LLM
FILTER_MAX_TOKENS=1000
FILTER_TEMPERATURE=0.1

# Optional — image processing
FILTER_MAX_IMAGE_SIZE=0    # 0 = original
FILTER_IMAGE_QUALITY=85

# Optional — output
FILTER_SAVE_FRAMES=true
FILTER_OUTPUT_DIR=./output_frames
FILTER_OUTPUT_SCHEMA={"lettuce":{"present":false,"confidence":0.0},"tomato":{"present":false,"confidence":0.0}}

# Optional — topic filtering / forwarding
FILTER_TOPIC_PATTERN=.*
FILTER_EXCLUDE_TOPICS=debug,test
FILTER_FORWARD_MAIN=false

# Optional — testing
FILTER_NO_OPS=false
```

### Configuration matrix

| Variable | Type | Default | Required | Notes |
|---|---|---|---|---|
| `chattag_model` | string | `openai:gpt-4o-mini` | Yes | LangChain `provider:model` string |
| `prompt` | string | `""` | Yes | Path to prompt file (.txt) |
| `output_schema` | dict | `{}` | No | Labels + defaults; enforced via Pydantic |
| `max_tokens` | int | `1000` | No | Max response tokens |
| `temperature` | float | `0.1` | No | Controls randomness |
| `max_image_size` | int | `0` | No | Max image side in px (0 = original) |
| `image_quality` | int | `85` | No | JPEG quality (1–100) |
| `save_frames` | bool | `true` | No | Persist per-frame results |
| `output_dir` | string | `./output_frames` | No | Where to save results |
| `forward_main` | bool | `false` | No | Forward main topic to output |
| `no_ops` | bool | `false` | No | Skip LLM calls (testing) |
| `confidence_threshold` | float | `0.9` | No | Positive-class threshold for dataset generators |

Credentials are NOT a config field — set the provider's native env var (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`). LangChain reads them automatically.

## Architecture

The filter follows the standard OpenFilter `setup → process → shutdown` lifecycle:

| Stage | Responsibility |
|---|---|
| `setup()` | Validate config; build the LangChain chat model via `init_chat_model`; wrap it with `with_structured_output(Pydantic)` derived from `FILTER_OUTPUT_SCHEMA`; load prompt file |
| `process()` | For each frame: BGR→base64, build a multimodal `HumanMessage`, invoke the chain, normalize annotations, attach to `frame.data["meta"]["chattag"]` |
| `shutdown()` | Generate binary + balanced + COCO multilabel datasets from `labels.jsonl` |

### Data signature

Processed frames carry results under `frame.data["meta"]["chattag"]` (see [docs/output_contract.md](docs/output_contract.md)):

- `schema_version` — contract version (e.g. `"1.0"`)
- `annotations` — `{label_name: {present: bool, confidence: float}}`
- `usage` — `{input_tokens, output_tokens, total_tokens}` (from `AIMessage.usage_metadata`)
- `processing_time`, `timestamp`, `model`, `frame_id`
- `error` — present when processing failed

### Topic forwarding

`forward_main=True` preserves the original `main` topic alongside processed topics in the output dict — useful when downstream filters need the unmodified frame.

## Output structure (with `save_frames=true`)

```
./output_frames/
├── data/                       # Processed images
├── labels.jsonl                # One JSON line per frame (see docs/output_contract.md)
├── binary_datasets/            # Generated on shutdown
├── binary_datasets_balanced/   # Balanced (equal class) variant
└── multilabel_datasets/        # COCO export when schema has >1 label
```

Binary datasets are overwritten on each run; `labels.jsonl` and `data/` are append-only.

## Confidence threshold

`FILTER_CONFIDENCE_THRESHOLD` controls the cutoff used by the dataset generators on shutdown — `confidence ≥ threshold` → positive class, otherwise `absent`. Defaults to `0.9` (high precision). Lower it for higher recall.

## No-ops mode

```bash
export FILTER_NO_OPS=true
```

Wires up the pipeline without making any LLM calls — images are still processed and saved, default annotations are emitted. Use for plumbing/integration tests without burning credits.

## Usage scenarios

```bash
# Food annotation
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export FILTER_OUTPUT_SCHEMA='{"lettuce":{"present":false,"confidence":0.0},"tomato":{"present":false,"confidence":0.0}}'
python scripts/filter_food_annotation.py

# Pet classification (Gemini)
export FILTER_CHATTAG_MODEL=google_genai:gemini-2.0-flash
export FILTER_PROMPT="./prompts/pet_classification_prompt.txt"
export FILTER_OUTPUT_SCHEMA='{"cat":{"present":false,"confidence":0.0},"dog":{"present":false,"confidence":0.0}}'
python scripts/filter_pet_classification.py

# Multilabel (Claude, with COCO export on shutdown)
export FILTER_CHATTAG_MODEL=anthropic:claude-3-5-sonnet-latest
python scripts/filter_multilabel.py
```

## Prompt format

Prompts should clearly describe the task and the expected labels. Because LangChain enforces the output structure via Pydantic, prompts no longer need to insist as heavily on "return only valid JSON" — the provider's tool-calling layer handles that — but it remains a good idea to include the expected label list and rules for uncertainty.

```text
You are a vision analyst. Given an image, decide whether each of the
following items is visibly present:

ITEMS = ["cat", "dog"]

For each item, return:
  present: true if you can see it in the image, else false
  confidence: 0.0–1.0 reflecting your certainty
```

## Output example

```json
{
  "schema_version": "1.0",
  "image": "001.png",
  "labels": {
    "cat": {"present": true,  "confidence": 0.92},
    "dog": {"present": false, "confidence": 0.15}
  },
  "usage": {"input_tokens": 26288, "output_tokens": 414, "total_tokens": 26702},
  "prompt_used": "pet_classification_prompt.txt"
}
```

Full contract: [docs/output_contract.md](docs/output_contract.md).

## Project layout

```
filter-chatgpt-annotator/        # repo root, also the PyPI distribution name
├── filter_chattag/                 # import package (renamed in v1.0.0)
│   └── filter.py              # Main filter implementation (LangChain)
├── scripts/                   # Example pipelines
├── prompts/                   # Example prompt files
├── tests/
├── schemas/chattag_output.schema.json  # JSON Schema for the output contract
├── docs/                      # Output contract, usage guide, examples, providers
├── env.example
└── pyproject.toml
```

### Key dependencies

- `langchain>=0.3,<0.4` + `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `langchain-ollama`
- `pydantic>=2.0,<3.0`
- `openfilter[all]>=1.1.0,<2.0.0`
- `opencv-python>=4.8.0`, `pillow>=9.0.0`, `python-dotenv>=1.0.0`

## Testing

```bash
make test
make test-coverage
```

The offline test suite mocks LangChain and never hits a real provider. Integration tests against real providers are not run by default.

## Troubleshooting

- **`Authentication`/`401` errors** — the provider's native env var (`OPENAI_API_KEY` etc.) isn't reaching the process. `make run` and `docker-compose` need it exported in the parent shell or set under `environment:` in the compose file.
- **`Provider 'X' not supported`** — the matching `langchain-X` package isn't installed; the four officially supported ones ship by default. To use something else, `pip install` it and use its provider prefix.
- **Garbled annotations from Ollama** — vision-capable Ollama models (e.g. `llava`, `llama3.2-vision`) are required; text-only models will refuse the image content.
- **Slow processing** — set `FILTER_MAX_IMAGE_SIZE=512` and use a smaller model (`gpt-4o-mini`, `gemini-2.0-flash`, `claude-3-5-haiku-latest`).

## Documentation

- [Output contract](docs/output_contract.md)
- [Usage guide](docs/filter_usage_guide.md)
- [Usage examples](docs/usage_examples.md)
- [Adding more providers](docs/adding_other_llms.md)
- [Architecture diagram](docs/architecture_diagram.txt)
- [Migration from `filter-chatgpt-annotator`](MIGRATION.md)

## License

See [LICENSE](LICENSE) for details.
