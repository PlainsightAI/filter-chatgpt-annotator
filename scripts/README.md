# FilterChatTag Example Scripts

Example scripts showing how to use `FilterChatTag` (powered by [LangChain](https://python.langchain.com/)) for different image annotation tasks. The same code works with OpenAI, Google Gemini, Anthropic Claude, or Ollama — pick the provider by setting `FILTER_CHATTAG_MODEL`.

## Available scripts

| Script | Use case |
| --- | --- |
| `filter_food_annotation.py` | Food / salad ingredient detection (video or images) |
| `filter_multilabel.py` | Multilabel salad ingredient detection (video) with COCO export |
| `filter_pet_classification.py` | Cat / dog detection (video) |

All three pipe their input through `FilterChatTag` and visualize results via `Webvis`.

## Prerequisites

```bash
cd /path/to/filter-chattag
make install
cp env.example .env
# Edit .env: set FILTER_CHATTAG_MODEL and the matching provider env var
```

Pick one model string + matching credential:

```bash
# OpenAI
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

## Running

```bash
# Food annotation (video)
export FILTER_PROMPT=./prompts/food_annotation_prompt.txt
export VIDEO_PATH=/path/to/food_video.mp4
python scripts/filter_food_annotation.py

# Food annotation (images)
export FILTER_PROMPT=./prompts/food_annotation_prompt.txt
export IMAGE_PATH=/path/to/food_images
export IMAGE_PATTERN=jpg
python scripts/filter_food_annotation.py

# Pet classification
export FILTER_PROMPT=./prompts/pet_classification_prompt.txt
export VIDEO_PATH=/path/to/pet_video.mp4
python scripts/filter_pet_classification.py
```

## Common configuration

| Variable | Default | Description |
|---|---|---|
| `FILTER_CHATTAG_MODEL` | `openai:gpt-4o-mini` | LangChain model string (`provider:model`) |
| `FILTER_PROMPT` | required | Path to prompt file |
| `VIDEO_PATH` | optional | Path to video file (for video processing) |
| `IMAGE_PATH` | optional | Path to image file or directory |
| `IMAGE_PATTERN` | `jpg` | File pattern for image filtering |
| `FILTER_RECURSIVE` | `false` | Scan subdirectories recursively |
| `FILTER_MAX_TOKENS` | `1000` | Maximum response tokens |
| `FILTER_TEMPERATURE` | `0.1` | Response randomness (0–2) |
| `FILTER_MAX_IMAGE_SIZE` | `0` | Max image side in px (0 = original) |
| `FILTER_IMAGE_QUALITY` | `85` | JPEG quality (1–100) |
| `FILTER_SAVE_FRAMES` | `true` | Persist per-frame JSON results |
| `FILTER_OUTPUT_DIR` | `./output_frames` | Where to save results |
| `FILTER_NO_OPS` | `false` | Skip LLM calls (testing) |

## Output format

See [`docs/output_contract.md`](../docs/output_contract.md) for the `meta.chattag` stream payload and the `labels.jsonl` line schema. Output structure is identical across providers — LangChain's `with_structured_output` enforces the Pydantic schema generated from `FILTER_OUTPUT_SCHEMA`.

## Custom output schemas

```python
output_schema={
    "custom_item_1": {"present": False, "confidence": 0.0},
    "custom_item_2": {"present": False, "confidence": 0.0}
}
```

## Tips

- For video, always use `!sync` to make sure VideoIn waits for each frame to be processed before sending the next. LLM calls are slow.
- Use `gpt-4o-mini` / `gemini-2.0-flash` / `claude-3-5-haiku-latest` for cost-effective batches; switch to a larger model only when quality demands it.
- Set `FILTER_NO_OPS=true` to wire up the pipeline without paying for any LLM calls.
