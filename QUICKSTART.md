# Quick Start

This guide has one goal: **run `FilterChatTag` quickly**.

`FilterChatTag` sends each video frame to a multimodal chat model and attaches structured `{present, confidence}` annotations to the frame metadata. It runs as a single stage in an OpenFilter pipeline.

Pipeline contract: `VideoIn → FilterChatTag → Webvis`

## Prerequisites

- Python 3.10+ and a virtualenv
- An API key for one provider (OpenAI, Google Gemini, Anthropic, or a local Ollama)

## Get a sample video

Any mp4 works. This one is public and needs no credentials:

```bash
curl -O https://storage.googleapis.com/plainsight-ml-assets-production/videos/car_truck_person.mp4
```

`train.mp4` is available at the same path if you prefer a different clip.

## Install and set a credential

```bash
make install
export OPENAI_API_KEY=sk-...   # or the variable matching your provider
```

See the provider table in [README.md](README.md#pick-a-provider) for the other three.

## Run

Two things decide the output: the **prompt** tells the model what to look at, and `output_schema` declares the label keys it must answer with. Both are passed explicitly below, so this command does not depend on `.env`:

```bash
openfilter run \
  - VideoIn \
      --sources 'file://car_truck_person.mp4!loop' \
      --outputs 'tcp://*:5550' \
  - filter_chattag.filter.FilterChatTag \
      --sources 'tcp://localhost:5550' \
      --outputs 'tcp://*:5552' \
      --chattag_model 'openai:gpt-4o-mini' \
      --prompt './prompts/vehicle_prompt.txt' \
      --output_schema '{"car":{"present":false,"confidence":0.0},"truck":{"present":false,"confidence":0.0},"person":{"present":false,"confidence":0.0}}' \
      --mq_log pretty \
  - Webvis \
      --sources 'tcp://localhost:5552'
```

Then open Webvis at http://localhost:8000.

To label something else, swap the prompt file and the `output_schema` keys together — they have to agree. `prompts/` ships several examples, and [Prompt format](README.md#prompt-format) documents the contract.

## Verify it worked

Each frame carries a `meta.chattag` payload. The per-label entries sit under
`annotations`, one per schema key, alongside the model and usage fields:

```json
{
  "schema_version": 1,
  "annotations": {
    "car": {"present": true, "confidence": 0.95},
    "truck": {"present": false, "confidence": 0.9},
    "person": {"present": true, "confidence": 0.88}
  },
  "usage": {"total_tokens": 412},
  "model": "openai:gpt-4o-mini",
  "processing_time": 1.31,
  "timestamp": 1787097600.42,
  "frame_id": 0
}
```

Three entries, because the `output_schema` above declares three keys.

`--mq_log pretty` prints it per frame, so the first frames appearing in the terminal mean the pipeline ran end to end.

## Without an API key

Add `--no_ops true` to the `FilterChatTag` stage to run the whole pipeline with the model call skipped. Useful to confirm wiring before spending tokens.

## Running from `.env` instead

The `scripts/filter_*.py` entry points call `load_dotenv()`, so they do read `.env`, unlike `make run` and `openfilter run`.

The script's own `output_schema` is a default, not the final word: `normalize_config` reads `FILTER_OUTPUT_SCHEMA` from the environment and overrides it (`filter_chattag/filter.py`). Since `env.example` sets that variable, a reader who copies it runs with the `.env` schema whatever script they pick. So set the schema next to the prompt, and keep the two in agreement:

```bash
cp env.example .env      # then edit it:
#   VIDEO_PATH=./car_truck_person.mp4
#   FILTER_PROMPT=./prompts/vehicle_prompt.txt
#   FILTER_OUTPUT_SCHEMA={"car":{"present":false,"confidence":0.0},"truck":{"present":false,"confidence":0.0},"person":{"present":false,"confidence":0.0}}
#   OPENAI_API_KEY=sk-...
python scripts/filter_food_annotation.py
```

The script name does not constrain the labels, since both the prompt and the schema come from `.env`.
