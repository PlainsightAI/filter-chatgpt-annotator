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

Each frame carries a `meta.chattag` payload with one entry per schema key:

```json
{"car": {"present": true, "confidence": 0.95}, "truck": {"present": false, "confidence": 0.9}}
```

`--mq_log pretty` prints it per frame, so the first frames appearing in the terminal mean the pipeline ran end to end.

## Without an API key

Add `--no_ops true` to the `FilterChatTag` stage to run the whole pipeline with the model call skipped. Useful to confirm wiring before spending tokens.

## Running from `.env` instead

The `scripts/filter_*.py` entry points call `load_dotenv()`, so they do read `.env`, unlike `make run` and `openfilter run`. Note each script hardcodes its own `output_schema`, so use one whose schema matches your prompt:

```bash
cp env.example .env      # set VIDEO_PATH, FILTER_PROMPT and the credential
python scripts/filter_food_annotation.py
```
