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

## Configure

```bash
make install
cp env.example .env
```

Edit `.env` and set three things:

```bash
FILTER_CHATTAG_MODEL=openai:gpt-4o-mini   # or another provider, see the table in README.md
OPENAI_API_KEY=sk-...                     # the credential matching the model above
VIDEO_PATH=./car_truck_person.mp4
```

The prompt is configurable and decides what gets labelled. `env.example` defaults to `./prompts/food_annotation_prompt.txt`; for the sample clip above use a prompt that asks about vehicles and people, or write your own:

```bash
cat > prompts/vehicle_prompt.txt <<'EOF'
Look at this image and determine whether each of the following is present:
car, truck, person.
EOF

# then in .env
FILTER_PROMPT=./prompts/vehicle_prompt.txt
```

See [Prompt format](README.md#prompt-format) for the exact contract the model must satisfy.

## Run

```bash
make run
```

Then open Webvis at http://localhost:8000.

## Verify it worked

Each frame carries a `meta.chattag` payload with one entry per label, for example:

```json
{"car": {"present": true, "confidence": 0.95}, "truck": {"present": false, "confidence": 0.9}}
```

With `FILTER_SAVE_FRAMES=true` the annotated frames and a `labels.jsonl` land under `FILTER_OUTPUT_DIR` (default `./output_frames`). One line per frame in `labels.jsonl` means the pipeline ran end to end.

## Without an API key

Set `FILTER_NO_OPS=true` to run the whole pipeline with the model call skipped. Useful to confirm wiring before spending tokens.
