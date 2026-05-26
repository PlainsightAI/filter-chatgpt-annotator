# ChatTag output contract (OpenFilter)

Versioned payloads produced by `FilterChatTag` for streaming frames and for saved JSONL. Downstream filters should read `schema_version` and **ignore unknown fields** (top-level or nested) unless they opt in to a documented extension.

The JSON Schema under `schemas/` lists required keys and known shapes; it **does not** forbid extra properties, so a validator must not reject payloads merely for carrying additional keys (forward compatibility within the same `schema_version` string).

The shape below is identical regardless of which LangChain-backed provider produced the annotations (OpenAI, Gemini, Claude, Ollama). LangChain's `with_structured_output(Pydantic)` enforces the schema on every provider.

## `schema_version`

| Value   | Meaning                                      |
|---------|----------------------------------------------|
| `"1.0"` | Current contract (classification labels only) |

## Stream: `frame.data["meta"]["chattag"]`

| Field              | Type    | Required | Description |
|--------------------|---------|----------|-------------|
| `schema_version`   | string  | yes      | Contract version, e.g. `"1.0"`. |
| `annotations`      | object  | yes      | Map of label name → per-label record (see below). |
| `usage`            | object  | yes      | `input_tokens`, `output_tokens`, `total_tokens` (integers). Populated from `AIMessage.usage_metadata`; may be all zero when the provider does not report token usage (some Ollama models). |
| `processing_time`  | number  | yes      | Seconds spent on the LLM call for this frame. |
| `timestamp`        | number  | yes      | Unix time when processing finished. |
| `model`            | string  | yes      | LangChain model string used for the request (e.g. `openai:gpt-4o-mini`). |
| `frame_id`         | string  | yes      | Frame identifier from input meta or topic. |
| `error`            | string  | no       | Present when the LLM call or structured-output parse failed. |

### Per-label record (`annotations[label]`)

| Field         | Type    | Required | Description |
|---------------|---------|----------|-------------|
| `present`     | boolean | yes      | Whether the label applies to the image. |
| `confidence`  | number  | yes      | Score in `[0.0, 1.0]`. |

## JSONL: `labels.jsonl` (one JSON object per line)

| Field            | Type   | Required | Description |
|------------------|--------|----------|-------------|
| `schema_version` | string | yes      | Same as stream payload. |
| `image`          | string | yes      | Path to the saved frame image (relative to cwd or as saved). |
| `labels`         | object | yes      | Same shape as `annotations`. |
| `usage`          | object | yes      | Same as stream `usage`. |
| `prompt_used`    | string | yes      | Basename of the configured prompt path (`config.prompt`); empty string if no path was set. Always set after `setup()`. |

## Examples

### Stream metadata

```json
{
  "schema_version": "1.0",
  "annotations": {
    "item_a": { "present": true, "confidence": 0.92 },
    "item_b": { "present": false, "confidence": 0.15 }
  },
  "usage": { "input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050 },
  "processing_time": 1.2,
  "timestamp": 1710000000.0,
  "model": "openai:gpt-4o-mini",
  "frame_id": "0"
}
```

### JSONL line

```json
{
  "schema_version": "1.0",
  "image": "output_frames/data/0_1710000000123.png",
  "labels": {
    "item_a": { "present": true, "confidence": 0.92 },
    "item_b": { "present": false, "confidence": 0.15 }
  },
  "usage": { "input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050 },
  "prompt_used": "example_prompt.txt"
}
```

## Derived exports

When `save_frames` is on and `output_schema` has **more than one** key, shutdown also writes `multilabel_datasets/annotations.json` (COCO-style JSON: full-image `bbox` per label that passes `confidence_threshold`). Stream and `labels.jsonl` payloads remain classification-only (`present` / `confidence`).

## Machine-readable schema

See [schemas/chattag_output.schema.json](../schemas/chattag_output.schema.json): documents required fields and types; additional properties are allowed so future optional fields (e.g. debug metadata) do not break validation.
