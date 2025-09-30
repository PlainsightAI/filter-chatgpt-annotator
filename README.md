# ChatTag

[![PyPI version](https://img.shields.io/pypi/v/filter-chatgpt-annotator.svg?style=flat-square)](https://pypi.org/project/filter-chatgpt-annotator/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-chatgpt-annotator?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-chatgpt-annotator)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-chatgpt-annotator/blob/main/LICENSE)

A generic filter that uses ChatGPT Vision API for image annotation and analysis across diverse datasets and domains.

## Features

- **Multi-domain Support**: Supports any domain requiring image classification and annotation (food, pets, medical, industrial, etc.)
- **Configurable Prompts**: Customizable prompts for different annotation tasks
- **Standardized Output**: Consistent JSON format with confidence scores
- **Image Optimization**: Automatic image resizing to reduce API costs
- **Fault Tolerant**: Logs and skips malformed data instead of crashing
- **Real-time Processing**: Processes video streams in real-time
- **Web Visualization**: Includes web interface for viewing results
- **Pipeline Integration**: Works with OpenFilter pipeline architecture
- **Environment Configuration**: Full configuration through environment variables
- **Frame Persistence**: Optional saving of JSON results per frame
- **Topic Filtering**: Process specific topics or exclude unwanted ones
- **Topic Forwarding**: Preserve main topic alongside processed results for pipeline compatibility
- **Cost Optimization**: Configurable image size and quality settings

## Architecture

The filter follows the OpenFilter pattern with three main stages:

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| `setup()` | Parse and validate configuration; initialize ChatGPT client; load prompt file |
| `process()` | Core operation: send images to ChatGPT Vision API, parse, validate, attach result |
| `shutdown()` | Clean up resources (close connections) when filter stops |

### Data Signature

The filter returns processed frames with the following data structure:

**Main Frame Data:**
- Original frame data preserved
- Processing results added to frame metadata:
  - `annotations`: Dict with item_name -> {"present": bool, "confidence": float}
  - `usage`: Dict with token usage information
  - `processing_time`: Processing time in seconds
  - `timestamp`: Processing timestamp
  - `error`: Error message if processing failed

**Topic Forwarding:**
The `forward_main` parameter controls whether the main topic from input frames is forwarded to the output:

- **`forward_main=True`**: The main topic from input frames is preserved and forwarded to the output alongside processed results
- **`forward_main=False`**: Only processed frames are returned (no main topic forwarding)

This is useful in pipeline scenarios where you want to preserve the original main frame alongside processed results for downstream filters.

## Installation

```bash
# Install with development dependencies
make install
```

## Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` file with your configuration:
```bash
# Required: OpenAI API Key
FILTER_CHATGPT_API_KEY=your_openai_api_key_here

# Required: Path to prompt file
FILTER_PROMPT=./prompts/annotation_prompt.txt

# Optional: ChatGPT model (default: gpt-4o-mini)
FILTER_CHATGPT_MODEL=gpt-4o-mini

# Optional: API parameters
FILTER_MAX_TOKENS=1000
FILTER_TEMPERATURE=0.1

# Optional: Image processing
FILTER_MAX_IMAGE_SIZE=512
FILTER_IMAGE_QUALITY=85

# Optional: Output configuration
FILTER_SAVE_FRAMES=false
FILTER_OUTPUT_DIR=./output_frames

# Optional: Output schema (JSON string)
FILTER_OUTPUT_SCHEMA={"item1": {"present": false, "confidence": 0.0}, "item2": {"present": false, "confidence": 0.0}}

# Optional: Topic filtering
FILTER_TOPIC_PATTERN=.*
FILTER_EXCLUDE_TOPICS=debug,test

# Optional: Topic forwarding (preserve main topic alongside processed results)
FILTER_FORWARD_MAIN=false

# Optional: No-ops mode (skip API calls for testing)
FILTER_NO_OPS=false
```

### Configuration Matrix

| Variable | Type | Default | Required | Notes |
|----------|------|---------|----------|-------|
| `chatgpt_model` | string | "gpt-4o-mini" | Yes | Model name |
| `chatgpt_api_key` | string | "" | Yes | API key |
| `prompt` | string | "" | Yes | Path to prompt file (.txt) |
| `output_schema` | dict | {} | No | Defines expected labels and defaults |
| `max_tokens` | int | 1000 | No | Max response tokens |
| `temperature` | float | 0.1 | No | Controls randomness |
| `max_image_size` | int | 0 | No | Max image size (0 = keep original) |
| `image_quality` | int | 85 | No | JPEG quality (1-100) |
| `save_frames` | bool | true | No | Save JSON per frame |
| `output_dir` | string | "./output_frames" | No | Where to save JSON output |
| `forward_main` | bool | false | No | Forward main topic to output |
| `no_ops` | bool | false | No | Skip API calls for testing |
| `confidence_threshold` | float | 0.9 | No | Confidence threshold for positive classification (0.0-1.0) |

## Usage

### No-Ops Mode (Testing)

For testing and development, you can enable no-ops mode to skip API calls:

```bash
# Enable no-ops mode
export FILTER_NO_OPS=true

# Run the filter (will skip API calls and use default annotations)
python scripts/filter_annotation_batch.py
```

In no-ops mode:
- ✅ Images are still processed and resized
- ✅ JSON files are still generated with default annotations
- ✅ Binary datasets are still created on shutdown
- ❌ No API calls are made to ChatGPT
- ❌ No API costs are incurred

This is useful for:
- Testing the pipeline without API costs
- Validating image processing and file generation
- Development and debugging

### Image Size Configuration

The `max_image_size` parameter controls image resizing for API cost optimization:

```bash
# Keep original image size (highest quality, highest cost)
export FILTER_MAX_IMAGE_SIZE=0

# Resize to 512px (good quality, moderate cost)
export FILTER_MAX_IMAGE_SIZE=512

# Resize to 256px (lower quality, lowest cost)
export FILTER_MAX_IMAGE_SIZE=256
```

**Cost Impact:**
- `0` (original): ~$0.15/image (high quality)
- `512px`: ~$0.01/image (good quality)
- `256px`: ~$0.005/image (lower quality)

### Topic Forwarding Configuration

The `forward_main` parameter controls whether the main topic from input frames is forwarded to the output:

```bash
# Forward main topic to preserve original frame (recommended for pipelines)
export FILTER_FORWARD_MAIN=true

# Don't forward main topic (only processed results)
export FILTER_FORWARD_MAIN=false
```

**Use Cases:**
- **Pipeline Processing**: When you want to preserve the original main frame for downstream filters
- **Multi-topic Processing**: When processing specific topics but want to keep the main frame intact
- **Data Preservation**: When you need both processed results and original frame data

**Output Behavior:**
- **With `forward_main=True`**: Output includes both processed topics and the original main topic
- **With `forward_main=False`**: Output includes only processed topics

**Example Output Structure:**
```python
# With forward_main=True
{
    "main": Frame(original_image, original_data, "BGR"),           # Original main frame
    "processed_topic_1": Frame(image, results_metadata, "BGR"),   # Processed frame
    "processed_topic_2": Frame(image, results_metadata, "BGR")    # Processed frame
}

# With forward_main=False
{
    "processed_topic_1": Frame(image, results_metadata, "BGR"),   # Processed frame
    "processed_topic_2": Frame(image, results_metadata, "BGR")    # Processed frame
}
```

### Save Frames Configuration

The `save_frames` parameter controls whether to save individual JSON files:

```bash
# Save JSON files (default - recommended)
export FILTER_SAVE_FRAMES=true

# Don't save files (only show in web interface)
export FILTER_SAVE_FRAMES=false
```

**Benefits of saving frames:**
- ✅ **Processed images** - Images saved in `data/` subfolder with unique names
- ✅ **JSONL dataset** - Results saved in dataset_langchain format
- ✅ **Binary datasets** - Automatically generated for ML training
- ✅ **Debugging** - Can inspect individual frame results and images
- ✅ **Batch processing** - Results available after pipeline ends

**When to disable:**
- Quick testing without file clutter
- Web visualization only
- Temporary analysis

### Confidence Threshold Configuration

The `confidence_threshold` parameter controls the minimum confidence score required to classify an item as "present" in the generated datasets:

```bash
# Default: 90% confidence required
export FILTER_CONFIDENCE_THRESHOLD=0.9

# More lenient: 70% confidence required
export FILTER_CONFIDENCE_THRESHOLD=0.7

# Very strict: 95% confidence required
export FILTER_CONFIDENCE_THRESHOLD=0.95
```

**How it works:**
- **Confidence ≥ threshold** → Item classified as **PRESENT** (positive class)
- **Confidence < threshold** → Item classified as **ABSENT** (negative class)

**Examples:**
```json
{
  "avocado": {
    "present": true,
    "confidence": 0.92  // ✅ 92% ≥ 90% → "avocado" (with threshold=0.9)
  },
  "tomato": {
    "present": true,
    "confidence": 0.85  // ❌ 85% < 90% → "absent" (with threshold=0.9)
  }
}
```

**Recommended values:**
- **0.9 (90%)** - Default, high precision
- **0.8 (80%)** - Balanced precision/recall
- **0.7 (70%)** - Higher recall, more lenient
- **0.95 (95%)** - Very high precision, strict

### Output Structure

When `save_frames=true`, the following structure is created:

```
./output_frames/
├── data/                     # Processed images subfolder
│   ├── 0_1758035382121.jpg  # Frame 0 with timestamp
│   ├── 1_1758035382122.jpg  # Frame 1 with timestamp
│   └── 2_1758035382123.jpg  # Frame 2 with timestamp
├── labels.jsonl              # Dataset in dataset_langchain format
└── binary_datasets/          # Generated automatically on shutdown (overwrites existing)
    ├── item1_labels.json
    ├── item2_labels.json
    ├── item3_labels.json
    ├── item4_labels.json
    └── _summary_report.json
└── binary_datasets_balanced/ # Balanced datasets (equal class representation)
    ├── item1_labels.json
    ├── item2_labels.json
    ├── item3_labels.json
    ├── item4_labels.json
    └── _summary_report.json  # Summary report (highlighted with underscore)
```

**Important Notes:**
- **Binary datasets are overwritten** on each run to ensure they reflect the latest processing results
- **Images are saved incrementally** during processing (append mode)
- **JSONL file is appended** during processing, not overwritten
- **Summary report is regenerated** on each shutdown
- **Balanced datasets** are generated automatically

### Basic Pipeline

Run the complete annotation pipeline:

```bash
python scripts/filter_food_annotation.py
```

This will:
1. Load video from `VIDEO_PATH` environment variable
2. Process frames with ChatGPT Vision API using the specified prompt
3. Display results in web interface at `http://localhost:8000`

### Using Makefile

```bash
# Run with example video
make run-example

# Run with custom video
VIDEO_PATH=/path/to/video.mp4 make run-custom

# Check environment
make check-env

# Run tests
make test
```

## Usage Scenarios

### 1. Example Dataset (Food Analysis)
Detect items with confidence levels (example):

```bash
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export FILTER_OUTPUT_SCHEMA='{"lettuce": {"present": false, "confidence": 0.0}, "tomato": {"present": false, "confidence": 0.0}}'
python scripts/filter_food_annotation.py
```

### 2. Pet Classification
Detect presence of cats/dogs:

```bash
export FILTER_PROMPT="./prompts/pet_classification_prompt.txt"
export FILTER_OUTPUT_SCHEMA='{"cat": {"present": false, "confidence": 0.0}, "dog": {"present": false, "confidence": 0.0}}'
python scripts/filter_pet_classification.py
```

### 3. Medical Imaging
Detect medical conditions (research/educational only):

```bash
export FILTER_PROMPT="./prompts/medical_imaging_prompt.txt"
export FILTER_OUTPUT_SCHEMA='{"tumor": {"present": false, "confidence": 0.0}, "calcification": {"present": false, "confidence": 0.0}}'
python scripts/filter_medical_imaging.py
```

### 4. Industrial Quality
Detect defects in assembly line images:

```bash
export FILTER_PROMPT="./prompts/industrial_quality_prompt.txt"
export FILTER_SAVE_FRAMES="true"
export FILTER_OUTPUT_DIR="./quality_results"
python scripts/filter_industrial_quality.py
```

### 5. Pipeline Integration with Topic Forwarding
Preserve main topic for downstream processing:

```bash
export FILTER_PROMPT="./prompts/annotation_prompt.txt"
export FILTER_FORWARD_MAIN="true"  # Preserve main topic
export FILTER_OUTPUT_SCHEMA='{"item1": {"present": false, "confidence": 0.0}, "item2": {"present": false, "confidence": 0.0}}'
python scripts/filter_annotation.py
```

This configuration ensures that:
- The original main frame is preserved for downstream filters
- Processed results are available alongside the original data
- Pipeline compatibility is maintained

### 6. Object Detection Tasks
Generate COCO format datasets for object detection training:

```bash
export FILTER_PROMPT="./prompts/food_annotation_prompt_bb.txt"
export FILTER_OUTPUT_SCHEMA='{"avocado": {"present": false, "confidence": 0.0, "bbox": null}}'
python scripts/filter_food_annotation.py
```

**Auto-detection**: The filter automatically detects when to generate detection datasets based on the presence of `bbox` fields in the output schema.

**Output Structure**:
```
output_frames/
├── data/                     # Processed images
├── labels.jsonl              # Main dataset with bbox coordinates
├── binary_datasets/          # Classification datasets (always generated)
│   ├── avocado_labels.json
│   └── _summary_report.json
└── detection_datasets/       # COCO format datasets (if bbox schema present)
    ├── annotations.json      # COCO format annotations
    └── _summary_report.json  # Detection dataset summary
```

**Key Features**:
- ✅ **Always generates classification datasets** for binary classification training
- ✅ **Auto-generates detection datasets** when bbox fields are present in schema
- ✅ **No manual task configuration** needed - fully automatic
- ✅ **Backward compatible** with existing configurations

**COCO Format Features**:
- Standard COCO JSON format with `images`, `annotations`, and `categories` sections
- Automatic image dimension detection
- Absolute coordinate conversion from normalized bbox coordinates
- Category mapping with unique IDs
- Compatible with popular frameworks (PyTorch, TensorFlow, etc.)

## Prompt Format & Importance

The prompt format is critical for annotation quality. Prompts must:

- Define the exact list of items to check
- Enforce output as strict JSON only (no extra text)
- Provide clear rules for uncertainty and confidence scoring

### Example Prompt (Generic Dataset)

```
You are a vision analyst. Given an image, determine whether each of the following items is visibly present.
Return ONLY valid JSON with keys: "present" (boolean) and "confidence" (0-1).
ITEMS = ["item1", "item2", "item3", "item4", "item5", ...]
```

### Example Prompt (Pets Dataset)

```
You are a vision analyst. Given an image, determine whether it contains a cat or a dog.
Return ONLY valid JSON with:
{
  "cat": {"present": <true|false>, "confidence": <0-1>},
  "dog": {"present": <true|false>, "confidence": <0-1>}
}
Rules:
- If unsure, set present=false and confidence ≤0.3.
- Base decision only on visible image content.
```

## Standard Output Format

All annotations follow this standardized format:

```json
{
  "item_name": {
    "present": true|false,
    "confidence": 0.0-1.0
  }
}
```

### Example Runtime Output

```json
{
  "image": "001.png",
  "labels": {
    "cat": {"present": true, "confidence": 0.92},
    "dog": {"present": false, "confidence": 0.15}
  },
  "usage": {
    "input_tokens": 26288,
    "output_tokens": 414,
    "total_tokens": 26702
  }
}
```

## Available Scripts

The `scripts/` directory contains example implementations for different use cases:

- **`filter_food_annotation.py`**: Example food item detection
- **`filter_pet_classification.py`**: Cat/dog classification
- **`filter_medical_imaging.py`**: Medical image analysis (research only)
- **`filter_industrial_quality.py`**: Quality inspection and defect detection

See [scripts/README.md](scripts/README.md) for detailed usage instructions.

## Cost Optimization

### Image Processing
- **Resize Images**: Use `FILTER_MAX_IMAGE_SIZE=256` for faster processing
- **Quality Settings**: Lower `FILTER_IMAGE_QUALITY` to reduce token usage
- **Model Selection**: Use `gpt-4o-mini` for cost-effective processing

### Token Management
- **Token Limits**: Reduce `FILTER_MAX_TOKENS` for simpler tasks
- **Prompt Optimization**: Keep prompts concise and focused
- **Batch Processing**: Process multiple frames efficiently

## Development

### Project Structure

```
filter-chatgpt-annotator/
├── filter_chatgpt_annotator/
│   └── filter.py              # Main filter implementation
├── scripts/                   # Example usage scripts
│   ├── filter_food_annotation.py
│   ├── filter_pet_classification.py
│   ├── filter_medical_imaging.py
│   ├── filter_industrial_quality.py
│   └── README.md
├── prompts/                   # Example prompt files
│   ├── food_annotation_prompt.txt
│   ├── pet_classification_prompt.txt
│   ├── medical_imaging_prompt.txt
│   └── industrial_quality_prompt.txt
├── tests/                     # Test files
├── env.example               # Environment configuration example
└── pyproject.toml           # Project dependencies
```

### Key Dependencies

- `openai>=1.0.0` - ChatGPT Vision API client
- `openfilter[all]>=0.1.0` - Filter framework
- `opencv-python>=4.8.0` - Image processing
- `pillow>=9.0.0` - Image manipulation
- `python-dotenv>=1.0.0` - Environment configuration

### Testing

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Check code quality
make lint

# Format code
make format
```

## Troubleshooting

### API Key Issues
If you get API key errors:
1. Check that `FILTER_CHATGPT_API_KEY` is set correctly in `.env`
2. Verify your OpenAI API key is valid and has sufficient credits
3. Ensure the key has access to the Vision API

### Prompt File Not Found
If you get prompt file errors:
1. Check that `FILTER_PROMPT` points to an existing file
2. Verify the prompt file contains valid text
3. Ensure the prompt returns valid JSON format

### JSON Parse Errors
If ChatGPT returns invalid JSON:
1. Review your prompt to ensure it enforces JSON-only output
2. Add validation rules in the prompt
3. Check the filter logs for the raw response

### Performance Issues
If processing is slow:
1. Reduce `FILTER_MAX_IMAGE_SIZE` to 256 or 128
2. Lower `FILTER_IMAGE_QUALITY` to 70-80
3. Use `gpt-4o-mini` instead of `gpt-4o`
4. Reduce `FILTER_MAX_TOKENS` for simpler tasks

### Cost Optimization
To reduce API costs:
1. Use smaller image sizes (`FILTER_MAX_IMAGE_SIZE=256`)
2. Lower image quality (`FILTER_IMAGE_QUALITY=70`)
3. Optimize prompts to be more concise
4. Use `gpt-4o-mini` model
5. Set appropriate token limits

## Open Questions & Next Steps

- Should the filter enforce JSON Schema validation instead of simple type casting?
- Should prompts be standardized into a prompt library by domain?
- Should batch multi-image requests be supported for efficiency?
- What metrics (tokens, cost, latency) should be exposed for monitoring?
- Should we allow provider abstraction (Gemini, Claude) in the next iteration?

## Documentation

For more detailed information, configuration examples, and advanced usage scenarios, see the [comprehensive documentation](https://github.com/PlainsightAI/filter-chatgpt-annotator/blob/main/docs/overview.md).

## License

See LICENSE file for details.