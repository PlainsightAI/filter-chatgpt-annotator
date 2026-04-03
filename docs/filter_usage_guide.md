# ChatTag Usage Guide

## Overview

The `ChatTag` is a powerful filter that uses ChatGPT Vision API for image annotation and analysis. It can process video streams or image collections to extract structured classification labels with confidence scores.

## Key Features

- **Multi-domain Support**: Works with any domain requiring image classification (food, pets, medical, industrial, etc.)
- **Configurable Prompts**: Customizable prompts for different annotation tasks
- **Standardized Output**: Versioned contract for stream metadata and JSONL (see [output_contract.md](output_contract.md))
- **Image Optimization**: Automatic resizing to reduce API costs
- **Fault Tolerant**: Logs errors and continues processing
- **Real-time Processing**: Processes video streams in real-time
- **Web Visualization**: Built-in web interface for viewing results
- **Dataset Generation**: Binary classification datasets from saved `labels.jsonl`

## Installation & Setup

### Prerequisites

The project uses a Makefile for installation. You have two options:

#### Option 1: Using Google Cloud Authentication (Recommended)

```bash
# Set up Google Cloud authentication
gcloud auth login
gcloud auth application-default login

# Install the package
make install
```

#### Option 2: Using Google Application Credentials

```bash
# Set up service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Install the package
make install
```

#### Option 3: Manual Installation (Alternative)

If you don't have access to the private repository, you can install manually:

```bash
# Install dependencies
pip install openai>=1.0.0
pip install opencv-python>=4.8.0
pip install python-dotenv>=1.0.0
pip install pillow>=9.0.0

# Install in development mode
pip install -e .
```

### What `make install` does:

The `make install` command:
1. Authenticates with Google Artifact Registry
2. Installs the package in development mode (`pip install -e .[dev]`)
3. Includes all development dependencies (pytest, build tools, etc.)
4. Sets up the package for local development

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
FILTER_CHATGPT_API_KEY=your_openai_api_key_here

# Input source (choose one)
VIDEO_PATH=/path/to/your/video.mp4
IMAGE_PATH=/path/to/your/images

# Prompt configuration
FILTER_PROMPT=./prompts/your_prompt.txt

# Model configuration
FILTER_CHATGPT_MODEL=gpt-4o-mini
FILTER_MAX_TOKENS=1000
FILTER_TEMPERATURE=0.1

# Image processing
FILTER_MAX_IMAGE_SIZE=512
FILTER_IMAGE_QUALITY=90

# Output configuration
FILTER_OUTPUT_DIR=./output_frames
FILTER_SAVE_FRAMES=true
FILTER_CONFIDENCE_THRESHOLD=0.9

# Dataset options
FILTER_BALANCE_DATASET=true

# Testing mode
FILTER_NO_OPS=false

# Debug options
FILTER_DEBUG_METADATA=false
```

## Configuration Parameters

### Core Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chatgpt_model` | string | "gpt-4o" | OpenAI model to use (gpt-4o, gpt-4o-mini) |
| `chatgpt_api_key` | string | "" | OpenAI API key (required) |
| `prompt` | string | "" | Path to prompt file (required) |
| `output_schema` | dict | {} | Expected output format schema |

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | integer | 1000 | Maximum tokens in response |
| `temperature` | float | 0.1 | Response randomness (0.0-2.0) |

### Image Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_image_size` | integer | 0 | Max image size in pixels (0=keep original) |
| `image_quality` | integer | 85 | JPEG quality (1-100) |

### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_frames` | boolean | true | Save processed images and results |
| `output_dir` | string | "./output_frames" | Output directory path |
| `confidence_threshold` | float | 0.9 | Minimum confidence for positive classification |

### Topic Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic_pattern` | string | null | Regex pattern to match topics |
| `exclude_topics` | list | [] | List of topics to exclude |
| `forward_main` | boolean | false | Forward main topic to output |

### Dataset Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `no_ops` | boolean | false | Skip API calls (testing mode) |

### Debug Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug_metadata` | boolean | false | Enable debug metadata logging (frame info and images) |
| `FILTER_DEBUG_METADATA` | boolean | false | Environment variable for debug metadata (auto-normalized) |

## Usage Examples

### Basic Usage

```python
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

# Configure the filter
config = FilterChatgptAnnotatorConfig(
    id="food_annotation",
    sources="tcp://localhost:5550",
    outputs="tcp://*:5552",
    chatgpt_api_key="your_api_key",
    prompt="./prompts/food_prompt.txt",
    output_schema={
        "avocado": {"present": False, "confidence": 0.0},
        "lettuce": {"present": False, "confidence": 0.0},
        "tomato": {"present": False, "confidence": 0.0}
    }
)

# Run the filter
FilterChatgptAnnotator.run(config)
```

### Using with OpenFilter Pipeline

```python
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

Filter.run_multi([
    (VideoIn, dict(
        sources="file:///path/to/video.mp4!resize=960x540!sync!no-loop;main",
        outputs="tcp://*:5550"
    )),
    (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
        id="annotation_filter",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        chatgpt_api_key="your_api_key",
        prompt="./prompts/food_prompt.txt",
        output_schema={
            "avocado": {"present": False, "confidence": 0.0},
            "lettuce": {"present": False, "confidence": 0.0}
        }
    )),
    (Webvis, dict(
        sources="tcp://localhost:5552"
    ))
])
```

## Output Formats

### Frame Metadata

Each processed frame includes the following metadata:

```json
{
  "meta": {
    "chatgpt_annotator": {
      "schema_version": "1.0",
      "annotations": {
        "avocado": {
          "present": true,
          "confidence": 0.95
        },
        "lettuce": {
          "present": true,
          "confidence": 0.88
        }
      },
      "usage": {
        "input_tokens": 26088,
        "output_tokens": 107,
        "total_tokens": 26195
      },
      "processing_time": 2.34,
      "timestamp": 1640995200.0,
      "model": "gpt-4o",
      "frame_id": "frame_001"
    }
  }
}
```

### Generated Datasets

The filter generates JSONL and binary classification exports when `save_frames` is enabled.

#### 1. JSONL Format (`labels.jsonl`)
```json
{"schema_version": "1.0", "image": "output_frames/data/0_1640995200.jpg", "labels": {"avocado": {"present": true, "confidence": 0.95}}, "usage": {"input_tokens": 26088, "output_tokens": 107, "total_tokens": 26195}, "prompt_used": "food_prompt.txt"}
```

#### 2. Binary Classification Datasets (`binary_datasets/`)
```json
{
  "annotations": [
    {"filename": "0_1640995200.jpg", "label": "avocado"},
    {"filename": "1_1640995201.jpg", "label": "absent"},
    {"filename": "2_1640995202.jpg", "label": "avocado"}
  ]
}
```

#### 3. Multilabel COCO (`multilabel_datasets/`)

If `output_schema` defines **more than one** label, shutdown also writes COCO-style `annotations.json`: each image gets one full-frame box per label that is present above `confidence_threshold`. Use this for multilabel tooling that expects COCO layout.

## Prompt Design

### Basic Prompt Structure

```
You are a vision analyst for [DOMAIN]. Given an image of [CONTEXT], determine whether each of the following [ITEMS] is present.

Return ONLY valid JSON with the exact keys:

{
  "item1": {"present": <true|false>, "confidence": <0.0-1.0>},
  "item2": {"present": <true|false>, "confidence": <0.0-1.0>}
}

TARGET ITEMS: ["item1", "item2"]

CRITICAL RULES:
- Only mark items as present if they are ACTUALLY IN THE [CONTEXT]
- Confidence should reflect your certainty
```

### Example Prompts

#### Food Annotation
```
You are a vision analyst for food annotation. Given an image of a salad, determine whether each of the following ingredients is present IN THE SALAD.

TARGET INGREDIENTS: ["avocado", "lettuce", "tomato"]

Return ONLY valid JSON with "present" and "confidence" (0.0-1.0) for each ingredient.
```

#### Pet Classification
```
You are a vision analyst for pet classification. Given an image, determine whether each of the following pets is present.

TARGET PETS: ["dog", "cat", "bird"]

Return ONLY valid JSON with "present" and "confidence" (0.0-1.0) for each pet.
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `chatgpt_api_key is required` | Missing API key | Set `FILTER_CHATGPT_API_KEY` environment variable |
| `prompt is required` | Missing prompt file | Set `FILTER_PROMPT` environment variable |
| `Failed to parse JSON response` | Invalid API response | Check prompt format and model capabilities |
| `API ERROR: Rate limit exceeded` | Too many requests | Implement retry logic or reduce request frequency |

### Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Enable Debug Metadata

Set the environment variable to enable detailed debug information:

```bash
# Enable debug metadata logging
export FILTER_DEBUG_METADATA=true

# Or in .env file
FILTER_DEBUG_METADATA=true
```

When `FILTER_DEBUG_METADATA=true`, the filter will create debug files in `output_frames/debug/`:
- `frames_received_*.txt`: Frame processing information
- `images/debug_*.jpg`: Processed frame images

**Note**: Debug metadata is disabled by default to avoid cluttering the output directory. Only enable when needed for troubleshooting.

## Performance Optimization

### Cost Optimization

1. **Image Resizing**: Set `max_image_size` to reduce API costs
2. **Quality Settings**: Adjust `image_quality` for balance between quality and cost
3. **Model Selection**: Use `gpt-4o-mini` for lower costs
4. **Token Limits**: Set appropriate `max_tokens` for your use case

### Speed Optimization

1. **Batch Processing**: Process multiple images in sequence
2. **Parallel Processing**: Use multiple filter instances
3. **Caching**: Implement result caching for repeated images
4. **No-ops Mode**: Use `no_ops=true` for testing without API calls

## Best Practices

### 1. Prompt Design
- Be specific about what to annotate
- Provide clear examples
- Use consistent terminology
- Test prompts with sample images

### 2. Schema Design
- Define clear output schemas (`present` / `confidence` per label)
- Use appropriate confidence thresholds
- Validate schema completeness

### 3. Error Handling
- Implement retry logic for API failures
- Log errors for debugging
- Use no-ops mode for testing
- Monitor API usage and costs

### 4. Dataset Quality
- Review generated annotations
- Use balanced datasets for training
- Check confidence score distributions

## Troubleshooting

### Common Issues

1. **Low Annotation Quality**
   - Improve prompt specificity
   - Adjust confidence threshold
   - Check image quality and size
   - Validate output schema

2. **API Rate Limits**
   - Implement exponential backoff
   - Reduce image size
   - Use different API keys
   - Monitor usage patterns

3. **Memory Issues**
   - Reduce image size
   - Process in smaller batches
   - Clear debug files regularly
   - Monitor system resources

4. **Inconsistent Results**
   - Lower temperature setting
   - Improve prompt clarity
   - Validate input images
   - Check model capabilities

## Support

For issues and questions:
1. Check the debug logs in `output_frames/debug/`
2. Review the generated datasets for quality
3. Test with no-ops mode first
4. Validate your prompt and schema design
5. Check OpenAI API status and limits
