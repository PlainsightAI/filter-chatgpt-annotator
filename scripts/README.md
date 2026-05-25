# ChatTag Filter Scripts

This directory contains example scripts demonstrating how to use the ChatTag with different types of image annotation tasks.

All scripts accept either `VIDEO_PATH` (a video file) or `IMAGE_PATH` (an image file or directory) and auto-select `VideoIn` vs `ImageIn` accordingly.

## Available Scripts

### 1. Food Annotation (`filter_food_annotation.py`)
Detects salad ingredients (avocado, lettuce, tomato) with confidence scores.

**Prompt**: `./prompts/food_annotation_prompt.txt`
**Output Schema**: avocado, lettuce, tomato

### 2. Pet Classification (`filter_pet_classification.py`)
Detects presence of cats and dogs.

**Prompt**: `./prompts/pet_classification_prompt.txt`
**Output Schema**: cat, dog
**Notes**: Lower token limits / smaller `max_image_size` defaults tuned for simple classification.

### 3. Multilabel Annotation (`filter_multilabel.py`)
Multilabel classification (e.g. avocado, fish, chicken). Emits a COCO export under `multilabel_datasets/` in addition to the per-label binary datasets.

**Prompt**: `./prompts/salad_prompt_multilabel.txt`
**Output Schema**: avocado, fish, chicken

## Usage

### Prerequisites

1. **Install Dependencies**:
   ```bash
   cd /path/to/filter-chatgpt-annotator
   make install
   ```

2. **Set up Environment Variables**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Required Environment Variables**:
   ```bash
   FILTER_CHATGPT_API_KEY=your_openai_api_key_here
   FILTER_PROMPT=./prompts/[prompt_file].txt
   VIDEO_PATH=/path/to/your/video.mp4
   ```

### Running Scripts

#### Food Annotation Example (Video):
```bash
# Set up environment for video processing
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export VIDEO_PATH="/path/to/food_video.mp4"

# Run the script
python scripts/filter_food_annotation.py
```

#### Food Annotation Example (Images):
```bash
# Set up environment for image processing
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export IMAGE_PATH="/path/to/food_images"  # Directory with multiple images
export IMAGE_PATTERN="jpg"  # Optional: file pattern (default: jpg)
export FILTER_RECURSIVE="false"  # Optional: scan subdirectories

# Run the script
python scripts/filter_food_annotation.py
```

#### Pet Classification Example (Video):
```bash
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/pet_classification_prompt.txt"
export VIDEO_PATH="/path/to/pet_video.mp4"

python scripts/filter_pet_classification.py
```

#### Pet Classification Example (Image):
```bash
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/pet_classification_prompt.txt"
export IMAGE_PATH="/path/to/pet_image.jpg"

python scripts/filter_pet_classification.py
```

#### Multilabel Annotation Example:
```bash
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/salad_prompt_multilabel.txt"
export IMAGE_PATH="/path/to/food_images"   # or VIDEO_PATH=/path/to/video.mp4
export FILTER_SAVE_FRAMES="true"

python scripts/filter_multilabel.py
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FILTER_CHATGPT_API_KEY` | Required | OpenAI API key |
| `FILTER_PROMPT` | Required | Path to prompt file |
| `VIDEO_PATH` | Optional | Path to video file (for video processing) |
| `IMAGE_PATH` | Optional | Path to image file or directory (for image processing) |
| `IMAGE_PATTERN` | `jpg` | File pattern for image filtering (e.g., jpg, png, *) |
| `FILTER_RECURSIVE` | `false` | Scan subdirectories recursively |
| `FILTER_CHATGPT_MODEL` | `gpt-4o-mini` | ChatGPT model to use |
| `FILTER_MAX_TOKENS` | `1000` | Maximum response tokens |
| `FILTER_TEMPERATURE` | `0.1` | Response randomness (0-2) |
| `FILTER_MAX_IMAGE_SIZE` | `0` | Max image size for processing (0 = keep original) |
| `FILTER_IMAGE_QUALITY` | `85` | JPEG quality (1-100) |
| `FILTER_SAVE_FRAMES` | `true` | Save JSON results per frame |
| `FILTER_OUTPUT_DIR` | `./output_frames` | Directory for saved results |
| `FILTER_NO_OPS` | `false` | Skip API calls for testing (use default annotations) |

### Input Source Configuration

The filter supports two types of input sources:

#### 1. Video Processing
```bash
export VIDEO_PATH="/path/to/video.mp4"
# Uses VideoIn filter for video frame extraction
```

#### 2. Image Processing
```bash
export IMAGE_PATH="/path/to/images"  # Directory with multiple images
export IMAGE_PATTERN="jpg"           # Optional: file pattern
export FILTER_RECURSIVE="false"      # Optional: scan subdirectories
# Uses ImageIn filter for image processing
```

**Image Processing Features:**
- **Single Image**: `IMAGE_PATH="/path/to/image.jpg"`
- **Directory**: `IMAGE_PATH="/path/to/images/"` (processes all images in directory)
- **Pattern Filtering**: `IMAGE_PATTERN="jpg"` (only process .jpg files)
- **Recursive Scanning**: `FILTER_RECURSIVE="true"` (scan subdirectories)
- **Looping**: Automatically loops through all images
- **Dynamic Monitoring**: Detects new images added to directory
- **FPS Control**: Processes images at controlled rate (2 FPS by default)

### Output Format

All scripts produce standardized JSON output with the following structure:

See [docs/output_contract.md](../docs/output_contract.md) for `meta.chatgpt_annotator` and `labels.jsonl` (includes `schema_version`).

## Real-World Usage Examples

### Example 1: Batch Processing Food Images
```bash
# Process all images in a directory
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export IMAGE_PATH="/path/to/salad_images"
export IMAGE_PATTERN="jpg"
export FILTER_SAVE_FRAMES="true"
export FILTER_OUTPUT_DIR="./food_analysis_results"

python scripts/filter_food_annotation.py
```

### Example 2: Monitoring Upload Directory
```bash
# Monitor a directory for new images
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/pet_classification_prompt.txt"
export IMAGE_PATH="/path/to/upload_folder"
export IMAGE_PATTERN="*"  # All file types
export FILTER_RECURSIVE="true"  # Scan subdirectories
export FILTER_SAVE_FRAMES="true"

python scripts/filter_pet_classification.py
```

### Example 3: Multilabel COCO Export
```bash
# Process a directory and emit per-label binary + COCO multilabel datasets
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/salad_prompt_multilabel.txt"
export IMAGE_PATH="/path/to/food_images"
export IMAGE_PATTERN="jpg"
export FILTER_SAVE_FRAMES="true"
export FILTER_OUTPUT_DIR="./multilabel_results"

python scripts/filter_multilabel.py
```

### Example 4: Single Image Processing
```bash
# Process a single image file
export FILTER_CHATGPT_API_KEY="your_api_key"
export FILTER_PROMPT="./prompts/food_annotation_prompt.txt"
export IMAGE_PATH="/path/to/single_image.jpg"
export FILTER_SAVE_FRAMES="true"

python scripts/filter_food_annotation.py
```

## Customization

### Creating Custom Prompts

1. Create a new prompt file in `./prompts/`:
   ```bash
   cp prompts/food_annotation_prompt.txt prompts/my_custom_prompt.txt
   ```

2. Edit the prompt to match your use case
3. Create a custom script or modify existing ones

### Custom Output Schemas

You can define custom output schemas in your scripts:

```python
output_schema={
    "custom_item_1": {"present": False, "confidence": 0.0},
    "custom_item_2": {"present": False, "confidence": 0.0}
}
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `FILTER_CHATGPT_API_KEY` is set correctly
2. **Prompt File Not Found**: Check that `FILTER_PROMPT` points to an existing file
3. **Video File Not Found**: Verify `VIDEO_PATH` points to a valid video file
4. **JSON Parse Error**: Check that your prompt returns valid JSON format

### Performance Tips

1. **Image Size**: Use `FILTER_MAX_IMAGE_SIZE=256` for faster processing
2. **Token Limits**: Reduce `FILTER_MAX_TOKENS` for simpler tasks
3. **Frame Saving**: Set `FILTER_SAVE_FRAMES=false` to reduce I/O overhead

### VideoIn Configuration

For reliable video processing, especially with API calls that take time, use the `!sync` option:

```python
# Correct VideoIn configuration for reliable processing
sources=f'file://{video_path}!resize=960x540!sync!no-loop;main'
```

**Key options:**
- `!sync`: Forces VideoIn to wait for each frame to be processed before sending the next
- `!no-loop`: Processes video once without looping
- `;main`: Specifies the topic name for frames

**Why `!sync` is important:**
- Without `!sync`: VideoIn may send frames faster than they can be processed, causing frame loss
- With `!sync`: Ensures all frames are processed, even with slow API calls
- Essential for ChatGPT API processing where each frame takes several seconds

### Cost Optimization

1. **Model Selection**: Use `gpt-4o-mini` for cost-effective processing
2. **Image Quality**: Lower `FILTER_IMAGE_QUALITY` to reduce token usage
3. **Batch Processing**: Process multiple frames in batches when possible

## Support

For issues and questions:
- Check the main README.md for general usage
- Review the filter implementation in `filter_chatgpt_annotator/filter.py`
- Test with the provided example prompts and scripts
