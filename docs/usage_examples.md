# ChatTag Usage Examples

## Installation

Before running the examples, make sure to install the package:

```bash
make install
```

## Quick Start Examples

### Example 1: Food Annotation with Bounding Boxes

```python
#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

# Load environment variables
load_dotenv()

# Configuration
video_path = os.getenv('VIDEO_PATH', '/path/to/salad_video.mp4')
api_key = os.getenv('FILTER_CHATGPT_API_KEY')
prompt_path = './prompts/simple_salad_prompt_bb.txt'

# Run the pipeline
Filter.run_multi([
    (VideoIn, dict(
        sources=f"file://{video_path}!resize=960x540!sync!no-loop;main",
        outputs="tcp://*:5550"
    )),
    (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
        id="food_annotation",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        chatgpt_api_key=api_key,
        prompt=prompt_path,
        output_schema={
            "avocado": {"present": False, "confidence": 0.0, "bbox": None},
            "lettuce": {"present": False, "confidence": 0.0, "bbox": None},
            "tomato": {"present": False, "confidence": 0.0, "bbox": None}
        },
        confidence_threshold=0.8,
        max_image_size=512,
        save_frames=True,
    )),
    (Webvis, dict(
        sources="tcp://localhost:5552"
    ))
])
```

### Example 2: Pet Classification

```python
#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.image_in import ImageIn
from openfilter.filter_runtime.filters.webvis import Webvis
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

# Load environment variables
load_dotenv()

# Configuration
image_path = os.getenv('IMAGE_PATH', '/path/to/pet_images')
api_key = os.getenv('FILTER_CHATGPT_API_KEY')
prompt_path = './prompts/pet_classification_prompt.txt'

# Run the pipeline
Filter.run_multi([
    (ImageIn, dict(
        sources=f"file://{image_path}",
        outputs="tcp://*:5550",
        poll_interval=0
    )),
    (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
        id="pet_classification",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        chatgpt_api_key=api_key,
        prompt=prompt_path,
        output_schema={
            "dog": {"present": False, "confidence": 0.0, "bbox": None},
            "cat": {"present": False, "confidence": 0.0, "bbox": None},
            "bird": {"present": False, "confidence": 0.0, "bbox": None}
        },
        confidence_threshold=0.7,
        max_image_size=1024,
        save_frames=True
    )),
    (Webvis, dict(
        sources="tcp://localhost:5552"
    ))
])
```

### Example 3: Medical Imaging Analysis

```python
#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.image_in import ImageIn
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig

# Load environment variables
load_dotenv()

# Configuration
image_path = os.getenv('IMAGE_PATH', '/path/to/medical_images')
api_key = os.getenv('FILTER_CHATGPT_API_KEY')
prompt_path = './prompts/medical_imaging_prompt.txt'

# Run the pipeline
Filter.run_multi([
    (ImageIn, dict(
        sources=f"file://{image_path}",
        outputs="tcp://*:5550",
        poll_interval=0
    )),
    (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
        id="medical_analysis",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        chatgpt_api_key=api_key,
        prompt=prompt_path,
        output_schema={
            "tumor": {"present": False, "confidence": 0.0, "bbox": None},
            "lesion": {"present": False, "confidence": 0.0, "bbox": None},
            "normal_tissue": {"present": False, "confidence": 0.0, "bbox": None}
        },
        confidence_threshold=0.9,
        max_image_size=0,  # Keep original size for medical images
        save_frames=True,
        temperature=0.0  # Low temperature for medical accuracy
    ))
])
```

## Environment Configuration Examples

### Basic Configuration (.env)

```bash
# Required
FILTER_CHATGPT_API_KEY=sk-your-openai-api-key-here

# Input source
VIDEO_PATH=/path/to/your/video.mp4
# OR
IMAGE_PATH=/path/to/your/images

# Prompt
FILTER_PROMPT=./prompts/your_prompt.txt

# Model settings
FILTER_CHATGPT_MODEL=gpt-4o-mini
FILTER_MAX_TOKENS=1000
FILTER_TEMPERATURE=0.1

# Image processing
FILTER_MAX_IMAGE_SIZE=512
FILTER_IMAGE_QUALITY=90

# Output settings
FILTER_OUTPUT_DIR=./output_frames
FILTER_SAVE_FRAMES=true
FILTER_CONFIDENCE_THRESHOLD=0.8

# Dataset options
FILTER_BALANCE_DATASET=true

# Testing
FILTER_NO_OPS=false

# Debug options
FILTER_DEBUG_METADATA=false
```

### Advanced Configuration (.env)

```bash
# API Configuration
FILTER_CHATGPT_API_KEY=sk-your-openai-api-key-here
FILTER_CHATGPT_MODEL=gpt-4o
FILTER_MAX_TOKENS=2000
FILTER_TEMPERATURE=0.0

# Image Processing
FILTER_MAX_IMAGE_SIZE=1024
FILTER_IMAGE_QUALITY=95

# Output Configuration
FILTER_OUTPUT_DIR=./output_frames
FILTER_SAVE_FRAMES=true
FILTER_CONFIDENCE_THRESHOLD=0.9

# Topic Filtering
FILTER_TOPIC_PATTERN=^frame_\d+$
FILTER_EXCLUDE_TOPICS=debug,test
FILTER_FORWARD_MAIN=true

# Dataset Options
FILTER_BALANCE_DATASET=true

# Testing Mode
FILTER_NO_OPS=false
```

## Prompt Examples

### Food Annotation Prompt

```
You are a precision vision analyst for food annotation. Given an image of a salad, determine whether each of the following ingredients is present IN THE SALAD and provide EXACT bounding box coordinates.

Return ONLY valid JSON with the exact keys:

{
  "avocado": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "lettuce": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "tomato": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null}
}

TARGET INGREDIENTS: ["avocado", "lettuce", "tomato"]

CRITICAL ANNOTATION RULES:
- Only mark ingredients as present if they are ACTUALLY IN THE SALAD PLATE/BOWL
- Bounding boxes must TIGHTLY FIT the specific ingredient
- For AVOCADO: Only include the green avocado pieces/slices
- For LETTUCE: Only include the green leafy parts
- For TOMATO: Only include red tomato pieces/slices
- Use normalized coordinates (0.0 to 1.0)
- Confidence should reflect your certainty
```

### Pet Classification Prompt

```
You are a precision vision analyst for pet classification. Given an image, determine whether each of the following pets is present and provide EXACT bounding box coordinates.

Return ONLY valid JSON with the exact keys:

{
  "dog": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "cat": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "bird": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null}
}

TARGET PETS: ["dog", "cat", "bird"]

CRITICAL ANNOTATION RULES:
- Only mark pets as present if they are clearly visible in the image
- Bounding boxes must ENCOMPASS ONLY the specific pet
- For DOG: Include the entire dog body
- For CAT: Include the entire cat body
- For BIRD: Include the entire bird body
- Use normalized coordinates (0.0 to 1.0)
- Confidence should reflect your certainty
```

### Medical Imaging Prompt

```
You are a precision vision analyst for medical imaging. Given an X-ray image, determine whether each of the following conditions is present and provide EXACT bounding box coordinates.

Return ONLY valid JSON with the exact keys:

{
  "tumor": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "lesion": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null},
  "normal_tissue": {"present": <true|false>, "confidence": <0.0-1.0>, "bbox": [<x_min>, <y_min>, <x_max>, <y_max>] or null}
}

TARGET CONDITIONS: ["tumor", "lesion", "normal_tissue"]

CRITICAL ANNOTATION RULES:
- Only mark conditions as present if they are clearly visible
- Bounding boxes must ENCOMPASS ONLY the specific condition
- For TUMOR: Include the entire tumor area
- For LESION: Include the entire lesion area
- For NORMAL_TISSUE: Include areas of normal tissue
- Use normalized coordinates (0.0 to 1.0)
- Confidence should reflect your certainty
- Be conservative in your assessments
```

## Output Analysis Examples

### Analyzing Generated Datasets

```python
import json
import os
from pathlib import Path

def analyze_dataset(output_dir):
    """Analyze the generated dataset for quality and distribution."""
    
    # Read JSONL file
    jsonl_file = Path(output_dir) / "labels.jsonl"
    records = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    print(f"Total records: {len(records)}")
    
    # Analyze each label
    labels = ["avocado", "lettuce", "tomato"]
    
    for label in labels:
        present_count = 0
        bbox_count = 0
        confidence_scores = []
        
        for record in records:
            if label in record["labels"]:
                data = record["labels"][label]
                if data.get('present', False):
                    present_count += 1
                    confidence_scores.append(data.get('confidence', 0.0))
                    if data.get('bbox') is not None:
                        bbox_count += 1
        
        print(f"\n{label.upper()}:")
        print(f"  Present: {present_count}/{len(records)} ({present_count/len(records)*100:.1f}%)")
        print(f"  With bbox: {bbox_count}/{present_count} ({bbox_count/present_count*100:.1f}%)")
        if confidence_scores:
            print(f"  Avg confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")

# Usage
analyze_dataset("./output_frames")
```

### Quality Check Script

```python
import json
from pathlib import Path

def quality_check(output_dir):
    """Check the quality of generated annotations."""
    
    jsonl_file = Path(output_dir) / "labels.jsonl"
    issues = []
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                record = json.loads(line.strip())
                
                for label, data in record["labels"].items():
                    present = data.get('present', False)
                    confidence = data.get('confidence', 0.0)
                    bbox = data.get('bbox', None)
                    
                    # Check consistency
                    if present and confidence < 0.5:
                        issues.append(f"Line {line_num}: {label} present but low confidence ({confidence:.2f})")
                    
                    if not present and confidence > 0.7:
                        issues.append(f"Line {line_num}: {label} not present but high confidence ({confidence:.2f})")
                    
                    if present and bbox is None:
                        issues.append(f"Line {line_num}: {label} present but no bbox")
                    
                    if not present and bbox is not None:
                        issues.append(f"Line {line_num}: {label} not present but has bbox")
    
    if issues:
        print("Quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No quality issues found!")

# Usage
quality_check("./output_frames")
```

## Testing and Debugging

### Testing with No-ops Mode

```python
# Set in .env file
FILTER_NO_OPS=true

# Or in code
config = FilterChatgptAnnotatorConfig(
    # ... other config
    no_ops=True
)
```

### Enabling Debug Metadata

```python
# Method 1: Set in .env file
FILTER_DEBUG_METADATA=true

# Method 2: Set environment variable
import os
os.environ['FILTER_DEBUG_METADATA'] = 'true'

# Method 3: Set directly in config
config = FilterChatgptAnnotatorConfig(
    # ... other config
    debug_metadata=True
)

# Run your filter
# Debug files will be created in output_frames/debug/
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run your filter
# Check debug files in output_frames/debug/
```

### Performance Monitoring

```python
import time
import psutil

def monitor_performance():
    """Monitor system performance during processing."""
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Run your filter here
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {end_memory - start_memory:.2f} MB")

# Usage
monitor_performance()
```

## Common Use Cases

### 1. Food Quality Control
- Annotate ingredients in food products
- Detect contaminants or foreign objects
- Monitor food preparation processes

### 2. Medical Imaging
- Analyze X-rays, CT scans, MRIs
- Detect tumors, lesions, abnormalities
- Assist in diagnostic processes

### 3. Industrial Quality Control
- Inspect manufactured products
- Detect defects or anomalies
- Monitor production lines

### 4. Pet and Animal Classification
- Classify different animal species
- Detect pets in images
- Monitor animal behavior

### 5. Retail and E-commerce
- Product recognition and classification
- Inventory management
- Visual search capabilities

## Best Practices Summary

1. **Start Simple**: Begin with basic classification before adding bounding boxes
2. **Test Prompts**: Validate prompts with sample images before full processing
3. **Monitor Costs**: Use appropriate image sizes and model settings
4. **Quality Check**: Review generated annotations for accuracy
5. **Error Handling**: Implement proper error handling and retry logic
6. **Documentation**: Keep track of prompt versions and results
7. **Validation**: Use no-ops mode for testing and validation
8. **Performance**: Monitor processing time and memory usage
