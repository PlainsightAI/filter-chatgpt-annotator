#!/usr/bin/env python

"""
Example script for running avocado, boiled eggs and fish annotation with bounding boxes using FilterChatgptAnnotator.

This script demonstrates how to use the FilterChatgptAnnotator with avocado, boiled eggs and fish annotation
prompts that include bounding box coordinates for object detection tasks.

Required environment variables in .env file:
    FILTER_CHATGPT_API_KEY: OpenAI API key
    FILTER_PROMPT: Path to the prompt file (should include bbox)
    VIDEO_PATH: Path to the input video file
    # Task type is auto-detected based on bbox presence in output_schema


Example .env file content:
    FILTER_CHATGPT_API_KEY=your_openai_api_key_here
    FILTER_PROMPT=./prompts/salad_prompt_multilabel_class_demo.txt
    VIDEO_PATH=/path/to/your/video.mp4
    FILTER_CHATGPT_MODEL=gpt-4o-mini
    FILTER_MAX_TOKENS=1000
    FILTER_TEMPERATURE=0.1
    FILTER_MAX_IMAGE_SIZE=512
    FILTER_SAVE_FRAMES=true
    FILTER_RECURSIVE=false
    FILTER_CONFIDENCE_THRESHOLD=0.7

Output:
    - Always: Binary datasets in binary_datasets/ and binary_datasets_balanced/
    - If bbox schema present: COCO format dataset in detection_datasets/annotations.json
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from openfilter.filter_runtime.filter import Filter
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis


if __name__ == '__main__':
    # Get video path from environment variable
    video_path = os.getenv('VIDEO_PATH', '')
    
    # Get prompt path from environment variable
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/salad_prompt_multilabel.txt')
    
    # Get API key from environment variable
    api_key = os.getenv('FILTER_CHATGPT_API_KEY', '')
    
    
    if not api_key:
        print("Error: FILTER_CHATGPT_API_KEY environment variable is required")
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    # Check if we have video path
    if not video_path:
        print("Error: VIDEO_PATH environment variable is required")
        print("Please set the path to your input video in the .env file")
        exit(1)

    # Use VideoIn for video processing (processes all frames)
    input_source = (VideoIn, dict(
        sources=f'file://{video_path}!sync!no-loop;main',  # Process video, no loop, topic: main, sync
        outputs='tcp://*:5550',
    ))
    print(f"Using VideoIn with path: {video_path} (no loop, sync)")
    
    # Show output configuration
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output_frames')
    save_frames = os.getenv('FILTER_SAVE_FRAMES', 'true').lower() == 'true'
    if save_frames:
        print(f"Results will be saved to: {output_dir}")
        print("📊 Binary classification datasets will be generated")
        print("📦 COCO format detection dataset will be generated (if bbox schema present)")
    else:
        print("Results will only be shown in web interface (not saved to files)")

    Filter.run_multi([
        input_source,
        (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
            id="filter_chatgpt_avocado_tomato_fish_bb",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chatgpt_api_key=api_key,
            prompt=prompt_path,
            # HIGH QUALITY SETTINGS - Maximum quality for ChatGPT Vision API
            max_image_size=0,  # Keep original size (no downscaling)
            image_quality=98,  # Maximum JPEG quality (98% for best results)
            preserve_original_format=True,  # Preserve original format when possible
            # Task type is auto-detected based on bbox presence in output_schema
            # output_dir=os.getenv('FILTER_OUTPUT_DIR', './output_frames'),
            confidence_threshold=float(os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.7')),
            # Set output schema for avocado and fish with bounding boxes
            output_schema={
                "avocado": {"present": False, "confidence": 0.0, "bbox": None},
                "fish": {"present": False, "confidence": 0.0, "bbox": None},
            }
        )),
        (Webvis, dict(
            # sources='tcp://localhost:5550',
            sources='tcp://localhost:5552',  # Main stream with annotations
        )),
    ])
