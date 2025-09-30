#!/usr/bin/env python

"""
Example script for running food annotation with bounding boxes using FilterChatgptAnnotator.

This script demonstrates how to use the FilterChatgptAnnotator with food annotation
prompts that include bounding box coordinates for object detection tasks.

Required environment variables in .env file:
    FILTER_CHATGPT_API_KEY: OpenAI API key
    FILTER_PROMPT: Path to the prompt file (should include bbox)
    VIDEO_PATH: Path to the input video file (for video processing)
    IMAGE_PATH: Path to image file or directory (for image processing)
    # Task type is auto-detected based on bbox presence in output_schema

Example .env file content:
    FILTER_CHATGPT_API_KEY=your_openai_api_key_here
    FILTER_PROMPT=./prompts/food_annotation_prompt_bb.txt
    VIDEO_PATH=/path/to/your/video.mp4
    IMAGE_PATH=/path/to/your/images
    FILTER_CHATGPT_MODEL=gpt-4o-mini
    FILTER_MAX_TOKENS=1000
    FILTER_TEMPERATURE=0.1
    FILTER_MAX_IMAGE_SIZE=512
    FILTER_SAVE_FRAMES=true
    FILTER_RECURSIVE=false
    FILTER_CONFIDENCE_THRESHOLD=0.9

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
from openfilter.filter_runtime.filters.image_in import ImageIn


if __name__ == '__main__':
    # Get video path from environment variable or use default
    video_path = os.getenv('VIDEO_PATH', '')

    # Get image path from environment variable (can be file or directory)
    image_path = os.getenv('IMAGE_PATH', '')
    
    # Get prompt path from environment variable
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/simple_salad_prompt_bb_v3.txt')
    
    # Get API key from environment variable
    api_key = os.getenv('FILTER_CHATGPT_API_KEY', '')
    
    if not api_key:
        print("Error: FILTER_CHATGPT_API_KEY environment variable is required")
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    # Check if we have either video or image path
    if not video_path and not image_path:
        print("Error: Either VIDEO_PATH or IMAGE_PATH environment variable is required")
        print("Please set the path to your input video or image directory in the .env file")
        exit(1)

    # Determine which input source to use
    if image_path:
        # Use ImageIn for image processing
        input_source = (ImageIn, dict(
            sources=f'file://{image_path}',  # Process images once, no loop, topic: main
            outputs='tcp://*:5550',
            poll_interval=0,  # No continuous polling
        ))
        print(f"Using ImageIn with path: {image_path} (no loop, no polling)")
    else:
        # Use VideoIn for video processing
        input_source = (VideoIn, dict(
            sources=f'file://{video_path}!resize=960x540!sync!no-loop;main',  # Process video once, no loop, topic: main
            outputs='tcp://*:5550',
        ))
        print(f"Using VideoIn with path: {video_path} (no loop)")
    
    # Show output configuration
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output_frames')
    save_frames = os.getenv('FILTER_SAVE_FRAMES', 'true').lower() == 'true'
    no_ops = os.getenv('FILTER_NO_OPS', 'false').lower() == 'true'
    if save_frames:
        print(f"Results will be saved to: {output_dir}")
        print("📊 Binary classification datasets will be generated")
        print("📦 COCO format detection dataset will be generated (if bbox schema present)")
    else:
        print("Results will only be shown in web interface (not saved to files)")
    if no_ops:
        print("⚠️  NO-OPS MODE ENABLED - API calls will be skipped (testing mode)")

    Filter.run_multi([
        input_source,
        (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
            id="filter_chatgpt_food_annotation_bb",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chatgpt_api_key=api_key,
            prompt=prompt_path,
            # Task type is auto-detected based on bbox presence in output_schema
            # FILTER_MAX_IMAGE_SIZE=512, # Resize image to 512px
            # Set output schema for food ingredients with bounding boxes
            output_schema={
                "avocado": {"present": False, "confidence": 0.0, "bbox": None},
                "lettuce": {"present": False, "confidence": 0.0, "bbox": None},
                "tomato": {"present": False, "confidence": 0.0, "bbox": None},
            }
        )),
        (Webvis, dict(
            # sources='tcp://localhost:5550',
            sources='tcp://localhost:5552',  # Main stream with annotations
        )),
    ])
