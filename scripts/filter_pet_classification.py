#!/usr/bin/env python

"""
Example script for running pet classification using FilterChatgptAnnotator.

This script demonstrates how to use the FilterChatgptAnnotator with pet classification
prompts in a complete pipeline. Accepts either a video file (VIDEO_PATH) or an
image file/directory (IMAGE_PATH).

Required environment variables in .env file:
    FILTER_CHATGPT_API_KEY: OpenAI API key
    FILTER_PROMPT: Path to the prompt file
    VIDEO_PATH or IMAGE_PATH: Path to a video file or to an image file/directory

Example .env file content:
    FILTER_CHATGPT_API_KEY=your_openai_api_key_here
    FILTER_PROMPT=./prompts/pet_classification_prompt.txt
    IMAGE_PATH=/path/to/pet_image.jpg     # or VIDEO_PATH=/path/to/pets.mp4
    FILTER_CHATGPT_MODEL=gpt-4o-mini
    FILTER_MAX_TOKENS=500
    FILTER_TEMPERATURE=0.1
    FILTER_MAX_IMAGE_SIZE=512
"""

import os
from dotenv import load_dotenv

load_dotenv()

from openfilter.filter_runtime.filter import Filter
from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.image_in import ImageIn


if __name__ == '__main__':
    video_path = os.getenv('VIDEO_PATH', '')
    image_path = os.getenv('IMAGE_PATH', '')
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/pet_classification_prompt.txt')
    api_key = os.getenv('FILTER_CHATGPT_API_KEY', '')

    if not api_key:
        print("Error: FILTER_CHATGPT_API_KEY environment variable is required")
        print("Please set your OpenAI API key in the .env file")
        exit(1)

    if not video_path and not image_path:
        print("Error: Either VIDEO_PATH or IMAGE_PATH environment variable is required")
        print("Please set the path to your input video or image in the .env file")
        exit(1)

    if image_path:
        input_source = (ImageIn, dict(
            sources=f'file://{image_path}!maxfps=2!no-loop;main',
            outputs='tcp://*:5550',
            poll_interval=0,
        ))
        print(f"Using ImageIn with path: {image_path} (no loop, no polling)")
    else:
        input_source = (VideoIn, dict(
            sources=f'file://{video_path}!resize=960x540!sync!no-loop;main',
            outputs='tcp://*:5550',
        ))
        print(f"Using VideoIn with path: {video_path} (no loop, sync)")

    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output_frames')
    save_frames = os.getenv('FILTER_SAVE_FRAMES', 'true').lower() == 'true'
    no_ops = os.getenv('FILTER_NO_OPS', 'false').lower() == 'true'
    if save_frames:
        print(f"Results will be saved to: {output_dir}")
        print("Binary classification datasets will be generated")
    else:
        print("Results will only be shown in web interface (not saved to files)")
    if no_ops:
        print("NO-OPS MODE ENABLED - API calls will be skipped (testing mode)")

    Filter.run_multi([
        input_source,
        (FilterChatgptAnnotator, FilterChatgptAnnotatorConfig(
            id="filter_chatgpt_pet_classification",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chatgpt_api_key=api_key,
            prompt=prompt_path,
            chatgpt_model=os.getenv('FILTER_CHATGPT_MODEL', 'gpt-4o-mini'),
            max_tokens=int(os.getenv('FILTER_MAX_TOKENS', '500')),
            temperature=float(os.getenv('FILTER_TEMPERATURE', '0.1')),
            max_image_size=int(os.getenv('FILTER_MAX_IMAGE_SIZE', '512')),
            image_quality=int(os.getenv('FILTER_IMAGE_QUALITY', '85')),
            save_frames=save_frames,
            output_dir=output_dir,
            no_ops=no_ops,
            output_schema={
                "cat": {"present": False, "confidence": 0.0},
                "dog": {"present": False, "confidence": 0.0},
            },
        )),
        (Webvis, dict(
            sources='tcp://localhost:5552',
        )),
    ])
