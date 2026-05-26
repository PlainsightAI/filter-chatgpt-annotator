#!/usr/bin/env python

"""
Example script for running food annotation using FilterChatTag (LangChain-backed).

Required environment variables in .env file:
    FILTER_CHATTAG_MODEL: LangChain model string (e.g. "openai:gpt-4o-mini")
    OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST: provider credential
    FILTER_PROMPT: Path to the prompt file
    VIDEO_PATH: Path to the input video file (for video processing)
    IMAGE_PATH: Path to image file or directory (for image processing)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from openfilter.filter_runtime.filter import Filter
from filter_chattag.filter import FilterChatTag, FilterChatTagConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.image_in import ImageIn


if __name__ == '__main__':
    video_path = os.getenv('VIDEO_PATH', '')
    image_path = os.getenv('IMAGE_PATH', '')
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/food_annotation_prompt.txt')
    chattag_model = os.getenv('FILTER_CHATTAG_MODEL', 'openai:gpt-4o-mini')

    if not video_path and not image_path:
        print("Error: Either VIDEO_PATH or IMAGE_PATH environment variable is required")
        print("Please set the path to your input video or image directory in the .env file")
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
    else:
        print("Results will only be shown in web interface (not saved to files)")
    if no_ops:
        print("NO-OPS MODE ENABLED - LLM calls will be skipped (testing mode)")

    Filter.run_multi([
        input_source,
        (FilterChatTag, FilterChatTagConfig(
            id="filter_chattag_food_annotation",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chattag_model=chattag_model,
            prompt=prompt_path,
            output_schema={
                "avocado": {"present": False, "confidence": 0.0},
                "lettuce": {"present": False, "confidence": 0.0},
                "tomato": {"present": False, "confidence": 0.0},
            }
        )),
        (Webvis, dict(
            sources='tcp://localhost:5552',
        )),
    ])
