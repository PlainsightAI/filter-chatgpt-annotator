#!/usr/bin/env python

"""
Example script for running pet classification using FilterChatTag (LangChain-backed).

Required environment variables in .env file:
    FILTER_CHATTAG_MODEL: LangChain model string (e.g. "openai:gpt-4o-mini")
    OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST: provider credential
    FILTER_PROMPT: Path to the prompt file
    VIDEO_PATH: Path to the input video file
"""

import os
from dotenv import load_dotenv

load_dotenv()

from openfilter.filter_runtime.filter import Filter
from filter_chattag.filter import FilterChatTag, FilterChatTagConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis


if __name__ == '__main__':
    video_path = os.getenv('VIDEO_PATH', '')
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/pet_classification_prompt.txt')
    chattag_model = os.getenv('FILTER_CHATTAG_MODEL', 'openai:gpt-4o-mini')

    if not video_path:
        print("Error: VIDEO_PATH environment variable is required")
        print("Please set the path to your input video in the .env file")
        exit(1)

    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output_frames')
    save_frames = os.getenv('FILTER_SAVE_FRAMES', 'true').lower() == 'true'
    no_ops = os.getenv('FILTER_NO_OPS', 'false').lower() == 'true'
    if save_frames:
        print(f"Results will be saved to: {output_dir}")
        print("Binary classification datasets will be generated")
    else:
        print("Results will only be shown in web interface (not saved to files)")
    if no_ops:
        print("NO-OPS MODE ENABLED - LLM calls will be skipped (testing mode)")

    Filter.run_multi([
        (VideoIn, dict(
            sources=f'file://{video_path}!resize=960x540!sync!no-loop;main',
            outputs='tcp://*:5550',
        )),
        (FilterChatTag, FilterChatTagConfig(
            id="filter_chattag_pet_classification",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chattag_model=chattag_model,
            prompt=prompt_path,
            max_tokens=int(os.getenv('FILTER_MAX_TOKENS', '500')),
            temperature=float(os.getenv('FILTER_TEMPERATURE', '0.1')),
            max_image_size=int(os.getenv('FILTER_MAX_IMAGE_SIZE', '512')),
            image_quality=int(os.getenv('FILTER_IMAGE_QUALITY', '85')),
            save_frames=save_frames,
            output_dir=os.getenv('FILTER_OUTPUT_DIR', './output_frames'),
            output_schema={
                "cat": {"present": False, "confidence": 0.0},
                "dog": {"present": False, "confidence": 0.0}
            }
        )),
        (Webvis, dict(
            sources='tcp://localhost:5552',
        )),
    ])
