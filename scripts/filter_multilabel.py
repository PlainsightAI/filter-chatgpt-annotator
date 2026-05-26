#!/usr/bin/env python

"""
Example script for multilabel classification (e.g. avocado, fish) with FilterChatTag.

Required environment variables in .env file:
    FILTER_CHATTAG_MODEL: LangChain model string (e.g. "openai:gpt-4o-mini")
    OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST: provider credential
    FILTER_PROMPT: Path to the prompt file
    VIDEO_PATH: Path to the input video file

Output when saving frames:
    - labels.jsonl and binary_datasets/ (and balanced variants)
    - COCO multilabel export under multilabel_datasets/ when multiple labels are configured
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
    prompt_path = os.getenv('FILTER_PROMPT', './prompts/salad_prompt_multilabel.txt')
    chattag_model = os.getenv('FILTER_CHATTAG_MODEL', 'openai:gpt-4o-mini')

    if not video_path:
        print("Error: VIDEO_PATH environment variable is required")
        print("Please set the path to your input video in the .env file")
        exit(1)

    input_source = (VideoIn, dict(
        sources=f'file://{video_path}!sync!no-loop;main',
        outputs='tcp://*:5550',
    ))
    print(f"Using VideoIn with path: {video_path} (no loop, sync)")

    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output_frames')
    save_frames = os.getenv('FILTER_SAVE_FRAMES', 'true').lower() == 'true'
    if save_frames:
        print(f"Results will be saved to: {output_dir}")
        print("Binary classification datasets will be generated")
        print("COCO format multilabel dataset will be generated (multilabel_datasets/)")
    else:
        print("Results will only be shown in web interface (not saved to files)")

    Filter.run_multi([
        input_source,
        (FilterChatTag, FilterChatTagConfig(
            id="filter_chattag_avocado_fish_multilabel",
            sources="tcp://localhost:5550",
            outputs="tcp://*:5552",
            chattag_model=chattag_model,
            prompt=prompt_path,
            max_image_size=0,
            image_quality=98,
            preserve_original_format=True,
            confidence_threshold=float(os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.7')),
            output_schema={
                "avocado": {"present": False, "confidence": 0.0},
                "fish": {"present": False, "confidence": 0.0},
                "chicken": {"present": False, "confidence": 0.0},
            }
        )),
        (Webvis, dict(
            sources='tcp://localhost:5552',
        )),
    ])
