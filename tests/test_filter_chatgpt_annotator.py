#!/usr/bin/env python

import json
import logging
import multiprocessing
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from filter_chatgpt_annotator.filter import FilterChatgptAnnotator, FilterChatgptAnnotatorConfig
from openfilter.filter_runtime.filter import Frame

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv('LOG_LEVEL') or 'INFO').upper())))

VERBOSE   = '-v' in sys.argv or '--verbose' in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class TestFilterChatgptAnnotator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a temporary prompt file
        self.prompt_file = os.path.join(self.temp_dir, "test_prompt.txt")
        with open(self.prompt_file, 'w') as f:
            f.write("Test prompt for image analysis")
        
        # Create test configuration
        self.config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-api-key",
            prompt=self.prompt_file,
            output_schema={
                "item1": {"present": False, "confidence": 0.0},
                "item2": {"present": False, "confidence": 0.0}
            },
            save_frames=False,  # Don't save files during tests
            no_ops=True,  # Use no-ops mode to avoid API calls
            forward_main=False
        )
        
        # Create a test image (simple RGB array)
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create test frame
        self.test_frame = Frame(
            image=self.test_image,
            data={"meta": {"id": "test_frame_001"}},
            format="BGR"
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test missing API key
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="",  # Empty API key
            prompt=self.prompt_file
        )
        
        with self.assertRaises(ValueError) as context:
            FilterChatgptAnnotator.normalize_config(config)
        
        self.assertIn("chatgpt_api_key is required", str(context.exception))
        
        # Test missing prompt file
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-api-key",
            prompt="/nonexistent/prompt.txt"
        )
        
        with self.assertRaises(FileNotFoundError) as context:
            FilterChatgptAnnotator.normalize_config(config)
        
        self.assertIn("Prompt file not found", str(context.exception))
    
    def test_no_ops_mode(self):
        """Test no-ops mode functionality."""
        # Create and setup filter
        filter_instance = FilterChatgptAnnotator(self.config)
        filter_instance.setup(self.config)
        
        # Process frame
        frames = {"test_frame": self.test_frame}
        result = filter_instance.process(frames)
        
        # Verify result structure
        self.assertIn("test_frame", result)
        result_frame = result["test_frame"]
        
        # Verify metadata was added
        self.assertIn("meta", result_frame.data)
        self.assertIn("chatgpt_annotator", result_frame.data["meta"])
        
        # Verify result uses default annotations
        annotations = result_frame.data["meta"]["chatgpt_annotator"]["annotations"]
        self.assertEqual(annotations["item1"]["present"], False)
        self.assertEqual(annotations["item1"]["confidence"], 0.0)
        self.assertEqual(annotations["item2"]["present"], False)
        self.assertEqual(annotations["item2"]["confidence"], 0.0)
        
        # Verify usage shows zero tokens
        usage = result_frame.data["meta"]["chatgpt_annotator"]["usage"]
        self.assertEqual(usage["total_tokens"], 0)
    
    def test_forward_main_functionality(self):
        """Test forward_main functionality."""
        # Create config with forward_main=True
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-api-key",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            forward_main=True
        )
        
        # Create and setup filter
        filter_instance = FilterChatgptAnnotator(config)
        filter_instance.setup(config)
        
        # Create frames with main topic
        main_frame = Frame(
            image=self.test_image,
            data={"meta": {"id": "main_frame"}},
            format="BGR"
        )
        
        frames = {
            "main": main_frame,
            "other_topic": self.test_frame
        }
        
        # Process frames
        result = filter_instance.process(frames)
        
        # Verify main topic is preserved and comes first
        result_keys = list(result.keys())
        self.assertEqual(result_keys[0], "main")
        
        # Verify main frame data is preserved
        main_result = result["main"]
        self.assertEqual(main_result.data["meta"]["id"], "main_frame")
        
        # Verify other topic was processed
        self.assertIn("other_topic", result)
        other_result = result["other_topic"]
        self.assertIn("chatgpt_annotator", other_result.data["meta"])
    
    def test_validate_annotations(self):
        """Test annotation validation logic."""
        # Create filter instance with minimal config
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-key",
            prompt=self.prompt_file
        )
        filter_instance = FilterChatgptAnnotator(config)
        filter_instance.output_schema = {
            "item1": {"present": False, "confidence": 0.0},
            "item2": {"present": False, "confidence": 0.0}
        }
        # Set has_bbox_schema since we're not calling setup()
        filter_instance.has_bbox_schema = False
        
        # Test with valid annotations
        valid_annotations = {
            "item1": {"present": True, "confidence": 0.9},
            "item2": {"present": False, "confidence": 0.1}
        }
        
        result = filter_instance._validate_annotations(valid_annotations)
        self.assertEqual(result["item1"]["present"], True)
        self.assertEqual(result["item1"]["confidence"], 0.9)
        self.assertEqual(result["item2"]["present"], False)
        self.assertEqual(result["item2"]["confidence"], 0.1)
        
        # Test with boolean annotations (should be converted)
        boolean_annotations = {
            "item1": True,
            "item2": False
        }
        
        result = filter_instance._validate_annotations(boolean_annotations)
        self.assertEqual(result["item1"]["present"], True)
        self.assertEqual(result["item1"]["confidence"], 1.0)
        self.assertEqual(result["item2"]["present"], False)
        self.assertEqual(result["item2"]["confidence"], 0.0)
    
    def test_get_default_annotations(self):
        """Test default annotations generation."""
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-key",
            prompt=self.prompt_file
        )
        filter_instance = FilterChatgptAnnotator(config)
        filter_instance.output_schema = {
            "item1": {"present": False, "confidence": 0.0},
            "item2": {"present": False, "confidence": 0.0}
        }
        
        defaults = filter_instance._get_default_annotations()
        self.assertEqual(defaults["item1"]["present"], False)
        self.assertEqual(defaults["item1"]["confidence"], 0.0)
        self.assertEqual(defaults["item2"]["present"], False)
        self.assertEqual(defaults["item2"]["confidence"], 0.0)
        
        # Test without schema
        filter_instance.output_schema = None
        defaults = filter_instance._get_default_annotations()
        self.assertEqual(defaults, {})
    
    def test_topic_filtering(self):
        """Test topic filtering functionality."""
        # Create config with topic pattern
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-api-key",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            topic_pattern="test_.*"  # Only process topics starting with "test_"
        )
        
        # Create and setup filter
        filter_instance = FilterChatgptAnnotator(config)
        filter_instance.setup(config)
        
        # Create frames with different topic names
        frames = {
            "test_frame": self.test_frame,
            "other_frame": Frame(
                image=self.test_image,
                data={"meta": {"id": "other_frame"}},
                format="BGR"
            )
        }
        
        # Process frames
        result = filter_instance.process(frames)
        
        # Verify only test_frame was processed
        self.assertIn("test_frame", result)
        self.assertNotIn("other_frame", result)
        
        # Verify test_frame was processed
        test_result = result["test_frame"]
        self.assertIn("chatgpt_annotator", test_result.data["meta"])
    
    def test_exclude_topics(self):
        """Test exclude topics functionality."""
        # Create config with exclude topics
        config = FilterChatgptAnnotatorConfig(
            chatgpt_api_key="test-api-key",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            exclude_topics=["excluded_frame"]
        )
        
        # Create and setup filter
        filter_instance = FilterChatgptAnnotator(config)
        filter_instance.setup(config)
        
        # Create frames with different topic names
        frames = {
            "test_frame": self.test_frame,
            "excluded_frame": Frame(
                image=self.test_image,
                data={"meta": {"id": "excluded_frame"}},
                format="BGR"
            )
        }
        
        # Process frames
        result = filter_instance.process(frames)
        
        # Verify excluded_frame was not processed
        self.assertIn("test_frame", result)
        self.assertNotIn("excluded_frame", result)
        
        # Verify test_frame was processed
        test_result = result["test_frame"]
        self.assertIn("chatgpt_annotator", test_result.data["meta"])
    
    def test_environment_variables(self):
        """Test environment variable configuration."""
        # Set environment variables
        os.environ["FILTER_CHATGPT_API_KEY"] = "env-api-key"
        os.environ["FILTER_FORWARD_MAIN"] = "true"
        os.environ["FILTER_NO_OPS"] = "true"
        
        try:
            # Create config without explicit values
            config = FilterChatgptAnnotatorConfig(
                prompt=self.prompt_file,
                output_schema={"item1": {"present": False, "confidence": 0.0}}
            )
            
            # Normalize config (this should pick up env vars)
            normalized_config = FilterChatgptAnnotator.normalize_config(config)
            
            # Verify environment variables were applied
            self.assertEqual(normalized_config.chatgpt_api_key, "env-api-key")
            self.assertEqual(normalized_config.forward_main, True)
            self.assertEqual(normalized_config.no_ops, True)
            
        finally:
            # Clean up environment variables
            for key in ["FILTER_CHATGPT_API_KEY", "FILTER_FORWARD_MAIN", "FILTER_NO_OPS"]:
                if key in os.environ:
                    del os.environ[key]


try:
    multiprocessing.set_start_method('spawn')  # CUDA doesn't like fork()
except Exception:
    pass

if __name__ == '__main__':
    unittest.main()