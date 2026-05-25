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

from filter_chattag.filter import (
    FilterChatTag,
    FilterChatTagConfig,
    CHATTAG_META_KEY,
    CHATTAG_OUTPUT_SCHEMA_VERSION,
)
from openfilter.filter_runtime.filter import Frame

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv('LOG_LEVEL') or 'INFO').upper())))

VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class TestFilterChatTag(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.prompt_file = os.path.join(self.temp_dir, "test_prompt.txt")
        with open(self.prompt_file, 'w') as f:
            f.write("Test prompt for image analysis")

        self.config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file,
            output_schema={
                "item1": {"present": False, "confidence": 0.0},
                "item2": {"present": False, "confidence": 0.0}
            },
            save_frames=False,
            no_ops=True,
            forward_main=False
        )

        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        self.test_frame = Frame(
            image=self.test_image,
            data={"meta": {"id": "test_frame_001"}},
            format="BGR"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_validation(self):
        """Missing model string and missing prompt should raise."""
        config = FilterChatTagConfig(
            chattag_model="",
            prompt=self.prompt_file
        )

        with self.assertRaises(ValueError) as context:
            FilterChatTag.normalize_config(config)

        self.assertIn("chattag_model is required", str(context.exception))

        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt="/nonexistent/prompt.txt"
        )

        with self.assertRaises(FileNotFoundError) as context:
            FilterChatTag.normalize_config(config)

        self.assertIn("Prompt file not found", str(context.exception))

    def test_no_ops_mode(self):
        filter_instance = FilterChatTag(self.config)
        filter_instance.setup(self.config)

        frames = {"test_frame": self.test_frame}
        result = filter_instance.process(frames)

        self.assertIn("test_frame", result)
        result_frame = result["test_frame"]

        self.assertIn("meta", result_frame.data)
        self.assertIn(CHATTAG_META_KEY, result_frame.data["meta"])

        meta_ann = result_frame.data["meta"][CHATTAG_META_KEY]
        self.assertEqual(meta_ann["schema_version"], CHATTAG_OUTPUT_SCHEMA_VERSION)

        annotations = meta_ann["annotations"]
        self.assertEqual(annotations["item1"]["present"], False)
        self.assertEqual(annotations["item1"]["confidence"], 0.0)
        self.assertEqual(annotations["item2"]["present"], False)
        self.assertEqual(annotations["item2"]["confidence"], 0.0)

        usage = meta_ann["usage"]
        self.assertEqual(usage["total_tokens"], 0)

    def test_forward_main_functionality(self):
        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            forward_main=True
        )

        filter_instance = FilterChatTag(config)
        filter_instance.setup(config)

        main_frame = Frame(
            image=self.test_image,
            data={"meta": {"id": "main_frame"}},
            format="BGR"
        )

        frames = {
            "main": main_frame,
            "other_topic": self.test_frame
        }

        result = filter_instance.process(frames)

        result_keys = list(result.keys())
        self.assertEqual(result_keys[0], "main")

        main_result = result["main"]
        self.assertEqual(main_result.data["meta"]["id"], "main_frame")

        self.assertIn("other_topic", result)
        other_result = result["other_topic"]
        self.assertIn(CHATTAG_META_KEY, other_result.data["meta"])

    def test_validate_annotations(self):
        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file
        )
        filter_instance = FilterChatTag(config)
        filter_instance.output_schema = {
            "item1": {"present": False, "confidence": 0.0},
            "item2": {"present": False, "confidence": 0.0}
        }

        valid_annotations = {
            "item1": {"present": True, "confidence": 0.9},
            "item2": {"present": False, "confidence": 0.1}
        }

        result = filter_instance._validate_annotations(valid_annotations)
        self.assertEqual(result["item1"]["present"], True)
        self.assertEqual(result["item1"]["confidence"], 0.9)
        self.assertEqual(result["item2"]["present"], False)
        self.assertEqual(result["item2"]["confidence"], 0.1)

        boolean_annotations = {
            "item1": True,
            "item2": False
        }

        result = filter_instance._validate_annotations(boolean_annotations)
        self.assertEqual(result["item1"]["present"], True)
        self.assertEqual(result["item1"]["confidence"], 1.0)
        self.assertEqual(result["item2"]["present"], False)
        self.assertEqual(result["item2"]["confidence"], 0.0)

    def test_default_for_schema_key(self):
        d = FilterChatTag._default_for_schema_key

        self.assertEqual(d(None), {"present": False, "confidence": 0.0})
        self.assertEqual(d(0), {"present": False, "confidence": 0.0})
        self.assertEqual(d("x"), {"present": False, "confidence": 0.0})
        self.assertEqual(d([]), {"present": False, "confidence": 0.0})

        self.assertEqual(d({}), {"present": False, "confidence": 0.0})

        legacy = {
            "present": True,
            "confidence": 0.5,
            "bbox": None,
            "extra": "ignored",
        }
        out = d(legacy)
        self.assertEqual(out, {"present": True, "confidence": 0.5})
        self.assertEqual(set(out.keys()), {"present", "confidence"})

    def test_get_default_annotations(self):
        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file
        )
        filter_instance = FilterChatTag(config)
        filter_instance.output_schema = {
            "item1": {"present": False, "confidence": 0.0},
            "item2": {"present": False, "confidence": 0.0}
        }

        defaults = filter_instance._get_default_annotations()
        self.assertEqual(defaults["item1"]["present"], False)
        self.assertEqual(defaults["item1"]["confidence"], 0.0)
        self.assertEqual(defaults["item2"]["present"], False)
        self.assertEqual(defaults["item2"]["confidence"], 0.0)

        filter_instance.output_schema = {
            "avocado": {"present": False, "confidence": 0.0, "bbox": None},
        }
        defaults = filter_instance._get_default_annotations()
        self.assertEqual(defaults["avocado"], {"present": False, "confidence": 0.0})
        self.assertNotIn("bbox", defaults["avocado"])

        filter_instance.output_schema = None
        defaults = filter_instance._get_default_annotations()
        self.assertEqual(defaults, {})

    def test_topic_filtering(self):
        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            topic_pattern="test_.*"
        )

        filter_instance = FilterChatTag(config)
        filter_instance.setup(config)

        frames = {
            "test_frame": self.test_frame,
            "other_frame": Frame(
                image=self.test_image,
                data={"meta": {"id": "other_frame"}},
                format="BGR"
            )
        }

        result = filter_instance.process(frames)

        self.assertIn("test_frame", result)
        self.assertNotIn("other_frame", result)

        test_result = result["test_frame"]
        self.assertIn(CHATTAG_META_KEY, test_result.data["meta"])

    def test_exclude_topics(self):
        config = FilterChatTagConfig(
            chattag_model="openai:gpt-4o-mini",
            prompt=self.prompt_file,
            output_schema={"item1": {"present": False, "confidence": 0.0}},
            save_frames=False,
            no_ops=True,
            exclude_topics=["excluded_frame"]
        )

        filter_instance = FilterChatTag(config)
        filter_instance.setup(config)

        frames = {
            "test_frame": self.test_frame,
            "excluded_frame": Frame(
                image=self.test_image,
                data={"meta": {"id": "excluded_frame"}},
                format="BGR"
            )
        }

        result = filter_instance.process(frames)

        self.assertIn("test_frame", result)
        self.assertNotIn("excluded_frame", result)

        test_result = result["test_frame"]
        self.assertIn(CHATTAG_META_KEY, test_result.data["meta"])

    def test_environment_variables(self):
        os.environ["FILTER_CHATTAG_MODEL"] = "anthropic:claude-3-5-sonnet-latest"
        os.environ["FILTER_FORWARD_MAIN"] = "true"
        os.environ["FILTER_NO_OPS"] = "true"

        try:
            config = FilterChatTagConfig(
                prompt=self.prompt_file,
                output_schema={"item1": {"present": False, "confidence": 0.0}}
            )

            normalized_config = FilterChatTag.normalize_config(config)

            self.assertEqual(normalized_config.chattag_model, "anthropic:claude-3-5-sonnet-latest")
            self.assertEqual(normalized_config.forward_main, True)
            self.assertEqual(normalized_config.no_ops, True)

        finally:
            for key in ["FILTER_CHATTAG_MODEL", "FILTER_FORWARD_MAIN", "FILTER_NO_OPS"]:
                if key in os.environ:
                    del os.environ[key]

    def test_build_schema_returns_pydantic_model(self):
        """_build_schema generates a Pydantic model with a field per label."""
        schema = FilterChatTag._build_schema({
            "cat": {"present": False, "confidence": 0.0},
            "dog": {"present": False, "confidence": 0.0},
        })

        self.assertIsNotNone(schema)
        instance = schema(
            cat={"present": True, "confidence": 0.9},
            dog={"present": False, "confidence": 0.1},
        )
        dumped = instance.model_dump()
        self.assertEqual(dumped["cat"]["present"], True)
        self.assertEqual(dumped["cat"]["confidence"], 0.9)
        self.assertEqual(dumped["dog"]["present"], False)

    def test_build_schema_rejects_confidence_out_of_range(self):
        """The generated schema enforces 0 ≤ confidence ≤ 1."""
        from pydantic import ValidationError

        schema = FilterChatTag._build_schema({
            "cat": {"present": False, "confidence": 0.0},
        })

        with self.assertRaises(ValidationError):
            schema(cat={"present": True, "confidence": 1.5})

    def test_build_schema_empty_returns_none(self):
        self.assertIsNone(FilterChatTag._build_schema({}))
        self.assertIsNone(FilterChatTag._build_schema(None))

    def test_build_model_dispatch_per_provider(self):
        """_build_model passes the model string through to LangChain's init_chat_model
        and wraps it with structured output when a schema is provided."""
        from filter_chattag import filter as filter_mod

        wrapped_runnable = Mock(name="WrappedRunnable")
        chat_model = Mock(name="ChatModel")
        chat_model.with_structured_output.return_value = wrapped_runnable

        with patch.object(filter_mod, "init_chat_model", create=True, return_value=chat_model) as mock_init:
            # Patch where init_chat_model is imported inside _build_model
            with patch("langchain.chat_models.init_chat_model", return_value=chat_model) as mock_init2:
                schema = FilterChatTag._build_schema({"x": {"present": False, "confidence": 0.0}})
                config = FilterChatTagConfig(
                    chattag_model="google_genai:gemini-2.0-flash",
                    prompt=self.prompt_file,
                    max_tokens=500,
                    temperature=0.2,
                )

                runnable = FilterChatTag._build_model(config, schema)

                mock_init2.assert_called_once_with(
                    "google_genai:gemini-2.0-flash",
                    max_tokens=500,
                    temperature=0.2,
                )
                chat_model.with_structured_output.assert_called_once()
                # include_raw=True is required so we can read usage_metadata off the raw AIMessage
                _, kwargs = chat_model.with_structured_output.call_args
                self.assertTrue(kwargs.get("include_raw"))
                self.assertIs(runnable, wrapped_runnable)


try:
    multiprocessing.set_start_method('spawn')  # CUDA doesn't like fork()
except Exception:
    pass

if __name__ == '__main__':
    unittest.main()
