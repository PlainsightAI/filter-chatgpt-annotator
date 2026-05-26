import logging
import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, Type
from PIL import Image
import io

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

__all__ = ['FilterChatTagConfig', 'FilterChatTag', 'CHATTAG_OUTPUT_SCHEMA_VERSION']

logger = logging.getLogger(__name__)

# OpenFilter / JSONL output contract (see docs/output_contract.md)
CHATTAG_OUTPUT_SCHEMA_VERSION = "1.0"

# Frame metadata key under frame.data["meta"][...] where ChatTag stores its results.
CHATTAG_META_KEY = "chattag"


class FilterChatTagConfig(FilterConfig):
    # LangChain model string: "provider:model" (e.g. "openai:gpt-4o-mini",
    # "google_genai:gemini-2.0-flash", "anthropic:claude-3-5-sonnet-latest",
    # "ollama:llava"). Credentials come from the provider's native env vars
    # (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, OLLAMA_HOST).
    chattag_model: str = "openai:gpt-4o-mini"

    prompt: str = ""
    output_schema: Dict[str, Any] = {}

    # LLM parameters
    max_tokens: int = 1000
    temperature: float = 0.1

    # Image processing
    max_image_size: int = 0  # 0 = keep original size
    image_quality: int = 95
    preserve_original_format: bool = True

    # Output options
    save_frames: bool = True
    output_dir: str = "./output_frames"

    # Topic filtering
    topic_pattern: str = None
    exclude_topics: list = []

    # Forward main topic to output
    forward_main: bool = False

    # No-ops mode (skip LLM calls for testing)
    no_ops: bool = False

    # Debug metadata logging
    debug_metadata: bool = False

    # Confidence threshold for positive classification (0.0 to 1.0)
    confidence_threshold: float = 0.9


class FilterChatTag(Filter):
    """
    LangChain-powered multi-provider vision annotator.

    Sends each frame to a configurable chat model (OpenAI, Google Gemini,
    Anthropic Claude, or Ollama) and stores structured annotations in frame
    metadata under ``frame.data["meta"]["chattag"]``.

    The model is selected via ``chattag_model`` (env: ``FILTER_CHATTAG_MODEL``)
    using LangChain's ``init_chat_model`` "provider:model" syntax. Output
    structure is enforced via ``with_structured_output(Pydantic)`` so every
    provider returns the same shape.
    """

    @classmethod
    def normalize_config(cls, config: FilterChatTagConfig):
        config = FilterChatTagConfig(super().normalize_config(config))

        env_mapping = {
            "chattag_model": (str, str.strip),
            "prompt": (str, str.strip),
            "max_tokens": (int, lambda x: int(x.strip())),
            "temperature": (float, lambda x: float(x.strip())),
            "max_image_size": (int, lambda x: int(x.strip())),
            "image_quality": (int, lambda x: int(x.strip())),
            "preserve_original_format": (bool, lambda x: x.strip().lower() in ('true', '1', 'yes', 'on')),
            "save_frames": (bool, lambda x: x.strip().lower() == "true"),
            "output_dir": (str, str.strip),
            "topic_pattern": (str, str.strip),
            "forward_main": (bool, lambda x: x.strip().lower() in ('true', '1', 'yes', 'on')),
            "no_ops": (bool, lambda x: x.strip().lower() in ('true', '1', 'yes', 'on')),
            "debug_metadata": (bool, lambda x: x.strip().lower() in ('true', '1', 'yes', 'on')),
            "confidence_threshold": (float, lambda x: float(x.strip())),
        }

        for key, (expected_type, converter) in env_mapping.items():
            env_key = f"FILTER_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    converted_val = converter(env_val)
                    if not isinstance(converted_val, expected_type):
                        raise TypeError(
                            f"Environment variable {env_key} must be of type {expected_type.__name__}"
                        )
                    setattr(config, key, converted_val)
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert environment variable {env_key}: {str(e)}"
                    )

        output_schema_env = os.getenv("FILTER_OUTPUT_SCHEMA")
        if output_schema_env:
            try:
                config.output_schema = json.loads(output_schema_env)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in FILTER_OUTPUT_SCHEMA: {str(e)}")

        exclude_topics_env = os.getenv("FILTER_EXCLUDE_TOPICS")
        if exclude_topics_env:
            config.exclude_topics = [topic.strip() for topic in exclude_topics_env.split(",") if topic.strip()]

        if not config.chattag_model:
            raise ValueError("chattag_model is required (set FILTER_CHATTAG_MODEL, e.g. 'openai:gpt-4o-mini')")

        if not config.prompt:
            raise ValueError("prompt is required (set FILTER_PROMPT)")

        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if config.temperature < 0 or config.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")

        if config.max_image_size < 0:
            raise ValueError("max_image_size must be non-negative (0 = keep original size)")

        if config.image_quality < 1 or config.image_quality > 100:
            raise ValueError("image_quality must be between 1 and 100")

        if config.confidence_threshold < 0.0 or config.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        if config.prompt and not os.path.exists(config.prompt):
            raise FileNotFoundError(f"Prompt file not found: {config.prompt}")

        logger.debug(f"Normalized config: {config}")
        return config

    def setup(self, config: FilterChatTagConfig):
        logger.info("========= Setting up FilterChatTag =========")

        self.config: FilterChatTagConfig = config
        logger.info(f"FilterChatTag config: {config}")

        self.chattag_model = config.chattag_model
        self.model = config.chattag_model  # used in result metadata
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_image_size = config.max_image_size
        self.image_quality = config.image_quality
        self.preserve_original_format = config.preserve_original_format
        self.save_frames = config.save_frames
        self.output_dir = Path(config.output_dir) if config.save_frames else None
        self.no_ops = config.no_ops
        self.output_schema = config.output_schema
        self.confidence_threshold = config.confidence_threshold

        if not self.no_ops:
            self._pydantic_schema = self._build_schema(self.output_schema)
            self._chain = self._build_model(config, self._pydantic_schema)
            logger.info(f"Initialized LangChain chat model: {config.chattag_model}")
        else:
            self._pydantic_schema = None
            self._chain = None
            logger.info("Skipping LLM initialization (no-ops mode)")

        _prompt_path = (getattr(config, "prompt", None) or "").strip()
        self.prompt_filename = os.path.basename(_prompt_path) if _prompt_path else ""

        try:
            with open(config.prompt, 'r', encoding='utf-8') as f:
                self.prompt_text = f.read().strip()
            logger.debug(f"Loaded prompt from: {config.prompt}")
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt file: {str(e)}")

        if self.save_frames and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory: {self.output_dir}")

        self.topic_pattern = config.topic_pattern
        self.exclude_topics = config.exclude_topics
        self.forward_main = config.forward_main

        if self.topic_pattern:
            try:
                import re
                self.topic_regex = re.compile(self.topic_pattern)
                logger.info(f"Using topic pattern: {self.topic_pattern}")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{self.topic_pattern}': {e}")
                raise ValueError(f"Invalid regex pattern: {e}")
        else:
            self.topic_regex = None
            logger.debug("No topic pattern specified, will process all topics")

        logger.info("FilterChatTag setup complete.")

    @staticmethod
    def _build_schema(output_schema: Dict[str, Any]) -> Optional[Type]:
        """
        Build a Pydantic model from the user-supplied output_schema dict.

        Input shape: ``{"label_name": {"present": False, "confidence": 0.0}, ...}``.
        Each top-level key becomes a ``LabelAnnotation`` field (``present: bool``,
        ``confidence: float`` ∈ [0, 1]). Passed to ``with_structured_output`` so
        every provider returns the same shape.
        """
        if not output_schema:
            return None

        from pydantic import BaseModel, Field, create_model

        class LabelAnnotation(BaseModel):
            present: bool = Field(description="Whether the label is present in the image")
            confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")

        fields = {
            label: (
                LabelAnnotation,
                Field(description=f"Annotation for label '{label}'"),
            )
            for label in output_schema.keys()
        }

        return create_model("ChatTagAnnotations", **fields)

    @staticmethod
    def _build_model(config: FilterChatTagConfig, schema: Optional[Type]):
        """
        Build the LangChain chat model and wrap it with structured output.

        With a schema, returns a Runnable that yields
        ``{"raw": AIMessage, "parsed": <pydantic>, "parsing_error": ...}``
        (via ``include_raw=True``) so usage metadata is preserved alongside
        the parsed annotations.
        """
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as e:
            raise ImportError(
                "langchain is required. Install with: pip install langchain langchain-openai "
                "langchain-google-genai langchain-anthropic langchain-ollama"
            ) from e

        model = init_chat_model(
            config.chattag_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        if schema is not None:
            return model.with_structured_output(schema, include_raw=True)
        return model

    def process(self, frames: dict[str, Frame]):
        """
        Process frames through the configured LangChain chat model.

        Returns processed frames with annotation results stored under
        ``frame.data["meta"]["chattag"]``.
        """
        processed_frames = {}

        logger.debug(f"PROCESS CALL: Received {len(frames)} frames with keys: {list(frames.keys())}")

        if not hasattr(self, '_total_frames_processed'):
            self._total_frames_processed = 0

        if self.config.debug_metadata:
            debug_dir = Path(self.output_dir) / "debug" if self.save_frames else Path("./debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            debug_file = debug_dir / f"frames_received_{int(time.time())}.txt"
            with open(debug_file, 'w') as f:
                f.write(f"PROCESS CALL DEBUG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total frames received: {len(frames)}\n")
                f.write(f"Frame IDs: {list(frames.keys())}\n")
                f.write("="*50 + "\n")
                for frame_id, frame in frames.items():
                    f.write(f"Frame ID: {frame_id}\n")
                    f.write(f"Frame data keys: {list(frame.data.keys()) if hasattr(frame.data, 'keys') else 'No data keys'}\n")
                    if hasattr(frame.data, 'get') and 'meta' in frame.data:
                        f.write(f"Frame meta: {frame.data.get('meta', {})}\n")
                    f.write(f"Image shape: {frame.rw_bgr.image.shape if hasattr(frame.rw_bgr, 'image') else 'No image'}\n")
                    f.write("-" * 30 + "\n")

            logger.debug(f"DEBUG: Saved frame info to {debug_file}")

            debug_images_dir = debug_dir / "images"
            debug_images_dir.mkdir(exist_ok=True)

            for frame_id, frame in frames.items():
                try:
                    debug_timestamp = int(time.time() * 1000000)
                    debug_image_path = debug_images_dir / f"debug_{frame_id}_{debug_timestamp}.jpg"
                    image_rgb = frame.rw_bgr.image[:, :, ::-1]
                    pil_image = Image.fromarray(image_rgb)
                    pil_image.save(debug_image_path, "JPEG", quality=90)
                    logger.debug(f"DEBUG: Saved frame image to {debug_image_path}")
                except Exception as e:
                    logger.error(f"DEBUG: Failed to save frame image for {frame_id}: {e}")

        for frame_id, frame in frames.items():
            logger.debug(f"STARTING frame processing: {frame_id}")
            should_exclude = False
            for pattern in self.exclude_topics:
                try:
                    import re
                    if re.match(pattern, frame_id):
                        should_exclude = True
                        break
                except re.error:
                    if pattern == frame_id:
                        should_exclude = True
                        break

            if should_exclude:
                logger.info(f"SKIPPING topic {frame_id} as it matches exclude pattern")
                continue

            if self.topic_regex and not self.topic_regex.search(frame_id):
                logger.info(f"SKIPPING topic {frame_id} due to topic_regex mismatch")
                continue

            image = frame.rw_bgr.image
            frame_meta = frame.data.get('meta', {})
            frame_id_meta = frame_meta.get('id', frame_id)

            start_time = time.time()
            try:
                logger.debug(f"CALLING LLM for frame {frame_id_meta}")
                annotations, usage = self._analyze_image(image, frame_id_meta)
                processing_time = time.time() - start_time

                results = {
                    "schema_version": CHATTAG_OUTPUT_SCHEMA_VERSION,
                    "annotations": annotations,
                    "usage": usage,
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "model": self.model,
                    "frame_id": frame_id_meta
                }

                logger.info(f"LLM SUCCESS for frame {frame_id_meta}: {len(annotations)} annotations, {usage['total_tokens']} tokens, {processing_time:.2f}s")

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"LLM ERROR for frame {frame_id_meta}: {str(e)}")

                results = {
                    "schema_version": CHATTAG_OUTPUT_SCHEMA_VERSION,
                    "annotations": self._get_default_annotations(),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "model": self.model,
                    "frame_id": frame_id_meta,
                    "error": str(e)
                }

            frame_data = frame.data.copy() if hasattr(frame.data, 'copy') else dict(frame.data)

            if "meta" not in frame_data:
                frame_data["meta"] = {}

            frame_data["meta"][CHATTAG_META_KEY] = results

            updated_frame = Frame(image, frame_data, "BGR")

            processed_frames[frame_id] = updated_frame
            self._total_frames_processed += 1
            logger.info(f"ADDED frame {frame_id} to output (batch: {len(processed_frames)}, total: {self._total_frames_processed})")

            if self.save_frames:
                image_path = self._save_processed_image(frame_id_meta, frame.image)
                self._save_frame_results(frame_id_meta, results, image_path)


        if self.forward_main:
            main_found = False
            for frame_id, frame in frames.items():
                if frame_id == "main":
                    main_frame = Frame(frame.rw_bgr.image, frame.data, "BGR")
                    ordered_frames = {"main": main_frame}
                    for key, value in processed_frames.items():
                        if key != "main":
                            ordered_frames[key] = value
                    processed_frames = ordered_frames
                    main_found = True
                    break
            if not main_found:
                logger.warning("No main topic found in frames, skipping forward_main")

        logger.debug(f"BATCH COMPLETE: Input frames: {len(frames)}, Output frames: {len(processed_frames)}, Total processed so far: {self._total_frames_processed}")
        return processed_frames

    def _analyze_image(self, image, frame_id: str) -> tuple[Dict[str, Any], Dict[str, int]]:
        """
        Analyze an image with the configured LangChain chat model.

        Returns ``(annotations_dict, usage_dict)``. Annotations are always
        normalized through ``_validate_annotations`` so downstream consumers
        see the same shape regardless of provider.
        """
        if self.no_ops:
            logger.info(f"NO-OPS: Skipping LLM call for frame {frame_id}")
            return self._get_default_annotations(), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        image_rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        if self.max_image_size > 0:
            pil_image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to max {self.max_image_size}px for frame {frame_id}")
        else:
            logger.debug(f"Keeping original image size for frame {frame_id}")

        buffer = io.BytesIO()
        quality = max(self.image_quality, 90)
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )

        logger.debug(f"Invoking LLM for frame {frame_id}")
        response = self._chain.invoke([message])
        logger.debug(f"LLM response received for frame {frame_id}")

        if self._pydantic_schema is not None:
            parsed = response.get("parsed")
            raw = response.get("raw")
            parsing_error = response.get("parsing_error")

            if parsing_error is not None or parsed is None:
                logger.error(f"Structured output parse failed for frame {frame_id}: {parsing_error}")
                annotations = self._get_default_annotations()
            else:
                annotations = {
                    label: {"present": bool(ann["present"]), "confidence": float(ann["confidence"])}
                    for label, ann in parsed.model_dump().items()
                }
                annotations = self._validate_annotations(annotations)
                self._perform_annotation_quality_checks(annotations, frame_id)

            usage_md = getattr(raw, "usage_metadata", None) or {}
            usage = {
                "input_tokens": int(usage_md.get("input_tokens", 0)),
                "output_tokens": int(usage_md.get("output_tokens", 0)),
                "total_tokens": int(usage_md.get("total_tokens", 0)),
            }
        else:
            raw_text = response.content if hasattr(response, "content") else str(response)
            try:
                annotations = json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for frame {frame_id}: {str(e)}")
                logger.error(f"Raw response: {raw_text}")
                annotations = self._get_default_annotations()
            annotations = self._validate_annotations(annotations)
            self._perform_annotation_quality_checks(annotations, frame_id)

            usage_md = getattr(response, "usage_metadata", None) or {}
            usage = {
                "input_tokens": int(usage_md.get("input_tokens", 0)),
                "output_tokens": int(usage_md.get("output_tokens", 0)),
                "total_tokens": int(usage_md.get("total_tokens", 0)),
            }

        return annotations, usage

    def _validate_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize annotations to the canonical {present, confidence} shape."""
        validated = {}

        if self.output_schema:
            for key, default_value in self.output_schema.items():
                if key in annotations:
                    if isinstance(annotations[key], dict) and "present" in annotations[key] and "confidence" in annotations[key]:
                        validated[key] = {
                            "present": bool(annotations[key]["present"]),
                            "confidence": float(annotations[key]["confidence"])
                        }
                    elif isinstance(annotations[key], bool):
                        validated[key] = {
                            "present": annotations[key],
                            "confidence": 1.0 if annotations[key] else 0.0
                        }
                    else:
                        validated[key] = self._default_for_schema_key(default_value)
                else:
                    validated[key] = self._default_for_schema_key(default_value)
        else:
            for key, value in annotations.items():
                if isinstance(value, dict) and "present" in value and "confidence" in value:
                    validated[key] = {
                        "present": bool(value["present"]),
                        "confidence": float(value["confidence"])
                    }
                elif isinstance(value, bool):
                    validated[key] = {
                        "present": value,
                        "confidence": 1.0 if value else 0.0
                    }
                else:
                    validated[key] = {
                        "present": bool(value),
                        "confidence": 0.5
                    }

        return validated

    @staticmethod
    def _default_for_schema_key(default_value: Any) -> Dict[str, Any]:
        """Normalize an output_schema entry to a classification-only default."""
        if isinstance(default_value, dict):
            return {
                "present": bool(default_value.get("present", False)),
                "confidence": float(default_value.get("confidence", 0.0)),
            }
        return {"present": False, "confidence": 0.0}

    def _perform_annotation_quality_checks(self, annotations: Dict[str, Any], frame_id: str):
        try:
            quality_issues = []

            for label, data in annotations.items():
                if not isinstance(data, dict):
                    continue

                present = data.get('present', False)
                confidence = data.get('confidence', 0.0)

                if present and confidence < 0.5:
                    quality_issues.append(f"{label}: Present but low confidence ({confidence:.2f})")

                if not present and confidence > 0.7:
                    quality_issues.append(f"{label}: Not present but high confidence ({confidence:.2f})")

            if quality_issues:
                logger.warning(f"Quality issues detected for frame {frame_id}:")
                for issue in quality_issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.debug(f"No quality issues detected for frame {frame_id}")

        except Exception as e:
            logger.error(f"Error performing quality checks for frame {frame_id}: {e}")

    def _get_default_annotations(self) -> Dict[str, Any]:
        if self.output_schema:
            return {
                k: self._default_for_schema_key(v)
                for k, v in self.output_schema.items()
            }
        return {}

    def _save_frame_results(self, frame_id: str, results: Dict[str, Any], image_path: str = None):
        """Save frame results to JSONL in dataset_langchain format."""
        try:
            dataset_entry = {
                "schema_version": results["schema_version"],
                "image": image_path or f"{frame_id}.jpg",
                "labels": results.get("annotations", {}),
                "usage": results.get("usage", {}),
                "prompt_used": self.prompt_filename
            }

            output_file = self.output_dir / "labels.jsonl"
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dataset_entry, ensure_ascii=False) + '\n')

            logger.debug(f"Saved frame results to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save frame results for {frame_id}: {str(e)}")

    def _save_processed_image(self, frame_id: str, image):
        try:
            image_rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(image_rgb)

            data_dir = self.output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)

            if self.preserve_original_format and pil_image.mode in ('RGB', 'RGBA'):
                filename = f"{frame_id}_{timestamp}.png"
                image_path = data_dir / filename
                pil_image.save(image_path, "PNG", optimize=False)
                logger.debug(f"Saved processed image to: {image_path} as PNG (lossless)")
            else:
                filename = f"{frame_id}_{timestamp}.jpg"
                image_path = data_dir / filename
                pil_image.save(
                    image_path,
                    "JPEG",
                    quality=max(self.image_quality, 95),
                    optimize=False,
                    subsampling=0,
                    progressive=False
                )
                logger.debug(f"Saved processed image to: {image_path} as JPEG with quality {max(self.image_quality, 95)}")

            return str(image_path)
        except Exception as e:
            logger.error(f"Failed to save processed image for {frame_id}: {str(e)}")
            return None

    def _generate_binary_datasets(self):
        """Generate binary datasets from saved JSONL in dataset_langchain format."""
        try:
            logger.info("Generating binary datasets from saved JSONL file...")

            jsonl_file = self.output_dir / "labels.jsonl"
            if not jsonl_file.exists():
                logger.warning("No labels.jsonl file found in output directory")
                return

            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))

            if not records:
                logger.warning("No records found in JSONL file")
                return

            labels = set()
            for record in records:
                labels.update(record["labels"].keys())

            if not labels:
                logger.warning("No labels found in records")
                return

            binary_datasets_dir = self.output_dir / "binary_datasets"
            binary_datasets_dir.mkdir(exist_ok=True)

            for label_name in labels:
                dataset = {"annotations": []}

                for record in records:
                    if label_name in record["labels"]:
                        present = record["labels"][label_name].get('present', False)
                        confidence = record["labels"][label_name].get('confidence', 0.0)

                        binary_label = label_name if present and confidence >= self.confidence_threshold else "absent"

                        image_path = record["image"]
                        filename = os.path.basename(image_path)

                        annotation = {
                            "filename": filename,
                            "label": binary_label
                        }
                        dataset["annotations"].append(annotation)

                dataset_file = binary_datasets_dir / f"{label_name}_labels.json"

                with open(dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)

                positive_count = sum(1 for ann in dataset["annotations"] if ann["label"] == label_name)
                negative_count = sum(1 for ann in dataset["annotations"] if ann["label"] == "absent")

                logger.info(f"Generated {label_name} dataset: {positive_count} {label_name}, {negative_count} absent samples (overwrote existing file)")

            summary = {
                "total_datasets": len(labels),
                "labels": sorted(list(labels)),
                "total_frames": len(records),
                "output_directory": str(binary_datasets_dir),
                "generated_at": time.time()
            }

            summary_file = binary_datasets_dir / "_summary_report.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Binary datasets generated successfully in: {binary_datasets_dir}")
            logger.info(f"Summary report saved to: {summary_file}")

            self._generate_balanced_datasets(records, labels, binary_datasets_dir)

        except Exception as e:
            logger.error(f"Failed to generate binary datasets: {str(e)}")

    def _generate_balanced_datasets(self, records, labels, binary_datasets_dir):
        """Generate balanced binary datasets where each class has equal representation."""
        try:
            logger.info("Generating balanced binary datasets...")

            balanced_datasets_dir = binary_datasets_dir.parent / "binary_datasets_balanced"
            balanced_datasets_dir.mkdir(exist_ok=True)

            for label in labels:
                positive_samples = []
                negative_samples = []

                for record in records:
                    if label in record["labels"]:
                        present = record["labels"][label].get('present', False)
                        confidence = record["labels"][label].get('confidence', 0.0)

                        image_path = record["image"]
                        filename = os.path.basename(image_path)

                        if present and confidence >= self.confidence_threshold:
                            positive_samples.append(filename)
                        else:
                            negative_samples.append(filename)

                min_samples = min(len(positive_samples), len(negative_samples))

                if min_samples == 0:
                    logger.warning(f"No samples found for {label}, skipping balanced dataset")
                    continue

                import random
                balanced_positive = random.sample(positive_samples, min_samples) if len(positive_samples) >= min_samples else positive_samples
                balanced_negative = random.sample(negative_samples, min_samples) if len(negative_samples) >= min_samples else negative_samples

                balanced_dataset = {"annotations": []}

                for filename in balanced_positive:
                    balanced_dataset["annotations"].append({
                        "filename": filename,
                        "label": label
                    })

                for filename in balanced_negative:
                    balanced_dataset["annotations"].append({
                        "filename": filename,
                        "label": "absent"
                    })

                random.shuffle(balanced_dataset["annotations"])

                balanced_dataset_file = balanced_datasets_dir / f"{label}_labels.json"

                with open(balanced_dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(balanced_dataset, f, indent=2, ensure_ascii=False)

                logger.info(f"Generated balanced {label} dataset: {len(balanced_positive)} {label}, {len(balanced_negative)} absent samples")

            balanced_summary = {
                "total_datasets": len(labels),
                "labels": sorted(list(labels)),
                "total_frames": len(records),
                "output_directory": str(balanced_datasets_dir),
                "labeling_scheme": {
                    "positive_class": "label_name",
                    "negative_class": "absent"
                },
                "balancing": {
                    "enabled": True,
                    "method": "equal_sampling",
                    "description": "Each class has equal representation (balanced)"
                },
                "generated_at": time.time()
            }

            balanced_summary_file = balanced_datasets_dir / "_summary_report.json"
            with open(balanced_summary_file, 'w', encoding='utf-8') as f:
                json.dump(balanced_summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Balanced datasets generated successfully in: {balanced_datasets_dir}")
            logger.info(f"Balanced summary report saved to: {balanced_summary_file}")

        except Exception as e:
            logger.error(f"Failed to generate balanced datasets: {str(e)}")

    def _generate_multilabel_coco_datasets(self):
        """
        Build a COCO-style JSON from labels.jsonl for multilabel workflows.
        Each positive label (per confidence threshold) gets one full-image box.
        """
        try:
            logger.info("Generating multilabel COCO datasets...")

            jsonl_file = self.output_dir / "labels.jsonl"
            if not jsonl_file.exists():
                logger.warning("No labels.jsonl file found in output directory")
                return

            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))

            if not records:
                logger.warning("No records found in JSONL file")
                return

            labels = set()
            for record in records:
                labels.update(record["labels"].keys())

            if not labels:
                logger.warning("No labels found in records")
                return

            multilabel_datasets_dir = self.output_dir / "multilabel_datasets"
            multilabel_datasets_dir.mkdir(exist_ok=True)

            coco_dataset = {
                "info": {
                    "description": "ChatTag Multilabel Dataset",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "FilterChatTag",
                    "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                "images": [],
                "annotations": [],
                "categories": []
            }

            for idx, label in enumerate(sorted(labels), 1):
                coco_dataset["categories"].append({
                    "id": idx,
                    "name": label,
                    "supercategory": "object"
                })

            category_mapping = {label: idx for idx, label in enumerate(sorted(labels), 1)}

            try:
                import cv2
            except ImportError:
                cv2 = None

            annotation_id = 1
            for image_id, record in enumerate(records, 1):
                image_path = record["image"]
                filename = os.path.basename(image_path)

                width, height = 640, 480
                if cv2 is not None:
                    try:
                        full_image_path = self.output_dir / image_path
                        if full_image_path.exists():
                            img = cv2.imread(str(full_image_path))
                            if img is not None:
                                height, width = img.shape[:2]
                            else:
                                logger.warning(
                                    "cv2.imread returned no data for %s (corrupt or unsupported format); "
                                    "using fallback dimensions 640x480 for COCO export",
                                    filename,
                                )
                        else:
                            logger.warning(
                                "Image not found at %s for COCO export; using fallback dimensions 640x480",
                                full_image_path,
                            )
                    except Exception as e:
                        logger.warning(f"Could not read image dimensions for {filename}: {e}")

                coco_dataset["images"].append({
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": filename,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                })

                for label_name, label_data in record["labels"].items():
                    if (label_data.get('present', False) and
                            label_data.get('confidence', 0.0) >= self.confidence_threshold):
                        bbox = [0, 0, width, height]
                        area = width * height
                        coco_dataset["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_mapping[label_name],
                            "segmentation": [],
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        })
                        annotation_id += 1

            coco_file = multilabel_datasets_dir / "annotations.json"
            with open(coco_file, 'w', encoding='utf-8') as f:
                json.dump(coco_dataset, f, indent=2, ensure_ascii=False)

            summary = {
                "task_type": "multilabel_classification",
                "format": "COCO",
                "total_classes": len(labels),
                "classes": sorted(list(labels)),
                "category_mapping": category_mapping,
                "total_images": len(records),
                "total_annotations": annotation_id - 1,
                "output_directory": str(multilabel_datasets_dir),
                "confidence_threshold": self.confidence_threshold,
                "coco_file": str(coco_file),
                "bbox_type": "full_image_bbox",
                "description": "Each present label gets a bounding box covering the entire image",
                "generated_at": time.time()
            }

            summary_file = multilabel_datasets_dir / "_summary_report.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Multilabel COCO dataset generated in: {multilabel_datasets_dir}")

        except Exception as e:
            logger.error(f"Failed to generate multilabel COCO datasets: {str(e)}")

    def shutdown(self):
        logger.info("========= Shutting down FilterChatTag =========")

        if self.save_frames and self.output_dir and self.output_dir.exists():
            self._generate_binary_datasets()
            if self.output_schema and len(self.output_schema) > 1:
                logger.info("Multiple classes detected — generating multilabel COCO export...")
                self._generate_multilabel_coco_datasets()

        self._chain = None
        logger.info("FilterChatTag shutdown complete.")


if __name__ == '__main__':
    FilterChatTag.run()
