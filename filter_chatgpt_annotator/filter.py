import logging
import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image
import io

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

__all__ = ['FilterChatgptAnnotatorConfig', 'FilterChatgptAnnotator']

logger = logging.getLogger(__name__)


class FilterChatgptAnnotatorConfig(FilterConfig):
    # ChatGPT API configuration
    chatgpt_model: str = "gpt-4o-mini"
    # chatgpt_model: str = "gpt-4o"
    chatgpt_api_key: str = ""
    prompt: str = ""
    output_schema: Dict[str, Any] = {}
    
    # API parameters
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # Image processing
    max_image_size: int = 0  # 0 = keep original size
    image_quality: int = 95  # High quality to preserve original image quality
    preserve_original_format: bool = True  # Try to preserve original image format when possible
    
    # Output options
    save_frames: bool = True
    output_dir: str = "./output_frames"
    
    # Topic filtering
    topic_pattern: str = None
    exclude_topics: list = []
    
    # Forward main topic to output
    forward_main: bool = False
    
    # No-ops mode (skip API calls for testing)
    no_ops: bool = False
    
    # Debug metadata logging
    debug_metadata: bool = False
    
    # Confidence threshold for positive classification (0.0 to 1.0)
    confidence_threshold: float = 0.9
    
    # Task type is auto-detected based on bbox presence in output_schema
    # No longer needed as parameter


class FilterChatgptAnnotator(Filter):
    """
    Filter that uses ChatGPT Vision API for image annotation and analysis.
    
    This filter processes video frames using ChatGPT Vision API to extract structured 
    annotations and labels based on configurable prompts and standardized output schemas.
    
    Data Signature:
    --------------
    The filter returns processed frames with the following data structure:
    
    Main Frame Data:
    - Original frame data preserved
    - Processing results added to frame metadata:
      - annotations: Dict with item_name -> {"present": bool, "confidence": float}
      - usage: Dict with token usage information
      - processing_time: Processing time in seconds
      - timestamp: Processing timestamp
      - error: Error message if processing failed
    
    Topic Forwarding:
    ----------------
    The `forward_main` parameter controls whether the main topic from input frames 
    is forwarded to the output:
    
    - When `forward_main=True`: The main topic from input frames is preserved and 
      forwarded to the output alongside processed results
    - When `forward_main=False`: Only processed frames are returned (no main topic forwarding)
    
    This is useful in pipeline scenarios where you want to preserve the original 
    main frame alongside processed results for downstream filters.
    
    Key Features:
    - Configurable prompts for different annotation tasks
    - Standardized JSON output format with confidence scores
    - Image resizing to optimize API costs
    - Fault tolerant: logs and skips malformed data
    - Support for diverse datasets (any domain with image classification needs)
    - Optional frame persistence for auditing/debugging
    - Topic forwarding for pipeline compatibility
    """

    @classmethod
    def normalize_config(cls, config: FilterChatgptAnnotatorConfig):
        config = FilterChatgptAnnotatorConfig(super().normalize_config(config))

        # Environment variable mapping with type information
        env_mapping = {
            "chatgpt_model": (str, str.strip),
            "chatgpt_api_key": (str, str.strip),
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

        # Process environment variables
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

        # Handle output_schema from environment (JSON string)
        output_schema_env = os.getenv("FILTER_OUTPUT_SCHEMA")
        if output_schema_env:
            try:
                config.output_schema = json.loads(output_schema_env)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in FILTER_OUTPUT_SCHEMA: {str(e)}")

        # Handle exclude_topics from environment (comma-separated)
        exclude_topics_env = os.getenv("FILTER_EXCLUDE_TOPICS")
        if exclude_topics_env:
            config.exclude_topics = [topic.strip() for topic in exclude_topics_env.split(",") if topic.strip()]

        # Validate required parameters
        if not config.chatgpt_api_key:
            raise ValueError("chatgpt_api_key is required (set FILTER_CHATGPT_API_KEY)")
        
        if not config.prompt:
            raise ValueError("prompt is required (set FILTER_PROMPT)")

        # Validate parameter ranges
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
        
        # Task type is now auto-detected based on bbox presence
        # No validation needed

        # Validate prompt file exists
        if config.prompt and not os.path.exists(config.prompt):
            raise FileNotFoundError(f"Prompt file not found: {config.prompt}")

        # Log normalized config with masked API key
        masked_config = cls._mask_api_key_in_config(config)
        logger.debug(f"Normalized config: {masked_config}")
        return config

    @classmethod
    def _mask_api_key_in_config(cls, config: FilterChatgptAnnotatorConfig) -> dict:
        """
        Create a masked version of config for logging (hides API key).
        
        Args:
            config: FilterChatgptAnnotatorConfig object
            
        Returns:
            dict: Config dict with masked API key
        """
        config_dict = {
            'id': config.id,
            'sources': config.sources,
            'outputs': config.outputs,
            'chatgpt_api_key': cls._mask_api_key(config.chatgpt_api_key),
            'prompt': config.prompt,
            'save_frames': config.save_frames,
            'output_schema': config.output_schema,
            'pipeline_id': config.pipeline_id,
            'device_name': config.device_name
        }
        return config_dict

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """
        Mask API key showing only first 8 and last 4 characters.
        
        Args:
            api_key: Original API key
            
        Returns:
            str: Masked API key
        """
        if not api_key or len(api_key) < 12:
            return "***masked***"
        
        return f"{api_key[:8]}...{api_key[-4:]}"

    def setup(self, config: FilterChatgptAnnotatorConfig):
        logger.info("========= Setting up FilterChatgptAnnotator =========")
        
        # Store config for later use
        self.config: FilterChatgptAnnotatorConfig = config
        
        # Log config with masked API key
        masked_config = self._mask_api_key_in_config(config)
        logger.info(f"FilterChatgptAnnotator config: {masked_config}")
        
        # Store commonly used config values as instance attributes
        self.chatgpt_model = config.chatgpt_model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_image_size = config.max_image_size
        self.image_quality = config.image_quality
        self.preserve_original_format = config.preserve_original_format
        self.save_frames = config.save_frames
        self.output_dir = Path(config.output_dir) if config.save_frames else None
        self.no_ops = config.no_ops
        self.output_schema = config.output_schema
        
        # Initialize ChatGPT client (skip if in no-ops mode)
        if not self.no_ops:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.chatgpt_api_key)
                logger.info(f"Initialized OpenAI client with model: {config.chatgpt_model}")
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        else:
            self.client = None
            logger.info("Skipping OpenAI client initialization (no-ops mode)")

        # Load prompt from file
        try:
            with open(config.prompt, 'r', encoding='utf-8') as f:
                self.prompt_text = f.read().strip()
            logger.debug(f"Loaded prompt from: {config.prompt}")
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt file: {str(e)}")

        # Initialize output directory if saving frames
        if self.save_frames and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory: {self.output_dir}")

        # Initialize topic filtering
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

        # Store other configuration
        self.model = config.chatgpt_model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_image_size = config.max_image_size
        self.image_quality = config.image_quality
        self.preserve_original_format = config.preserve_original_format
        self.output_schema = config.output_schema
        self.confidence_threshold = config.confidence_threshold
        # Auto-detect task type based on output_schema
        self.has_bbox_schema = False
        if self.output_schema:
            # Check if any schema item has bbox field
            for item_schema in self.output_schema.values():
                if isinstance(item_schema, dict) and "bbox" in item_schema:
                    self.has_bbox_schema = True
                    logger.info("Auto-detected bbox schema - will generate both classification and detection datasets")
                    break
        
        if not self.has_bbox_schema:
            logger.info("No bbox schema detected - will generate classification datasets only")

        logger.info("FilterChatgptAnnotator setup complete.")

    def process(self, frames: dict[str, Frame]):
        """
        Process frames using ChatGPT Vision API.
        
        Args:
            frames: Dictionary with input frames
            
        Returns:
            Processed frames with annotation results in metadata.
        """
        processed_frames = {}
        
        logger.debug(f"PROCESS CALL: Received {len(frames)} frames with keys: {list(frames.keys())}")
        
        # Store total frames processed across all calls for debugging
        if not hasattr(self, '_total_frames_processed'):
            self._total_frames_processed = 0
        
        # DEBUG: Save frame info to debug file (only if debug_metadata is enabled)
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
            
            # DEBUG: Save frame images for visual debugging
            debug_images_dir = debug_dir / "images"
            debug_images_dir.mkdir(exist_ok=True)
            
            for frame_id, frame in frames.items():
                try:
                    # Save debug image with unique timestamp (microseconds for uniqueness)
                    debug_timestamp = int(time.time() * 1000000)  # Use microseconds for uniqueness
                    debug_image_path = debug_images_dir / f"debug_{frame_id}_{debug_timestamp}.jpg"
                    image_rgb = frame.rw_bgr.image[:, :, ::-1]  # BGR to RGB
                    from PIL import Image
                    pil_image = Image.fromarray(image_rgb)
                    pil_image.save(debug_image_path, "JPEG", quality=90)
                    logger.debug(f"DEBUG: Saved frame image to {debug_image_path}")
                except Exception as e:
                    logger.error(f"DEBUG: Failed to save frame image for {frame_id}: {e}")
        
        # Process each frame received in this call
        for frame_id, frame in frames.items():
            logger.debug(f"STARTING frame processing: {frame_id}")
            # Check if topic should be excluded
            should_exclude = False
            for pattern in self.exclude_topics:
                try:
                    import re
                    if re.match(pattern, frame_id):
                        should_exclude = True
                        break
                except re.error:
                    # If pattern is not a valid regex, treat it as an exact match
                    if pattern == frame_id:
                        should_exclude = True
                        break
            
            if should_exclude:
                logger.info(f"SKIPPING topic {frame_id} as it matches exclude pattern")
                continue
            
            # Skip if topic doesn't match pattern (if pattern is specified)
            if self.topic_regex and not self.topic_regex.search(frame_id):
                logger.info(f"SKIPPING topic {frame_id} due to topic_regex mismatch")
                continue
            
            # Get image from frame
            image = frame.rw_bgr.image
            
            # Get frame metadata
            frame_meta = frame.data.get('meta', {})
            frame_id_meta = frame_meta.get('id', frame_id)
            
            # Process frame with ChatGPT Vision API
            start_time = time.time()
            try:
                logger.debug(f"CALLING API for frame {frame_id_meta}")
                annotations, usage = self._analyze_image_with_chatgpt(image, frame_id_meta)
                processing_time = time.time() - start_time
                
                # Create results dictionary
                results = {
                    "annotations": annotations,
                    "usage": usage,
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "model": self.model,
                    "frame_id": frame_id_meta
                }
                
                logger.info(f"API SUCCESS for frame {frame_id_meta}: {len(annotations)} annotations, {usage['total_tokens']} tokens, {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"API ERROR for frame {frame_id_meta}: {str(e)}")
                
                # Create error results
                results = {
                    "annotations": self._get_default_annotations(),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "model": self.model,
                    "frame_id": frame_id_meta,
                    "error": str(e)
                }
            
            # Preserve original frame data and add new results
            frame_data = frame.data.copy() if hasattr(frame.data, 'copy') else dict(frame.data)
            
            # Ensure meta exists in frame data
            if "meta" not in frame_data:
                frame_data["meta"] = {}
            
            # Add ChatGPT results to metadata
            frame_data["meta"]["chatgpt_annotator"] = results
            
            # Create new frame with preserved data and updated metadata
            updated_frame = Frame(image, frame_data, "BGR")
            
            # Add the processed frame to output
            processed_frames[frame_id] = updated_frame
            self._total_frames_processed += 1
            logger.info(f"ADDED frame {frame_id} to output (batch: {len(processed_frames)}, total: {self._total_frames_processed})")
            
            # Save frame results and image if enabled
            if self.save_frames:
                # Save processed image with unique name
                image_path = self._save_processed_image(frame_id_meta, frame.image)
                
                # Save results in dataset_langchain format
                self._save_frame_results(frame_id_meta, results, image_path)
        
        
        # Handle forward_main logic
        if self.forward_main:
            main_found = False
            for frame_id, frame in frames.items():
                if frame_id == "main":
                    main_frame = Frame(frame.rw_bgr.image, frame.data, "BGR")
                    ordered_frames = {"main": main_frame}
                    # Add all other frames after main
                    for key, value in processed_frames.items():
                        if key != "main":  # Avoid duplicating main if it already exists
                            ordered_frames[key] = value
                    processed_frames = ordered_frames
                    main_found = True
                    break
            if not main_found:
                logger.warning("No main topic found in frames, skipping forward_main")
        
        logger.debug(f"BATCH COMPLETE: Input frames: {len(frames)}, Output frames: {len(processed_frames)}, Total processed so far: {self._total_frames_processed}")
        return processed_frames

    def _analyze_image_with_chatgpt(self, image, frame_id: str) -> tuple[Dict[str, Any], Dict[str, int]]:
        """
        Analyze image using ChatGPT Vision API.
        
        Args:
            image: OpenCV image (BGR format)
            frame_id: Frame identifier for logging
            
        Returns:
            Tuple of (annotations_dict, usage_dict)
        """
        # Check if no-ops mode is enabled
        if self.no_ops:
            logger.info(f"NO-OPS: Skipping API call for frame {frame_id}")
            return self._get_default_annotations(), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        # Convert OpenCV BGR image to PIL RGB
        image_rgb = image[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(image_rgb)
        
        # Resize image to optimize API costs (only if max_image_size > 0)
        # For better annotation precision, use higher quality resizing
        if self.max_image_size > 0:
            # Use LANCZOS resampling for better quality when downscaling
            pil_image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to max {self.max_image_size}px for frame {frame_id}")
        else:
            logger.debug(f"Keeping original image size for frame {frame_id}")
        
        # Convert to base64 with optimized settings for better quality
        buffer = io.BytesIO()
        # Use higher quality for better annotation precision
        quality = max(self.image_quality, 90)  # Ensure minimum quality of 90
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Call ChatGPT Vision API
        logger.debug(f"Making API request for frame {frame_id}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        logger.debug(f"API response received for frame {frame_id}")
        
        # Extract response content
        raw_response = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Parse JSON response
        try:
            annotations = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for frame {frame_id}: {str(e)}")
            logger.error(f"Raw response: {raw_response}")
            annotations = self._get_default_annotations()
        
        # Validate and normalize annotations
        annotations = self._validate_annotations(annotations)
        
        # Perform quality checks on annotations
        self._perform_annotation_quality_checks(annotations, frame_id)
        
        return annotations, usage

    def _validate_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize annotation format.
        
        Args:
            annotations: Raw annotations from ChatGPT
            
        Returns:
            Validated annotations in standard format
        """
        validated = {}
        
        # If output_schema is provided, use it as template
        if self.output_schema:
            for key, default_value in self.output_schema.items():
                if key in annotations:
                    # Validate the annotation format
                    if isinstance(annotations[key], dict) and "present" in annotations[key] and "confidence" in annotations[key]:
                        validated_item = {
                            "present": bool(annotations[key]["present"]),
                            "confidence": float(annotations[key]["confidence"])
                        }
                        
                        # Add bbox if present (for detection tasks)
                        if "bbox" in annotations[key]:
                            bbox = annotations[key]["bbox"]
                            if bbox is not None and isinstance(bbox, list) and len(bbox) == 4:
                                # Validate bbox coordinates with enhanced checks
                                if self._validate_bbox_coordinates(bbox, key):
                                    validated_item["bbox"] = bbox
                                else:
                                    logger.warning(f"Invalid bbox coordinates for {key}: {bbox}")
                                    validated_item["bbox"] = None
                            else:
                                validated_item["bbox"] = None
                        
                        validated[key] = validated_item
                    elif isinstance(annotations[key], bool):
                        # Convert boolean to standard format
                        validated_item = {
                            "present": annotations[key],
                            "confidence": 1.0 if annotations[key] else 0.0
                        }
                        # Add bbox as null for boolean format if schema expects bbox
                        if self.has_bbox_schema:
                            validated_item["bbox"] = None
                        validated[key] = validated_item
                    else:
                        # Use default value
                        validated[key] = default_value
                else:
                    # Use default value
                    validated[key] = default_value
        else:
            # No schema provided, try to normalize existing annotations
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
                    # Convert to boolean with default confidence
                    validated[key] = {
                        "present": bool(value),
                        "confidence": 0.5
                    }
        
        return validated

    def _validate_bbox_coordinates(self, bbox: list, label: str) -> bool:
        """
        Validate bounding box coordinates with enhanced checks.
        
        Args:
            bbox: List of 4 coordinates [x_min, y_min, x_max, y_max]
            label: Label name for logging
            
        Returns:
            bool: True if bbox is valid, False otherwise
        """
        try:
            # Check if all coordinates are numbers
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                logger.warning(f"Bbox coordinates must be numbers for {label}: {bbox}")
                return False
            
            x_min, y_min, x_max, y_max = bbox
            
            # Check if coordinates are within valid range [0, 1]
            if not all(0 <= coord <= 1 for coord in bbox):
                logger.warning(f"Bbox coordinates must be between 0 and 1 for {label}: {bbox}")
                return False
            
            # Check if coordinates form a valid rectangle
            if x_min >= x_max or y_min >= y_max:
                logger.warning(f"Invalid bbox rectangle for {label}: x_min={x_min} >= x_max={x_max} or y_min={y_min} >= y_max={y_max}")
                return False
            
            # Check if bbox has reasonable size (not too small or too large)
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            # Minimum area threshold (0.1% of image)
            if area < 0.001:
                logger.warning(f"Bbox too small for {label}: area={area:.4f} < 0.001")
                return False
            
            # Maximum area threshold (80% of image)
            if area > 0.8:
                logger.warning(f"Bbox too large for {label}: area={area:.4f} > 0.8")
                return False
            
            # Check aspect ratio (not too extreme)
            aspect_ratio = width / height if height > 0 else float('inf')
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                logger.warning(f"Bbox aspect ratio too extreme for {label}: {aspect_ratio:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating bbox for {label}: {e}")
            return False

    def _perform_annotation_quality_checks(self, annotations: Dict[str, Any], frame_id: str):
        """
        Perform quality checks on annotations and log warnings for potential issues.
        
        Args:
            annotations: Validated annotations dictionary
            frame_id: Frame identifier for logging
        """
        try:
            quality_issues = []
            
            for label, data in annotations.items():
                if not isinstance(data, dict):
                    continue
                
                present = data.get('present', False)
                confidence = data.get('confidence', 0.0)
                bbox = data.get('bbox', None)
                
                # Check confidence vs presence consistency
                if present and confidence < 0.5:
                    quality_issues.append(f"{label}: Present but low confidence ({confidence:.2f})")
                
                if not present and confidence > 0.7:
                    quality_issues.append(f"{label}: Not present but high confidence ({confidence:.2f})")
                
                # Check bbox consistency with presence
                if present and bbox is None:
                    quality_issues.append(f"{label}: Present but no bounding box provided")
                
                if not present and bbox is not None:
                    quality_issues.append(f"{label}: Not present but bounding box provided")
                
                # Check for overlapping bounding boxes
                if present and bbox is not None:
                    for other_label, other_data in annotations.items():
                        if (other_label != label and 
                            isinstance(other_data, dict) and 
                            other_data.get('present', False) and 
                            other_data.get('bbox') is not None):
                            
                            overlap = self._calculate_bbox_overlap(bbox, other_data['bbox'])
                            if overlap > 0.5:  # 50% overlap threshold
                                quality_issues.append(f"{label} and {other_label}: High bounding box overlap ({overlap:.2f})")
            
            # Log quality issues
            if quality_issues:
                logger.warning(f"Quality issues detected for frame {frame_id}:")
                for issue in quality_issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.debug(f"No quality issues detected for frame {frame_id}")
                
        except Exception as e:
            logger.error(f"Error performing quality checks for frame {frame_id}: {e}")

    def _calculate_bbox_overlap(self, bbox1: list, bbox2: list) -> float:
        """
        Calculate the overlap ratio between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x_min, y_min, x_max, y_max]
            bbox2: Second bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            float: Overlap ratio (0.0 to 1.0)
        """
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Calculate intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            intersection = x_overlap * y_overlap
            
            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0

    def _get_default_annotations(self) -> Dict[str, Any]:
        """
        Get default annotations when processing fails.
        
        Returns:
            Default annotations dictionary
        """
        if self.output_schema:
            return self.output_schema.copy()
        else:
            return {}

    def _save_frame_results(self, frame_id: str, results: Dict[str, Any], image_path: str = None):
        """
        Save frame results to JSON file in dataset_langchain format.
        
        Args:
            frame_id: Frame identifier
            results: Processing results
            image_path: Path to the processed image (if save_frames=True)
        """
        try:
            # Create dataset_langchain format
            dataset_entry = {
                "image": image_path or f"{frame_id}.jpg",
                "labels": results.get("annotations", {}),
                "usage": results.get("usage", {})
            }
            
            # Save as JSONL (one line per entry)
            output_file = self.output_dir / "labels.jsonl"
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dataset_entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Saved frame results to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save frame results for {frame_id}: {str(e)}")
    
    def _save_processed_image(self, frame_id: str, image):
        """
        Save processed image with unique name in data subfolder.
        Preserves original image quality by using high-quality settings.
        
        Args:
            frame_id: Frame identifier
            image: OpenCV image (BGR format)
            
        Returns:
            str: Path to saved image
        """
        try:
            # Convert OpenCV BGR to RGB
            image_rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(image_rgb)
            
            # Create data subfolder
            data_dir = self.output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique filename
            timestamp = int(time.time() * 1000)  # milliseconds
            
            # Determine best format and extension
            if self.preserve_original_format and pil_image.mode in ('RGB', 'RGBA'):
                # Try to preserve original format - use PNG for lossless quality
                filename = f"{frame_id}_{timestamp}.png"
                image_path = data_dir / filename
                
                # Save as PNG for lossless quality
                pil_image.save(image_path, "PNG", optimize=False)
                logger.debug(f"Saved processed image to: {image_path} as PNG (lossless)")
            else:
                # Fallback to high-quality JPEG
                filename = f"{frame_id}_{timestamp}.jpg"
                image_path = data_dir / filename
                
                # Save image with high quality settings to preserve original quality
                # Use quality >= 95 and disable optimization to prevent quality loss
                pil_image.save(
                    image_path, 
                    "JPEG", 
                    quality=max(self.image_quality, 95),  # Ensure minimum quality of 95
                    optimize=False,  # Disable optimization to preserve quality
                    subsampling=0,   # Disable chroma subsampling for better quality
                    progressive=False  # Disable progressive encoding for better compatibility
                )
                logger.debug(f"Saved processed image to: {image_path} as JPEG with quality {max(self.image_quality, 95)}")
            
            return str(image_path)
        except Exception as e:
            logger.error(f"Failed to save processed image for {frame_id}: {str(e)}")
            return None

    def _generate_binary_datasets(self):
        """
        Generate binary datasets from saved JSONL file in dataset_langchain format.
        Creates separate datasets for each label/item.
        This method overwrites existing binary dataset files.
        """
        try:
            logger.info("Generating binary datasets from saved JSONL file...")
            
            # Read JSONL file
            jsonl_file = self.output_dir / "labels.jsonl"
            if not jsonl_file.exists():
                logger.warning("No labels.jsonl file found in output directory")
                return
            
            # Read all records from JSONL
            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            
            if not records:
                logger.warning("No records found in JSONL file")
                return
            
            # Get labels from first record
            labels = set()
            for record in records:
                labels.update(record["labels"].keys())
            
            if not labels:
                logger.warning("No labels found in records")
                return
            
            # Create binary datasets directory
            binary_datasets_dir = self.output_dir / "binary_datasets"
            binary_datasets_dir.mkdir(exist_ok=True)
            
            # Generate binary dataset for each label
            for label_name in labels:
                dataset = {"annotations": []}
                
                for record in records:
                    if label_name in record["labels"]:
                        # Convert present/confidence to binary label
                        present = record["labels"][label_name].get('present', False)
                        confidence = record["labels"][label_name].get('confidence', 0.0)
                        
                        # Use confidence threshold for binary classification
                        # Class positive: label name, Class negative: "absent"
                        binary_label = label_name if present and confidence >= self.confidence_threshold else "absent"
                        
                        # Extract filename from image path
                        image_path = record["image"]
                        filename = os.path.basename(image_path)
                            
                        annotation = {
                            "filename": filename,
                            "label": binary_label
                        }
                        dataset["annotations"].append(annotation)
                
                # Save binary dataset directly in binary_datasets folder (overwrites existing)
                dataset_file = binary_datasets_dir / f"{label_name}_labels.json"
                
                with open(dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                
                # Count samples
                positive_count = sum(1 for ann in dataset["annotations"] if ann["label"] == label_name)
                negative_count = sum(1 for ann in dataset["annotations"] if ann["label"] == "absent")
                
                logger.info(f"Generated {label_name} dataset: {positive_count} {label_name}, {negative_count} absent samples (overwrote existing file)")
            
            # Generate summary report
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
            
            # Always generate balanced datasets
            self._generate_balanced_datasets(records, labels, binary_datasets_dir)
            
        except Exception as e:
            logger.error(f"Failed to generate binary datasets: {str(e)}")

    def _generate_balanced_datasets(self, records, labels, binary_datasets_dir):
        """
        Generate balanced binary datasets where each class has equal representation.
        Creates a new directory 'binary_datasets_balanced' with balanced datasets.
        Works for any type of binary classification problem across any domain.
        
        Args:
            records: List of records from JSONL file
            labels: Set of all labels/classes found in the data
            binary_datasets_dir: Path to the original binary datasets directory
        """
        try:
            logger.info("Generating balanced binary datasets...")
            
            # Create balanced datasets directory
            balanced_datasets_dir = binary_datasets_dir.parent / "binary_datasets_balanced"
            balanced_datasets_dir.mkdir(exist_ok=True)
            
            # Generate balanced dataset for each label/class
            for label in labels:
                # Collect all annotations for this label
                positive_samples = []
                negative_samples = []
                
                for record in records:
                    if label in record["labels"]:
                        present = record["labels"][label].get('present', False)
                        confidence = record["labels"][label].get('confidence', 0.0)
                        
                        # Extract filename from image path
                        image_path = record["image"]
                        filename = os.path.basename(image_path)
                        
                        # Use confidence threshold for binary classification
                        if present and confidence >= self.confidence_threshold:
                            positive_samples.append(filename)
                        else:
                            negative_samples.append(filename)
                
                # Balance the dataset (use the smaller class size)
                min_samples = min(len(positive_samples), len(negative_samples))
                
                if min_samples == 0:
                    logger.warning(f"No samples found for {label}, skipping balanced dataset")
                    continue
                
                # Sample equal amounts from both classes
                import random
                balanced_positive = random.sample(positive_samples, min_samples) if len(positive_samples) >= min_samples else positive_samples
                balanced_negative = random.sample(negative_samples, min_samples) if len(negative_samples) >= min_samples else negative_samples
                
                # Create balanced dataset
                balanced_dataset = {"annotations": []}
                
                # Add positive samples
                for filename in balanced_positive:
                    balanced_dataset["annotations"].append({
                        "filename": filename,
                        "label": label
                    })
                
                # Add negative samples
                for filename in balanced_negative:
                    balanced_dataset["annotations"].append({
                        "filename": filename,
                        "label": "absent"
                    })
                
                # Shuffle the dataset
                random.shuffle(balanced_dataset["annotations"])
                
                # Save balanced dataset
                balanced_dataset_file = balanced_datasets_dir / f"{label}_labels.json"
                
                with open(balanced_dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(balanced_dataset, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Generated balanced {label} dataset: {len(balanced_positive)} {label}, {len(balanced_negative)} absent samples")
            
            # Generate balanced summary report
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

    def _generate_detection_datasets(self):
        """
        Generate object detection datasets from saved JSONL file.
        Creates datasets in COCO format for object detection training.
        """
        try:
            logger.info("Generating object detection datasets in COCO format...")
            
            # Read JSONL file
            jsonl_file = self.output_dir / "labels.jsonl"
            if not jsonl_file.exists():
                logger.warning("No labels.jsonl file found in output directory")
                return
            
            # Read all records from JSONL
            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            
            if not records:
                logger.warning("No records found in JSONL file")
                return
            
            # Get labels from all records
            labels = set()
            for record in records:
                labels.update(record["labels"].keys())
            
            if not labels:
                logger.warning("No labels found in records")
                return
            
            # Create detection datasets directory
            detection_datasets_dir = self.output_dir / "detection_datasets"
            detection_datasets_dir.mkdir(exist_ok=True)
            
            # Initialize COCO format structure
            coco_dataset = {
                "info": {
                    "description": "ChatGPT Annotator Detection Dataset",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "ChatGPT Annotator Filter",
                    "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Unknown",
                        "url": ""
                    }
                ],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Create categories
            for idx, label in enumerate(sorted(labels), 1):
                coco_dataset["categories"].append({
                    "id": idx,
                    "name": label,
                    "supercategory": "object"
                })
            
            # Create category mapping
            category_mapping = {label: idx for idx, label in enumerate(sorted(labels), 1)}
            
            # Process each record
            annotation_id = 1
            for image_id, record in enumerate(records, 1):
                # Extract image information
                image_path = record["image"]
                filename = os.path.basename(image_path)
                
                # Get image dimensions (we'll need to read the actual image)
                try:
                    import cv2
                    full_image_path = self.output_dir / image_path
                    if full_image_path.exists():
                        img = cv2.imread(str(full_image_path))
                        if img is not None:
                            height, width = img.shape[:2]
                        else:
                            # Fallback dimensions if image can't be read
                            width, height = 640, 480
                    else:
                        # Fallback dimensions if image doesn't exist
                        width, height = 640, 480
                except Exception as e:
                    logger.warning(f"Could not read image dimensions for {filename}: {e}")
                    width, height = 640, 480
                
                # Add image to COCO dataset
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
                
                # Process annotations for this image
                for label_name, label_data in record["labels"].items():
                    if (label_data.get('present', False) and 
                        label_data.get('confidence', 0.0) >= self.confidence_threshold and
                        'bbox' in label_data and 
                        label_data['bbox'] is not None):
                        
                        # Convert normalized bbox to COCO format
                        x_min_norm, y_min_norm, x_max_norm, y_max_norm = label_data['bbox']
                        
                        # Convert to absolute coordinates
                        x_min = x_min_norm * width
                        y_min = y_min_norm * height
                        bbox_width = (x_max_norm - x_min_norm) * width
                        bbox_height = (y_max_norm - y_min_norm) * height
                        
                        # COCO format: [x, y, width, height] (top-left corner + width/height)
                        bbox = [x_min, y_min, bbox_width, bbox_height]
                        area = bbox_width * bbox_height
                        
                        # Add annotation to COCO dataset
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
            
            # Save COCO dataset
            coco_file = detection_datasets_dir / "annotations.json"
            with open(coco_file, 'w', encoding='utf-8') as f:
                json.dump(coco_dataset, f, indent=2, ensure_ascii=False)
            
            # Generate summary report
            summary = {
                "task_type": "object_detection",
                "format": "COCO",
                "total_classes": len(labels),
                "classes": sorted(list(labels)),
                "category_mapping": category_mapping,
                "total_images": len(records),
                "total_annotations": annotation_id - 1,
                "output_directory": str(detection_datasets_dir),
                "confidence_threshold": self.confidence_threshold,
                "coco_file": str(coco_file),
                "generated_at": time.time()
            }
            
            summary_file = detection_datasets_dir / "_summary_report.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"COCO format detection dataset generated successfully in: {detection_datasets_dir}")
            logger.info(f"COCO annotations file: {coco_file}")
            logger.info(f"Summary report: {summary_file}")
            logger.info(f"Total images: {len(records)}, Total annotations: {annotation_id - 1}")
            
        except Exception as e:
            logger.error(f"Failed to generate detection datasets: {str(e)}")

    def shutdown(self):
        """
        Called once when the filter is shutting down.
        """
        logger.info("========= Shutting down FilterChatgptAnnotator =========")
        
        # Generate datasets if save_frames is enabled and output_dir exists
        # This should happen regardless of no_ops mode
        if self.save_frames and self.output_dir and self.output_dir.exists():
            # Always generate classification datasets
            self._generate_binary_datasets()
            
            # Generate detection datasets only if bbox schema is present
            if self.has_bbox_schema:
                self._generate_detection_datasets()
        
        # Clean up resources
        self.client = None
        logger.info("FilterChatgptAnnotator shutdown complete.")
    

if __name__ == '__main__':
    FilterChatgptAnnotator.run()