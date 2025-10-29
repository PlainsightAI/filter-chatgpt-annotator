#!/usr/bin/env python3
"""
Simple script to show images with classification labels overlaid
Shows all images in sequence with classification labels drawn on them

Usage:
    python show_labels.py [options]

Options:
    --labels-file PATH     Path to labels.jsonl file (default: ../output_frames/labels.jsonl)
    --images-dir PATH      Base directory for images (default: auto-detected from labels file)
    --help                 Show this help message

Environment Variables:
    LABELS_FILE            Path to labels.jsonl file
    IMAGES_DIR             Base directory for images
"""

import json
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

def _load_env_from_files():
    """Load environment variables from a .env file if present.
    
    Search order (first found is used):
    1) Current working directory /.env
    2) Repository root (parent of this script's directory) /.env

    Existing environment variables are not overridden.
    Only simple KEY=VALUE lines are supported; lines starting with '#' are ignored.
    """
    try:
        script_dir = Path(__file__).parent
        candidates = [
            Path.cwd() / ".env",
            script_dir.parent / ".env",
        ]
        for env_path in candidates:
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Do not override already-set variables
                        if key and key not in os.environ:
                            os.environ[key] = value
                break  # stop after the first .env found
    except Exception:
        # Fail silently; fallback paths/env will still work
        pass

def load_labels(labels_file_path):
    """Load labels from .jsonl (JSON Lines) or .json (array/object) file.

    - If extension is .jsonl or content is line-delimited JSON, parse line by line.
    - Otherwise, parse as a single JSON value (array or object) and normalize to a list.
    """
    data = []
    try:
        path = Path(labels_file_path)
        is_jsonl_ext = path.suffix.lower() == '.jsonl'
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            if is_jsonl_ext:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                content = f.read().strip()
                # Heuristic: if content seems to have newlines with braces frequently, try jsonl first
                if '\n{' in content or '\n[' in content:
                    # Try parse as jsonl; if fails, fallback to single json
                    try:
                        for line in content.splitlines():
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
                    except json.JSONDecodeError:
                        obj = json.loads(content)
                        if isinstance(obj, list):
                            data = obj
                        else:
                            data = [obj]
                else:
                    obj = json.loads(content)
                    if isinstance(obj, list):
                        data = obj
                    else:
                        data = [obj]

        # Normalize to expected structure: { 'image': <path>, 'labels': {<name>: {present, confidence}} }
        def normalize_item(raw_item):
            # Already in expected format
            if isinstance(raw_item, dict) and 'image' in raw_item and 'labels' in raw_item:
                return raw_item
            # Common alternative: {filename: str, label: str}
            if isinstance(raw_item, dict) and 'filename' in raw_item and 'label' in raw_item:
                label_name = str(raw_item['label'])
                return {
                    'image': raw_item['filename'],
                    'labels': {label_name: {'present': True, 'confidence': 1.0}}
                }
            # Fallback: if there is a single key that looks like image path
            for key in ['image_path', 'file', 'filepath', 'path', 'img', 'frame', 'filename', 'file_name']:
                if isinstance(raw_item, dict) and key in raw_item:
                    return {
                        'image': raw_item[key],
                        'labels': {}
                    }
            return raw_item

        # If top-level is a single dict with 'annotations', expand it
        if len(data) == 1 and isinstance(data[0], dict) and 'annotations' in data[0]:
            annotations = data[0]['annotations']
            if isinstance(annotations, list):
                data = [normalize_item(a) for a in annotations]
            else:
                data = [normalize_item(annotations)]
        else:
            data = [normalize_item(x) for x in data]
        print(f"Loaded {len(data)} images")
        return data
    except FileNotFoundError:
        print(f"File not found: {labels_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

def get_image_path(image_path, base_dir):
    """Get absolute path of image"""
    if not os.path.isabs(image_path):
        # Handle different path formats
        if image_path.startswith('output_frames/'):
            # Remove 'output_frames/' prefix (length 14)
            image_path = image_path[14:]
        elif image_path.startswith('./output_frames/'):
            # Remove './output_frames/' prefix
            image_path = image_path[16:]
        
        # Remove leading slash if it exists
        if image_path.startswith('/'):
            image_path = image_path[1:]
        
        # Join with base_dir
        return os.path.join(base_dir, image_path)
    return image_path

def draw_labels_on_image(image, labels, all_classes=None):
    """Draw classification labels on image with colors.

    If all_classes is provided, show all classes (present or absent).
    Otherwise, show only keys in labels.
    """
    height, width = image.shape[:2]
    
    # Text settings - make proportional to image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, width / 400))  # Scale between 0.5 and 2.0 based on image width
    thickness = max(1, int(width / 300))  # Thickness proportional to image width
    
    # Position for labels - proportional to image size
    y_offset = int(height * 0.05)  # 5% of image height
    x_offset = int(width * 0.02)   # 2% of image width
    
    class_iterable = list(all_classes) if all_classes else list(labels.keys())
    for label_name in class_iterable:
        label_data = labels.get(label_name, {})
        present = label_data.get('present', False)
        confidence = label_data.get('confidence', 0.0)
        
        # Colors: green for present, red for absent
        color = (0, 255, 0) if present else (0, 0, 255)
        
        # Show ALL labels (both present and absent)
        text = f"{label_name}: {'PRESENT' if present else 'ABSENT'} ({confidence:.2f})"
        
        # Draw text with background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        padding = max(5, int(width * 0.01))  # Padding proportional to image width
        cv2.rectangle(image, (x_offset - padding, y_offset - text_height - padding), 
                     (x_offset + text_width + padding, y_offset + padding), (0, 0, 0), -1)
        cv2.putText(image, text, (x_offset, y_offset), font, font_scale, color, thickness)
        y_offset += int(height * 0.08)  # 8% of image height between labels
    
    return image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Show images with labels overlaid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default paths
    python show_labels.py
    
    # Specify custom labels file
    python show_labels.py --labels-file /path/to/custom/labels.jsonl
    
    # Specify custom images directory
    python show_labels.py --images-dir /path/to/images
    
    # Use environment variables
    LABELS_FILE=/path/to/labels.jsonl IMAGES_DIR=/path/to/images python show_labels.py
        """
    )
    
    parser.add_argument(
        '--labels-file', 
        type=str,
        help='Path to labels.jsonl file (default: ../output_frames/labels.jsonl)'
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Base directory for images (default: auto-detected from labels file)'
    )
    
    return parser.parse_args()

def get_default_paths():
    """Get default paths based on script location"""
    script_dir = Path(__file__).parent
    return {
        'labels_file': script_dir.parent / "output_frames" / "labels.jsonl",
        'images_dir': script_dir.parent / "output_frames"
    }

def show_images_with_labels(labels_file_path=None, images_dir_path=None):
    """Show all images with labels and navigation"""
    # Load .env first (if present) so env vars become available
    _load_env_from_files()

    # Get paths from arguments, environment variables, or defaults
    if labels_file_path is None:
        labels_file_path = os.getenv('LABELS_FILE')
    
    if images_dir_path is None:
        images_dir_path = os.getenv('IMAGES_DIR')
    
    # Use defaults if not specified
    if labels_file_path is None or images_dir_path is None:
        defaults = get_default_paths()
        if labels_file_path is None:
            labels_file_path = str(defaults['labels_file'])
        if images_dir_path is None:
            images_dir_path = str(defaults['images_dir'])
    
    # Convert to Path objects
    labels_file = Path(labels_file_path)
    images_dir = Path(images_dir_path)
    
    # Check if labels file exists
    if not labels_file.exists():
        print(f"Error: labels.jsonl not found at: {labels_file}")
        print("Please specify the correct path using --labels-file or LABELS_FILE environment variable")
        return
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"Warning: Images directory not found at: {images_dir}")
        print("Please specify the correct path using --images-dir or IMAGES_DIR environment variable")
        print("Will attempt to use paths from labels file...")
    
    # Load data
    data = load_labels(str(labels_file))
    
    # Derive the set of all classes across the dataset
    all_classes = set()
    for it in data:
        lbls = it.get('labels', {}) or {}
        for k in lbls.keys():
            all_classes.add(k)
    base_dir = str(images_dir)
    current_index = 0
    consecutive_not_found = 0
    
    print(f"Labels file: {labels_file}")
    print(f"Images directory: {base_dir}")
    print("Navigation: ← → ↑ ↓ arrows or A/D keys to navigate, ESC to exit")
    
    while True:
        if current_index >= len(data):
            current_index = 0
        elif current_index < 0:
            current_index = len(data) - 1
            
        item = data[current_index]
        image_path = get_image_path(item['image'], base_dir)
        
        print(f"Looking for image: {image_path}")
        print(f"Original path from JSON: {item['image']}")
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            consecutive_not_found += 1
            if consecutive_not_found >= len(data):
                print("No images found! Check if the images exist in the correct path.")
                break
            current_index += 1
            continue
        
        consecutive_not_found = 0  # Reset counter when image is found
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            current_index += 1
            continue
        
        # Draw labels (ensure all classes are shown)
        image_with_labels = draw_labels_on_image(image.copy(), item.get('labels', {}), all_classes=all_classes)
        
        # Add image info and navigation with better visibility
        info_text = f"Image {current_index+1}/{len(data)} - {os.path.basename(image_path)}"
        nav_text = "< > ^ v or A/D to navigate, ESC to exit"
        
        # Draw background rectangles for better text visibility
        info_y = image_with_labels.shape[0] - 60
        nav_y = image_with_labels.shape[0] - 20
        
        # Background for info text
        (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_with_labels, (15, info_y - info_h - 5), 
                     (25 + info_w, info_y + 5), (0, 0, 0), -1)
        cv2.rectangle(image_with_labels, (15, info_y - info_h - 5), 
                     (25 + info_w, info_y + 5), (255, 255, 255), 2)
        
        # Background for navigation text
        (nav_w, nav_h), _ = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image_with_labels, (15, nav_y - nav_h - 5), 
                     (25 + nav_w, nav_y + 5), (0, 0, 0), -1)
        cv2.rectangle(image_with_labels, (15, nav_y - nav_h - 5), 
                     (25 + nav_w, nav_y + 5), (255, 255, 255), 2)
        
        # Draw text with better contrast
        cv2.putText(image_with_labels, info_text, (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image_with_labels, nav_text, (20, nav_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize if too large
        max_width = 1200
        max_height = 800
        height, width = image_with_labels.shape[:2]
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_with_labels = cv2.resize(image_with_labels, (new_width, new_height))
        
        # Show image
        cv2.imshow('Image Labels Viewer', image_with_labels)
        
        # Wait for key and handle navigation
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 83 or key == 2 or key == ord('d'):  # Right arrow or D
            current_index += 1
        elif key == 81 or key == 0 or key == ord('a'):  # Left arrow or A
            current_index -= 1
        elif key == 84 or key == 3:  # Down arrow
            current_index += 1
        elif key == 82 or key == 1:  # Up arrow
            current_index -= 1
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Show images with specified or default paths
    show_images_with_labels(
        labels_file_path=args.labels_file,
        images_dir_path=args.images_dir
    )
