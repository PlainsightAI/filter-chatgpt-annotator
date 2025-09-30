#!/usr/bin/env python3
"""
Simple script to show images with bounding box labels overlaid
Shows all images in sequence with bounding boxes drawn on them

Usage:
    python show_bbox.py [options]

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

def load_labels(labels_file_path):
    """Load labels from jsonl file"""
    data = []
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
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
            # Remove 'output_frames/' prefix
            image_path = image_path[13:]
        elif image_path.startswith('./output_frames/'):
            # Remove './output_frames/' prefix
            image_path = image_path[16:]
        
        # Remove leading slash if it exists
        if image_path.startswith('/'):
            image_path = image_path[1:]
        
        # Join with base_dir
        return os.path.join(base_dir, image_path)
    return image_path

def draw_bboxes_on_image(image, labels):
    """Draw bounding boxes and labels on image"""
    height, width = image.shape[:2]
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    color_index = 0
    
    for label_name, label_data in labels.items():
        present = label_data.get('present', False)
        confidence = label_data.get('confidence', 0.0)
        bbox = label_data.get('bbox', None)
        
        # Only draw if present and has bounding box
        if present and bbox is not None and len(bbox) == 4:
            # Convert normalized coordinates to absolute coordinates
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            x_max = int(bbox[2] * width)
            y_max = int(bbox[3] * height)
            
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))
            
            # Get color for this class
            color = colors[color_index % len(colors)]
            color_index += 1
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Prepare label text
            label_text = f"{label_name} ({confidence:.2f})"
            
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(image, (x_min, y_min - text_height - 10), 
                         (x_min + text_width + 10, y_min), color, -1)
            
            # Draw text
            cv2.putText(image, label_text, (x_min + 5, y_min - 5), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Show images with bounding box labels overlaid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default paths
    python show_bbox.py
    
    # Specify custom labels file
    python show_bbox.py --labels-file /path/to/custom/labels.jsonl
    
    # Specify custom images directory
    python show_bbox.py --images-dir /path/to/images
    
    # Use environment variables
    LABELS_FILE=/path/to/labels.jsonl IMAGES_DIR=/path/to/images python show_bbox.py
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

def show_images_with_bboxes(labels_file_path=None, images_dir_path=None):
    """Show all images with bounding boxes and navigation"""
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
        
        # Draw bounding boxes
        image_with_bboxes = draw_bboxes_on_image(image.copy(), item['labels'])
        
        # Add image info and navigation
        info_text = f"Image {current_index+1}/{len(data)} - {os.path.basename(image_path)}"
        nav_text = "← → ↑ ↓ or A/D to navigate, ESC to exit"
        
        cv2.putText(image_with_bboxes, info_text, (20, image_with_bboxes.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_bboxes, nav_text, (20, image_with_bboxes.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Resize if too large
        max_width = 1200
        max_height = 800
        height, width = image_with_bboxes.shape[:2]
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_with_bboxes = cv2.resize(image_with_bboxes, (new_width, new_height))
        
        # Show image
        cv2.imshow('Bounding Box Viewer', image_with_bboxes)
        
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
    show_images_with_bboxes(
        labels_file_path=args.labels_file,
        images_dir_path=args.images_dir
    )
