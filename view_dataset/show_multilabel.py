#!/usr/bin/env python3
"""
Simple script to show images with multilabel classification results from COCO format
Shows all images in sequence with only present classes displayed

Usage:
    python show_multilabel.py [options]

Options:
    --annotations-file PATH   Path to COCO annotations.json file (default: ../output_frames/detection_datasets/annotations.json)
    --images-dir PATH         Base directory for images (default: ../output_frames/data)
    --help                    Show this help message

Environment Variables:
    ANNOTATIONS_FILE          Path to COCO annotations.json file
    IMAGES_DIR                Base directory for images
"""

import json
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

def load_coco_annotations(annotations_file_path):
    """Load COCO format annotations from JSON file and convert to labels format"""
    try:
        with open(annotations_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert COCO format to labels format
        labels_data = []
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        for image in data['images']:
            image_id = image['id']
            
            # Get annotations for this image
            image_annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
            
            # Create labels dict showing presence of each category
            labels = {}
            for cat_id, cat_name in categories.items():
                # Check if this category is present in this image
                present = any(ann['category_id'] == cat_id for ann in image_annotations)
                confidence = 1.0 if present else 0.0
                labels[cat_name] = {'present': present, 'confidence': confidence}
            
            labels_data.append({
                'image': image['file_name'],
                'labels': labels
            })
        
        print(f"Loaded {len(labels_data)} images with {len(categories)} categories")
        return labels_data, list(categories.values())
    except FileNotFoundError:
        print(f"File not found: {annotations_file_path}")
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

def draw_labels_on_image(image, labels):
    """Draw only PRESENT classification labels on image with colors."""
    height, width = image.shape[:2]
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Position for labels
    y_offset = 40
    x_offset = 20
    
    # Get only present labels
    present_labels = []
    for label_name, label_data in labels.items():
        if label_data.get('present', False):
            confidence = label_data.get('confidence', 0.0)
            present_labels.append((label_name, confidence))
    
    # If no labels present, show "No objects detected"
    if not present_labels:
        text = "No objects detected"
        color = (128, 128, 128)  # Gray
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (x_offset - 5, y_offset - text_height - 5), 
                     (x_offset + text_width + 5, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (x_offset, y_offset), font, font_scale, color, thickness)
        return image
    
    # Draw only present labels
    for label_name, confidence in present_labels:
        # Color: green for present
        color = (0, 255, 0)
        
        # Show only present labels
        text = f"{label_name}"
        
        # Draw text with background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (x_offset - 5, y_offset - text_height - 5), 
                     (x_offset + text_width + 5, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (x_offset, y_offset), font, font_scale, color, thickness)
        y_offset += 35
    
    return image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Show images with multilabel classification results from COCO format')
    parser.add_argument('--annotations-file', 
                       help='Path to COCO annotations.json file')
    parser.add_argument('--images-dir', 
                       help='Base directory for images')
    return parser.parse_args()

def get_default_paths():
    """Get default paths for annotations and images"""
    script_dir = Path(__file__).parent
    return {
        'annotations_file': script_dir.parent / 'output_frames' / 'detection_datasets' / 'annotations.json',
        'images_dir': script_dir.parent / 'output_frames' / 'data'
    }

def show_images_with_multilabel(annotations_file_path=None, images_dir_path=None):
    """Show all images with multilabel annotations"""
    # Get paths from environment variables if available
    if annotations_file_path is None:
        annotations_file_path = os.getenv('ANNOTATIONS_FILE')
    if images_dir_path is None:
        images_dir_path = os.getenv('IMAGES_DIR')
    
    # Use defaults if not specified
    if annotations_file_path is None or images_dir_path is None:
        defaults = get_default_paths()
        if annotations_file_path is None:
            annotations_file_path = str(defaults['annotations_file'])
        if images_dir_path is None:
            images_dir_path = str(defaults['images_dir'])
    
    # Convert to Path objects
    annotations_file = Path(annotations_file_path)
    images_dir = Path(images_dir_path)
    
    # Check if annotations file exists
    if not annotations_file.exists():
        print(f"Error: annotations.json not found at: {annotations_file}")
        print("Please specify the correct path using --annotations-file or ANNOTATIONS_FILE environment variable")
        return
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"Warning: Images directory not found at: {images_dir}")
        print("Please specify the correct path using --images-dir or IMAGES_DIR environment variable")
        print("Will attempt to use paths from annotations file...")
    
    # Load data
    data, all_classes = load_coco_annotations(str(annotations_file))
    
    base_dir = str(images_dir)
    current_index = 0
    consecutive_not_found = 0
    
    print(f"Annotations file: {annotations_file}")
    print(f"Images directory: {base_dir}")
    print(f"All classes: {all_classes}")
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
        
        # Draw labels (show only present classes)
        image_with_labels = draw_labels_on_image(image.copy(), item.get('labels', {}))
        
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
        cv2.imshow('Multilabel Classification Viewer', image_with_labels)
        
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

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Show images with specified or default paths
    show_images_with_multilabel(
        annotations_file_path=args.annotations_file,
        images_dir_path=args.images_dir
    )
