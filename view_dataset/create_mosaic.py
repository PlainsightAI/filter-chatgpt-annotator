#!/usr/bin/env python3
"""
Script to create before/after mosaics for demo purposes.
Creates two mosaics: one without annotations and one with annotations.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def load_labels_jsonl(labels_file_path):
    """Load labels from JSONL file."""
    try:
        labels_data = []
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labels_data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(labels_data)} images")
        return labels_data
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
        
        # If the path starts with 'data/', remove it since base_dir already points to data directory
        if image_path.startswith('data/'):
            image_path = image_path[5:]  # Remove 'data/' prefix
        
        # Join with base_dir
        return os.path.join(base_dir, image_path)
    return image_path

def draw_labels_on_image(image, labels):
    """Draw only PRESENT classification labels on image with colors."""
    height, width = image.shape[:2]
    
    # Text settings - make labels very large to cover most of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(2.0, min(8.0, width / 100))  # Much larger scale for mosaic
    thickness = max(3, int(width / 100))  # Much thicker
    
    # Position for labels - center of image
    y_offset = int(height * 0.5)  # Center vertically
    x_offset = int(width * 0.1)  # Start from left side
    
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
        padding = max(10, int(width * 0.05))  # Larger padding
        cv2.rectangle(image, (x_offset - padding, y_offset - text_height - padding), 
                     (x_offset + text_width + padding, y_offset + padding), (0, 0, 0), -1)
        cv2.putText(image, text, (x_offset, y_offset), font, font_scale, color, thickness)
        return image
    
    # Draw only present labels
    for label_name, confidence in present_labels:
        # Color: green for present
        color = (0, 255, 0)
        
        # Show only present labels
        text = f"{label_name}"
        
        # Draw text with background - make it cover most of the image
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        padding = max(10, int(width * 0.05))  # Smaller padding for smaller rectangle
        cv2.rectangle(image, (x_offset - padding, y_offset - text_height - padding), 
                     (x_offset + text_width + padding, y_offset + padding), (0, 0, 0), -1)
        cv2.putText(image, text, (x_offset, y_offset), font, font_scale, color, thickness)
        y_offset += int(height * 0.3)  # Much more space between labels
    
    return image

def resize_image_for_mosaic(image, target_size=(400, 400)):
    """Resize image to target size while maintaining aspect ratio - images will be cropped to fit exactly."""
    height, width = image.shape[:2]
    
    # Calculate scale to fill the target size (may crop but won't stretch)
    scale = max(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Crop to exact target size from center
    start_x = (new_width - target_size[0]) // 2
    start_y = (new_height - target_size[1]) // 2
    
    cropped = resized[start_y:start_y + target_size[1], start_x:start_x + target_size[0]]
    
    return cropped

def create_mosaic(images, labels_data, base_dir, output_path, with_annotations=True, grid_size=(5, 2), target_size=(400, 400)):
    """Create a mosaic of images."""
    rows, cols = grid_size
    
    # Add spacing between rows (20 pixels)
    row_spacing = 20
    mosaic_width = cols * target_size[0]
    mosaic_height = rows * target_size[1] + (rows - 1) * row_spacing
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    
    # Process images
    for i, (image_path, labels) in enumerate(zip(images, labels_data)):
        if i >= rows * cols:
            break
            
        # Get full image path
        full_image_path = get_image_path(image_path, base_dir)
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        # Load image
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Warning: Could not load image: {full_image_path}")
            continue
        
        # Add annotations if requested
        if with_annotations:
            image = draw_labels_on_image(image, labels)
        
        # Resize for mosaic
        resized_image = resize_image_for_mosaic(image, target_size)
        
        # Calculate position in mosaic with row spacing
        row = i // cols
        col = i % cols
        
        y_start = row * (target_size[1] + row_spacing)
        y_end = y_start + target_size[1]
        x_start = col * target_size[0]
        x_end = x_start + target_size[0]
        
        # Place image in mosaic
        mosaic[y_start:y_end, x_start:x_end] = resized_image
    
    # Save mosaic
    cv2.imwrite(output_path, mosaic)
    print(f"Saved mosaic: {output_path}")
    
    return mosaic

def main():
    """Main function to create before/after mosaics."""
    parser = argparse.ArgumentParser(description='Create before/after mosaics for demo')
    parser.add_argument('--labels-file', 
                       help='Path to labels.jsonl file')
    parser.add_argument('--images-dir', 
                       help='Base directory for images')
    parser.add_argument('--output-dir', 
                       help='Output directory for mosaics (default: current directory)')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to include in mosaic (default: 10)')
    parser.add_argument('--grid-size', default='5x2',
                       help='Grid size for mosaic (default: 5x2)')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Create fullscreen mosaic (1920x1080)')
    
    args = parser.parse_args()
    
    # Parse grid size
    if args.fullscreen:
        # Fullscreen layout: 4x3 grid for 12 images
        grid_size = (4, 3)
        target_size = (640, 360)  # 1920/3 = 640, 1080/3 = 360
    else:
        try:
            grid_parts = args.grid_size.split('x')
            grid_size = (int(grid_parts[0]), int(grid_parts[1]))
        except:
            grid_size = (5, 2)
        target_size = (400, 400)
    
    # Get paths
    if args.labels_file:
        labels_file = args.labels_file
    else:
        labels_file = "../output_frames_demo_qrs1/labels.jsonl"
    
    if args.images_dir:
        images_dir = args.images_dir
    else:
        images_dir = "../output_frames_demo_qrs1/data"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "."
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels
    labels_data = load_labels_jsonl(labels_file)
    
    # Select random images
    if args.fullscreen:
        num_images = min(12, len(labels_data))  # 4x3 = 12 images for fullscreen
    else:
        num_images = min(args.num_images, len(labels_data))
    selected_indices = random.sample(range(len(labels_data)), num_images)
    selected_data = [labels_data[i] for i in selected_indices]
    
    print(f"Selected {num_images} images for mosaic")
    
    # Create before mosaic (without annotations)
    before_path = os.path.join(output_dir, "mosaic_before.png")
    print("Creating 'before' mosaic (without annotations)...")
    create_mosaic(
        [item['image'] for item in selected_data],
        [item['labels'] for item in selected_data],
        images_dir,
        before_path,
        with_annotations=False,
        grid_size=grid_size,
        target_size=target_size
    )
    
    # Create after mosaic (with annotations)
    after_path = os.path.join(output_dir, "mosaic_after.png")
    print("Creating 'after' mosaic (with annotations)...")
    create_mosaic(
        [item['image'] for item in selected_data],
        [item['labels'] for item in selected_data],
        images_dir,
        after_path,
        with_annotations=True,
        grid_size=grid_size,
        target_size=target_size
    )
    
    print(f"\n✅ Mosaics created successfully!")
    print(f"📁 Before (no annotations): {before_path}")
    print(f"📁 After (with annotations): {after_path}")
    print(f"\n🎯 These can be used for the demo 'before/after' comparison!")

if __name__ == '__main__':
    main()
