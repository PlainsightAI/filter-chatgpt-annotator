#!/usr/bin/env python3
"""
Script to generate multilabel datasets from existing labels.jsonl files.
This avoids calling the API again and generates the multilabel datasets directly.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time


def load_labels_jsonl(labels_file):
    """Load labels from jsonl file."""
    labels_data = []
    with open(labels_file, 'r') as f:
        for line in f:
            if line.strip():
                labels_data.append(json.loads(line.strip()))
    return labels_data


def extract_image_info(image_path):
    """Extract image filename and dimensions from path."""
    filename = os.path.basename(image_path)
    # Default dimensions - you might want to get actual dimensions
    return filename, 640, 480


def generate_multilabel_dataset(labels_data, output_dir, confidence_threshold=0.9):
    """Generate multilabel dataset in COCO format from labels.jsonl data."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all unique classes
    all_classes = set()
    for item in labels_data:
        for class_name in item['labels'].keys():
            all_classes.add(class_name)
    
    classes = sorted(list(all_classes))
    category_mapping = {class_name: idx + 1 for idx, class_name in enumerate(classes)}
    
    # Generate COCO format data
    coco_data = {
        "info": {
            "description": "ChatGPT Annotator Multilabel Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "ChatGPT Annotator Filter",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        "categories": [
            {
                "id": category_id,
                "name": class_name,
                "supercategory": "object"
            }
            for class_name, category_id in category_mapping.items()
        ]
    }
    
    # Process each image
    image_id = 1
    annotation_id = 1
    
    for item in labels_data:
        image_path = item['image']
        filename, width, height = extract_image_info(image_path)
        
        # Add image info
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        coco_data["images"].append(image_info)
        
        # Add annotations for present labels
        for class_name, label_info in item['labels'].items():
            if label_info['present'] and label_info['confidence'] >= confidence_threshold:
                # Use full image bounding box for multilabel
                bbox = [0, 0, width, height]
                area = width * height
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_mapping[class_name],
                    "segmentation": [],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    # Save COCO annotations
    annotations_file = os.path.join(output_dir, "annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # Generate summary report
    summary = {
        "task_type": "multilabel_classification",
        "format": "COCO",
        "total_classes": len(classes),
        "classes": classes,
        "category_mapping": category_mapping,
        "total_images": len(coco_data["images"]),
        "total_annotations": len(coco_data["annotations"]),
        "output_directory": output_dir,
        "confidence_threshold": confidence_threshold,
        "coco_file": annotations_file,
        "bbox_type": "full_image_bbox",
        "description": "Each present label gets a bounding box covering the entire image",
        "generated_at": time.time()
    }
    
    summary_file = os.path.join(output_dir, "_summary_report.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated multilabel dataset:")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Classes: {classes}")
    print(f"  - Output: {output_dir}")
    
    return summary


def main():
    """Main function to process directories."""
    if len(sys.argv) < 2:
        print("Usage: python generate_multilabel_from_labels.py <directory1> [directory2] ...")
        print("Example: python generate_multilabel_from_labels.py output_frames_mult_bow_crop_v1 output_frames_mult_bow_crop_v2")
        sys.exit(1)
    
    directories = sys.argv[1:]
    
    for directory in directories:
        labels_file = os.path.join(directory, "labels.jsonl")
        multilabel_dir = os.path.join(directory, "multilabel_datasets")
        
        if not os.path.exists(labels_file):
            print(f"Warning: {labels_file} not found, skipping {directory}")
            continue
        
        print(f"\nProcessing {directory}...")
        
        # Load labels
        labels_data = load_labels_jsonl(labels_file)
        print(f"Loaded {len(labels_data)} labels from {labels_file}")
        
        # Generate multilabel dataset
        summary = generate_multilabel_dataset(labels_data, multilabel_dir)
        
        print(f"✓ Completed {directory}")


if __name__ == "__main__":
    main()
