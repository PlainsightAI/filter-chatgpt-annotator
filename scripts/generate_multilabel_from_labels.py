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


def analyze_confidence_distribution(labels_data):
    """Analyze confidence distribution to suggest optimal threshold."""
    confidence_scores = []
    present_objects = []
    
    for item in labels_data:
        for class_name, label_info in item['labels'].items():
            confidence = label_info['confidence']
            confidence_scores.append(confidence)
            if label_info['present']:
                present_objects.append(confidence)
    
    if not present_objects:
        return 0.5, "No present objects found"
    
    # Calculate statistics
    present_objects.sort()
    n = len(present_objects)
    
    # Suggest thresholds
    thresholds = {
        "0.5": len([c for c in present_objects if c >= 0.5]),
        "0.6": len([c for c in present_objects if c >= 0.6]),
        "0.7": len([c for c in present_objects if c >= 0.7]),
        "0.8": len([c for c in present_objects if c >= 0.8]),
        "0.9": len([c for c in present_objects if c >= 0.9])
    }
    
    # Find optimal threshold (include most objects while filtering noise)
    optimal_threshold = 0.7
    for threshold, count in thresholds.items():
        if count >= n * 0.8:  # Include at least 80% of present objects
            optimal_threshold = float(threshold)
            break
    
    return optimal_threshold, f"Found {n} present objects. Suggested threshold: {optimal_threshold}"


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
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate multilabel datasets from existing labels.jsonl files')
    parser.add_argument('directories', nargs='+', help='Directories containing labels.jsonl files')
    parser.add_argument('--confidence', '-c', type=float, default=0.7, 
                       help='Confidence threshold for filtering labels (default: 0.7)')
    
    args = parser.parse_args()
    
    directories = args.directories
    confidence_threshold = args.confidence
    
    for directory in directories:
        labels_file = os.path.join(directory, "labels.jsonl")
        multilabel_dir = os.path.join(directory, "multilabel_datasets")
        
        if not os.path.exists(labels_file):
            print(f"Warning: {labels_file} not found, skipping {directory}")
            continue
        
        print(f"\nProcessing {directory}...")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Load labels
        labels_data = load_labels_jsonl(labels_file)
        print(f"Loaded {len(labels_data)} labels from {labels_file}")
        
        # Analyze confidence distribution
        optimal_threshold, analysis_msg = analyze_confidence_distribution(labels_data)
        print(f"Confidence analysis: {analysis_msg}")
        
        # Use optimal threshold if user didn't specify one
        if confidence_threshold == 0.7:  # Default value
            confidence_threshold = optimal_threshold
            print(f"Using optimal threshold: {confidence_threshold}")
        
        # Generate multilabel dataset
        summary = generate_multilabel_dataset(labels_data, multilabel_dir, confidence_threshold)
        
        print(f"✓ Completed {directory}")


if __name__ == "__main__":
    main()
