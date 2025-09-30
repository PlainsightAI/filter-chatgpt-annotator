# Image Labels Viewer

This directory contains interactive tools for visualizing images with their respective overlaid annotations, allowing navigation through datasets processed by the ChatTag Filter.

## Scripts Available

### 1. `show_labels.py` - Classification Labels Viewer
Interactive tool for visualizing **classification** results with overlaid text labels.

### 2. `show_bbox.py` - Bounding Box Viewer  
Interactive tool for visualizing **object detection** results with overlaid bounding boxes.

## Features

### Classification Viewer (`show_labels.py`)
- **Interactive Visualization**: Navigate through images using arrow keys or A/D
- **Overlaid Labels**: Displays classification results with colors (green for present, red for absent)
- **Smart Filtering**: Only shows items that are present or have confidence > 0.3
- **Flexible Paths**: Support for specifying different image directories and label files
- **Automatic Resizing**: Adjusts large images to fit the screen
- **Detailed Information**: Shows image counter, filename, and confidence scores

### Bounding Box Viewer (`show_bbox.py`)
- **Interactive Visualization**: Navigate through images using arrow keys or A/D
- **Bounding Boxes**: Draws colored rectangles around detected objects
- **Multi-Class Support**: Different colors for different object classes
- **Confidence Display**: Shows confidence scores for each detection
- **Flexible Paths**: Support for specifying different image directories and label files
- **Automatic Resizing**: Adjusts large images to fit the screen
- **Detailed Information**: Shows image counter, filename, and detection info

## Usage

### Basic Usage (Default Paths)

#### For Classification Results
```bash
cd view_dataset
python show_labels.py
```

#### For Object Detection Results
```bash
cd view_dataset
python show_bbox.py
```

By default, both scripts look for:
- `../output_frames/labels.jsonl` (labels file)
- `../output_frames/` (images directory)

The scripts are configured to work automatically with the standard ChatGPT Annotator Filter structure, where the `output_frames/` directory contains both the `labels.jsonl` file and the `data/` folder with processed images.

### Specifying Custom Paths

#### Via Command Line Arguments

```bash
# Specify custom labels file
python show_labels.py --labels-file /path/to/custom/labels.jsonl

# Specify custom images directory
python show_labels.py --images-dir /path/to/images

# Specify both
python show_labels.py --labels-file /path/to/labels.jsonl --images-dir /path/to/images
```

#### Via Environment Variables

```bash
# Use environment variables
export LABELS_FILE="/path/to/custom/labels.jsonl"
export IMAGES_DIR="/path/to/custom/images"
python show_labels.py

# Or in one line
LABELS_FILE="/path/to/labels.jsonl" IMAGES_DIR="/path/to/images" python show_labels.py
```

### Usage Examples

#### Classification Examples
```bash
# View food classification dataset
python show_labels.py --labels-file ../output_frames/food_analysis/labels.jsonl --images-dir ../output_frames/food_analysis

# View pet classification dataset
python show_labels.py --labels-file ../output_frames/pet_classification/labels.jsonl --images-dir ../output_frames/pet_classification

# View industrial quality dataset
python show_labels.py --labels-file /data/quality_inspection/labels.jsonl --images-dir /data/quality_inspection/images
```

#### Object Detection Examples
```bash
# View food detection dataset with bounding boxes
python show_bbox.py --labels-file ../output_frames/food_detection/labels.jsonl --images-dir ../output_frames/food_detection

# View avocado detection dataset
python show_bbox.py --labels-file ../output_frames/avocado_detection/labels.jsonl --images-dir ../output_frames/avocado_detection

# View custom detection dataset
python show_bbox.py --labels-file /data/object_detection/labels.jsonl --images-dir /data/object_detection/images
```

## Navigation Controls

| Key | Action |
|-----|--------|
| `←` `→` `↑` `↓` | Navigate between images |
| `A` / `D` | Navigate left/right |
| `ESC` | Exit viewer |

## Data Format

Both scripts expect a `labels.jsonl` file with the following format:

### Classification Format
```json
{"image": "path/to/image1.jpg", "labels": {"item1": {"present": true, "confidence": 0.9}, "item2": {"present": false, "confidence": 0.1}}, "usage": {...}}
{"image": "path/to/image2.jpg", "labels": {"item1": {"present": false, "confidence": 0.2}, "item2": {"present": true, "confidence": 0.8}}, "usage": {...}}
```

### Object Detection Format (with Bounding Boxes)
```json
{"image": "path/to/image1.jpg", "labels": {"item1": {"present": true, "confidence": 0.9, "bbox": [0.1, 0.2, 0.8, 0.9]}, "item2": {"present": false, "confidence": 0.1, "bbox": null}}, "usage": {...}}
{"image": "path/to/image2.jpg", "labels": {"item1": {"present": false, "confidence": 0.2, "bbox": null}, "item2": {"present": true, "confidence": 0.8, "bbox": [0.3, 0.4, 0.7, 0.8]}}, "usage": {...}}
```

**Note**: Bounding box coordinates are normalized (0-1) in format `[x_min, y_min, x_max, y_max]`.

## Path Handling

The script is intelligent when dealing with different path formats:

- **Absolute Paths**: Used directly
- **Relative Paths**: Combined with the specified base directory
- **Common Prefixes**: Automatically removes prefixes like `output_frames/` or `./output_frames/`

## Troubleshooting

### Labels File Not Found

```
Error: labels.jsonl not found at: /path/to/labels.jsonl
Please specify the correct path using --labels-file or LABELS_FILE environment variable
```

**Solution**: Check the file path or use `--labels-file` to specify the correct path.

### Images Directory Not Found

```
Warning: Images directory not found at: /path/to/images
Please specify the correct path using --images-dir or IMAGES_DIR environment variable
```

**Solution**: Check the directory path or use `--images-dir` to specify the correct path.

### Images Not Found

```
Image not found: /path/to/image.jpg
```

**Solution**: Verify that the images exist in the specified directory and that the paths in the `labels.jsonl` file are correct.

## Help

To see all available options:

```bash
# For classification viewer
python show_labels.py --help

# For bounding box viewer
python show_bbox.py --help
```

## Dependencies

- `opencv-python` (cv2)
- `numpy`
- Python 3.6+

## Integration with ChatTag Filter

Both scripts are designed to work with ChatTag Filter outputs:

- **`show_labels.py`**: Use with classification datasets (text labels)
- **`show_bbox.py`**: Use with object detection datasets (bounding boxes)

The scripts can also be used with any dataset that follows the specified JSONL format.
