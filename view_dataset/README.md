# Image Labels Viewer

Interactive tool for visualizing **classification** results with overlaid text labels and navigation through datasets produced by the ChatTag filter.

## Script

### `show_labels.py`
- Navigate with arrow keys or A/D; ESC to exit.
- Overlaid labels: green for present, red for absent; shows confidence.
- Paths: `--labels-file`, `--images-dir`, or env `LABELS_FILE` / `IMAGES_DIR`.

## Usage

```bash
cd view_dataset
python show_labels.py
```

Defaults: `../output_frames/labels.jsonl` and image paths relative to `../output_frames/`.

```bash
python show_labels.py --labels-file /path/to/labels.jsonl --images-dir /path/to/images
```

## Data format (`labels.jsonl`)

One JSON object per line. Each line includes `schema_version`, `image`, `labels`, `usage`, and `prompt_used` (see [docs/output_contract.md](../docs/output_contract.md)).

Classification labels map:

```json
{"schema_version": "1.0", "image": "path/to/image1.jpg", "labels": {"item1": {"present": true, "confidence": 0.9}, "item2": {"present": false, "confidence": 0.1}}, "usage": {}, "prompt_used": "prompt.txt"}
```

## Dependencies

- `opencv-python` (cv2)
- `numpy`
