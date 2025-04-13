# COCO Dataset Format Converter

This repository contains a collection of tools for converting and verifying COCO dataset annotations into different formats, specifically Pascal VOC and YOLO formats. It provides utilities for both conversion and visualization of the converted datasets.

> **Note**: All source code in this project was developed with the assistance of GitHub Copilot.

## Features

- Convert COCO format to Pascal VOC format (`coco_to_pascal.py`)
- Convert COCO format to YOLO format (`coco_to_yolo.py`)
- Visualize and verify Pascal VOC annotations (`show_pascal.py`)
- Verify YOLO dataset annotations (`verify_yolo_dataset.py`)

## Directory Structure

```
.
├── annotations/                 # COCO annotation files
│   ├── instances_train2014.json
│   └── instances_val2014.json
├── train2014/                  # COCO training images
├── val2014/                    # COCO validation images
├── pascal_coco/               # Converted Pascal VOC format
│   ├── classes.txt
│   ├── train.txt
│   ├── val.txt
│   ├── images/
│   └── labels/
└── yolov3_dataset/           # Converted YOLO format
    ├── classes.txt
    ├── images/
    └── labels/
```

## Usage

### 1. Converting COCO to Pascal VOC

```bash
python coco_to_pascal.py
```

This script:
- Converts COCO annotations to Pascal VOC XML format
- Creates symbolic links to original images
- Generates train.txt and val.txt files
- Handles invalid bounding boxes by fixing or filtering them
- Provides conversion statistics

### 2. Converting COCO to YOLO

```bash
python coco_to_yolo.py
```

This script:
- Converts COCO annotations to YOLO txt format
- Creates symbolic links to original images
- Generates class mapping in classes.txt
- Organizes data into train and validation sets

### 3. Visualizing Pascal VOC Annotations

```bash
python show_pascal.py [-a] [-b]
```

Options:
- `-a`: Enable auto-advance mode (pause only on validation errors)
- `-b`: Run in batch mode (validate all images without displaying)

### 4. Verifying YOLO Dataset

```bash
python verify_yolo_dataset.py
```

This tool:
- Visualizes YOLO format annotations
- Displays bounding boxes and class labels
- Helps verify the correctness of the conversion

## Features of the Conversion Tools

### COCO to Pascal VOC Converter
- Validates and fixes invalid bounding boxes
- Handles edge cases like zero-width/height boxes
- Maintains original image references through symbolic links
- Generates proper Pascal VOC XML structure
- Provides detailed conversion statistics

### COCO to YOLO Converter
- Converts coordinates to YOLO format (normalized)
- Creates proper directory structure
- Maintains dataset splits (train/val)
- Generates continuous class indices

### Visualization Tools
- Interactive visualization with OpenCV
- Support for batch validation
- Error checking and reporting
- Class-colored bounding boxes
- Support for symbolic links

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- XML ElementTree

## License

MIT License

## Note

This toolset assumes you have the COCO dataset (2014) downloaded and properly organized in the specified directory structure. The converters create symbolic links to the original images to save disk space.