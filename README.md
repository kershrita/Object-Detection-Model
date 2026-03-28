# Real-Time Object Detection System (TensorFlow SSD + OpenCV)

> A production-oriented computer vision pipeline for real-time object detection, localization, and classification on image, video, and webcam streams using TensorFlow SSD MobileNet v3 and OpenCV DNN.

## Overview

This project implements an end-to-end object detection system using a TensorFlow SSD model with OpenCV-based inference.

Developed in Nov 2022 and associated with Kafr El-Sheikh University, the system is engineered as a full detection pipeline, not only a model training exercise. It ingests visual input, applies standardized preprocessing, runs neural inference, and returns structured detections for downstream usage.

The implementation emphasizes:

- reliable inference with pre-trained TensorFlow artifacts
- clear preprocessing and label mapping stages
- interpretable output with bounding boxes, class labels, and confidence scores
- design extensibility for image, video, and live camera scenarios

Typical use cases include smart surveillance prototypes, classroom/lab computer vision experiments, and rapid proof-of-concept systems for visual automation.

## Use Cases

- Smart surveillance and scene monitoring prototypes
- Retail and logistics visual counting/inspection proof-of-concepts
- Academic computer vision labs and engineering demonstrations
- Rapid MVP development for AI-powered image/video analytics

## Keywords

Object detection, computer vision, TensorFlow, OpenCV, SSD MobileNet v3, COCO dataset, deep learning, real-time inference, image processing, video analytics, bounding box localization, AI engineering.

## Architecture

The system follows a layered inference pipeline:

![Object Detection Process Funnel](./assets/Object%20Detection%20Process%20Funnel.png)

### Components

- Input Layer
	- Captures frames from webcam or can be adapted for static images/video streams.
- Preprocessing Layer
	- Applies model-compatible transformations: resize to `320x320`, normalization via scale/mean, RGB channel ordering.
- Inference Layer
	- Executes object detection with `cv2.dnn_DetectionModel` using `frozen_inference_graph.pb` and `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`.
- Post-processing Layer
	- Applies confidence filtering (`confThreshold=0.45`) and maps class IDs to human-readable labels.
- Output Layer
	- Produces both visual overlays (bounding boxes + labels + confidence) and structured arrays (`classIds`, `confs`, `bbox`) for integration.

### Architecture Diagram

![Object Detection Process Cycle](./assets/Object%20Detection%20Process%20Cycle.png)

## Features

- Real-time object detection with bounding box localization
- Multi-class prediction using a COCO label map
- Confidence scoring for each detected object
- Modular pipeline that can support webcam, image, and video sources
- Readable and transferable inference workflow for academic and applied AI settings

## Technical Highlights

- Architecture-first implementation
	- The notebook is structured as an explicit inference system with separable stages (input, preprocess, infer, post-process, output), enabling straightforward conversion into a script/service.
- Deployment-aware model choice
	- SSD MobileNet v3 Large was selected as a practical speed/accuracy trade-off for real-time CPU-friendly inference scenarios.
- Externalized model metadata
	- Label names are loaded from a dedicated label file, improving maintainability and reducing hard-coded class dependencies.
- Operational thresholding
	- A confidence threshold is applied to reduce low-quality detections and improve output usefulness in live streams.
- Extensible output contract
	- The system returns both rendered frames and structured detection tensors, supporting downstream analytics or API integration.

## Tech Stack

- Python
- TensorFlow (model artifacts)
- OpenCV (`cv2.dnn` inference engine)
- NumPy
- Jupyter Notebook
- COCO class labels

## Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/kershrita/Object-Detection-Model.git
cd Object-Detection-Model
```

### 2. Create and Activate Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python tensorflow numpy jupyter
```

### 4. Verify Required Artifacts

Ensure these files are present in the project root:

- `frozen_inference_graph.pb`
- `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
- `labels`

### 5. Run Inference Workflow

```bash
jupyter notebook "Object Detection.ipynb"
```

Then execute notebook cells in order to initialize the model and start real-time detection.

## Results

Current implementation output:

- Real-time detections over live camera frames
- Bounding boxes and class labels rendered on each frame
- Per-object confidence scores displayed during inference
- Structured detection arrays (`classIds`, `confs`, `bbox`) available for logging or downstream processing

Example result format:

```text
Detected: PERSON | Confidence: 92.31% | Box: [x, y, w, h]
```

- Role: Applied AI Engineer (implementation, preprocessing pipeline, inference workflow, evaluation)
- Timeline: Nov 2022
- Affiliation: Kafr El-Sheikh University

## License

This project is available under the terms defined in [LICENSE](LICENSE).
