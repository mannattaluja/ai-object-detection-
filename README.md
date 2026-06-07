# AI-Powered Object Detection

Real-time object detection pipeline using **YOLOv8 + OpenCV**, achieving ~89% mAP50 at ~45 FPS on GPU. Includes a Flask REST API and ONNX export for edge deployment.

## Features

- **Real-time detection** — webcam, video file, or image input
- **YOLOv8** backbone with configurable confidence threshold
- **Flask REST API** — POST an image, receive JSON detections + annotated image (base64)
- **ONNX export** — ~40% inference speed boost for edge devices

## Project Structure

```
ai-object-detection/
├── detector.py        # Core detection pipeline (webcam / video / image)
├── app.py             # Flask REST API
├── export_onnx.py     # ONNX export for edge deployment
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) will be auto-downloaded on first run.

## Usage

### Command Line
```bash
# Webcam
python detector.py --source webcam

# Image
python detector.py --source photo.jpg

# Video
python detector.py --source video.mp4 --output output.mp4
```

### Flask API
```bash
python app.py
# Server runs on http://localhost:5000

# Detect objects in an image
curl -X POST -F "file=@image.jpg" http://localhost:5000/detect

# Health check
curl http://localhost:5000/health

# List classes
curl http://localhost:5000/classes
```

### ONNX Export
```bash
python export_onnx.py --model yolov8n.pt
```

## API Response Format
```json
{
  "count": 3,
  "detections": [
    {"label": "person", "confidence": 0.923, "bbox": [120, 80, 340, 560]},
    {"label": "car",    "confidence": 0.871, "bbox": [400, 200, 700, 420]}
  ],
  "image_b64": "<base64-encoded annotated JPEG>"
}
```

## Requirements

- Python 3.9+
- CUDA GPU recommended for ~45 FPS (CPU works too, slower)
