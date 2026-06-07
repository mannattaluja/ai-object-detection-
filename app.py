"""
Flask REST API for Object Detection
=====================================
Endpoints:
  POST /detect       — JSON detections + base64 annotated image
  GET  /health       — health check
  GET  /classes      — list of COCO class names

Usage:
    python app.py
    curl -X POST -F "file=@image.jpg" http://localhost:5000/detect
"""

import io
import base64
import json
from flask import Flask, request, jsonify
import cv2
import numpy as np
from detector import ObjectDetector

app = Flask(__name__)
detector = ObjectDetector(model_path="yolov8n.pt", conf=0.45)


def frame_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "yolov8n"})


@app.get("/classes")
def classes():
    return jsonify({"classes": list(detector.model.names.values())})


@app.post("/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a multipart/form-data request with key 'file'."}), 400

    file_bytes = request.files["file"].read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image."}), 400

    results = detector.model(img, conf=detector.conf, verbose=False)[0]
    annotated, detections = detector._annotate(img, results)

    return jsonify({
        "count":      len(detections),
        "detections": detections,
        "image_b64":  frame_to_b64(annotated),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
