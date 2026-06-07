"""
Export YOLOv8 to ONNX for edge deployment.
ONNX export gives ~40% inference speed boost over PyTorch on CPU.

Usage:
    python export_onnx.py
    python export_onnx.py --model yolov8s.pt --imgsz 640
"""

import argparse
from ultralytics import YOLO


def export(model_path: str, imgsz: int, dynamic: bool, simplify: bool):
    model = YOLO(model_path)
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=17,
    )
    print(f"\nONNX model exported → {out}")
    print("Run inference with onnxruntime:")
    print(f"  import onnxruntime as ort")
    print(f"  sess = ort.InferenceSession('{out}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="yolov8n.pt")
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--dynamic",  action="store_true", default=True)
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()
    export(args.model, args.imgsz, args.dynamic, args.simplify)
