from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ServiceError(RuntimeError):
    pass


def json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "message": message}), status


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    upload_dir = Path(args.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    detector: Any | None = None
    load_error: str | None = None

    if args.algorithm == "object_detection":
        try:
            detector = create_detector(args.runtime_type, Path(args.model_path))
        except Exception as exc:
            load_error = str(exc)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": load_error is None,
                "model_id": args.model_id,
                "algorithm": args.algorithm,
                "loaded": load_error is None,
                "message": load_error,
            }
        ), 200 if load_error is None else 503

    @app.post("/predict")
    def predict():
        if load_error:
            return json_error(load_error, 503)
        image = request.files.get("image")
        if not image or not image.filename:
            return json_error("请上传 image 文件")
        filename = secure_filename(Path(image.filename).name)
        if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return json_error("仅支持 jpg、png、bmp、webp 图片")
        image_path = upload_dir / f"{time.time_ns()}_{filename}"
        image.save(image_path)
        started = time.perf_counter()
        try:
            payload = infer(args.algorithm, args.model_name, image_path, detector)
        except ServiceError as exc:
            return json_error(str(exc), 500)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify({"ok": True, "algorithm": args.algorithm, "latency_ms": latency_ms, "result": payload})

    @app.post("/infer")
    def infer_base64():
        try:
            if load_error:
                return jsonify({"success": False, "msg": load_error})
            if args.algorithm != "object_detection":
                return jsonify({"success": False, "msg": "Only object detection models support /infer"})
            payload = request.get_json(silent=True) or {}
            image_base64 = str(payload.get("image", "")).strip()
            if not image_base64:
                return jsonify({"success": False, "msg": "image is required"})

            image = decode_base64_cv_image(image_base64)
            detections = infer_array(args.algorithm, image, detector)
            result = format_legacy_detection_result(detections, detector)
            if not result["objectVec"]:
                return jsonify({"success": False, "msg": "No detection results!"})
            result["success"] = True
            result["msg"] = "Detection target output"
            return jsonify(result)
        except Exception as exc:
            return jsonify({"success": False, "msg": f"Exception: {exc}"})

    return app


def infer(
    algorithm: str,
    model_name: str,
    image_path: Path,
    detector: Any | None,
) -> Any:
    if algorithm == "object_detection":
        if detector is None:
            raise ServiceError("目标检测模型未初始化")
        return detector.detect(image_path)
    if algorithm == "classification":
        return {
            "top1": {"label": "unknown", "score": 0.72},
            "top5": [
                {"label": "unknown", "score": 0.72},
                {"label": "sample", "score": 0.11},
                {"label": "background", "score": 0.08},
            ],
            "note": f"{model_name} 的分类推理桩，可替换为真实分类模型。",
        }
    if algorithm == "vlm":
        return {
            "text": f"已接收图片 {image_path.name}，{model_name} 的 VLM 推理桩返回此描述文本。",
            "answer": "当前沙箱接口已打通，可在 model_service.py 中接入真实 VLM。",
        }
    raise ServiceError(f"不支持的算法类型：{algorithm}")


def infer_array(algorithm: str, image: np.ndarray, detector: Any | None) -> Any:
    if algorithm == "object_detection":
        if detector is None:
            raise ServiceError("目标检测模型未初始化")
        return detector.detect_array(image)
    raise ServiceError(f"不支持的算法类型：{algorithm}")


def decode_base64_image(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    return base64.b64decode(image_base64, validate=True)


def decode_base64_cv_image(image_base64: str) -> np.ndarray:
    image_bytes = decode_base64_image(image_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ServiceError("无法解码图片")
    return image


def format_legacy_detection_result(detections: list[dict[str, Any]], detector: Any | None) -> dict[str, Any]:
    names = detector_names(detector)
    object_vec = []
    for det in detections:
        bbox = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
        label = str(det.get("label") or det.get("class") or "")
        class_id = int(det.get("class_id", class_id_for_label(label, names)))
        item = [
            int(round(float(bbox[0]))),
            int(round(float(bbox[1]))),
            int(round(float(bbox[2]))),
            int(round(float(bbox[3]))),
            float(det.get("score", det.get("confidence", 0))),
            class_id,
        ]
        if is_small_filtered(item, 3):
            continue
        item.append(names[class_id] if 0 <= class_id < len(names) else label)
        keys = ["x0", "y0", "x1", "y1", "confidence", "classId", "className"]
        object_vec.append({"classVec": {key: value for key, value in zip(keys, item)}})
    return {"objectVec": object_vec}


def detector_names(detector: Any | None) -> list[str]:
    if detector is None:
        return []
    if hasattr(detector, "names"):
        return normalize_names(detector.names)
    model = getattr(detector, "model", None)
    if model is None:
        return []
    names = getattr(getattr(model, "module", model), "names", None)
    return normalize_names(names)


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def create_detector(runtime_type: str, model_path: Path) -> Any:
    if runtime_type == "yolov5_v5":
        from detector import YoloV5Detector

        detector = YoloV5Detector()
        detector.load(model_path)
        return detector
    if runtime_type == "yolov5_v6":
        return YoloV5V6Runtime(model_path)
    if runtime_type == "yolov8_obb":
        return YoloV8OBBRuntime(model_path)
    raise ServiceError(f"不支持的推理后端：{runtime_type}")


class YoloV5V6Runtime:
    def __init__(self, model_path: Path) -> None:
        self.runtime_dir = Path(__file__).resolve().parent / "runtime-type" / "yolov5-6.0"
        if not self.runtime_dir.exists():
            raise ServiceError("未找到 YOLOv5-6.0 runtime 工程")
        old_cwd = Path.cwd()
        old_model_path = os.environ.get("MODEL_PATH")
        try:
            os.environ["MODEL_PATH"] = str(model_path)
            if str(self.runtime_dir) not in sys.path:
                sys.path.insert(0, str(self.runtime_dir))
            os.chdir(self.runtime_dir)
            from yolov5 import YOLOv5

            self.detector = YOLOv5()
            self.names = detector_names(self.detector)
        finally:
            if old_model_path is None:
                os.environ.pop("MODEL_PATH", None)
            else:
                os.environ["MODEL_PATH"] = old_model_path
            os.chdir(old_cwd)

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ServiceError(f"无法读取图片：{image_path.name}")
        return self.detect_array(image)

    def detect_array(self, image: np.ndarray) -> list[dict[str, Any]]:
        outs, names, _ = self.detector.infer(image)
        self.names = normalize_names(names)
        if outs is None or len(outs) == 0:
            return []
        detections = []
        for item in np.asarray(outs).tolist():
            detections.append(
                {
                    "label": self.names[int(item[5])] if 0 <= int(item[5]) < len(self.names) else str(int(item[5])),
                    "score": round(float(item[4]), 4),
                    "bbox": [int(round(float(item[0]))), int(round(float(item[1]))), int(round(float(item[2]))), int(round(float(item[3])))],
                    "class_id": int(item[5]),
                    "area": float(item[6]) if len(item) > 6 else None,
                }
            )
        return detections


class YoloV8OBBRuntime:
    def __init__(self, model_path: Path) -> None:
        self.runtime_dir = Path(__file__).resolve().parent / "runtime-type" / "yolov8-obb"
        if not self.runtime_dir.exists():
            raise ServiceError("未找到 YOLOv8-OBB runtime 工程")
        if str(self.runtime_dir) not in sys.path:
            sys.path.insert(0, str(self.runtime_dir))
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.names = normalize_names(getattr(self.model, "names", None))

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ServiceError(f"无法读取图片：{image_path.name}")
        return self.detect_array(image)

    def detect_array(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = self.model(image)[0]
        names = normalize_names(getattr(result, "names", None) or getattr(self.model, "names", None))
        if names:
            self.names = names
        obb = getattr(result, "obb", None)
        if obb is None or obb.data is None or len(obb.data) == 0:
            return []
        boxes = obb.data.cpu()
        points = xywhr2xyxyxyxy(boxes[..., :5]).view(len(boxes), -1).tolist()
        confs = boxes[..., 5].tolist()
        classes = list(map(int, boxes[..., 6].tolist()))
        detections = []
        for index, flat_points in enumerate(points):
            pairs = [[int(round(float(flat_points[i]))), int(round(float(flat_points[i + 1])))] for i in range(0, 8, 2)]
            xs = [point[0] for point in pairs]
            ys = [point[1] for point in pairs]
            class_id = classes[index]
            detections.append(
                {
                    "label": self.names[class_id] if 0 <= class_id < len(self.names) else str(class_id),
                    "score": round(float(confs[index]), 4),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "obb": pairs,
                    "class_id": class_id,
                }
            )
        return detections


def xywhr2xyxyxyxy(center: Any) -> Any:
    import torch

    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)
    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def class_id_for_label(label: str, names: list[str]) -> int:
    try:
        return names.index(label)
    except ValueError:
        return -1


def is_small_filtered(item: list[Any], class_id: int) -> bool:
    return int(item[5]) == class_id and (int(item[2] - item[0]) < 20 or int(item[3] - item[1]) < 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--algorithm", choices=["object_detection", "classification", "vlm"], required=True)
    parser.add_argument("--runtime-type", default="yolov5_v5", choices=["yolov5_v5", "yolov5_v6", "yolov8_obb"])
    parser.add_argument("--upload-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    create_app(parsed).run(host=parsed.host, port=parsed.port, debug=False, threaded=True, use_reloader=False)
