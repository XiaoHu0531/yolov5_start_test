from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


class DetectorError(RuntimeError):
    pass


class YoloV5Detector:
    def __init__(self) -> None:
        self.repo_path = self._find_repo()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model: Any | None = None
        self.model_path: Path | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def repo_ready(self) -> bool:
        return bool(self.repo_path and (self.repo_path / "hubconf.py").exists())

    def _find_repo(self) -> Path | None:
        base_dir = Path(__file__).resolve().parent
        candidates = [
            os.getenv("YOLOV5_REPO"),
            str(base_dir / "yolov5"),
            str(base_dir.parent / "model_test" / "yolov5"),
            str(base_dir.parent / "model_test" / "yolov5" / "yolov5-v5.0"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "hubconf.py").exists():
                return path
            for child in path.iterdir() if path.exists() else []:
                if child.is_dir() and (child / "hubconf.py").exists():
                    return child.resolve()
        return None

    def load(self, model_path: Path) -> None:
        if not self.repo_ready or self.repo_path is None:
            raise DetectorError(
                "未找到完整 YOLOv5 仓库。请设置 YOLOV5_REPO，或将 yolov5 仓库放到本项目目录。"
            )
        try:
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
            try:
                model = torch.hub.load(
                    str(self.repo_path),
                    "custom",
                    path_or_model=str(model_path),
                    source="local",
                    force_reload=False,
                )
            except TypeError as exc:
                if "path_or_model" not in str(exc):
                    raise
                model = torch.hub.load(str(self.repo_path), "custom", path=str(model_path), source="local", force_reload=False)
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_path = model_path
        except Exception as exc:
            raise DetectorError(f"模型加载失败：{exc}") from exc

    def unload(self) -> None:
        self.model = None
        self.model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, image_path: Path, *, conf: float = 0.25, iou: float = 0.45, image_size: int = 640) -> list[dict[str, Any]]:
        if self.model is None:
            raise DetectorError("模型尚未加载")
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DetectorError(f"无法读取图片：{image_path.name}")
        try:
            self.model.conf = conf
            self.model.iou = iou
            results = self.model(str(image_path), size=image_size)
        except Exception as exc:
            raise DetectorError(f"推理失败：{exc}") from exc

        return self._detections_from_results(results)

    @torch.no_grad()
    def detect_array(self, image: np.ndarray, *, conf: float = 0.25, iou: float = 0.45, image_size: int = 640) -> list[dict[str, Any]]:
        if self.model is None:
            raise DetectorError("模型尚未加载")
        if image is None:
            raise DetectorError("无法读取图片")
        try:
            self.model.conf = conf
            self.model.iou = iou
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_image, size=image_size)
        except Exception as exc:
            raise DetectorError(f"推理失败：{exc}") from exc

        return self._detections_from_results(results)

    def _detections_from_results(self, results: Any) -> list[dict[str, Any]]:
        table = results.pandas().xyxy[0]
        detections: list[dict[str, Any]] = []
        for row in table.to_dict("records"):
            x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("xmin", "ymin", "xmax", "ymax")]
            score = float(row["confidence"])
            label = str(row.get("name", row.get("class", "object")))
            detections.append({"label": label, "score": round(score, 4), "bbox": [x1, y1, x2, y2]})
        return detections
