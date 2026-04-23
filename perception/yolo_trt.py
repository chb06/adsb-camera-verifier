from __future__ import annotations

"""YOLO inference wrapper.

This file is intentionally a scaffold. We will implement the fast path using
TensorRT engines on the Jetson.

To keep the early bring-up easy, this wrapper currently supports:
- backend = 'none'       : returns no detections (useful for ROI overlay bring-up)
- backend = 'ultralytics': uses the Ultralytics YOLO python package (optional)

When you're ready for deployment, we'll add:
- backend = 'tensorrt'   : loads a .engine built with trtexec and runs fast FP16
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from common_types import Detection


@dataclass
class YoloConfig:
    backend: str = "none"  # none | ultralytics | tensorrt
    model_path: Optional[str] = None  # .pt for ultralytics, .engine for tensorrt
    imgsz: int = 640
    conf: float = 0.25


class YoloDetector:
    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        self.backend = cfg.backend.lower().strip()

        self._ultra = None
        if self.backend == "ultralytics":
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise RuntimeError(
                    "Ultralytics backend requested but ultralytics is not installed. "
                    "Install on Mac for training/quick tests: pip install ultralytics"
                ) from e
            if not cfg.model_path:
                cfg.model_path = "yolov8s.pt"
            self._ultra = YOLO(cfg.model_path)

        elif self.backend == "tensorrt":
            raise NotImplementedError(
                "TensorRT backend not implemented in this scaffold yet. "
                "We will add it once your ROI/calibration and dataset pipeline are working."
            )

        elif self.backend == "none":
            pass
        else:
            raise ValueError(f"Unknown YOLO backend: {cfg.backend}")

    def infer_bgr(self, img_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR image.

        Returns ROI-local detections.
        """
        if self.backend == "none":
            return []

        if self.backend == "ultralytics":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"[yolo] using device={device}")

            res = self._ultra.predict(
                img_bgr,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                device=device,
                verbose=False,
            )

            if not res:
                return []

            r0 = res[0]
            dets: List[Detection] = []

            if r0.boxes is None:
                return []

            names = r0.names
            for b in r0.boxes:
                cls_id = int(b.cls)
                cls_name = str(names.get(cls_id, cls_id))
                conf = float(b.conf)
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                dets.append(Detection(cls=cls_name, conf=conf, xyxy=(x1, y1, x2, y2)))

            return dets

        raise AssertionError("unreachable")