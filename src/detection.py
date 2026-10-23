from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import FACE_DETECTOR_WEIGHTS, DetectionConfig
from src.logging_config import get_logger

logger = get_logger(__name__)

DETECTION_COLUMNS = ["x1", "y1", "x2", "y2", "confidence", "faces_found", "detection_status"]


def read_image(image_path: Path) -> np.ndarray:
    # cv2.imread fails on non-ASCII Windows paths; imdecode does not.
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


class FaceDetector:
    def __init__(
        self,
        weights_path: Path = FACE_DETECTOR_WEIGHTS,
        config: DetectionConfig | None = None,
        device: str | None = None,
    ):
        if not weights_path.exists():
            raise FileNotFoundError(f"Face detector weights not found: {weights_path}")
        self.config = config or DetectionConfig()
        self.device = device
        self.model = YOLO(str(weights_path))
        if self.model.names != {0: "face"}:
            raise ValueError(f"{weights_path.name} is not a single-class face detector: {self.model.names}")

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float, int]:
        # Many crops are under 40 px and fill the whole frame. YOLO was trained on full
        # scenes, so upscale small images and pad them before detection.
        height, width = image.shape[:2]
        scale = max(1.0, self.config.min_input_side / min(height, width))
        if scale > 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = int(self.config.padding_fraction * max(image.shape[:2]))
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, pad

    def detect(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Return one (n, 5) array of x1, y1, x2, y2, confidence per image, in that image's coordinates."""
        prepared = [self._prepare(image) for image in images]
        results = self.model.predict(
            [padded for padded, _, _ in prepared],
            imgsz=self.config.detector_image_size,
            conf=self.config.confidence,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result, (_, scale, pad) in zip(results, prepared):
            boxes = (result.boxes.xyxy.cpu().numpy() - pad) / scale
            confidences = result.boxes.conf.cpu().numpy()
            detections.append(np.column_stack([boxes, confidences]))
        return detections


def select_primary_face(boxes: np.ndarray, image_shape: tuple[int, ...], allow_fallback: bool) -> dict:
    height, width = image_shape[:2]
    if len(boxes) > 0:
        x1, y1, x2, y2, confidence = boxes[np.argmax(boxes[:, 4])]
        status = "detected"
    elif allow_fallback:
        x1, y1, x2, y2, confidence = 0.0, 0.0, float(width), float(height), np.nan
        status = "full_image_fallback"
    else:
        x1 = y1 = x2 = y2 = confidence = np.nan
        status = "no_face"
    return dict(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        confidence=float(confidence),
        faces_found=len(boxes),
        detection_status=status,
    )


def detect_faces(detector: FaceDetector, image_paths: list[Path]) -> pd.DataFrame:
    """Most confident face per image; rows line up with image_paths."""
    batch_size = detector.config.batch_size
    records = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images, readable_paths = [], []
        for path in batch_paths:
            try:
                images.append(read_image(path))
                readable_paths.append(path)
            except ValueError:
                logger.warning("image_unreadable", path=str(path))
        boxes_by_path = dict(zip(readable_paths, detector.detect(images))) if images else {}
        shapes = {path: image.shape for path, image in zip(readable_paths, images)}

        for path in batch_paths:
            if path not in boxes_by_path:
                records.append(
                    dict.fromkeys(DETECTION_COLUMNS[:-2], np.nan) | {"faces_found": 0, "detection_status": "unreadable"}
                )
                continue
            records.append(
                select_primary_face(boxes_by_path[path], shapes[path], detector.config.fallback_to_full_image)
            )

    detections = pd.DataFrame(records, columns=DETECTION_COLUMNS)
    logger.info(
        "face_detection_complete", images=len(detections), **detections["detection_status"].value_counts().to_dict()
    )
    return detections


def crop_face(image: np.ndarray, box: tuple[float, float, float, float], margin: float, size: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    side = int(round(max(x2 - x1, y2 - y1) * (1 + margin)))
    side = max(side, 1)
    left = int(round((x1 + x2) / 2 - side / 2))
    top = int(round((y1 + y2) / 2 - side / 2))

    # Pad by repeating the border so faces near the edge keep the same scale.
    height, width = image.shape[:2]
    pad = max(0, -left, -top, left + side - width, top + side - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    face = image[top + pad : top + pad + side, left + pad : left + pad + side]

    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(face, (size, size), interpolation=interpolation)
