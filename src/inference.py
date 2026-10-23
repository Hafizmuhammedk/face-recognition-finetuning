from pathlib import Path

import numpy as np
import pandas as pd

from src.classifier import file_sha256, load_classifier, predict_identities
from src.config import BEST_MODEL, KNN_CLASSIFIER, DetectionConfig
from src.detection import FaceDetector, detect_faces
from src.embeddings import extract_embeddings, load_embedding_model
from src.evaluation import USABLE_STATUSES
from src.logging_config import get_logger

logger = get_logger(__name__)


class FaceRecognizer:
    def __init__(self, checkpoint_path: Path = BEST_MODEL, classifier_path: Path = KNN_CLASSIFIER, device: str = "cpu"):
        self.classifier = load_classifier(classifier_path)
        # Refuse to pair a KNN with a checkpoint whose embeddings it was not fitted on.
        if file_sha256(checkpoint_path) != self.classifier["embedding_checkpoint_sha256"]:
            raise ValueError(
                f"{classifier_path.name} was fitted on embeddings from a different checkpoint than {checkpoint_path.name}"
            )
        self.model, self.checkpoint = load_embedding_model(checkpoint_path, device)
        preprocessing = {
            key: value
            for key, value in self.checkpoint["preprocessing"].items()
            if key in DetectionConfig.__dataclass_fields__
        }
        self.detection_config = DetectionConfig(**preprocessing)
        self.detector = FaceDetector(config=self.detection_config, device=device)
        self.device = device
        logger.info(
            "recognizer_loaded",
            checkpoint=checkpoint_path.name,
            checkpoint_epoch=self.checkpoint["epoch"],
            neighbors=self.classifier["neighbors"],
            unknown_threshold=round(self.classifier["unknown_threshold"], 4),
        )

    def recognize(self, image_paths: list[Path]) -> pd.DataFrame:
        detections = detect_faces(self.detector, image_paths)
        usable = detections["detection_status"].isin(USABLE_STATUSES).to_numpy()

        results = detections.assign(image=[path.name for path in image_paths])
        for column in ("closed_set_prediction", "predicted_identity"):
            results[column] = None
        results[["nearest_similarity", "vote_share"]] = np.nan
        results["rejected_as_unknown"] = False

        if usable.any():
            boxes = detections.loc[usable, ["x1", "y1", "x2", "y2"]].to_numpy()
            paths = [path for path, keep in zip(image_paths, usable) if keep]
            embeddings = extract_embeddings(self.model, paths, boxes, self.detection_config, self.device)
            predictions = predict_identities(self.classifier["knn"], embeddings, self.classifier["unknown_threshold"])
            for column in predictions.columns:
                results.loc[usable, column] = predictions[column].to_numpy()

        logger.info(
            "recognition_complete",
            images=len(results),
            predicted=int(usable.sum()),
            rejected_as_unknown=int(results["rejected_as_unknown"].sum()),
        )
        return results
