from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Images"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
CHART_DIR = OUTPUT_DIR / "charts"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
REPORTS_DIR = PROJECT_ROOT / "reports"

FACE_DETECTOR_WEIGHTS = MODELS_DIR / "yolov8n-face-lindevs.pt"
ARCFACE_ONNX_WEIGHTS = MODELS_DIR / "w600k_r50.onnx"

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.csv"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.csv"
EXCLUDED_MANIFEST = MANIFEST_DIR / "excluded_manifest.csv"
SPLIT_SUMMARY = MANIFEST_DIR / "split_summary.json"
VALIDATION_PAIRS = MANIFEST_DIR / "validation_pairs.csv"
TEST_PAIRS = MANIFEST_DIR / "test_pairs.csv"

BEST_MODEL = CHECKPOINT_DIR / "best_model.pt"
FINAL_MODEL = CHECKPOINT_DIR / "final_model.pt"
KNN_CLASSIFIER = CHECKPOINT_DIR / "knn_classifier.joblib"

TRAINING_HISTORY = METRICS_DIR / "training_history.csv"
TRAINING_SUMMARY = METRICS_DIR / "training_summary.json"
INFERENCE_PREDICTIONS = PREDICTIONS_DIR / "test_predictions.csv"
INFERENCE_METRICS = METRICS_DIR / "knn_inference_metrics.json"

AUGMENTATIONS = ("none", "bright", "dark", "contrast", "grayscale", "shadow")
UNKNOWN_IDENTITY = "unknown"


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    validation_fraction: float = 0.15
    max_segment_length: int = 50
    min_segments_per_sequence: int = 10
    # Thumbnail cosine similarity. Consecutive video frames score 0.93-0.99,
    # different moments of the same person rarely go above 0.9.
    near_duplicate_threshold: float = 0.95
    seed: int = 42


@dataclass
class DetectionConfig:
    confidence: float = 0.25
    detector_image_size: int = 320
    min_input_side: int = 160
    padding_fraction: float = 0.25
    # Picked on validation pairs with the pre-trained model; 0.0 beat -0.1, 0.1, 0.2 and 0.3.
    crop_margin: float = 0.0
    face_size: int = 112
    # The given images are already face crops, so when YOLO finds nothing we use the whole image.
    fallback_to_full_image: bool = True
    batch_size: int = 32


@dataclass
class TrainingConfig:
    image_size: int = 112
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    embedding_dimension: int = 512
    arcface_scale: float = 32.0
    arcface_margin: float = 0.30
    trainable_layers: tuple[str, ...] = ("layer4", "bn2", "fc", "features")
    early_stopping_patience: int = 3
    knn_neighbors: tuple[int, ...] = (1, 3, 5, 7, 9, 15)
    validation_pairs_per_identity: int = 300
    random_seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0

    @classmethod
    def for_available_hardware(cls) -> "TrainingConfig":
        config = cls()
        if config.device == "cpu":
            config.batch_size = 32
            config.epochs = 3
            config.trainable_layers = ("bn2", "fc", "features")
        return config

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationConfig:
    test_pairs_per_identity: int = 500
    embedding_batch_size: int = 128
    seed: int = 42
