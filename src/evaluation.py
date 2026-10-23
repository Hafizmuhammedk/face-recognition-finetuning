import hashlib
import json

import numpy as np
import pandas as pd
from torch import nn

from src.classifier import build_knn, select_neighbors
from src.config import (
    CHART_DIR,
    EMBEDDINGS_DIR,
    MANIFEST_DIR,
    METRICS_DIR,
    PROJECT_ROOT,
    TEST_PAIRS,
    VALIDATION_PAIRS,
    DetectionConfig,
    EvaluationConfig,
    TrainingConfig,
)
from src.dataset import load_manifests, verify_split
from src.detection import FaceDetector, detect_faces
from src.embeddings import extract_embeddings
from src.logging_config import get_logger
from src.metrics import (
    PAIR_TYPES,
    identification_metrics,
    make_verification_pairs,
    pair_scores,
    select_threshold,
    verification_metrics,
)
from src.plots import (
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_similarity_histograms,
)

logger = get_logger(__name__)

USABLE_STATUSES = ("detected", "full_image_fallback")


def attach_detections(images: pd.DataFrame, subset_name: str, detector: FaceDetector) -> pd.DataFrame:
    # Cached so the baseline and the fine-tuned model are scored on exactly the same crops.
    cache_path = MANIFEST_DIR / f"detections_{subset_name}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["source_id"].tolist() == images["source_id"].tolist():
            return pd.concat([images.reset_index(drop=True), cached.drop(columns="source_id")], axis=1)
        logger.warning("detection_cache_stale", path=str(cache_path))

    detections = detect_faces(detector, [PROJECT_ROOT / path for path in images["path"]])
    detections.insert(0, "source_id", images["source_id"].to_numpy())
    detections.to_csv(cache_path, index=False)
    return pd.concat([images.reset_index(drop=True), detections.drop(columns="source_id")], axis=1)


def usable_faces(images: pd.DataFrame) -> pd.DataFrame:
    return images[images["detection_status"].isin(USABLE_STATUSES)].reset_index(drop=True)


def embed_faces(
    model: nn.Module, faces: pd.DataFrame, detection_config: DetectionConfig, device: str, batch_size: int = 128
) -> np.ndarray:
    paths = [PROJECT_ROOT / path for path in faces["path"]]
    boxes = faces[["x1", "y1", "x2", "y2"]].to_numpy()
    return extract_embeddings(model, paths, boxes, detection_config, device, batch_size)


def load_or_create_pairs(
    path, faces: pd.DataFrame, pairs_per_identity: int, seed: int, allow_create: bool
) -> pd.DataFrame:
    if path.exists():
        pairs = pd.read_csv(path)
        missing = set(pairs["left_id"]) | set(pairs["right_id"])
        missing -= set(faces["source_id"])
        if missing:
            raise ValueError(f"{path.name} references {len(missing)} images outside the expected subset")
        return pairs
    if not allow_create:
        raise FileNotFoundError(f"{path} is missing. Run scripts/baseline_evaluation.py first so both models share it.")
    pairs = make_verification_pairs(faces, pairs_per_identity, seed)
    pairs.to_csv(path, index=False)
    logger.info("pairs_created", path=str(path), pairs=len(pairs), same_person=int(pairs["same_person"].sum()))
    return pairs


def file_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_model(model: nn.Module, model_name: str, allow_create_test_pairs: bool, device: str) -> dict:
    """Same detections, pairs, threshold calibration and KNN procedure for every model; only the weights differ."""
    train_df, test_df = load_manifests()
    verify_split(train_df, test_df)
    detection_config = DetectionConfig()
    evaluation_config = EvaluationConfig()
    training_config = TrainingConfig()

    detector = FaceDetector(config=detection_config, device=device)
    subsets = {
        "train": train_df[train_df["subset"] == "train"],
        "validation": train_df[train_df["subset"] == "validation"],
        "test": test_df,
    }
    detected = {name: attach_detections(images, name, detector) for name, images in subsets.items()}
    faces = {name: usable_faces(images) for name, images in detected.items()}
    embeddings = {
        name: embed_faces(model, images, detection_config, device, evaluation_config.embedding_batch_size)
        for name, images in faces.items()
    }

    validation_pairs = load_or_create_pairs(
        VALIDATION_PAIRS,
        faces["validation"],
        training_config.validation_pairs_per_identity,
        evaluation_config.seed,
        allow_create=True,
    )
    test_pairs = load_or_create_pairs(
        TEST_PAIRS,
        faces["test"],
        evaluation_config.test_pairs_per_identity,
        evaluation_config.seed,
        allow_create=allow_create_test_pairs,
    )

    validation_scores = pair_scores(
        validation_pairs, faces["validation"]["source_id"].to_numpy(), embeddings["validation"]
    )
    threshold = select_threshold(validation_pairs["same_person"].to_numpy(), validation_scores)
    test_pairs["score"] = pair_scores(test_pairs, faces["test"]["source_id"].to_numpy(), embeddings["test"])
    labels = test_pairs["same_person"].to_numpy()

    verification = verification_metrics(labels, test_pairs["score"].to_numpy(), threshold)
    by_pair_type = {
        pair_type: verification_metrics(rows["same_person"].to_numpy(), rows["score"].to_numpy(), threshold)
        for pair_type, rows in test_pairs.groupby("pair_type")
        if pair_type in PAIR_TYPES
    }

    neighbors, neighbor_search = select_neighbors(
        embeddings["train"],
        faces["train"]["identity"].to_numpy(),
        embeddings["validation"],
        faces["validation"]["identity"].to_numpy(),
        training_config.knn_neighbors,
    )
    knn = build_knn(neighbors).fit(embeddings["train"], faces["train"]["identity"].to_numpy())
    test_faces = faces["test"].assign(predicted=knn.predict(embeddings["test"]))
    identification = identification_metrics(test_faces["identity"], test_faces["predicted"])
    identification["neighbors"] = neighbors
    identification["by_quality"] = {
        quality: identification_metrics(rows["identity"], rows["predicted"])
        for quality, rows in test_faces.groupby("quality")
    }

    results = {
        "model": model_name,
        "threshold_source": "validation_pairs",
        "validation_pairs": len(validation_pairs),
        "test_pairs_sha256": file_digest(TEST_PAIRS),
        "validation_pairs_sha256": file_digest(VALIDATION_PAIRS),
        "verification": verification,
        "verification_by_pair_type": by_pair_type,
        "identification": identification,
        "knn_neighbor_search": neighbor_search.to_dict("records"),
        "test_detection_status": detected["test"]["detection_status"].value_counts().to_dict(),
        "test_images_excluded": int(len(detected["test"]) - len(faces["test"])),
    }
    save_results(results, model_name, test_pairs, faces["test"], embeddings["test"])
    logger.info(
        "evaluation_complete",
        model=model_name,
        roc_auc=round(verification["roc_auc"], 4),
        verification_accuracy=round(verification["accuracy"], 4),
        knn_accuracy=round(identification["accuracy"], 4),
    )
    return results


COMPARISON_METRICS = {
    "ROC-AUC": ("verification", "roc_auc"),
    "Average precision": ("verification", "average_precision"),
    "Verification accuracy": ("verification", "accuracy"),
    "Precision": ("verification", "precision"),
    "Recall": ("verification", "recall"),
    "F1-score": ("verification", "f1"),
    "False acceptance rate": ("verification", "false_acceptance_rate"),
    "False rejection rate": ("verification", "false_rejection_rate"),
    "KNN identification accuracy": ("identification", "accuracy"),
    "KNN macro F1": ("identification", "macro_f1"),
}


def metrics_table(results_by_model: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            model: {label: results[section][key] for label, (section, key) in COMPARISON_METRICS.items()}
            for model, results in results_by_model.items()
        }
    )


def save_results(
    results: dict, model_name: str, test_pairs: pd.DataFrame, test_faces: pd.DataFrame, test_embeddings: np.ndarray
) -> None:
    for directory in (METRICS_DIR, CHART_DIR, EMBEDDINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{model_name}_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    test_pairs.to_csv(METRICS_DIR / f"{model_name}_test_pair_scores.csv", index=False)
    # Embeddings are biometric data, so outputs/embeddings is git-ignored.
    np.savez_compressed(
        EMBEDDINGS_DIR / f"{model_name}_test_embeddings.npz",
        source_ids=test_faces["source_id"].to_numpy(),
        embeddings=test_embeddings,
    )

    plot_model_charts(model_name, test_pairs, results["verification"]["threshold"])


DISPLAY_NAMES = {"baseline": "Baseline", "finetuned": "Fine-tuned"}


def plot_model_charts(model_name: str, test_pairs: pd.DataFrame, threshold: float) -> None:
    labels, scores = test_pairs["same_person"].to_numpy(), test_pairs["score"].to_numpy()
    title_name = DISPLAY_NAMES.get(model_name, model_name)
    plot_roc_curves({title_name: (labels, scores)}, CHART_DIR / f"{model_name}_roc.png", f"{title_name}: ROC curve")
    plot_precision_recall_curves(
        {title_name: (labels, scores)},
        CHART_DIR / f"{model_name}_precision_recall.png",
        f"{title_name}: precision-recall",
    )
    plot_similarity_histograms(
        {"Same person": scores[labels == 1], "Different person": scores[labels == 0]},
        CHART_DIR / f"{model_name}_similarity_distribution.png",
        f"{title_name}: test-pair similarity",
        thresholds={"Validation": threshold},
    )
