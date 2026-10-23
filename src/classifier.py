import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.config import UNKNOWN_IDENTITY
from src.logging_config import get_logger
from src.metrics import identification_metrics

logger = get_logger(__name__)


def build_knn(neighbors: int) -> KNeighborsClassifier:
    # Cosine distance so the neighbour similarity can be compared with the verification threshold.
    return KNeighborsClassifier(n_neighbors=neighbors, metric="cosine", weights="distance", algorithm="brute")


def select_neighbors(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    candidates: tuple[int, ...],
) -> tuple[int, pd.DataFrame]:
    results = []
    for neighbors in candidates:
        knn = build_knn(neighbors).fit(train_embeddings, train_labels)
        predicted = pd.Series(knn.predict(validation_embeddings))
        results.append({"neighbors": neighbors, **identification_metrics(pd.Series(validation_labels), predicted)})
    results = pd.DataFrame(results)
    best = int(results.sort_values(["accuracy", "neighbors"], ascending=[False, True]).iloc[0]["neighbors"])
    logger.info(
        "knn_neighbors_selected",
        neighbors=best,
        validation_accuracy=float(results.set_index("neighbors").loc[best, "accuracy"]),
    )
    return best, results


def predict_identities(
    knn: KNeighborsClassifier, embeddings: np.ndarray, unknown_threshold: float | None
) -> pd.DataFrame:
    predicted = knn.predict(embeddings)
    distances, _ = knn.kneighbors(embeddings, n_neighbors=1)
    nearest_similarity = 1.0 - distances[:, 0]
    # Share of the (distance-weighted) neighbour votes. Not a calibrated probability.
    vote_share = knn.predict_proba(embeddings).max(axis=1)

    rejected = np.zeros(len(predicted), bool)
    if unknown_threshold is not None:
        rejected = nearest_similarity < unknown_threshold
    return pd.DataFrame(
        {
            "closed_set_prediction": predicted,
            "predicted_identity": np.where(rejected, UNKNOWN_IDENTITY, predicted),
            "nearest_similarity": nearest_similarity,
            "vote_share": vote_share,
            "rejected_as_unknown": rejected,
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_classifier(
    path: Path,
    knn: KNeighborsClassifier,
    unknown_threshold: float,
    gallery_source_ids: list[str],
    embedding_checkpoint: Path,
    selection_results: pd.DataFrame,
) -> None:
    # The checkpoint hash ties this KNN to the network whose embeddings it was fitted on.
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "knn": knn,
            "identities": sorted(knn.classes_.tolist()),
            "neighbors": knn.n_neighbors,
            "unknown_threshold": unknown_threshold,
            "threshold_source": "validation_pairs",
            "gallery_source_ids": gallery_source_ids,
            "embedding_checkpoint": embedding_checkpoint.name,
            "embedding_checkpoint_sha256": file_sha256(embedding_checkpoint),
            "neighbor_selection": selection_results.to_dict("records"),
        },
        path,
    )
    logger.info("knn_saved", path=str(path), gallery=len(gallery_source_ids), neighbors=knn.n_neighbors)


def load_classifier(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"KNN classifier not found: {path}. Run Part 2 of the notebook first.")
    return joblib.load(path)
