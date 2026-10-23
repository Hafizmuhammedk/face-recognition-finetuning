"""Check every saved artefact for contact with the test set.

    python -m scripts.check_data_leakage

Checks for files that do not exist yet are skipped. Exits with status 1 if any check fails.
"""

import hashlib
import json
import sys

import pandas as pd
import torch

from src.classifier import file_sha256, load_classifier
from src.config import (
    BEST_MODEL,
    FINAL_MODEL,
    KNN_CLASSIFIER,
    METRICS_DIR,
    TEST_DIR,
    TEST_PAIRS,
    TRAIN_DIR,
    VALIDATION_PAIRS,
    SplitConfig,
)
from src.dataset import (
    DataLeakageError,
    compute_thumbnails,
    load_manifests,
    nearest_same_identity_similarity,
    verify_split,
)
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def check_manifests(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[str, str]:
    try:
        overlaps = verify_split(train_df, test_df)
    except DataLeakageError as error:
        return "failed", str(error)
    return (
        "passed",
        f"no shared source IDs, files, SHA-256 hashes or segments ({len(overlaps)} overlap counts are all 0)",
    )


def check_near_duplicates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[str, str]:
    # Recomputed from the image files rather than trusting the numbers stored in the manifest.
    threshold = SplitConfig().near_duplicate_threshold
    similarity = nearest_same_identity_similarity(
        train_df,
        compute_thumbnails(train_df["path"].tolist()),
        test_df,
        compute_thumbnails(test_df["path"].tolist()),
    )
    flagged = int((similarity >= threshold).sum())
    detail = f"highest train-to-test thumbnail similarity {similarity.max():.5f}; {flagged} training images at or above {threshold}"
    return ("failed" if flagged else "passed"), detail


def folder_hashes(folder) -> set[str]:
    return {hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.rglob("*") if path.is_file()}


def check_split_folders(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[str, str]:
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        return "skipped", "data/train or data/test has not been created"
    train_hashes, test_hashes = folder_hashes(TRAIN_DIR), folder_hashes(TEST_DIR)
    shared = train_hashes & test_hashes
    if shared or train_hashes != set(train_df["sha256"]) or test_hashes != set(test_df["sha256"]):
        return "failed", f"{len(shared)} identical files in both folders, or a folder does not match its manifest"
    return "passed", f"{len(train_hashes)} train and {len(test_hashes)} test files, no file content in both"


def check_checkpoint(path, train_df: pd.DataFrame, test_ids: set[str]) -> tuple[str, str]:
    if not path.exists():
        return "skipped", f"{path.name} not found"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    used = set(checkpoint["train_source_ids"])
    expected = set(train_df.loc[train_df["subset"] == "train", "source_id"])
    if used & test_ids or not used <= expected:
        return (
            "failed",
            f"{len(used & test_ids)} test images and {len(used - expected)} non-training images used for fine-tuning",
        )
    return (
        "passed",
        f"fine-tuned on {len(used)} training-subset images (epoch {checkpoint['epoch']}); no validation or test images",
    )


def check_classifier(train_df: pd.DataFrame, test_ids: set[str]) -> tuple[str, str]:
    if not KNN_CLASSIFIER.exists():
        return "skipped", "knn_classifier.joblib not found"
    bundle = load_classifier(KNN_CLASSIFIER)
    gallery = set(bundle["gallery_source_ids"])
    problems = []
    if gallery & test_ids:
        problems.append(f"{len(gallery & test_ids)} test images in the KNN gallery")
    if bundle["threshold_source"] != "validation_pairs":
        problems.append(f"threshold calibrated on {bundle['threshold_source']}")
    if BEST_MODEL.exists() and file_sha256(BEST_MODEL) != bundle["embedding_checkpoint_sha256"]:
        problems.append("KNN was fitted with a different checkpoint than best_model.pt")
    if problems:
        return "failed", "; ".join(problems)
    return (
        "passed",
        f"gallery of {len(gallery)} training images; threshold from validation pairs; matches best_model.pt",
    )


def check_pairs(train_df: pd.DataFrame, test_ids: set[str]) -> tuple[str, str]:
    if not TEST_PAIRS.exists() or not VALIDATION_PAIRS.exists():
        return "skipped", "pair manifests not created yet"
    test_pairs, validation_pairs = pd.read_csv(TEST_PAIRS), pd.read_csv(VALIDATION_PAIRS)
    validation_ids = set(train_df.loc[train_df["subset"] == "validation", "source_id"])
    test_refs = set(test_pairs["left_id"]) | set(test_pairs["right_id"])
    validation_refs = set(validation_pairs["left_id"]) | set(validation_pairs["right_id"])
    if not test_refs <= test_ids or not validation_refs <= validation_ids:
        return "failed", "a pair manifest references images outside its subset"
    return (
        "passed",
        f"{len(test_pairs)} test pairs use only test images; {len(validation_pairs)} threshold-calibration pairs use only validation images",
    )


def check_shared_protocol() -> tuple[str, str]:
    paths = [METRICS_DIR / "baseline_metrics.json", METRICS_DIR / "finetuned_metrics.json"]
    available = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]
    if not available:
        return "skipped", "no evaluation results yet"
    current = file_sha256(TEST_PAIRS)
    if any(results["test_pairs_sha256"] != current for results in available):
        return "failed", "an evaluation used a different test pair manifest"
    if any(results["threshold_source"] != "validation_pairs" for results in available):
        return "failed", "an evaluation threshold was not calibrated on validation pairs"
    return "passed", f"{len(available)} evaluation(s) used the same test pairs and validation-calibrated thresholds"


def main() -> None:
    configure_logging("check_data_leakage")
    train_df, test_df = load_manifests()
    test_ids = set(test_df["source_id"])

    checks = {
        "Train/test manifests are disjoint": check_manifests(train_df, test_df),
        "No near-duplicates across train/test": check_near_duplicates(train_df, test_df),
        "data/train and data/test share no files": check_split_folders(train_df, test_df),
        "best_model.pt trained without test data": check_checkpoint(BEST_MODEL, train_df, test_ids),
        "final_model.pt trained without test data": check_checkpoint(FINAL_MODEL, train_df, test_ids),
        "KNN gallery and threshold exclude test data": check_classifier(train_df, test_ids),
        "Pair manifests stay within their subsets": check_pairs(train_df, test_ids),
        "Baseline and final use one fixed protocol": check_shared_protocol(),
    }
    summary = [{"check": name, "status": status, "detail": detail} for name, (status, detail) in checks.items()]
    for row in summary:
        logger.info("integrity_check", check=row["check"], status=row["status"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))

    if any(row["status"] == "failed" for row in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
