import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    EXCLUDED_MANIFEST,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SPLIT_SUMMARY,
    TEST_DIR,
    TEST_MANIFEST,
    TRAIN_DIR,
    TRAIN_MANIFEST,
    SplitConfig,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
QUALITY_TAGS = {"high_quality": "hq", "low_quality": "lq"}


class DataLeakageError(RuntimeError):
    pass


def scan_dataset(root: Path = RAW_DATA_DIR) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}. Extract Images.zip into data/raw/.")

    rows = []
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        identity = identity_dir.name.lower()
        for quality_dir in sorted(path for path in identity_dir.iterdir() if path.is_dir()):
            # The archive mixes "High_quality" and "high_quality".
            quality = quality_dir.name.lower()
            if quality not in QUALITY_TAGS:
                raise ValueError(f"Unexpected quality folder: {quality_dir}")
            tag = QUALITY_TAGS[quality]

            for path in sorted(quality_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                data = path.read_bytes()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                frame_number = re.search(r"\d+", path.stem)
                slug = re.sub(r"[^a-z0-9]+", "_", path.name.lower()).strip("_")
                rows.append(
                    {
                        "source_id": f"{identity}_{tag}_{slug}",
                        "identity": identity,
                        "quality": quality,
                        # "d (1).jpg" and "d (1).png" are different frames, so each
                        # file type is its own sequence.
                        "sequence": f"{identity}_{tag}_{path.suffix.lower().lstrip('.')}",
                        "frame_index": int(frame_number.group()) if frame_number else 0,
                        "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "readable": image is not None,
                        "width": image.shape[1] if image is not None else 0,
                        "height": image.shape[0] if image is not None else 0,
                    }
                )

    inventory = pd.DataFrame(rows)
    logger.info(
        "dataset_scanned",
        images=len(inventory),
        identities=inventory["identity"].nunique(),
        unreadable=int((~inventory["readable"]).sum()),
    )
    return inventory


def mark_exclusions(inventory: pd.DataFrame) -> pd.DataFrame:
    inventory = inventory.sort_values(["sequence", "frame_index", "source_id"]).reset_index(drop=True)
    inventory["exclusion_reason"] = None
    inventory["duplicate_of"] = None
    inventory.loc[~inventory["readable"], "exclusion_reason"] = "unreadable"

    readable = inventory[inventory["readable"]]
    identities_per_hash = readable.groupby("sha256")["identity"].nunique()
    conflicting = inventory["sha256"].isin(identities_per_hash[identities_per_hash > 1].index)
    inventory.loc[conflicting & inventory["readable"], "exclusion_reason"] = "conflicting_identity_labels"

    # Keep only the first copy of an exact duplicate so it can't end up on both sides.
    first_copy = readable.groupby("sha256")["source_id"].first()
    is_later_copy = inventory["readable"] & inventory.duplicated("sha256", keep="first") & ~conflicting
    inventory.loc[is_later_copy, "exclusion_reason"] = "exact_duplicate"
    inventory.loc[is_later_copy, "duplicate_of"] = inventory.loc[is_later_copy, "sha256"].map(first_copy)

    logger.info("exclusions_marked", **inventory["exclusion_reason"].value_counts().to_dict())
    return inventory


def assign_segments(images: pd.DataFrame, config: SplitConfig) -> pd.Series:
    segments = pd.Series(index=images.index, dtype=object)
    for sequence, rows in images.groupby("sequence"):
        ordered = rows.sort_values(["frame_index", "source_id"]).index
        length = max(1, min(config.max_segment_length, len(ordered) // config.min_segments_per_sequence))
        for position, index in enumerate(ordered):
            segments[index] = f"{sequence}_seg{position // length:03d}"
    return segments


def select_holdout_segments(images: pd.DataFrame, fraction: float, rng: np.random.Generator) -> pd.Series:
    holdout = pd.Series(False, index=images.index)
    for _, rows in images.groupby("sequence", sort=True):
        sizes = rows.groupby("segment").size()
        target = fraction * len(rows)
        chosen, total = [], 0
        for segment in rng.permutation(sizes.index.to_numpy()):
            if abs(total + sizes[segment] - target) <= abs(total - target):
                chosen.append(segment)
                total += sizes[segment]
        holdout[rows.index[rows["segment"].isin(chosen)]] = True
    return holdout


def thumbnail_vector(gray_image: np.ndarray) -> np.ndarray:
    vector = cv2.resize(gray_image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    vector -= vector.mean()
    return vector / (np.linalg.norm(vector) + 1e-6)


def compute_thumbnails(image_paths: list[str]) -> np.ndarray:
    thumbnails = np.zeros((len(image_paths), 32 * 32), np.float32)
    for row, path in enumerate(image_paths):
        image = cv2.imdecode(np.fromfile(str(PROJECT_ROOT / path), np.uint8), cv2.IMREAD_GRAYSCALE)
        thumbnails[row] = thumbnail_vector(image)
    return thumbnails


def nearest_same_identity_similarity(
    query: pd.DataFrame,
    query_thumbnails: np.ndarray,
    reference: pd.DataFrame,
    reference_thumbnails: np.ndarray,
) -> np.ndarray:
    best = np.full(len(query), -1.0, np.float32)
    query_identity = query["identity"].to_numpy()
    reference_identity = reference["identity"].to_numpy()
    for identity in np.unique(query_identity):
        query_rows = np.flatnonzero(query_identity == identity)
        reference_rows = np.flatnonzero(reference_identity == identity)
        if len(reference_rows) == 0:
            continue
        similarity = query_thumbnails[query_rows] @ reference_thumbnails[reference_rows].T
        best[query_rows] = similarity.max(axis=1)
    return best


def split_dataset(inventory: pd.DataFrame, config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by blocks of consecutive frames, then drop training images that are near-copies of test images.

    The images are video frames, so splitting single files would put almost the
    same picture in train and test. The test set is never changed by the
    near-duplicate step; only training images are removed.
    """
    rng = np.random.default_rng(config.seed)
    images = inventory[inventory["exclusion_reason"].isna()].copy()
    images["segment"] = assign_segments(images, config)

    images["subset"] = np.where(select_holdout_segments(images, config.test_fraction, rng), "test", "train")
    training_portion = images[images["subset"] == "train"]
    is_validation = select_holdout_segments(training_portion, config.validation_fraction, rng)
    images.loc[is_validation[is_validation].index, "subset"] = "validation"

    thumbnails = compute_thumbnails(images["source_path"].tolist())
    position = pd.Series(np.arange(len(images)), index=images.index)
    images["max_similarity_to_test"] = np.nan
    images["max_similarity_to_validation"] = np.nan

    test = images[images["subset"] == "test"]
    candidates = images[images["subset"] != "test"]
    images.loc[candidates.index, "max_similarity_to_test"] = nearest_same_identity_similarity(
        candidates, thumbnails[position[candidates.index]], test, thumbnails[position[test.index]]
    )
    near_test = images["max_similarity_to_test"] >= config.near_duplicate_threshold

    validation = images[(images["subset"] == "validation") & ~near_test]
    fit = images[(images["subset"] == "train") & ~near_test]
    images.loc[fit.index, "max_similarity_to_validation"] = nearest_same_identity_similarity(
        fit, thumbnails[position[fit.index]], validation, thumbnails[position[validation.index]]
    )
    near_validation = (
        (images["subset"] == "train")
        & ~near_test
        & (images["max_similarity_to_validation"] >= config.near_duplicate_threshold)
    )

    removed = images[near_test | near_validation].copy()
    removed["exclusion_reason"] = np.where(
        near_test[removed.index], "near_duplicate_of_test", "near_duplicate_of_validation"
    )
    kept = images.drop(removed.index)

    train_df = kept[kept["subset"] != "test"].reset_index(drop=True)
    test_df = kept[kept["subset"] == "test"].reset_index(drop=True)
    excluded_df = pd.concat([inventory[inventory["exclusion_reason"].notna()], removed], ignore_index=True)

    logger.info(
        "dataset_split_complete",
        train=int((train_df["subset"] == "train").sum()),
        validation=int((train_df["subset"] == "validation").sum()),
        test=len(test_df),
        excluded=len(excluded_df),
    )
    return train_df, test_df, excluded_df


def copy_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy images into data/train and data/test, keeping the <person>/<quality> layout of the original."""
    results = []
    for images, target_root in ((train_df, TRAIN_DIR), (test_df, TEST_DIR)):
        if target_root.exists():
            shutil.rmtree(target_root)
        paths = []
        for row in images.itertuples():
            source = PROJECT_ROOT / row.source_path
            target = target_root / row.identity / row.quality / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            paths.append(target.relative_to(PROJECT_ROOT).as_posix())
        results.append(images.assign(path=paths))
        logger.info("split_copied", folder=str(target_root), images=len(paths))
    return results[0], results[1]


def verify_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    overlaps = {
        f"shared_{column}": len(set(train_df[column]) & set(test_df[column]))
        for column in ("source_id", "path", "sha256", "segment")
    }
    validation = train_df[train_df["subset"] == "validation"]
    fit = train_df[train_df["subset"] == "train"]
    overlaps["shared_validation_train_segments"] = len(set(validation["segment"]) & set(fit["segment"]))
    overlaps["test_identities_missing_from_train"] = len(set(test_df["identity"]) - set(fit["identity"]))

    failures = {name: count for name, count in overlaps.items() if count}
    if failures:
        raise DataLeakageError(f"Split integrity check failed: {failures}")
    logger.info("split_verified", train=len(train_df), test=len(test_df))
    return overlaps


def summarize_split(
    train_df: pd.DataFrame, test_df: pd.DataFrame, excluded_df: pd.DataFrame, config: SplitConfig
) -> dict:
    combined = pd.concat([train_df, test_df])
    counts = combined.groupby(["identity", "quality", "subset"]).size().unstack(fill_value=0)
    kept = len(combined)
    near_duplicates = excluded_df["exclusion_reason"].str.startswith("near_duplicate").sum()
    return {
        "config": vars(config),
        "images_scanned": kept + len(excluded_df),
        "images_kept": kept,
        "subset_counts": combined["subset"].value_counts().to_dict(),
        "test_fraction_at_split": round(len(test_df) / (kept + near_duplicates), 4),
        "test_fraction_of_kept": round(len(test_df) / kept, 4),
        "segments": combined.groupby("subset")["segment"].nunique().to_dict(),
        "excluded_by_reason": excluded_df["exclusion_reason"].value_counts().to_dict(),
        "per_identity_quality": {
            f"{identity}/{quality}": row.to_dict() for (identity, quality), row in counts.iterrows()
        },
    }


def save_manifests(train_df: pd.DataFrame, test_df: pd.DataFrame, excluded_df: pd.DataFrame, summary: dict) -> None:
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_MANIFEST, index=False)
    test_df.to_csv(TEST_MANIFEST, index=False)
    excluded_df.to_csv(EXCLUDED_MANIFEST, index=False)
    SPLIT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("manifests_saved", directory=str(TRAIN_MANIFEST.parent))


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_MANIFEST.exists() or not TEST_MANIFEST.exists():
        raise FileNotFoundError("Split manifests not found. Run Part 1 of face_recognition.ipynb first.")
    return pd.read_csv(TRAIN_MANIFEST), pd.read_csv(TEST_MANIFEST)
