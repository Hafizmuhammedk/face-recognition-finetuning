"""Detect and recognise the face in one image, or in every image inside a folder.

    python -m scripts.predict --input "data/test/thomas/low_quality/g (12).png"
    python -m scripts.predict --input data/test/thomas
    python -m scripts.predict --input data/test --output outputs/predictions/test_folder.csv

When the path contains a person's folder name (e.g. data/test/thomas/...), that
name is shown as the actual identity and the accuracy is printed.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset import IMAGE_SUFFIXES
from src.inference import FaceRecognizer
from src.logging_config import configure_logging


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"No such file or folder: {input_path}")


def identity_from_path(path: Path, identities: list[str]) -> str | None:
    folders = [part.lower() for part in reversed(path.parts[:-1])]
    return next((name for name in folders if name in identities), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and recognise faces in an image or a folder of images.")
    parser.add_argument("--input", required=True, type=Path, help="Image file or folder of images")
    parser.add_argument("--output", type=Path, help="Optional CSV file for the results")
    args = parser.parse_args()

    configure_logging("predict")
    images = find_images(args.input)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    recognizer = FaceRecognizer(device="cuda" if torch.cuda.is_available() else "cpu")
    results = recognizer.recognize(images)
    identities = recognizer.classifier["identities"]

    table = pd.DataFrame(
        {
            "image": [str(path) for path in images],
            "detection": results["detection_status"],
            "predicted": results["predicted_identity"].fillna("no face"),
            "similarity": results["nearest_similarity"].astype(float).round(3),
            "actual": [identity_from_path(path, identities) for path in images],
        }
    )
    has_actual = table["actual"].notna()
    table.loc[has_actual, "correct"] = table.loc[has_actual, "predicted"] == table.loc[has_actual, "actual"]
    if not has_actual.any():
        table = table.drop(columns=["actual"])

    print(table.to_string(index=False))
    if has_actual.any():
        correct = table.loc[has_actual, "correct"].astype(bool)
        print(f"\nAccuracy: {correct.mean():.3f} ({correct.sum()} of {len(correct)} images)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
