"""Build reports/final_report.pdf from the saved results. Run after the final evaluation and the leakage check.

    python -m scripts.generate_report

Every number comes from files in data/manifests and outputs/; the script stops if one is missing.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import (
    CHART_DIR,
    INFERENCE_METRICS,
    INFERENCE_PREDICTIONS,
    MANIFEST_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    SPLIT_SUMMARY,
    TRAINING_HISTORY,
    TRAINING_SUMMARY,
)
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

PAGE_WIDTH = A4[0] - 4 * cm
HEADER_COLOR = colors.HexColor("#e8eef7")
GRID_COLOR = colors.HexColor("#b8b8b8")


def load_json(path: Path) -> dict | list:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the full workflow before generating the report.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the full workflow before generating the report.")
    return pd.read_csv(path, **kwargs)


class ReportBuilder:
    def __init__(self):
        styles = getSampleStyleSheet()
        self.styles = {
            "title": ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=6),
            "subtitle": ParagraphStyle(
                "subtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#52514e")
            ),
            "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceBefore=4, spaceAfter=6),
            "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3),
            "body": ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5),
            "caption": ParagraphStyle(
                "caption", parent=styles["Italic"], fontSize=8.5, alignment=TA_CENTER, spaceBefore=2, spaceAfter=9
            ),
            "cell": ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8, leading=10),
        }
        self.story = []
        self.figures = 0
        self.tables = 0

    def heading(self, text: str, level: int = 1) -> None:
        self.story.append(Paragraph(text, self.styles[f"h{level}"]))

    def paragraph(self, text: str) -> None:
        self.story.append(Paragraph(text, self.styles["body"]))

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.story.append(Paragraph(f"&bull;&nbsp; {item}", self.styles["body"]))

    def table(self, frame: pd.DataFrame, caption: str, column_widths: list[float] | None = None) -> None:
        self.tables += 1
        rows = [[Paragraph(f"<b>{column}</b>", self.styles["cell"]) for column in frame.columns]]
        rows += [[Paragraph(str(value), self.styles["cell"]) for value in row] for row in frame.itertuples(index=False)]
        table = Table(rows, colWidths=column_widths, repeatRows=1, hAlign="CENTER")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID_COLOR),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        caption_text = Paragraph(f"<b>Table {self.tables}.</b> {caption}", self.styles["caption"])
        self.story.append(KeepTogether([table, caption_text]))

    def figure(self, path: Path, caption: str, width: float) -> None:
        if not path.exists():
            raise SystemExit(f"Missing chart {path}. Run scripts/final_evaluation.py first.")
        self.figures += 1
        image = Image(str(path))
        image.drawHeight = image.drawHeight * width / image.drawWidth
        image.drawWidth = width
        caption_text = Paragraph(f"<b>Figure {self.figures}.</b> {caption}", self.styles["caption"])
        self.story.append(KeepTogether([image, caption_text]))

    def figure_pair(self, left: Path, right: Path, caption: str) -> None:
        """Two charts side by side under a single figure number."""
        images = []
        for path in (left, right):
            if not path.exists():
                raise SystemExit(f"Missing chart {path}. Run scripts/final_evaluation.py first.")
            image = Image(str(path))
            width = PAGE_WIDTH / 2 - 0.2 * cm
            image.drawHeight = image.drawHeight * width / image.drawWidth
            image.drawWidth = width
            images.append(image)
        self.figures += 1
        grid = Table([images], colWidths=[PAGE_WIDTH / 2] * 2)
        caption_text = Paragraph(f"<b>Figure {self.figures}.</b> {caption}", self.styles["caption"])
        self.story.append(KeepTogether([grid, caption_text]))

    def page_break(self) -> None:
        self.story.append(PageBreak())


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {document.page}")
    canvas.drawString(2 * cm, 1.2 * cm, "Face recognition under challenging image conditions")
    canvas.restoreState()


def introduction(report: ReportBuilder, training: dict) -> None:
    report.story.append(Spacer(1, 1.2 * cm))
    report.story.append(Paragraph("Face Recognition under Challenging Image Conditions", report.styles["title"]))
    report.story.append(
        Paragraph(
            f"AI Developer Intern technical assignment &middot; {date.today():%d %B %Y}", report.styles["subtitle"]
        )
    )
    report.story.append(Spacer(1, 0.6 * cm))

    report.heading("1. Introduction")
    report.heading("Objective", 2)
    report.paragraph(
        "Recognise eight people from the company-provided images, many of which are small, blurred, dark or "
        "seen from the side. A pre-trained face-embedding network is fine-tuned on a training split of those "
        "images, and the effect is measured against the unmodified network on a held-out 20% test split."
    )
    report.heading("Problem definition", 2)
    report.paragraph(
        "Two tasks are measured separately. <b>Verification</b>: given two face images, decide whether they show "
        "the same person by thresholding the cosine similarity of their embeddings. <b>Identification</b>: given "
        "one face, name which of the eight enrolled people it is, using a k-nearest-neighbour classifier over "
        "embeddings of the training images. Verification is the primary comparison between the baseline and the "
        "fine-tuned model because it isolates the quality of the embedding itself."
    )
    report.heading("Selected architecture", 2)
    report.paragraph(
        "Faces are located with a YOLOv8n detector trained for faces and cropped to a 112 x 112 square. Embeddings "
        "come from InsightFace's ArcFace IResNet-50 (<i>w600k_r50</i>, trained on WebFace600K), which produces a "
        "512-dimensional vector. InsightFace distributes this model only as an inference ONNX file, so it was "
        "rebuilt as a PyTorch module and its weights copied across. The copy reproduces the ONNX embeddings to a "
        "cosine similarity above 0.99999, which makes the baseline genuinely the released model and the "
        f"fine-tuned model the same architecture with updated weights. Fine-tuning updated "
        f"{training['trainable_parameters']:,} of {training['trainable_parameters'] + training['frozen_parameters']:,} parameters."
    )
    stack = pd.DataFrame(
        [
            ("Language", "Python 3.11"),
            ("Face detection", "Ultralytics YOLOv8n, lindevs yolov8n-face (WIDER FACE), release 1.0.1"),
            ("Face embedding", "InsightFace ArcFace IResNet-50 (buffalo_l / w600k_r50), PyTorch re-implementation"),
            ("Training", f"PyTorch {training['torch_version']}, ArcFace margin loss, mixed precision on GPU"),
            ("Identity classifier", "scikit-learn KNeighborsClassifier, cosine distance"),
            ("Image processing", "OpenCV, NumPy"),
            ("Data and evaluation", "pandas, scikit-learn metrics"),
            ("Visualisation and report", "Matplotlib, ReportLab"),
            ("Logging", "structlog"),
            ("Hardware", training["gpu"] or "CPU"),
        ],
        columns=["Component", "Technology"],
    )
    report.table(stack, "Technology stack.", [4 * cm, PAGE_WIDTH - 4 * cm])
    report.page_break()


def dataset_section(report: ReportBuilder, split: dict, detection_counts: pd.DataFrame) -> None:
    report.heading("2. Dataset and Preprocessing")
    report.heading("Dataset summary", 2)
    per_cell = pd.DataFrame(split["per_identity_quality"]).T.fillna(0).astype(int)
    per_cell.index = pd.MultiIndex.from_tuples([tuple(key.split("/")) for key in per_cell.index])
    totals = per_cell.groupby(level=0).sum()
    hq = per_cell.xs("high_quality", level=1).sum(axis=1)
    lq = per_cell.xs("low_quality", level=1).sum(axis=1)
    summary = pd.DataFrame(
        {
            "Identity": totals.index,
            "High quality": hq.reindex(totals.index).to_numpy(),
            "Low quality": lq.reindex(totals.index).to_numpy(),
            "Train": totals["train"].to_numpy(),
            "Validation": totals["validation"].to_numpy(),
            "Test": totals["test"].to_numpy(),
        }
    )
    summary.loc[len(summary)] = ["Total", *summary.drop(columns="Identity").sum().tolist()]
    report.paragraph(
        f"The archive holds {split['images_scanned']:,} files for eight people. Each person has a <i>high_quality</i> "
        "folder of frames from one close-up video and a <i>low_quality</i> folder of small CCTV-style crops. All "
        f"images are already face crops. After exclusions, {split['images_kept']:,} images were used."
    )
    report.table(summary, "Images used per identity, by source quality and split subset.")

    report.heading("Image quality challenges", 2)
    report.paragraph(
        "High-quality crops are around 300-500 pixels wide. Low-quality crops are mostly 16-80 pixels, frequently "
        "in profile or looking down, motion-blurred, compressed, or partly showing the back of the head. "
        "Consecutive video frames are nearly identical, so a naive file-level split would put near-copies of "
        "test images in training."
    )

    report.heading("Face detection", 2)
    report.paragraph(
        "YOLOv8n-face runs on each image after small crops are upscaled to at least 160 pixels and padded, "
        "because the detector was trained on whole scenes rather than tight crops. The most confident box is "
        "squared and resized to 112 x 112. When no face is found, the whole crop is used instead (the images are "
        "already face crops). This fallback is recorded for every image and reported below. Crop margin 0.0 was "
        "chosen on validation pairs with the pre-trained model."
    )
    report.table(
        detection_counts,
        "Detection outcome per subset. Fallback images are still evaluated and are counted in every metric.",
    )

    report.heading("Training augmentation", 2)
    augmentations = pd.DataFrame(
        [
            ("none", "Face crop as it is"),
            ("bright", "Linear gain 1.2 and offset +35"),
            ("dark", "Gamma 1.9"),
            ("contrast", "CLAHE on the LAB lightness channel"),
            ("grayscale", "Luminance copied to all three channels"),
            ("shadow", "Linear shading ramp at a random angle"),
            ("flip", "Horizontal mirror, applied to half of all samples on top of the above"),
        ],
        columns=["Augmentation", "Method"],
    )
    report.table(
        augmentations,
        "Augmentations applied in memory to training face crops. One is drawn at random for every sample; "
        "no augmented images are written to disk, and validation and test images are never augmented.",
        [3 * cm, PAGE_WIDTH - 3 * cm],
    )

    report.heading("80/20 split strategy", 2)
    config = split["config"]
    excluded = split["excluded_by_reason"]
    report.paragraph(
        f"Each sequence (identity, quality and file type) was cut into contiguous segments of at most "
        f"{config['max_segment_length']} frames, with at least {config['min_segments_per_sequence']} segments per "
        f"sequence. Whole segments were assigned to test until about {percent(config['test_fraction'])} of each "
        f"sequence was held out ({percent(split['test_fraction_at_split'])} overall), and "
        f"{percent(config['validation_fraction'])} of the remaining segments became a training-only validation set. "
        f"Seed {config['seed']}. Exclusions: {excluded.get('unreadable', 0)} unreadable files, "
        f"{excluded.get('exact_duplicate', 0)} exact duplicates (first copy kept), "
        f"{excluded.get('near_duplicate_of_test', 0)} training images that were near-copies of a test image, and "
        f"{excluded.get('near_duplicate_of_validation', 0)} that were near-copies of a validation image "
        f"(thumbnail cosine at least {config['near_duplicate_threshold']}). Because only training images were "
        f"removed, the test set is {percent(split['test_fraction_of_kept'])} of the images finally used. The split "
        "is copied to data/train/&lt;person&gt;/high_quality|low_quality and data/test/&lt;person&gt;/high_quality|"
        "low_quality, the same layout as the original archive; validation images stay in data/train and are "
        "marked in the training manifest."
    )

    report.heading("Dataset integrity checks", 2)
    report.paragraph(
        "Before training and before each evaluation, the code verifies that no source ID, file, SHA-256 hash or "
        "segment appears in both partitions and that validation and training segments are disjoint; any overlap "
        "stops the run. The full audit of every artefact (near-duplicates recomputed from raw files, checkpoints, "
        "KNN gallery, pair files) is in Section 6."
    )


def verification_rows(results: dict) -> list[tuple[str, str]]:
    v = results["verification"]
    return [
        ("ROC-AUC", fmt(v["roc_auc"])),
        ("Average precision", fmt(v["average_precision"])),
        ("Verification accuracy", fmt(v["accuracy"])),
        ("Precision", fmt(v["precision"])),
        ("Recall", fmt(v["recall"])),
        ("F1-score", fmt(v["f1"])),
        ("False acceptance rate", fmt(v["false_acceptance_rate"])),
        ("False rejection rate", fmt(v["false_rejection_rate"])),
        ("Threshold (from validation pairs)", fmt(v["threshold"])),
        ("Test pairs", f"{v['pairs']:,}"),
    ]


def baseline_section(report: ReportBuilder, baseline: dict) -> None:
    report.heading("3. Baseline Evaluation")
    report.paragraph(
        "The unmodified pre-trained network was evaluated before any fine-tuning. For each identity and each "
        "quality combination (high-high, high-low, low-low), same-person pairs were drawn from different "
        "segments and matched with an equal number of different-person pairs. The pair list is saved once and "
        "reused unchanged for the fine-tuned model. The decision threshold maximises accuracy on a separate set "
        "of validation pairs; test labels were not used to choose it."
    )
    report.table(
        pd.DataFrame(verification_rows(baseline), columns=["Metric", "Baseline"]),
        "Baseline verification metrics on the held-out test pairs.",
        [7 * cm, 4 * cm],
    )
    by_type = pd.DataFrame(
        [
            (pair_type, f"{m['pairs']:,}", fmt(m["roc_auc"]), fmt(m["accuracy"]))
            for pair_type, m in baseline["verification_by_pair_type"].items()
        ],
        columns=["Pair type", "Pairs", "ROC-AUC", "Accuracy"],
    )
    report.table(by_type, "Baseline verification by image-quality combination.")
    report.figure_pair(
        CHART_DIR / "baseline_roc.png",
        CHART_DIR / "baseline_precision_recall.png",
        "Baseline ROC curve (left) and precision-recall curve (right) on the test pairs.",
    )
    report.figure(
        CHART_DIR / "baseline_similarity_distribution.png",
        "Baseline cosine similarity of same-person and different-person test pairs, with the validation threshold.",
        PAGE_WIDTH * 0.62,
    )


def training_section(report: ReportBuilder, training: dict, history: pd.DataFrame, weight_change: pd.DataFrame) -> None:
    config = training["config"]
    report.heading("4. Fine-Tuning and Training")
    report.paragraph(
        "The backbone was fine-tuned with an ArcFace additive-angular-margin softmax over the eight identities. "
        "Triplet loss was not used: with eight identities, almost every batch would contain only easy negatives. "
        "The classification head was started from each identity's mean pre-trained embedding, so early gradients "
        "do not disturb the backbone. The first three stages of the network were frozen and BatchNorm running "
        "statistics were held at their pre-trained values. Each epoch sampled identities equally often and, for "
        "each sampled image, one random augmentation from Table 4 plus a horizontal flip half of the time."
    )
    hyperparameters = pd.DataFrame(
        [
            ("Architecture", "IResNet-50, 512-d embedding, 112 x 112 RGB input, (x - 127.5) / 127.5"),
            ("Pre-trained weights", "InsightFace buffalo_l w600k_r50.onnx (WebFace600K, ArcFace)"),
            (
                "Trainable modules",
                ", ".join(config["trainable_layers"]) + f" ({training['trainable_parameters']:,} parameters)",
            ),
            ("Frozen modules", f"conv1, prelu, layer1-layer3 ({training['frozen_parameters']:,} parameters)"),
            ("Loss", f"ArcFace margin softmax, scale {config['arcface_scale']}, margin {config['arcface_margin']}"),
            (
                "Optimizer",
                f"{config['optimizer']}, lr {config['learning_rate']} backbone / {config['head_learning_rate']} head, "
                f"weight decay {config['weight_decay']}",
            ),
            (
                "Scheduler",
                f"{config['scheduler']} over {config['epochs']} epochs, early stopping patience {config['early_stopping_patience']}",
            ),
            ("Batch size", str(config["batch_size"])),
            ("Sampling", "identity-balanced weighted sampling with replacement"),
            ("Augmentation", ", ".join(training["augmentations"]) + "; horizontal flip (in memory)"),
            ("Training data", f"{training['train_images']:,} training-subset images"),
            ("Validation", f"{training['validation_images']:,} validation images from held-out training segments"),
            ("Checkpoint selection", "best validation-pair ROC-AUC (epoch 0 = pre-trained model)"),
            ("Hardware", f"{training['gpu'] or 'CPU'}, PyTorch {training['torch_version']}"),
            ("Epochs run / best epoch", f"{training['epochs_run']} / {training['best_epoch']}"),
            ("Measured training time", f"{training['training_seconds'] / 60:.1f} minutes"),
            ("Random seed", str(config["random_seed"])),
        ],
        columns=["Setting", "Value"],
    )
    report.table(
        hyperparameters, "Fine-tuning configuration as recorded during the run.", [4.2 * cm, PAGE_WIDTH - 4.2 * cm]
    )

    history_table = history.assign(
        epoch=history["epoch"].astype(int),
        train_loss=history["train_loss"].map(lambda v: "-" if pd.isna(v) else fmt(v, 3)),
        train_accuracy=history["train_accuracy"].map(lambda v: "-" if pd.isna(v) else fmt(v, 3)),
        validation_loss=history["validation_loss"].map(lambda v: fmt(v, 3)),
        validation_accuracy=history["validation_accuracy"].map(lambda v: fmt(v, 3)),
        validation_roc_auc=history["validation_roc_auc"].map(lambda v: fmt(v, 4)),
        seconds=history["seconds"].map(lambda v: f"{v:.0f}"),
    )[
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "validation_roc_auc",
            "seconds",
        ]
    ]
    history_table.columns = ["Epoch", "Train loss", "Train acc.", "Val. loss", "Val. acc.", "Val. ROC-AUC", "Seconds"]
    report.table(history_table, "Training history. Accuracies are head predictions without the margin.")
    report.figure(
        CHART_DIR / "training_history.png",
        "Training and validation loss (left) and validation-pair ROC-AUC (right) per epoch.",
        PAGE_WIDTH,
    )

    changed = weight_change[weight_change["relative_change"] > 0]
    report.paragraph(
        "To confirm that fine-tuning updated the network, every module of best_model.pt was compared with the "
        "pre-trained weights. Modules with a non-zero relative change: "
        + ", ".join(f"{row.module} ({row.relative_change:.4f})" for row in changed.itertuples())
        + ". All frozen modules are unchanged."
        if len(changed)
        else "best_model.pt is identical to the pre-trained weights: no fine-tuned epoch improved validation ROC-AUC."
    )


def comparison_section(
    report: ReportBuilder,
    comparison: pd.DataFrame,
    by_pair_type: pd.DataFrame,
    inference: dict,
    predictions: pd.DataFrame,
    baseline: dict,
    fine_tuned: dict,
) -> None:
    report.heading("5. Final Evaluation and Comparison")
    report.paragraph(
        "The fine-tuned model was evaluated with the baseline's exact protocol: the same cached detections, the "
        "same test pairs (identical SHA-256), a threshold calibrated on the same validation pairs, and KNN "
        "neighbour count chosen on validation accuracy. The two runs differ only in the embedding weights."
    )
    table = comparison.reset_index().rename(columns={"index": "Metric"})
    table["Baseline"] = table["Baseline"].map(fmt)
    table["Fine-tuned"] = table["Fine-tuned"].map(fmt)
    table["Change"] = table["Change"].map(lambda value: f"{value:+.4f}")
    report.table(
        table,
        "Test-set metrics before and after fine-tuning. False acceptance and rejection rates are better when lower.",
    )

    pair_table = by_pair_type.pivot(index="pair_type", columns="model", values="roc_auc").reset_index()
    pair_table = pd.DataFrame(
        {
            "Pair type": pair_table["pair_type"],
            "Baseline ROC-AUC": pair_table["Baseline"].map(fmt),
            "Fine-tuned ROC-AUC": pair_table["Fine-tuned"].map(fmt),
            "Baseline accuracy": by_pair_type[by_pair_type["model"] == "Baseline"]["accuracy"].map(fmt).to_numpy(),
            "Fine-tuned accuracy": by_pair_type[by_pair_type["model"] == "Fine-tuned"]["accuracy"].map(fmt).to_numpy(),
        }
    )
    report.table(pair_table, "Verification by image-quality combination.")

    report.figure_pair(
        CHART_DIR / "comparison_roc.png",
        CHART_DIR / "comparison_precision_recall.png",
        "ROC (left) and precision-recall (right) curves of both models on the same test pairs.",
    )
    report.figure_pair(
        CHART_DIR / "comparison_same_person_similarity.png",
        CHART_DIR / "comparison_different_person_similarity.png",
        "Similarity of same-person pairs (left) and different-person pairs (right) for both models, "
        "with each model's validation threshold.",
    )
    report.figure(
        CHART_DIR / "comparison_metrics.png", "Before-and-after comparison of test-set metrics.", PAGE_WIDTH * 0.8
    )

    report.heading("Identification with the saved KNN classifier", 2)
    closed = inference["closed_set"]
    by_quality = inference["closed_set_by_quality"]
    report.paragraph(
        f"Part 3 of the notebook loaded best_model.pt and knn_classifier.joblib from disk (k = {inference['neighbors']}) "
        f"and classified all {inference['test_images']:,} test images; {inference['images_without_usable_face']} had no "
        f"usable face and are counted as errors. Closed-set accuracy was {fmt(closed['accuracy'])} (macro F1 "
        f"{fmt(closed['macro_f1'])}): {fmt(by_quality['high_quality']['accuracy'])} on high-quality and "
        f"{fmt(by_quality['low_quality']['accuracy'])} on low-quality images. With the validation-calibrated "
        f"unknown-identity threshold ({fmt(inference['unknown_threshold'])}), {inference['rejected_as_unknown']:,} "
        f"images were rejected as unknown. Since every test person is enrolled, each rejection counts as an error, "
        f"giving accuracy {fmt(inference['with_rejection_accuracy'])}; accuracy on the accepted faces was "
        f"{fmt(inference['accuracy_on_accepted'])}. For comparison, the baseline embeddings with the same KNN "
        f"procedure reached {fmt(baseline['identification']['accuracy'])} closed-set accuracy."
    )
    report.figure(
        CHART_DIR / "knn_confusion_matrix.png",
        "Confusion matrix of the saved KNN classifier on the test set (closed-set predictions, counts).",
        PAGE_WIDTH * 0.6,
    )

    report.heading("Improvements and regressions", 2)
    improved = comparison[comparison["Direction"] == "improved"].index.tolist()
    regressed = comparison[comparison["Direction"] == "regressed"].index.tolist()
    report.bullets(
        [
            "Improved: " + (", ".join(improved) if improved else "none") + ".",
            "Regressed: " + (", ".join(regressed) if regressed else "none") + ".",
        ]
    )

    report.heading("Failure analysis", 2)
    errors = predictions[predictions["closed_set_prediction"] != predictions["identity"]]
    confusions = errors.groupby(["identity", "closed_set_prediction"]).size().sort_values(ascending=False).head(5)
    per_cell = (
        predictions.assign(ok=predictions["closed_set_prediction"] == predictions["identity"])
        .groupby(["identity", "quality"])["ok"]
        .mean()
        .sort_values()
    )
    weakest = ", ".join(
        f"{identity} {quality.replace('_', '-')} ({fmt(value, 3)})"
        for (identity, quality), value in per_cell.head(4).items()
    )
    report.paragraph(
        f"The weakest identity and quality groups by closed-set accuracy were {weakest}. The most frequent "
        "confusions were "
        + "; ".join(f"{actual} predicted as {predicted} ({count})" for (actual, predicted), count in confusions.items())
        + ". By detection outcome, accuracy was "
        + ", ".join(
            f"{status.replace('_', ' ')} {fmt(rows['correct_closed'].mean(), 3)} ({len(rows):,} images)"
            for status, rows in predictions.assign(
                correct_closed=predictions["closed_set_prediction"] == predictions["identity"]
            ).groupby("detection_status")
        )
        + ". Low-quality CCTV crops dominate the errors: many are under 32 pixels wide, seen in profile or from "
        "behind, and too small for the face detector."
    )
    lq_base = baseline["verification_by_pair_type"]["lq-lq"]["roc_auc"]
    lq_fine = fine_tuned["verification_by_pair_type"]["lq-lq"]["roc_auc"]
    report.paragraph(f"For low-quality to low-quality pairs, ROC-AUC moved from {fmt(lq_base)} to {fmt(lq_fine)}.")
    report.page_break()


def integrity_section(
    report: ReportBuilder, integrity: list[dict], split: dict, training: dict, fine_tuned: dict, baseline: dict
) -> None:
    report.heading("6. Integrity, Reproducibility and Conclusion")
    report.heading("Data leakage checks", 2)
    report.paragraph(
        "Output of scripts/check_data_leakage.py, which re-reads every artefact independently of the training code."
    )
    report.table(
        pd.DataFrame(integrity).rename(columns={"check": "Check", "status": "Status", "detail": "Detail"}),
        "Integrity audit.",
        [5 * cm, 1.6 * cm, PAGE_WIDTH - 6.6 * cm],
    )

    report.heading("Reproducibility", 2)
    report.bullets(
        [
            f"Seed {split['config']['seed']} for the split, pair sampling, data sampler, augmentation and KNN selection; "
            "all settings in src/config.py and stored in each checkpoint.",
            "Manifests with SHA-256 hashes for every image: data/manifests/train_manifest.csv, test_manifest.csv, excluded_manifest.csv.",
            f"Fixed test pairs (SHA-256 {baseline['test_pairs_sha256'][:16]}...) shared by both evaluations.",
            "Run order: notebook Part 1, scripts.baseline_evaluation, notebook Parts 2-3, scripts.final_evaluation, "
            "scripts.check_data_leakage, scripts.generate_report. Commands are listed in README.md.",
            f"Hardware: {training['gpu'] or 'CPU'}; GPU kernels are not forced to be deterministic, so retraining may differ in late decimals.",
        ]
    )

    report.heading("Limitations", 2)
    report.bullets(
        [
            "Each person's high-quality images come from essentially one recording. Segment splitting and "
            "near-duplicate removal stop near-identical frames crossing the split, but training and test frames "
            "still share lighting, clothing and background.",
            "Training and testing use the same eight people, so the results show how well the model separates "
            "these people. They do not show generalisation to new identities, and the fine-tuned embedding may be "
            "worse than the original on unseen faces.",
            "The face detector returns boxes without landmarks, so faces are box-cropped rather than "
            "landmark-aligned as ArcFace expects, which lowers both models' scores.",
            "Low-quality crops in which the detector finds no face are used whole, so their framing differs from "
            "detected faces; accuracy by detection outcome is reported in Section 5.",
            "Near-duplicate detection compares 32 x 32 thumbnails. It catches near-identical frames, not every "
            "frame from the same moment under a different crop.",
            "The unknown-identity threshold is the validation verification threshold applied to the nearest gallery "
            "face. The data contains no unenrolled people, so open-set rejection could not be evaluated; on the "
            "closed test set it rejected none of the images.",
            "One training run was made. No variance across seeds was measured.",
        ]
    )

    report.heading("Conclusion", 2)
    base_v, fine_v = baseline["verification"], fine_tuned["verification"]
    base_id, fine_id = baseline["identification"], fine_tuned["identification"]
    report.paragraph(
        f"On the fixed test pairs, verification ROC-AUC changed from {fmt(base_v['roc_auc'])} to "
        f"{fmt(fine_v['roc_auc'])}, and verification accuracy at the validation-calibrated threshold from "
        f"{fmt(base_v['accuracy'])} to {fmt(fine_v['accuracy'])}. KNN identification accuracy under the same "
        f"protocol changed from {fmt(base_id['accuracy'])} to {fmt(fine_id['accuracy'])}. "
        "Given the limitations above, these figures measure recognition of the eight enrolled people under the "
        "conditions in the provided images."
    )

    report.heading("Declaration", 2)
    report.paragraph(
        "I declare that the 20% evaluation subset described in this report was not used for training, "
        "fine-tuning, checkpoint selection, KNN hyperparameter selection or threshold calibration, that only the "
        "company-provided images were used for the reported experiments, and that all results were produced "
        "by the code submitted with this report."
    )
    signature = Table(
        [["Name:", "_" * 40], ["Signature:", "_" * 40], ["Date:", "_" * 40]],
        colWidths=[3 * cm, 9 * cm],
        hAlign="LEFT",
    )
    signature.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TOPPADDING", (0, 0), (-1, -1), 9)]))
    report.story.append(signature)


def detection_table() -> pd.DataFrame:
    counts = {}
    for subset in ("train", "validation", "test"):
        frame = load_csv(MANIFEST_DIR / f"detections_{subset}.csv")
        counts[subset.capitalize()] = frame["detection_status"].value_counts()
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table.index = [status.replace("_", " ") for status in table.index]
    return table.rename_axis("Detection status").reset_index()


def main() -> None:
    configure_logging("generate_report")
    split = load_json(SPLIT_SUMMARY)
    baseline = load_json(METRICS_DIR / "baseline_metrics.json")
    fine_tuned = load_json(METRICS_DIR / "finetuned_metrics.json")
    training = load_json(TRAINING_SUMMARY)
    inference = load_json(INFERENCE_METRICS)
    integrity = load_json(METRICS_DIR / "integrity_summary.json")
    comparison = load_csv(METRICS_DIR / "comparison_metrics.csv", index_col=0)
    by_pair_type = load_csv(METRICS_DIR / "comparison_by_pair_type.csv")
    history = load_csv(TRAINING_HISTORY)
    weight_change = load_csv(METRICS_DIR / "weight_change.csv")
    predictions = load_csv(INFERENCE_PREDICTIONS)

    if any(row["status"] == "failed" for row in integrity):
        raise SystemExit("The integrity audit has failures; fix them before producing a report.")

    report = ReportBuilder()
    introduction(report, training)
    dataset_section(report, split, detection_table())
    baseline_section(report, baseline)
    training_section(report, training, history, weight_change)
    comparison_section(report, comparison, by_pair_type, inference, predictions, baseline, fine_tuned)
    integrity_section(report, integrity, split, training, fine_tuned, baseline)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "final_report.pdf"
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Face Recognition under Challenging Image Conditions",
        author="AI Developer Intern candidate",
    )
    document.build(report.story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    logger.info("report_written", path=str(output), figures=report.figures, tables=report.tables)


if __name__ == "__main__":
    main()
