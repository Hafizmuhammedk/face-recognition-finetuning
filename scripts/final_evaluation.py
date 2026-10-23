"""Score the fine-tuned model on the baseline's test pairs and compare the two. Run after Part 3.

python -m scripts.final_evaluation
"""

import json

import pandas as pd
import torch

from src.config import (
    BEST_MODEL,
    CHART_DIR,
    INFERENCE_PREDICTIONS,
    METRICS_DIR,
    TRAINING_HISTORY,
)
from src.embeddings import load_embedding_model
from src.evaluation import evaluate_model, metrics_table
from src.logging_config import configure_logging, get_logger
from src.metrics import PAIR_TYPES
from src.plots import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_similarity_histograms,
    plot_training_history,
)

logger = get_logger(__name__)

LOWER_IS_BETTER = {"False acceptance rate", "False rejection rate"}


def load_baseline() -> dict:
    path = METRICS_DIR / "baseline_metrics.json"
    if not path.exists():
        raise FileNotFoundError("Baseline metrics not found. Run scripts/baseline_evaluation.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison(baseline: dict, fine_tuned: dict) -> pd.DataFrame:
    table = metrics_table({"Baseline": baseline, "Fine-tuned": fine_tuned})
    table["Change"] = table["Fine-tuned"] - table["Baseline"]
    better = [(change < 0) if metric in LOWER_IS_BETTER else (change > 0) for metric, change in table["Change"].items()]
    table["Direction"] = [
        "improved" if good else ("unchanged" if change == 0 else "regressed")
        for good, change in zip(better, table["Change"])
    ]
    return table


def pair_type_comparison(baseline: dict, fine_tuned: dict) -> pd.DataFrame:
    rows = []
    for pair_type in PAIR_TYPES:
        for model, results in (("Baseline", baseline), ("Fine-tuned", fine_tuned)):
            metrics = results["verification_by_pair_type"][pair_type]
            rows.append(
                {
                    "pair_type": pair_type,
                    "model": model,
                    **{
                        key: metrics[key]
                        for key in ("pairs", "roc_auc", "accuracy", "false_acceptance_rate", "false_rejection_rate")
                    },
                }
            )
    return pd.DataFrame(rows)


def plot_comparisons(comparison: pd.DataFrame) -> None:
    scores = {
        "Baseline": pd.read_csv(METRICS_DIR / "baseline_test_pair_scores.csv"),
        "Fine-tuned": pd.read_csv(METRICS_DIR / "finetuned_test_pair_scores.csv"),
    }
    curves = {name: (frame["same_person"].to_numpy(), frame["score"].to_numpy()) for name, frame in scores.items()}
    plot_roc_curves(curves, CHART_DIR / "comparison_roc.png", "ROC: baseline vs fine-tuned")
    plot_precision_recall_curves(
        curves, CHART_DIR / "comparison_precision_recall.png", "Precision-recall: baseline vs fine-tuned"
    )

    thresholds = {
        name: json.loads((METRICS_DIR / f"{key}_metrics.json").read_text())["verification"]["threshold"]
        for name, key in (("Baseline", "baseline"), ("Fine-tuned", "finetuned"))
    }
    for same, label in ((1, "same_person"), (0, "different_person")):
        plot_similarity_histograms(
            {name: frame.loc[frame["same_person"] == same, "score"].to_numpy() for name, frame in scores.items()},
            CHART_DIR / f"comparison_{label}_similarity.png",
            f"{label.replace('_', '-').capitalize()} pair similarity",
            thresholds=thresholds,
        )

    plot_metric_comparison(
        comparison[["Baseline", "Fine-tuned"]].rename(
            index=lambda name: f"{name} (lower is better)" if name in LOWER_IS_BETTER else name
        ),
        CHART_DIR / "comparison_metrics.png",
        "Test-set metrics before and after fine-tuning",
    )

    if TRAINING_HISTORY.exists():
        plot_training_history(pd.read_csv(TRAINING_HISTORY), CHART_DIR / "training_history.png")
    if INFERENCE_PREDICTIONS.exists():
        predictions = pd.read_csv(INFERENCE_PREDICTIONS).dropna(subset=["closed_set_prediction"])
        labels = sorted(predictions["identity"].unique())
        plot_confusion_matrix(
            predictions["identity"],
            predictions["closed_set_prediction"],
            labels,
            CHART_DIR / "knn_confusion_matrix.png",
            "KNN identification on the test set",
        )
    else:
        logger.warning("inference_predictions_missing", note="run Part 3 of the notebook for the confusion matrix")


def main() -> None:
    configure_logging("final_evaluation")
    baseline = load_baseline()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint = load_embedding_model(BEST_MODEL, device)
    logger.info("fine_tuned_model_loaded", checkpoint=BEST_MODEL.name, epoch=checkpoint["epoch"])
    fine_tuned = evaluate_model(model, "finetuned", allow_create_test_pairs=False, device=device)

    if fine_tuned["test_pairs_sha256"] != baseline["test_pairs_sha256"]:
        raise RuntimeError("Baseline and fine-tuned evaluations used different test pairs; results are not comparable")

    comparison = build_comparison(baseline, fine_tuned)
    by_pair_type = pair_type_comparison(baseline, fine_tuned)
    comparison.to_csv(METRICS_DIR / "comparison_metrics.csv")
    by_pair_type.to_csv(METRICS_DIR / "comparison_by_pair_type.csv", index=False)
    plot_comparisons(comparison)

    print(comparison.round(4).to_string())
    print()
    print(by_pair_type.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
