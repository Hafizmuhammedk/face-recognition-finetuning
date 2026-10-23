"""Score the pre-trained model on the test set before any fine-tuning. Run after Part 1 of the notebook.

python -m scripts.baseline_evaluation
"""

import torch

from src.config import BEST_MODEL
from src.embeddings import load_pretrained_arcface
from src.evaluation import evaluate_model, metrics_table
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    configure_logging("baseline_evaluation")
    if BEST_MODEL.exists():
        logger.warning("fine_tuned_checkpoint_already_exists", note="the baseline does not use it")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_model(load_pretrained_arcface(), "baseline", allow_create_test_pairs=True, device=device)

    print(metrics_table({"Baseline": results}).round(4).to_string())
    print("Validation-calibrated threshold:", round(results["verification"]["threshold"], 4))


if __name__ == "__main__":
    main()
