import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Equal pairs per quality combination, otherwise the much larger low-quality folders dominate.
PAIR_TYPES = {
    "hq-hq": ("high_quality", "high_quality"),
    "hq-lq": ("high_quality", "low_quality"),
    "lq-lq": ("low_quality", "low_quality"),
}


def make_verification_pairs(images: pd.DataFrame, pairs_per_identity: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_type = pairs_per_identity // len(PAIR_TYPES)
    pairs = []
    for identity, own in images.groupby("identity", sort=True):
        others = images[images["identity"] != identity]
        for pair_type, (left_quality, right_quality) in PAIR_TYPES.items():
            left_pool = own[own["quality"] == left_quality]
            right_pool = own[own["quality"] == right_quality]
            impostor_pool = others[others["quality"] == right_quality]
            if left_pool.empty or right_pool.empty:
                continue

            genuine = set()
            for _ in range(per_type * 20):
                if len(genuine) == per_type:
                    break
                left = left_pool.iloc[rng.integers(len(left_pool))]
                right = right_pool.iloc[rng.integers(len(right_pool))]
                # Different segments, so a same-person pair is never two neighbouring frames.
                if left["segment"] != right["segment"]:
                    genuine.add(tuple(sorted((left["source_id"], right["source_id"]))))
            impostors = set(
                zip(
                    rng.choice(left_pool["source_id"].to_numpy(), len(genuine)),
                    rng.choice(impostor_pool["source_id"].to_numpy(), len(genuine)),
                )
            )

            pairs += [(left, right, 1, pair_type) for left, right in sorted(genuine)]
            pairs += [(left, right, 0, pair_type) for left, right in sorted(impostors)]

    return pd.DataFrame(pairs, columns=["left_id", "right_id", "same_person", "pair_type"])


def pair_scores(pairs: pd.DataFrame, source_ids: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    row_of = pd.Series(np.arange(len(source_ids)), index=source_ids)
    left = embeddings[row_of[pairs["left_id"]].to_numpy()]
    right = embeddings[row_of[pairs["right_id"]].to_numpy()]
    return (left * right).sum(axis=1)


def select_threshold(same_person: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(same_person, scores)
    positives = same_person.sum()
    negatives = len(same_person) - positives
    accuracy = (true_positive_rate * positives + (1 - false_positive_rate) * negatives) / len(same_person)
    best = int(np.argmax(accuracy[1:])) + 1  # index 0 is the artificial +inf threshold
    return float(thresholds[best])


def verification_metrics(same_person: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    accepted = scores >= threshold
    genuine = same_person == 1
    precision, recall, f1, _ = precision_recall_fscore_support(same_person, accepted, average="binary", zero_division=0)
    return {
        "pairs": int(len(scores)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(same_person, scores)),
        "average_precision": float(average_precision_score(same_person, scores)),
        "accuracy": float((accepted == genuine).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_acceptance_rate": float(accepted[~genuine].mean()),
        "false_rejection_rate": float((~accepted[genuine]).mean()),
        "mean_same_person_similarity": float(scores[genuine].mean()),
        "mean_different_person_similarity": float(scores[~genuine].mean()),
    }


def identification_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "images": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }
