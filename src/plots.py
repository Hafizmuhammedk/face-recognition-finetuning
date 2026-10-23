from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

SERIES_COLORS = ["#2a78d6", "#eb6834"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    for color, (label, (labels, scores)) in zip(SERIES_COLORS, curves.items()):
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        axis.plot(false_positive_rate, true_positive_rate, color=color, label=label)
    axis.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="Chance")
    axis.set(xlabel="False acceptance rate", ylabel="True acceptance rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(loc="lower right")
    return _save(figure, path)


def plot_precision_recall_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    for color, (label, (labels, scores)) in zip(SERIES_COLORS, curves.items()):
        precision, recall, _ = precision_recall_curve(labels, scores)
        axis.plot(recall, precision, color=color, label=label)
    axis.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(loc="lower left")
    return _save(figure, path)


def plot_similarity_histograms(
    series: dict[str, np.ndarray],
    path: Path,
    title: str,
    thresholds: dict[str, float] | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    bins = np.linspace(-0.4, 1.0, 57)
    colors = dict(zip(series, SERIES_COLORS))
    for label, scores in series.items():
        axis.hist(scores, bins=bins, color=colors[label], alpha=0.55, label=label, edgecolor="white", linewidth=0.5)
    for label, threshold in (thresholds or {}).items():
        # A threshold named after a series takes its colour; anything else is drawn in grey.
        color = colors.get(label, TEXT_SECONDARY)
        axis.axvline(
            threshold, color=color, linestyle="--", linewidth=1.5, label=f"{label} threshold = {threshold:.3f}"
        )
    axis.set(xlabel="Cosine similarity", ylabel="Pairs", title=title)
    axis.legend(loc="upper right", fontsize=9)
    return _save(figure, path)


def plot_training_history(history: pd.DataFrame, path: Path) -> Path:
    trained = history[history["epoch"] > 0]
    figure, (loss_axis, auc_axis) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    loss_axis.plot(trained["epoch"], trained["train_loss"], color=SERIES_COLORS[0], marker="o", label="Training")
    loss_axis.plot(history["epoch"], history["validation_loss"], color=SERIES_COLORS[1], marker="o", label="Validation")
    loss_axis.set(xlabel="Epoch", ylabel="ArcFace loss", title="Training and validation loss")
    loss_axis.legend()

    auc_axis.plot(history["epoch"], history["validation_roc_auc"], color=SERIES_COLORS[1], marker="o")
    best = history.loc[history["validation_roc_auc"].idxmax()]
    auc_axis.annotate(
        f"best epoch {int(best['epoch'])}: {best['validation_roc_auc']:.4f}",
        (best["epoch"], best["validation_roc_auc"]),
        textcoords="offset points",
        xytext=(0, -40),
        ha="center",
        color=TEXT_PRIMARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8},
    )
    auc_axis.set(xlabel="Epoch (0 = pre-trained)", ylabel="ROC-AUC", title="Validation verification ROC-AUC")
    for axis in (loss_axis, auc_axis):
        axis.set_xticks(history["epoch"])
    return _save(figure, path)


def plot_metric_comparison(table: pd.DataFrame, path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(7.0, 0.42 * len(table) + 1.2))
    positions = np.arange(len(table))
    height = 0.38
    for offset, (color, column) in enumerate(zip(SERIES_COLORS, table.columns)):
        bars = axis.barh(
            positions + (offset - 0.5) * height, table[column], height=height * 0.92, color=color, label=column
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=TEXT_PRIMARY)
    axis.set_yticks(positions, table.index)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.12)
    axis.grid(axis="y", visible=False)
    axis.set(xlabel="Value", title=title)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(table.columns))
    return _save(figure, path)


def plot_confusion_matrix(actual: pd.Series, predicted: pd.Series, labels: list[str], path: Path, title: str) -> Path:
    # Colour shows the share of each row; the numbers are raw counts.
    counts = confusion_matrix(actual, predicted, labels=labels)
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(0.62 * len(labels) + 2.6, 0.55 * len(labels) + 2.0))
    axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            if counts[row, column]:
                color = "white" if rates[row, column] > 0.6 else TEXT_PRIMARY
                axis.text(column, row, counts[row, column], ha="center", va="center", fontsize=8, color=color)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.grid(False)
    axis.set(xlabel="Predicted identity", ylabel="Actual identity", title=title)
    return _save(figure, path)
