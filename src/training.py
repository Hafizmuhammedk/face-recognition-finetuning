import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import (
    ARCFACE_ONNX_WEIGHTS,
    AUGMENTATIONS,
    BEST_MODEL,
    FINAL_MODEL,
    PROJECT_ROOT,
    TRAINING_HISTORY,
    DetectionConfig,
    TrainingConfig,
)
from src.detection import crop_face, read_image
from src.embeddings import IResNet, faces_to_tensor
from src.evaluation import embed_faces
from src.logging_config import get_logger
from src.metrics import pair_scores

logger = get_logger(__name__)

DARK_TABLE = (np.linspace(0, 1, 256) ** 1.9 * 255).round().astype(np.uint8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_shadow(face: np.ndarray, angle_degrees: float, strength: float = 0.55) -> np.ndarray:
    size = face.shape[0]
    ys, xs = np.mgrid[0:size, 0:size] / (size - 1)
    angle = np.deg2rad(angle_degrees)
    ramp = xs * np.cos(angle) + ys * np.sin(angle)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    mask = 1.0 - strength * np.clip((ramp - 0.35) / 0.4, 0, 1)
    return (face * mask[..., None]).astype(np.uint8)


def augment_face(face: np.ndarray, kind: str) -> np.ndarray:
    if kind == "bright":
        return cv2.convertScaleAbs(face, alpha=1.2, beta=35)
    if kind == "dark":
        return cv2.LUT(face, DARK_TABLE)
    if kind == "contrast":
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if kind == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if kind == "shadow":
        return add_shadow(face, float(torch.rand(1)) * 360)
    return face


class ArcFaceHead(nn.Module):
    """ArcFace additive angular margin classifier (Deng et al., 2019), used only during training."""

    def __init__(self, embedding_dimension: int, num_classes: int, scale: float, margin: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return cosine * self.scale
        theta = torch.acos(cosine)
        # cos(theta + m) is no longer monotonic once theta + m > pi, so switch to a linear penalty there.
        with_margin = torch.where(
            theta + self.margin < math.pi,
            torch.cos(theta + self.margin),
            cosine - self.margin * math.sin(self.margin),
        )
        is_target = F.one_hot(labels, cosine.shape[1]).bool()
        return torch.where(is_target, with_margin, cosine) * self.scale

    @torch.no_grad()
    def initialise_from_class_means(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        # A random head sends large gradients into the pre-trained backbone in the first steps.
        for label in range(self.weight.shape[0]):
            self.weight[label] = torch.from_numpy(embeddings[labels == label].mean(axis=0))


class AugmentedFaceDataset(Dataset):
    def __init__(self, faces: pd.DataFrame, labels: np.ndarray, config: DetectionConfig):
        self.faces = faces.reset_index(drop=True)
        self.labels = labels
        self.config = config

    def __len__(self) -> int:
        return len(self.faces)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.faces.iloc[index]
        image = read_image(PROJECT_ROOT / row["path"])
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        face = crop_face(image, box, self.config.crop_margin, self.config.face_size)

        # torch's random generator is seeded per DataLoader worker, so use it rather than NumPy.
        kind = AUGMENTATIONS[int(torch.randint(len(AUGMENTATIONS), (1,)))]
        face = augment_face(face, kind)
        if torch.rand(1).item() < 0.5:
            face = face[:, ::-1]
        return faces_to_tensor(face[None])[0], int(self.labels[index])


def set_trainable_layers(model: IResNet, trainable_layers: tuple[str, ...]) -> dict[str, int]:
    prefixes = tuple(f"{name}." for name in trainable_layers)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefixes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": trainable, "frozen_parameters": total - trainable}


def build_optimizer(model: IResNet, head: ArcFaceHead, config: TrainingConfig) -> torch.optim.Optimizer:
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": config.learning_rate},
        {"params": list(head.parameters()), "lr": config.head_learning_rate},
    ]
    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def train_one_epoch(
    model: IResNet,
    head: ArcFaceHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
) -> dict[str, float]:
    # eval() keeps the pre-trained BatchNorm statistics; batches of 64 are too small to
    # re-estimate them. The unfrozen weights still get gradients.
    model.eval()
    head.train()
    use_amp = device.startswith("cuda")
    total_loss, correct, seen = 0.0, 0, 0

    for faces, labels in loader:
        faces, labels = faces.to(device, non_blocking=True), labels.to(device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
            embeddings = model(faces)
        logits = head(embeddings.float(), labels)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            correct += (head(embeddings.float()).argmax(dim=1) == labels).sum().item()
        total_loss += loss.item() * len(labels)
        seen += len(labels)

    return {"train_loss": total_loss / seen, "train_accuracy": correct / seen}


@torch.no_grad()
def validate(
    model: IResNet,
    head: ArcFaceHead,
    validation_faces: pd.DataFrame,
    validation_labels: np.ndarray,
    validation_pairs: pd.DataFrame,
    detection_config: DetectionConfig,
    device: str,
) -> dict[str, float]:
    embeddings = embed_faces(model, validation_faces, detection_config, device)
    head.eval()
    tensor = torch.from_numpy(embeddings).to(device)
    labels = torch.tensor(validation_labels, device=device)
    scores = pair_scores(validation_pairs, validation_faces["source_id"].to_numpy(), embeddings)
    return {
        "validation_loss": F.cross_entropy(head(tensor, labels), labels).item(),
        "validation_accuracy": (head(tensor).argmax(dim=1) == labels).float().mean().item(),
        "validation_roc_auc": float(roc_auc_score(validation_pairs["same_person"], scores)),
    }


def save_checkpoint(
    path: Path,
    model: IResNet,
    head: ArcFaceHead,
    identities: list[str],
    config: TrainingConfig,
    detection_config: DetectionConfig,
    epoch: int,
    metrics: dict,
    train_source_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "identities": identities,
            "epoch": epoch,
            "validation_metrics": metrics,
            "config": config.to_dict(),
            "preprocessing": {
                **asdict(detection_config),
                "color_order": "RGB",
                "pixel_normalization": "(pixel - 127.5) / 127.5",
            },
            "pretrained_weights": ARCFACE_ONNX_WEIGHTS.name,
            "train_source_ids": train_source_ids,
        },
        path,
    )


def fine_tune(
    model: IResNet,
    head: ArcFaceHead,
    train_faces: pd.DataFrame,
    validation_faces: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    identities: list[str],
    config: TrainingConfig,
    detection_config: DetectionConfig,
) -> pd.DataFrame:
    """Train and keep the checkpoint with the best validation ROC-AUC.

    Epoch 0 is the pre-trained model, so if no epoch beats it best_model.pt stays unchanged.
    """
    device = config.device
    label_of = {identity: index for index, identity in enumerate(identities)}
    train_labels = train_faces["identity"].map(label_of).to_numpy()
    validation_labels = validation_faces["identity"].map(label_of).to_numpy()
    train_source_ids = train_faces["source_id"].tolist()

    dataset = AugmentedFaceDataset(train_faces, train_labels, detection_config)
    # People have between about 500 and 4,500 training images; sample each person equally often.
    sample_weights = 1.0 / train_faces["identity"].map(train_faces["identity"].value_counts()).to_numpy()
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    model.to(device)
    head.to(device)
    optimizer = build_optimizer(model, head, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    def checkpoint(path: Path, epoch: int, metrics: dict) -> None:
        save_checkpoint(path, model, head, identities, config, detection_config, epoch, metrics, train_source_ids)

    initial = validate(model, head, validation_faces, validation_labels, validation_pairs, detection_config, device)
    history = [
        {"epoch": 0, "train_loss": np.nan, "train_accuracy": np.nan, **initial, "learning_rate": 0.0, "seconds": 0.0}
    ]
    logger.info("pretrained_validation", **{key: round(value, 4) for key, value in initial.items()})
    best_auc = initial["validation_roc_auc"]
    checkpoint(BEST_MODEL, 0, initial)
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, head, loader, optimizer, scaler, device)
        validation_metrics = validate(
            model, head, validation_faces, validation_labels, validation_pairs, detection_config, device
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            **train_metrics,
            **validation_metrics,
            "learning_rate": learning_rate,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        logger.info("training_epoch_complete", **{key: round(value, 5) for key, value in row.items()})

        if validation_metrics["validation_roc_auc"] > best_auc:
            best_auc = validation_metrics["validation_roc_auc"]
            checkpoint(BEST_MODEL, epoch, validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info("early_stopping", epoch=epoch, best_validation_roc_auc=round(best_auc, 5))
                break

    checkpoint(FINAL_MODEL, epoch, validation_metrics)
    history = pd.DataFrame(history)
    TRAINING_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(TRAINING_HISTORY, index=False)
    return history


def weight_change_summary(pretrained: IResNet, fine_tuned: IResNet) -> pd.DataFrame:
    """Relative change ||w_new - w_old|| / ||w_old|| for each top-level module."""
    old = dict(pretrained.cpu().named_parameters())
    rows = []
    for module_name, module in fine_tuned.cpu().named_children():
        names = [f"{module_name}.{name}" for name, _ in module.named_parameters()]
        before = torch.cat([old[name].detach().flatten() for name in names])
        after = torch.cat([parameter.detach().flatten() for parameter in module.parameters()])
        rows.append(
            {
                "module": module_name,
                "parameters": before.numel(),
                "relative_change": float((after - before).norm() / before.norm()),
            }
        )
    return pd.DataFrame(rows)
