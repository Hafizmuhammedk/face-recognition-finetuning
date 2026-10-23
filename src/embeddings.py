from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import ARCFACE_ONNX_WEIGHTS, DetectionConfig
from src.detection import crop_face, read_image


class IBasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Conv2d(in_channels, channels, 1, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.conv2(self.prelu(self.conv1(self.bn1(x)))) + identity


class IResNet(nn.Module):
    # InsightFace's ONNX export folds the BatchNorm after each convolution into the
    # convolution bias. This module uses the same layout so the weights load exactly.

    def __init__(self, layers: tuple[int, ...] = (3, 4, 14, 3), embedding_dimension: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.prelu = nn.PReLU(64)
        in_channels = 64
        for index, (blocks, channels) in enumerate(zip(layers, (64, 128, 256, 512)), start=1):
            stage = [IBasicBlock(in_channels, channels, stride=2)]
            stage += [IBasicBlock(channels, channels, stride=1) for _ in range(blocks - 1)]
            setattr(self, f"layer{index}", nn.Sequential(*stage))
            in_channels = channels
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dimension)
        self.features = nn.BatchNorm1d(embedding_dimension)

    def blocks(self) -> list[IBasicBlock]:
        return [block for stage in (self.layer1, self.layer2, self.layer3, self.layer4) for block in stage]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.conv1(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.bn2(x), 1)
        return self.features(self.fc(x))


def load_pretrained_arcface(onnx_path: Path = ARCFACE_ONNX_WEIGHTS) -> IResNet:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ArcFace weights not found: {onnx_path}")
    graph = onnx.load(str(onnx_path)).graph
    weights = {tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy()) for tensor in graph.initializer}

    model = IResNet()
    state = model.state_dict()
    named = [name for name in state if name in weights]
    state.update({name: weights[name] for name in named})
    model.load_state_dict(state)

    # The folded convolutions and PReLU slopes lost their names in the export, so match
    # them by order: stem first, then conv1, conv2 and downsample of each block.
    convolutions = [model.conv1]
    for block in model.blocks():
        convolutions += [conv for conv in (block.conv1, block.conv2, block.downsample) if conv is not None]
    prelus = [model.prelu] + [block.prelu for block in model.blocks()]
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    prelu_nodes = [node for node in graph.node if node.op_type == "PRelu"]
    if len(conv_nodes) != len(convolutions) or len(prelu_nodes) != len(prelus):
        raise ValueError(f"{onnx_path.name} does not match the IResNet-50 layout")

    with torch.no_grad():
        for module, node in zip(convolutions, conv_nodes):
            module.weight.copy_(weights[node.input[1]])
            module.bias.copy_(weights[node.input[2]])
        for module, node in zip(prelus, prelu_nodes):
            module.weight.copy_(weights[node.input[1]].flatten())

    used = set(named) | {name for node in conv_nodes + prelu_nodes for name in node.input[1:]}
    unused = set(weights) - used
    if unused:
        raise ValueError(f"Unused tensors in {onnx_path.name}: {sorted(unused)[:5]}")
    return model


def faces_to_tensor(faces: np.ndarray) -> torch.Tensor:
    rgb = faces[..., ::-1].astype(np.float32)
    return torch.from_numpy((rgb - 127.5) / 127.5).permute(0, 3, 1, 2).contiguous()


class FaceCropDataset(Dataset):
    def __init__(self, image_paths: list[Path], boxes: np.ndarray, config: DetectionConfig):
        self.image_paths = image_paths
        self.boxes = boxes
        self.config = config

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = read_image(self.image_paths[index])
        face = crop_face(image, self.boxes[index], self.config.crop_margin, self.config.face_size)
        return faces_to_tensor(face[None])[0]


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    image_paths: list[Path],
    boxes: np.ndarray,
    detection_config: DetectionConfig,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(FaceCropDataset(image_paths, boxes, detection_config), batch_size=batch_size)
    model = model.to(device).eval()
    embeddings = [model(batch.to(device)).float().cpu() for batch in loader]
    embeddings = torch.cat(embeddings) if embeddings else torch.empty(0, 512)
    return nn.functional.normalize(embeddings, dim=1).numpy()


def onnx_agreement(model: IResNet, faces: np.ndarray, onnx_path: Path = ARCFACE_ONNX_WEIGHTS) -> float:
    """Lowest cosine similarity between our PyTorch embeddings and onnxruntime's for the same faces."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3  # the export says batch size 1 and warns on larger batches
    session = onnxruntime.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    inputs = faces_to_tensor(faces)
    reference = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})[0]
    with torch.no_grad():
        ours = model.cpu().eval()(inputs).numpy()
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    ours /= np.linalg.norm(ours, axis=1, keepdims=True)
    return float((reference * ours).sum(axis=1).min())


def load_embedding_model(checkpoint_path: Path, device: str = "cpu") -> tuple[IResNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = IResNet(embedding_dimension=checkpoint["config"]["embedding_dimension"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
