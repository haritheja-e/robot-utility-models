import torch
import torch.nn as nn
import torchvision
import einops
from torchvision.transforms.v2 import Normalize

import decord
from pathlib import Path

NORMALIZER = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

RED_GRIPPER_10PCT_VALUE = 0.18
RED_GRIPPER_90PCT_VALUE = 0.68
IMAGE_SIZE = 256


class NewGripperModel(nn.Module):
    def __init__(self, dropout: float = 0.5, bins: int = 15):
        super(NewGripperModel, self).__init__()

        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.resnet18.fc = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4 * (bins + 1)),
        )
        self.bins = bins + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18(x)
        x = self.regressor(x)
        x = einops.rearrange(x, "b (points bins) -> b points bins", points=4)
        x = torch.softmax(x, dim=2)
        return x

    @staticmethod
    def expected_value(probs: torch.Tensor) -> torch.Tensor:
        bins_shape = probs.shape[-1]
        bins = torch.linspace(0.0, 1.0, bins_shape, device=probs.device)
        return torch.sum(probs * bins, dim=-1)


def video_frames_extractor(video_path: Path):
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = []
    for i in range(len(vr)):
        frames.append(vr[i])
    frames_tensor = torch.stack(frames)
    frames_tensor = frames_tensor / 255.0
    frames_tensor = einops.rearrange(frames_tensor, "t h w c -> t c h w")
    frames_tensor = NORMALIZER(frames_tensor)
    return frames_tensor


@torch.no_grad()
def extract_gripper_value_from_pretrained_model_and_path(
    model: NewGripperModel, video_path: Path, normalize: bool = True
):
    frames = video_frames_extractor(video_path)
    return extract_gripper_value_from_pretrained_model_and_frames(
        model, frames, normalize
    )


@torch.no_grad()
def extract_gripper_value_from_pretrained_model_and_frames(
    model: NewGripperModel, frames: torch.Tensor, normalize: bool = True
):
    gripper_value_probs = model(frames)
    gripper_value = NewGripperModel.expected_value(gripper_value_probs)
    gripper_value = torch.norm(gripper_value[:,:2] - gripper_value[:,2:], p=2, dim=-1)
    if normalize:
        gripper_value = (gripper_value - RED_GRIPPER_10PCT_VALUE) / (
            RED_GRIPPER_90PCT_VALUE - RED_GRIPPER_10PCT_VALUE
        )
    return gripper_value
