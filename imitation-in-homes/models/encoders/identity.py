"""
A generic model wrapper for dummy identity encoders.
"""

import pathlib
from typing import Union
from torch import nn
import torch
from .abstract_base_encoder import AbstractEncoder


class IdentityEncoder(AbstractEncoder):
    def __init__(
        self,
        model_name,
        pretrained: bool = True,
        weight_path: Union[None, str, pathlib.Path] = None,
    ):
        super().__init__()
        self._model_name = model_name

        self.model = nn.Identity()

        # Use a placeholder parameter if the model has no parameters
        if len(list(self.model.parameters())) == 0:
            self.placeholder_param = nn.Parameter(torch.zeros(1, requires_grad=True))
        else:
            self.placeholder_param = None

    def transform(self, x):
        return x

    @property
    def feature_dim(self):
        return 0

    def to(self, device):
        self.model.to(device)
        if self.placeholder_param is not None:
            self.placeholder_param.to(device)
        return self

    def forward(self, x):
        return self.transform(x)
