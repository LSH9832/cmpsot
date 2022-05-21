"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    model/__init__.py
    model class
"""
from .Backbone import CMPBackbone
from .Head import CMPHead

from .IOprocess import *
from .base_blocks import *


class CMPSot(nn.Module):

    def __init__(self, depth=3, width=32, length=100, depthwise=True):
        super().__init__()
        self.backbone = CMPBackbone(
            depth=depth,
            width=width,
            length=length,
            depthwise=depthwise
        )
        self.head = CMPHead(in_features=length)

    def forward(self, x, show=False):
        print("input shape:", x.shape) if show else None
        x = self.backbone(x)
        print("backbone output shape:", x.shape) if show else None
        result = self.head(x)   # [:, 0]
        print("head output shape:", result.shape, result) if show else None
        return result
