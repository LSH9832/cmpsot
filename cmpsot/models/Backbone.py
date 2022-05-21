"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    Backbone.py
    model backbone class
"""
from .base_blocks import *


class CMPBackbone(nn.Module):

    def __init__(self, depth=3, width=32, length=100, depthwise=True):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.stem = Focus(3, width)  # 3 x 224 x 224 -> 32 x 112 x 112

        #  -> 64 x 56 x 56
        self.net0 = nn.Sequential(
            Conv(width, width * 2, step=2, kernel=1),
            *[Bottleneck(width * 2, width * 2) for _ in range(depth)]
        )

        # -> 128 x 28 x 28
        self.net1 = nn.Sequential(
            Conv(width * 2, width * 4, step=2, kernel=1),
            *[Bottleneck(width * 4, width * 4) for _ in range(depth)]
        )

        # -> 256 x 14 x 14
        self.net2 = nn.Sequential(
            Conv(width * 4, width * 8, step=2, kernel=1),
            *[Bottleneck(width * 8, width * 8) for _ in range(depth)]
        )

        # -> 512 x 7 x 7
        self.net3 = nn.Sequential(
            Conv(width * 8, width * 16, step=2, kernel=1),
            *[Bottleneck(width * 16, width * 16) for _ in range(depth)]
        )

        # self.reduce = nn.Sequential(
        #     Conv(width * 16, length * 2, step=1, kernel=7, pad=0),
        #     Conv(length * 2, length, step=1, kernel=1),
        #     Bottleneck(length, length, depthwise=True)
        # )

        self.reduce = nn.Sequential(
            Conv(width * 16, length, step=1, kernel=3, pad=0),
            Conv(length, length, step=1, kernel=3, pad=0),
            Conv(length, length, step=1, kernel=3, pad=0),
            Bottleneck(length, length, depthwise=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.net0(x)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        y = self.reduce(x)
        return y
