"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    base_blocks.py
    several common network blocks used in this model
"""
from .Conv import BaseConv, DWConv, nn
import torch


class Focus(nn.Module):

    def __init__(self, in_channels, out_channels, step=1, kernel=1):
        super().__init__()
        self.conv = BaseConv(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel=kernel,
            step=step
        )

    def forward(self, x):
        tlx = x[..., 0::2, 0::2]
        trx = x[..., 0::2, 1::2]
        blx = x[..., 1::2, 0::2]
        brx = x[..., 1::2, 1::2]

        return self.conv(torch.cat((tlx, trx, blx, brx), dim=1))


class ResLayer(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, kernel=1, step=1)
        self.layer2 = BaseConv(mid_channels, in_channels, kernel=3, step=1)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel=1, step=1)
        self.conv2 = Conv(hidden_channels, out_channels, kernel=3, step=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
