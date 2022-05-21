"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    Conv.py
    several common convolution layers used in this model
"""
from torch import nn


class BaseConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            step=1,
            kernel=3,
            pad=None,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros"
    ):
        super().__init__()
        pad = int((kernel - 1) / 2) if pad is None else pad

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=step,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):

    def __init__(self, in_channels, out_channels, step=1, kernel=3, groups=1, **kwargs):
        super().__init__()
        self.depth_conv = BaseConv(
            in_channels=in_channels,
            out_channels=in_channels,
            step=step,
            kernel=kernel,
            groups=in_channels,
            **kwargs
        )
        self.wise_conv = BaseConv(
            in_channels=in_channels,
            out_channels=out_channels,
            step=1,
            kernel=1,
            groups=groups,
            **kwargs
        )

    def forward(self, x):
        return self.wise_conv(self.depth_conv(x))
