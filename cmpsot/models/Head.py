"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    Head.py
    model head class
"""
from .base_blocks import *


class CMPHead(nn.Module):

    def __init__(self, in_features=100):
        super().__init__()
        self.linear0 = nn.Linear(in_features=in_features, out_features=in_features // 2)
        self.bn0 = nn.BatchNorm1d(num_features=in_features // 2)
        self.linear1 = nn.Linear(in_features=in_features // 2, out_features=in_features)
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.linear2 = nn.Linear(in_features=in_features, out_features=1)
        self.act_h = nn.SiLU()    # nn.Sigmoid()
        self.act_o = nn.Sigmoid()

    def forward0(self, x):
        x = self.linear0(x)
        if x.shape[0] > 1:
            x = self.bn0(x)
        x = self.act_h(x)
        return x

    def forward1(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act_h(x)
        x = self.linear2(x)
        x = self.act_o(x)
        return x

    def forward(self, x):
        ori, imgs = x[0:1, :, 0, 0], x[1:, :, 0, 0]
        x = torch.abs(imgs - ori)
        x = self.forward0(x)
        return self.forward1(x)
