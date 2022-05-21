"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    Loss.py
    loss function of this model
"""
from torch import nn


class CMPLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()  # nn.CrossEntropyLoss()    # nn.MSELoss()

    def get_loss(self, output, target):
        loss = self.loss(output, target)
        return loss

    def forward(self, output, target):
        return self.get_loss(output, target)


# if __name__ == '__main__':
#     # import torch
#     a = CMPLoss()
