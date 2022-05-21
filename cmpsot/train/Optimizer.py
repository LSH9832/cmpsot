"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    Optimizer.py
    optimizer for training this model
"""
import torch
from torch import nn


def get_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=lr, momentum=momentum, nesterov=True
    )
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": weight_decay}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})

    return optimizer
