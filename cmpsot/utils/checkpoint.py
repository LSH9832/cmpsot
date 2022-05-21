"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
"""
import torch
from torch import nn

import os
from loguru import logger


def load_checkpoint_by_name(model: nn.Module, file_name: str):
    if os.path.exists(file_name):
        ckpt = torch.load(file_name)["model"]
        model = load_checkpoint(model, ckpt)
    return model


def load_checkpoint(model, checkpoint: dict):
    ckpt = checkpoint
    model_state_dict = model.state_dict()
    params_dict = {}
    for name, params in model_state_dict.items():
        if name in ckpt:
            if params.shape == ckpt[name].shape:
                params_dict[name] = ckpt[name]
            else:
                logger.info("shape of %s not match!" % name)
        else:
            logger.info("no params named %s" % name)
    model.load_state_dict(params_dict, strict=False)
    return model


def save_checkpoint(file_name, model: nn.Module, optimizer, model_size, classes, epoch):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer,
        "model_size": model_size,
        "classes": classes,
        "epoch": epoch
    }
    torch.save(data, file_name)
