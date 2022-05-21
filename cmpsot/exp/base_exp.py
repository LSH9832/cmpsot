"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    base_exp.py
    common operations of this_model
    - get_model/load_model
    - get_optimizer/load_optimizer
    - get_loss_function/load_loss_function
    - get_dataset/load_dataset
    - inference
    - choose device
"""
import torch
from torch import nn
from ..models import CMPSot, get_input
from ..train import CMPOptimizer, CMPDataset, CMPLoss


class Exp:

    def __init__(self, depth=3, width=32, length=100, depthwise=True, half=False):
        self.depth = depth
        self.width = width
        self.length = length
        self.depthwise = depthwise
        self.half = half
        self.model = None
        self.optimizer = None
        self.loss = None
        self.dataloader = None
        self.device = "cpu"

    def compare(self, img, imgs: list, show=False):
        input_data = get_input(img, imgs)

        if not self.device == "cpu":
            input_data = input_data.cuda(self.device)
            if self.half:
                input_data = input_data.half()
        if self.model.training:
            return self.model(input_data, show=show)
        else:
            with torch.no_grad():
                return self.model(input_data, show=show)

    #######################################
    def load_model(self):

        def init_net(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.model = CMPSot(
            depth=self.depth,
            width=self.width,
            length=self.length,
            depthwise=self.depthwise
        )
        self.model.apply(init_net)
        return self.model

    def load_optimizer(self):
        self.optimizer = CMPOptimizer(self.model, lr=0)
        return self.optimizer

    def load_loss(self):
        self.loss = CMPLoss()
        return self.loss

    def load_dataloader(
            self,
            image_dirs,
            anno_dirs,
            cache=False,
            preload=False,
            image_type="jpg",
            local_rank=0,
            world_size=1
    ):
        assert len(image_dirs) == len(anno_dirs), "number of image dirs and annotation dirs must be equal!"
        self.dataloader = CMPDataset(
            image_dirs=image_dirs,
            annotation_dirs=anno_dirs,
            image_type=image_type,
            local_rank=local_rank,
            world_size=world_size
        )
        self.dataloader.enable_cache(cache)
        if preload:
            self.dataloader.preload()
        return self.dataloader

    #######################################
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def cuda(self, device=None):
        self.device = device
        self.model.cuda(device)
        if self.half:
            self.model.half()

    def cpu(self):
        self.device = "cpu"
        self.model.cpu()

    #######################################
    def __call__(self, img, imgs: list, show=False):
        return self.compare(img, imgs, show=show)