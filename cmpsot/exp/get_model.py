"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    get_model.py
    add other operations of this_model based on base_exp.py
    - get_trainer/load_trainer
    - determine specific params of this model and then load model
    - choose device
"""
import os
import torch

from .base_exp import Exp
from ..train import CMPTrainer
from ..models import get_input


class get_exp(Exp):

    def __init__(self, model_size="s"):
        params = {
            "nano": [1, 16, 256, True],
            "tiny": [1, 16, 256, False],
            "s": [1, 32, 256, False],
            "m": [2, 64, 512, False],
            "l": [3, 128, 1024, False],
            "xl": [4, 256, 2048, False]
        }
        super(get_exp, self).__init__(*params[model_size])
        self.model_size = model_size

    def get_trainer(
            self,
            train_image_dir,
            train_anno_dir,
            image_type="jpg",
            cache=False,
            preload=False,
            device=0,
            local_rank=0,
            world_size=1
    ):
        if not torch.cuda.is_available():
            import warnings
            device = "cpu"
            warnings.warn("Cuda is not available, use cpu instead.")
        trainer = CMPTrainer(
            model=self.model if self.model is not None else self.load_model(),
            optimizer=self.optimizer if self.optimizer is not None else self.load_optimizer(),
            loss=self.loss if self.loss is not None else self.load_loss(),
            dataloader=self.dataloader
            if self.dataloader is not None
            else self.load_dataloader(
                image_dirs=train_image_dir,
                anno_dirs=train_anno_dir,
                image_type=image_type,
                cache=cache,
                preload=preload,
                local_rank=local_rank,
                world_size=world_size
            ),
            device=device,
            preprocess=get_input,
            model_size=self.model_size,
            local_rank=local_rank,
            world_size=world_size
        )
        return trainer


def load_exp(weight_name):
    assert os.path.isfile(weight_name), "No such file"
    ckpt = torch.load(weight_name)
    for name in ["model", "model_size"]:
        assert name in ckpt, "param %s not exist!"

    exp = get_exp(ckpt["model_size"])
    exp.load_model()
    exp.model.load_state_dict(ckpt["model"])
    return exp
