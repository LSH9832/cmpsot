"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    trainer.py
    main struct of training this model
"""
import os
import shutil
import time
from loguru import logger
import yaml
import random

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist


class TrainerParams:

    def __init__(self):
        self.max_epoch = 300
        self.fp16 = False

        self.total_iter = 0
        self.now_iter = 0
        self.now_epoch = 0
        self.start_epoch = 0

        self.warmup_epochs = 5
        self.lr_per_img = 0.01 / 64
        self.start_lr_rate = 0
        self.final_lr_rate = 0.05
        self.now_lr = 0

        self.momentum = 0.9
        self.weight_decay = 5e-4

        self.output_dir = "./output"
        self.pretrained_weight_file = None

        self.iter_per_show = 10

    def load_params_from_dict(self, data):
        for name in data:
            # print(name, data[name])
            setattr(self, name, data[name]) if name in self.__dict__ else None  # print("param \"%s\" not found" % name)

    def load_params_from_file(self, file_name):
        # print(self.__dict__)
        if not os.path.isfile(file_name):
            return False
        data = yaml.load(open(file_name), yaml.Loader)
        self.load_params_from_dict(data)
        return True

    def count_now_lr(self, batch_size=1):
        max_lr = self.lr_per_img * batch_size
        start_lr = self.start_lr_rate * max_lr

        if self.now_epoch <= self.warmup_epochs:
            epoch_start_lr = start_lr + \
                             (max_lr - start_lr) * (self.now_epoch - 1) / self.warmup_epochs
            step_per_iter = (max_lr - start_lr) / (self.total_iter * self.warmup_epochs)
        else:
            epoch_start_lr = max_lr * \
                             (
                                     1 -
                                     (1 - self.final_lr_rate) *
                                     (self.now_epoch - self.warmup_epochs - 1) /
                                     (self.max_epoch - self.warmup_epochs)
                             )
            step_per_iter = max_lr * (self.final_lr_rate - 1) / (self.total_iter * (self.max_epoch - self.warmup_epochs))

        lr = (epoch_start_lr + self.now_iter * step_per_iter) * random.randint(95, 105) / 100.0
        # print(max_lr, step_per_iter, lr)
        return lr


class CMPTrainer(TrainerParams):

    def __init__(
            self,
            model: nn.Module,
            optimizer,
            loss,
            dataloader,
            device,
            preprocess,
            model_size,
            local_rank=0,
            world_size=1
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.total_iter = self.dataloader.get_total_iter()
        self.device = device
        self.preprocess = preprocess
        self.model_size = model_size
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.start_time = 0

        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        self.world_size = world_size

    def train(self):
        def before_train():
            os.makedirs(self.output_dir, exist_ok=True)

            torch.cuda.set_device(self.device)
            self.model.to(self.device)

            if self.world_size > 1:
                dist.barrier()
            if self.pretrained_weight_file is not None and os.path.isfile(self.pretrained_weight_file):
                self.resume_train_from_ckpt(self.pretrained_weight_file)

            if self.is_distributed:
                self.model = DDP(
                    module=self.model,
                    device_ids=[self.device],
                    broadcast_buffers=False,
                )

            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
            self.model.train()

            self.start_time = time.time()

        def train_one_epoch():
            def before_epoch():
                if self.world_size > 1:
                    if self.local_rank == 0:
                        logger.info("wait for other subprocesses")
                    dist.barrier()

                if self.local_rank == 0:
                    logger.info("start train epoch %d" % self.now_epoch)
                self.dataloader.init_loader(rand=True)

            def train_one_iter():
                def before_iter():
                    pass

                def train_this_iter():
                    iter_start_time = time.time()

                    # get data and label
                    *inputs, targets = self.dataloader.get_data(self.now_iter)

                    inputs = self.preprocess(*inputs)

                    data_time = time.time() - iter_start_time
                    inputs = inputs.float().to(self.device)
                    targets = targets.float().to(self.device)
                    targets.requires_grad = False

                    # update learning rate
                    lr = self.count_now_lr(batch_size=targets.shape[0])
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                    # forward and get loss
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        outputs = self.model(inputs)

                    loss = self.loss(outputs, targets)

                    # print(self.now_iter, "before backward") if self.local_rank == 0 else None

                    # backward and optimize
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # print(self.now_iter, "after backward") if self.local_rank == 0 else None

                    # print result of this iter
                    iter_time = time.time() - iter_start_time
                    if self.now_iter % self.iter_per_show == 0:
                        logger.info(
                            "rank:%s epoch:%d/%d iter:%d/%d loss:%.5f data_time:%.3fs iter_time:%.3fs lr:%.10f ETA:%s" % (
                                self.local_rank,
                                self.now_epoch,
                                self.max_epoch,
                                self.now_iter,
                                self.total_iter,
                                float(loss),
                                data_time,
                                iter_time,
                                lr,
                                self.count_rest_time()
                            )
                        )

                def after_iter():
                    pass

                if (self.now_iter - 1) % self.world_size == self.local_rank:
                    before_iter()
                    train_this_iter()
                    after_iter()

            def after_epoch():
                if self.world_size > 1:
                    logger.info("rank %d finished epoch %d" % (self.local_rank, self.now_epoch))
                    dist.barrier()
                if self.local_rank == 0:
                    self.save_ckpt()

            before_epoch()
            for self.now_iter in range(1, self.total_iter + 1):
                train_one_iter()
            after_epoch()

        def after_train():
            if self.local_rank == 0:
                logger.info("training time cost: %s" % self.count_rest_time(int(time.time() - self.start_time)))

        before_train()
        for self.now_epoch in range(self.start_epoch + 1, self.max_epoch + 1):
            train_one_epoch()
        after_train()

    def save_ckpt(self):
        ckpt_model = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        save_data = {
            "model": ckpt_model,
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.now_epoch,
            "model_size": self.model_size
        }
        pth_file_name = "%s/epoch_%d.pth" % (self.output_dir, self.now_epoch)
        pth_last_file_name = "%s/last.pth" % self.output_dir
        torch.save(save_data, pth_file_name)
        shutil.copy(pth_file_name, pth_last_file_name)
        logger.info("weight saved to %s" % pth_file_name)

    def resume_train_from_ckpt(self, ckpt_name):
        if self.local_rank == 0:
            logger.info("resume train from file %s" % ckpt_name.replace("\\", "/"))
        ckpt = torch.load(ckpt_name, map_location="cpu")
        if "model" in ckpt:
            try:
                self.model.load_state_dict(ckpt["model"])
            except:
                from ..utils import load_checkpoint
                self.model = load_checkpoint(self.model, ckpt["model"])
            self.model.to(self.device)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except:
                pass
        if "epoch" in ckpt:
            self.start_epoch = ckpt["epoch"]
        del ckpt

    def count_rest_time(self, rest_time: int = None):
        if rest_time is None:
            cost_time = time.time() - self.start_time
            time_per_iter = cost_time / ((self.now_epoch - self.start_epoch - 1) * self.total_iter + self.now_iter)
            rest_iter = (self.total_iter - self.now_iter) + (self.max_epoch - self.now_epoch) * self.total_iter
            rest_time = int(rest_iter * time_per_iter)
        jz = [60, 60, 24]
        time_list = []
        for j in jz:
            time_list.append(rest_time % j)
            rest_time //= j
        time_list.append(rest_time)
        time_list.reverse()
        return "%s%s:%s:%s" % (
            ("%d %s " % (
                time_list[0],
                "days" if time_list[0] > 1 else "day"
            )) if time_list[0] > 0 else "",
            str(time_list[1]).zfill(2),
            str(time_list[2]).zfill(2),
            str(time_list[3]).zfill(2)
        )
