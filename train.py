"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    train.py
    no need for any changes of this file, changing train_settings.yaml is all you need to do
"""
import cmpsot

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
import torch.backends.cudnn as cudnn

from datetime import timedelta
from loguru import logger
from platform import system
import os


def train(
        local_rank=0,
        world_size=1,
        device: int or list = 0,
        dist_url=f"tcp://127.0.0.1:58081",
        model_size="s",
        train_settings="./train_settings.yaml",
        dataset_dir="E:/Dataset/SOT_UAV123"
):
    n_th = train_settings["num_threads"] if isinstance(train_settings, dict) else 1
    cache = train_settings["cache"] if isinstance(train_settings, dict) else True
    preload = train_settings["preload"] if isinstance(train_settings, dict) else True
    torch.set_num_threads(n_th)
    device = device[local_rank] if isinstance(device, list) else device
    torch.cuda.set_device(device)

    train_image_dirs, train_anno_dirs = cmpsot.get_img_anno_dirs(dataset_dir)

    if world_size > 1:
        try:
            dist.init_process_group(
                backend="nccl" if system() == "Linux" else "gloo",
                init_method=dist_url,
                world_size=world_size,
                rank=local_rank,
                timeout=timedelta(minutes=30),
            )
        except Exception:
            logger.error("Process group URL: {}".format(dist_url))
            raise

        dist.barrier()
        if isinstance(train_settings, dict):
            if train_settings["enable_cudnn_benchmark"]:
                cudnn.benchmark = True

    comparer = cmpsot.get_exp(model_size=model_size)
    trainer = comparer.get_trainer(
        train_image_dir=train_image_dirs,
        train_anno_dir=train_anno_dirs,
        cache=cache,
        preload=preload,
        image_type="jpg",
        device=device,
        local_rank=local_rank,
        world_size=world_size
    )
    if isinstance(train_settings, str):
        trainer.load_params_from_file(train_settings)
    elif isinstance(train_settings, dict):
        trainer.load_params_from_dict(train_settings)
    trainer.train()


def launch(
        devices: list = None,
        model_size="s",
        train_settings="./train_settings.yaml",
        dataset_dir="E:/Dataset/SOT_UAV123"
):
    if devices is None:
        devices = [i for i in range(torch.cuda.device_count())]
    elif isinstance(devices, int) or len(devices) == 1:
        train(
            train_settings=train_settings,
            dataset_dir=dataset_dir
        )
        return
    world_size = len(devices)

    dist_url = f"tcp://127.0.0.1:{cmpsot.find_free_port()}"
    start_method = "spawn"

    mp.start_processes(
        train,
        nprocs=len(devices),
        args=(
            world_size,
            devices,
            dist_url,
            model_size,
            train_settings,
            dataset_dir
        ),
        daemon=False,
        start_method=start_method,
    )


def main(settings_file_name):

    import yaml
    from tabulate import tabulate

    assert os.path.exists(settings_file_name), "file %s not exist!" % os.path.abspath(settings_file_name)

    data = yaml.load(open(settings_file_name), yaml.Loader)

    os.makedirs(data["output_dir"], exist_ok=True)
    
    logger.add(
        os.path.join(data["output_dir"], "train_log.txt"),
        format=("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    )

    logger.info("training settings:\n%s" % "{}".format(
        tabulate(
            [(name, data[name]) for name in data],
            headers=["Keys", "Values"],
            tablefmt="fancy_grid"
        )
    ))

    model_size = data["model_size"]
    dataset_path = data["dataset_path"]
    cuda_devices = data["cuda_devices"]      # [i for i in range(torch.cuda.device_count())]

    if cuda_devices == "full":
        cuda_devices = [i for i in range(torch.cuda.device_count())]
    elif isinstance(cuda_devices, int):
        assert cuda_devices < torch.cuda.device_count(), "device %d not found!" % cuda_devices
        cuda_devices = [cuda_devices]
    else:
        assert isinstance(cuda_devices, list) or isinstance(cuda_devices, tuple), "devices format not correct!"
        for this_device in cuda_devices:
            assert this_device < torch.cuda.device_count(), "device %d not found!" % this_device

    launch(
        devices=cuda_devices,
        model_size=model_size,
        train_settings=data,
        dataset_dir=dataset_path
    )


if __name__ == '__main__':
    settings_file = "./train_settings.yaml"
    main(settings_file)
