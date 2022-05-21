def get_default_training_params():
    return {
        # model settings
        "model_size": "s",  # nano, tiny, s, m, l, xl

        # device settings
        "cuda_devices": "full",  # 0 (or 1, 2, ...) for single gpu, [0, 1, ...] for multi-gpus, "full" for all gpus
        "enable_cudnn_benchmark": False,
        "num_threads": 1,

        # training settings
        "max_epoch": 300,
        "fp16": False,  # amp
        "start_epoch": 0,
        "warmup_epochs": 5,

        "lr_per_img": 0.0002,
        "start_lr_rate": 0.0,
        "final_lr_rate": 0.05,

        "momentum": 0.9,
        "weight_decay": 0.0005,

        # data settings
        "dataset_path": "E:/Dataset/SOT_UAV123",
        "cache": True,  # cache data into memory, make sure enough memory remained
        "preload": True,  # cache data (true:before/false:at) first training epoch

        "output_dir": "./train_output",
        "pretrained_weight_file": "./train_output/last.pth",

        # visual settings
        "iter_per_show": 100
    }


def create_default_training_params(filename="train_settings.yaml"):
    open(filename, "w", encoding="utf8").write(
        """# model settings
model_size: "s"    # nano, tiny, s, m, l, xl

# device settings
cuda_devices: "full"   # 0 (or 1, 2, ...) for single gpu, [0, 1, ...] for multi-gpus, "full" for all gpus
enable_cudnn_benchmark: false
num_threads: 1

# training settings
max_epoch: 300
fp16: false       # amp
start_epoch: 0
warmup_epochs: 5

lr_per_img: 0.0002
start_lr_rate: 0.0
final_lr_rate: 0.05

momentum: 0.9
weight_decay: 0.0005

# data settings
dataset_path: "E:/Dataset/SOT_UAV123"
cache: true         # cache data into memory, make sure enough memory remained
preload: true       # cache data (true:before/false:at) first training epoch

output_dir: "./train_output"
pretrained_weight_file: "./train_output/last.pth"

# visual settings
iter_per_show: 100"""
    )
