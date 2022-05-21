"""
Usage:
python3 export_trt.py -c train_output/last.pth -w 4
"""
import argparse
import os
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from cmpsot import load_exp


def make_parser():
    parser = argparse.ArgumentParser("tensorrt deploy")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("-w", '--workspace', type=int, default=8, help='max workspace size(GB)')
    parser.add_argument("-b", '--batch', type=int, default=3, help='max batch size in detect')
    return parser


@logger.catch
def main():
    output_dir_name = "./tensorrt_output"

    args = make_parser().parse_args()
    
    exp = load_exp(args.ckpt)
    model = exp.load_model()
    
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    x = torch.ones(args.batch, 3, 224, 224).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=((1 << 30) * args.workspace),
        max_batch_size=args.batch,
    )
    
    os.makedirs(output_dir_name, exist_ok=True)
    py_trt_model_path = os.path.join(output_dir_name, "model_trt.pth")
    torch.save(model_trt.state_dict(), py_trt_model_path)
    logger.success("TensorRT model for python inference is saved to %s." % py_trt_model_path)
    engine_file = os.path.join(output_dir_name, "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
    logger.success("TensorRT engine file for C++ inference is saved to %s." % engine_file)


if __name__ == "__main__":
    main()
