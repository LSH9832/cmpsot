"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    IOprocess.py
    some input/output process of this model
"""
import cv2
import torch


def get_input(ori_img, imgs: list):
    ori_img = preprocess(ori_img)
    imgs = preprocess(*imgs)
    ori = ori_img
    imgs = torch.cat(imgs, dim=0) if isinstance(imgs, list) else torch.cat([imgs] * 2, dim=0)
    return torch.cat([ori, imgs], dim=0)


def preprocess(*imgs):
    output = []
    for img in imgs:
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
        img = img.float()
        output.append(img)
    return output if len(output) > 1 else output[0] if len(output) else False


if __name__ == '__main__':
    import numpy as np
    out = preprocess(np.zeros([2, 2, 3]))
    print(out.shape)
