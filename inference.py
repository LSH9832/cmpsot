"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
"""
from cmpsot import load_exp
import cv2
from glob import iglob


def test(weight_name):

    ori_image = "assets/001.jpg"
    cmp_imgs = ["assets/002.jpg", "assets/003.jpg", "assets/004.jpg"]

    ori_image = "assets/car1.jpg"
    cmp_imgs = ["assets/car2.jpg", "assets/car3.jpg", "assets/car4.jpg"]

    inferencer = load_exp(weight_name)
    # inferencer.cuda()
    inferencer.eval()

    # import torch
    # torch.save(inferencer.model.backbone.state_dict(), "backbone.pth")
    # torch.save(inferencer.model.head.state_dict(), "head.pth")

    ori_image = cv2.imread(ori_image)
    cmp_imgs = [cv2.imread(img_name) for img_name in cmp_imgs]

    result = inferencer(ori_image, cmp_imgs)

    print(result)


for name in sorted(iglob("./train_output/last.pth")):
    print(name)
    test(name)
    print()
