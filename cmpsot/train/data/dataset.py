"""
Author: LSH9832
For more information, visit https://github.com/LSH9832
Introduction:
    dataset.py
    load dataset
"""
import cv2
import numpy as np
import torch
from torch import distributed as dist
from glob import glob
import os
import random
from loguru import logger
import pickle
import zlib


def get_img_anno_dirs(dataset_dir):
    from glob import glob
    import os
    img_dirs = []
    anno_dirs = []
    for name in sorted(glob(os.path.join(dataset_dir, "*"))):
        if os.path.isdir(name) and os.path.isdir(os.path.join(name, "annotations")):
            img_dirs.append(name.replace("\\", "/"))
            anno_dirs.append(os.path.join(name, "annotations").replace("\\", "/"))
    return img_dirs, anno_dirs


class CMPDataset:
    """
    annotation format (xxxx.txt) -> (xxxx.jpg)

    xxxx.jpg  (relating image file)
    x y w h 0 (tracking object)
    x y w h 1 (another object with same class of tracking object)
    x y w h 2 (other object)
    """

    def __init__(self, image_dirs, annotation_dirs, image_type="jpg", preprocess=None, local_rank=0, world_size=1):

        self.local_rank = local_rank
        self.world_size = world_size

        self.img_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.anno_dirs = annotation_dirs if isinstance(annotation_dirs, list) else [annotation_dirs]

        self.anno_list = []
        [self.anno_list.append(sorted(glob(os.path.join(anno_dir, "*.txt")))) for anno_dir in self.anno_dirs]

        self.total_iter = None

        self.order = [i for i in range(len(self.anno_dirs))]
        self.iter_map = []
        self.direction = 1
        self.now_image_dir = ""
        self.now_anno_files = []
        self.cache = False
        self.cache_data = {}
        self.preprocess = preprocess

    def enable_cache(self, flag=True):
        self.cache = flag
        if self.local_rank == 0:
            logger.warning("using cache will occupy huge memory, "
                           "please make sure you have enough memory.")
        if self.world_size > 1:
            dist.barrier()

    def get_total_iter(self, real=False):
        if self.total_iter is None:
            total_iter = 0
            for this_list in self.anno_list:
                total_iter += len(this_list) - 1
            self.total_iter = total_iter

        if real:
            return self.total_iter
        else:
            return self.total_iter - self.total_iter % self.world_size

    def init_loader(self, rand=True):
        self.direction = 1
        self.order = [i for i in range(len(self.anno_dirs))]
        if not len(self.iter_map):
            self.iter_map = []
            [self.iter_map.append(i) if i % self.world_size == self.local_rank else None
             for i in range(self.get_total_iter(True))]
        self.now_image_dir = self.img_dirs[self.order[0]]
        if rand:
            self.direction = [-1, 1][random.randint(0, 1)]
            random.shuffle(self.order)
            random.shuffle(self.iter_map)

    def _get_data_from_file(self, file_name, now_image_dir):
        data_list = open(file_name, encoding="utf8").read().split("\n")
        output = {
            "image": None,
            "bbox": [],
            "label": [],
            "h": 0,
            "w": 0
        }
        start = True
        for this_data in data_list:
            if len(this_data):
                if start:
                    start = False
                    image_path = os.path.join(now_image_dir, this_data)
                    assert os.path.exists(image_path), image_path + " does not exist from label file %s" % file_name
                    output["image"] = cv2.imread(image_path)
                    output["h"], output["w"], _ = np.shape(output["image"])
                else:
                    *xywh, label = [int(item) for item in this_data.split(" ")]
                    output["bbox"].append(xywh)
                    output["label"].append(label)

        return output

    def output_format(self, ori_data, cmp_data):
        img_h, img_w, _ = np.shape(ori_data["image"])
        img_h2, img_w2, _ = np.shape(cmp_data["image"])
        ori_img = None
        score = [1., 0.5, 0.]
        cmp_imgs = []
        targets = []
        for label, xywh in zip(ori_data["label"], ori_data["bbox"]):
            if label == 0:
                x, y, w, h = xywh

                x = max(1, min(img_w - w - 1, x))
                y = max(1, min(img_h - h - 1, y))

                ori_img = ori_data["image"][y:min(y + h, ori_data["h"] - 1), x:min(x + w, ori_data["w"] - 1)]
                break

        for label, xywh in zip(cmp_data["label"], cmp_data["bbox"]):
            x, y, w, h = xywh

            x = max(1, min(img_w2 - w - 1, x))
            y = max(1, min(img_h2 - h - 1, y))

            cmp_imgs.append(cmp_data["image"][y:min(y + h, cmp_data["h"] - 1), x:min(x + w, cmp_data["w"] - 1)])
            targets.append([score[label]])
        if len(targets) == 1:
            targets *= 2
            cmp_imgs *= 2
        targets = torch.from_numpy(np.array(targets))
        if self.preprocess is not None:
            return self.preprocess(ori_img, cmp_imgs), targets
        return ori_img, cmp_imgs, targets

    def preload(self):
        from threading import Thread
        from multiprocessing import cpu_count
        if self.cache:
            logger.info("rank %d: preloading data to cache..." % self.local_rank)
            total_thread = cpu_count() // self.world_size
            self.cache = False
            self.init_loader(rand=False)
            total_iter = self.get_total_iter(True)

            self.now_iter = 0

            def preload_one_thread(thread_id):
                for i in range(total_iter):
                    if i % total_thread == thread_id:
                        if i % self.world_size == self.local_rank:
                            self.cache_data[i] = zlib.compress(pickle.dumps(self.get_data(i + 1)), level=5)
                            if i > self.now_iter:
                                print("\r%d/%d" % (i, total_iter), end="")
                                self.now_iter = i

            Threads = [Thread(target=preload_one_thread, args=(index, )) for index in range(total_thread)]
            [thread.start() for thread in Threads]
            [thread.join() for thread in Threads]
            if self.world_size > 1:
                dist.barrier()
            if self.local_rank == 0:
                print("\r%d/%d" % (total_iter, total_iter))
            self.cache = True
            if self.world_size > 1:
                dist.barrier()
            logger.info("rank %d: data preload done." % self.local_rank)

    def get_data(self, now_iter):
        try:
            now_iter = self.iter_map[(now_iter - 1) // self.world_size]
        except:
            logger.error("list out of range: %d %d %d" % (now_iter, len(self.iter_map), (now_iter - 1) // self.world_size))
            raise
        ori_ni = now_iter

        if self.cache and ori_ni in self.cache_data:
            # print("\r%d" % int(ori_ni in self.cache_data), end="")
            return pickle.loads(zlib.decompress(self.cache_data[ori_ni]))
        else:
            index = 0
            for i in self.order:
                if now_iter > len(self.anno_list[i]) - 1:
                    now_iter -= len(self.anno_list[i]) - 1
                    continue
                index = i
                now_iter = now_iter * self.direction - int(self.direction > 0)
                break

            now_image_dir = self.img_dirs[index]
            ori_img_anno_file = self.anno_list[index][now_iter].replace("\\", "/")
            cmp_img_anno_file = self.anno_list[index][now_iter + self.direction].replace("\\", "/")

            self.now_anno_files = [ori_img_anno_file, cmp_img_anno_file]

            ori_data = self._get_data_from_file(ori_img_anno_file, now_image_dir)
            cmp_data = self._get_data_from_file(cmp_img_anno_file, now_image_dir)

            ret = self.output_format(ori_data, cmp_data)
            if self.cache:
                self.cache_data[ori_ni] = zlib.compress(pickle.dumps(ret), level=5)
            return ret

    def get_current_anno_files(self):
        return self.now_anno_files
