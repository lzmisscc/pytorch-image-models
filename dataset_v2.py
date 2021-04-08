import json
import jsonlines
import tqdm
import random
import re
from random import shuffle
import PIL
from PIL import Image
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import lmdb
import cv2
import math

random.seed(100)
FLAG_TRAIN = True
train = 'data_v3/label_ensemble_clean_600w_100char.txt.lz'
# label = open(train, 'r').readlines()
# new_label = []
# for index, l in enumerate(label):
#     filename, content = l.strip().split('.png ')
#     new_label.append(f'{filename}.png\t{content}\n')
# with open(f"{train}.lz", "w") as f:
#     f.writelines(new_label)

# exit()
val = 'data_v3/trans_val.txt'
eval = 'data_v3/trans_eval_classify.txt'
lamdb_path = 'data_v3/data_v3_00000'
predict_ = "/data/lz/jiangming/pytorch_lmdb_noposi/results/lz_13_table_ocr_lmdb_896_budingW_noposi_0207.txt"

db = lmdb.open(lamdb_path, readonly=True)
txn = db.begin()


class Cus_Dataset(Dataset):
    def __init__(self, flag='train', transform=None) -> None:
        super().__init__()

        if flag == 'train':
            self.label = open(train, 'r').readlines()
        else:
            self.label = open(val, 'r').readlines()

        # t, f = [], []
        # for index, l in enumerate(self.label):
        #     filename, content = l.strip().split('\t')
        #     if '卐' in content:
        #         t.append(index)
        #     else:
        #         f.append(index)

        # self.res = random.choices(t, k=100000) + random.choices(f, k=100000)
        # shuffle(self.res)
        shuffle(self.label)
        self.transform = transform
        self.flag = flag

    def __len__(self) -> int:
        if self.flag == 'eval':
            return len(self.label[:5000])
        return len(self.label)

    def __getitem__(self, index: int):
        # index = self.res[index]
        l = self.label[index]
        filename, content = l.strip().split('\t')
        im = txn.get(filename.encode(), False)
        if im == False:
            return self[random.choice(range(len(self)))]

        im = cv2.imdecode(np.frombuffer(im, np.uint8), 3)
        im = Image.fromarray(im)
        W, H = im.size
        im = im.resize((math.ceil(W*(64/H)), 64))
        new_im = Image.new(mode="RGB", size=(
            224, 244), color=(255, 255, 255))
        new_im.paste(im, (random.choice(range(0, 50)),
                          random.choice(range(0, 50))))

        im = new_im

        if self.transform:
            im = self.transform(new_im)

        if '卐' in content:
            label = 1
        else:
            label = 0
        if self.flag == 'train' or FLAG_TRAIN:
            return im, label
        else:
            return im, label, filename


class Cus_Dataset_v2(Dataset):
    def __init__(self, flag='train', transform=None) -> None:
        super().__init__()

        if flag == 'train':
            self.label = open(train, 'r').readlines()
        else:
            self.label = open(val, 'r').readlines()

        self.transform = transform
        self.flag = flag

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int):
        # index = self.res[index]
        l = self.label[index]
        filename, content = l.strip().split('\t')
        im = txn.get(filename.encode(), False)
        if im == False:
            return self[random.choice(range(len(self)))]

        im = cv2.imdecode(np.frombuffer(im, np.uint8), 3)
        im = Image.fromarray(im)
        W, H = im.size
        im = im.resize((math.ceil(W*(64/H)), 64))
        new_im = Image.new(mode="RGB", size=(
            224, 244), color=(255, 255, 255))
        new_im.paste(im, (random.choice(range(0, 100)),
                          random.choice(range(0, 100))))

        im = new_im

        if self.transform:
            im = self.transform(new_im)

        if '卐' in content:
            label = 1
        else:
            label = 0
        return im, label, filename, content


class Cus_Dataset_v3(Dataset):
    def __init__(self, flag='train', transform=None) -> None:
        super().__init__()
        self.filenames = []
        if flag == 'train':
            self.label = open(train, 'r').readlines()
        elif flag == 'val':
            self.label = open(val, 'r').readlines()
        elif flag == 'eval':
            self.label = open(eval, 'r').readlines()
        elif flag == 'predict':
            self.label = open(predict_, 'r').readlines()
            res = []
            for i in self.label:
                name, content = i.split('.png ')
                res.append(f"{name}.png\t{content}")
                self.filenames.append(name)
            self.label = res

        # t, f = [], []
        # for index, l in enumerate(self.label):
        #     filename, content = l.strip().split('\t')
        #     if '卐' in content:
        #         t.append(index)
        #     else:
        #         f.append(index)

        # self.res = random.choices(t, k=100000) + random.choices(f, k=100000)
        # shuffle(self.res)
        self.transform = transform
        self.flag = flag
        print(f"use Cus_Dataset_v3:{len(self)}")
    def __len__(self) -> int:
        return len(self.label[:1000])

    def __getitem__(self, index: int):
        # index = self.res[index]
        l = self.label[index]
        filename, content = l.strip().split('\t')
        im = txn.get(filename.encode(), False)
        if im == False:
            return self[random.choice(range(len(self)))]

        im = cv2.imdecode(np.frombuffer(im, np.uint8), 3)
        im = Image.fromarray(im)
        W, H = im.size
        im = im.resize((math.ceil(W*(64/H)), 64))
        new_im = Image.new(mode="RGB", size=(
            224, 244), color=(255, 255, 255))
        new_im.paste(im, (random.choice(range(0, 50)),
                          random.choice(range(0, 50))))

        im = new_im

        if self.transform:
            im = self.transform(new_im)

        if '卐' in content:
            label = 1
        else:
            label = 0
        if self.flag == 'train':
            return im, label
        else:
            return im, label, filename, content


if __name__ == '__main__':
    d = Cus_Dataset_v3('predict')
    print(d[3])
